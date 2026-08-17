"""
test_workspace_isolation.py — Workspace isolation is enforced in the data/
query layer (SQLite workspace_id + documents_in_workspace filtering), not
just by the frontend never displaying the wrong papers.

Runs entirely against the ISOLATED Phase 8 test Chroma/SQLite fixture (see
conftest.py) — never backend/chroma_store or the real data/registry.db.
Mocks only the Groq network boundary for the one generation test.

Covers:
  1. Creating a workspace never touches/clears another workspace's documents.
  2. Uploading a PDF while workspace_id=A makes it visible ONLY under
     GET /documents?workspace_id=A, never under workspace B.
  3. /query is rejected at the DATA layer (not just trusting the request) if
     asked to use a document_id that doesn't actually belong to the claimed
     workspace_id — even though the document_id itself is valid.
  4. A normal, correctly-scoped query against a real uploaded document still
     produces exactly ONE LLM call and a grounded answer.
  5. /sessions (conversations) are workspace-filterable the same way.
"""
from pathlib import Path
from unittest.mock import patch
import json

import pytest
from fastapi.testclient import TestClient

import config
from main import app
from database import documents_in_workspace, list_documents, list_sessions
from query_transform import start_call_tracking, get_call_log
from test_auth_helpers import register

client = TestClient(app)

FIXTURE_PDF = Path(__file__).parent / "tests" / "fixtures" / "attention.pdf"

# All tests in this file operate as a single authenticated user (workspace
# isolation here is BETWEEN workspaces this same user owns, not between
# different users — see test_authorization.py for cross-USER adversarial
# tests). Registered once per module, deferred to setup (not collection)
# time via this autouse fixture, same reasoning conftest.py documents for
# why side effects must never run at bare module scope.
HEADERS: dict = {}


@pytest.fixture(autouse=True, scope="module")
def _module_auth():
    global HEADERS
    _, HEADERS = register(client, "wsiso")
    yield


def _upload(workspace_id: str) -> dict:
    with open(FIXTURE_PDF, "rb") as f:
        resp = client.post(
            "/upload",
            files={"file": ("attention.pdf", f, "application/pdf")},
            data={"workspace_id": workspace_id},
            headers=HEADERS,
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_creating_workspace_never_clears_another():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"

    ws_a = client.post("/workspaces", json={"name": "Transformer Research"}, headers=HEADERS).json()
    upload_a = _upload(ws_a["workspace_id"])

    before = client.get("/documents", params={"workspace_id": ws_a["workspace_id"]}, headers=HEADERS).json()
    assert len(before) == 1

    ws_b = client.post("/workspaces", json={"name": "Computer Vision"}, headers=HEADERS).json()
    assert ws_b["workspace_id"] != ws_a["workspace_id"]

    after = client.get("/documents", params={"workspace_id": ws_a["workspace_id"]}, headers=HEADERS).json()
    assert len(after) == 1
    assert after[0]["document_id"] == upload_a["document_id"], "Workspace A's document must survive Workspace B's creation"

    b_docs = client.get("/documents", params={"workspace_id": ws_b["workspace_id"]}, headers=HEADERS).json()
    assert b_docs == [], "A freshly created workspace must start with zero papers"


def test_document_visible_only_in_its_own_workspace():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"

    ws_a = client.post("/workspaces", json={"name": "Workspace A"}, headers=HEADERS).json()
    ws_b = client.post("/workspaces", json={"name": "Workspace B"}, headers=HEADERS).json()

    uploaded = _upload(ws_a["workspace_id"])
    doc_id = uploaded["document_id"]

    a_docs = [d["document_id"] for d in client.get("/documents", params={"workspace_id": ws_a["workspace_id"]}, headers=HEADERS).json()]
    b_docs = [d["document_id"] for d in client.get("/documents", params={"workspace_id": ws_b["workspace_id"]}, headers=HEADERS).json()]

    assert doc_id in a_docs
    assert doc_id not in b_docs


def test_documents_in_workspace_data_layer_filter():
    """The low-level filter that backs /query's workspace_id parameter."""
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"

    ws_a = client.post("/workspaces", json={"name": "Filter Test A"}, headers=HEADERS).json()
    ws_b = client.post("/workspaces", json={"name": "Filter Test B"}, headers=HEADERS).json()
    doc_a = _upload(ws_a["workspace_id"])["document_id"]

    # Asking for doc_a under workspace_id=B (a real document_id, wrong workspace)
    # must be filtered out by the data layer, not trusted.
    assert documents_in_workspace([doc_a], ws_b["workspace_id"]) == []
    assert documents_in_workspace([doc_a], ws_a["workspace_id"]) == [doc_a]
    # An id that doesn't exist at all is also dropped, not just mismatched ones.
    assert documents_in_workspace(["not_a_real_doc_id"], ws_a["workspace_id"]) == []


def test_query_rejects_cross_workspace_document_id():
    """
    /query must never let a request pull evidence from a document that
    belongs to a DIFFERENT workspace than the one it claims, even if the
    caller (a buggy or tampered client) explicitly names that document_id.
    """
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"

    ws_a = client.post("/workspaces", json={"name": "Cross Test A"}, headers=HEADERS).json()
    ws_b = client.post("/workspaces", json={"name": "Cross Test B"}, headers=HEADERS).json()
    doc_a = _upload(ws_a["workspace_id"])["document_id"]

    resp = client.post("/query", json={
        "question": "What is the main contribution?",
        "document_ids": [doc_a],
        "workspace_id": ws_b["workspace_id"],  # mismatched on purpose
    }, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["documents_found"] == 0
    assert "upload" in data["answer"].lower(), "Must behave as if no documents are scoped, not silently search doc_a"


class _FakeMessage:
    def __init__(self, content):
        self.content = content

class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)

class _FakeUsage:
    prompt_tokens = 40
    completion_tokens = 12

class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()

_physical_call_count = {"n": 0}

class _FakeCompletions:
    def create(self, model, messages, temperature=0.0, max_tokens=None):
        # Counts physical calls via a plain dict, not query_transform's
        # contextvar-based call log — TestClient's ASGI transport may run
        # the request in a different thread than the test itself, and
        # contextvars don't cross threads, so asserting on get_call_log()
        # here would be a false negative, not a real absence of a call.
        _physical_call_count["n"] += 1
        return _FakeResponse(json.dumps({
            "answer": "The Transformer architecture relies entirely on attention mechanisms.",
            "grounded": True,
            "evidence_sufficient": True,
            "document_ids": [_FAKE_DOC_ID],
        }))

class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()

class _FakeGroqClient:
    def __init__(self, api_key=None):
        self.chat = _FakeChat()

_FAKE_DOC_ID = ""


def test_correctly_scoped_query_still_one_llm_call():
    global _FAKE_DOC_ID
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"

    ws = client.post("/workspaces", json={"name": "Generation Test"}, headers=HEADERS).json()
    doc_id = _upload(ws["workspace_id"])["document_id"]
    _FAKE_DOC_ID = doc_id
    _physical_call_count["n"] = 0

    with patch("groq.Groq", _FakeGroqClient):
        resp = client.post("/query", json={
            "question": "What is the main contribution of this paper?",
            "document_ids": [doc_id],
            "workspace_id": ws["workspace_id"],
        }, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "attention" in data["answer"].lower()
    assert data["documents_found"]

    assert _physical_call_count["n"] == 1, (
        f"Correctly-scoped normal query must be exactly 1 physical LLM call, got {_physical_call_count['n']}"
    )


def test_sessions_are_workspace_filterable():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"

    ws_a = client.post("/workspaces", json={"name": "Session Test A"}, headers=HEADERS).json()
    ws_b = client.post("/workspaces", json={"name": "Session Test B"}, headers=HEADERS).json()

    sess_a = client.post("/sessions", json={"workspace_id": ws_a["workspace_id"], "title": "Compare methodologies"}, headers=HEADERS).json()
    client.post("/sessions", json={"workspace_id": ws_b["workspace_id"], "title": "Find limitations"}, headers=HEADERS)

    a_sessions = client.get("/sessions", params={"workspace_id": ws_a["workspace_id"]}, headers=HEADERS).json()
    assert len(a_sessions) == 1
    assert a_sessions[0]["session_id"] == sess_a["session_id"]
    assert a_sessions[0]["title"] == "Compare methodologies"

    b_sessions = client.get("/sessions", params={"workspace_id": ws_b["workspace_id"]}, headers=HEADERS).json()
    assert len(b_sessions) == 1
    assert b_sessions[0]["title"] == "Find limitations"


def test_workspace_list_reports_real_paper_and_chat_counts():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"

    ws = client.post("/workspaces", json={"name": "Counting Test"}, headers=HEADERS).json()
    _upload(ws["workspace_id"])
    client.post("/sessions", json={"workspace_id": ws["workspace_id"]}, headers=HEADERS)
    client.post("/sessions", json={"workspace_id": ws["workspace_id"]}, headers=HEADERS)

    got = client.get(f"/workspaces/{ws['workspace_id']}", headers=HEADERS).json()
    assert got["paper_count"] == 1
    assert got["chat_count"] == 2
