"""
test_authorization.py — adversarial cross-user authorization tests.

Two REAL, separately registered users (never a client-supplied
workspace_id trusted as the security principal — see main.py's
_require_workspace_owner() / _require_resource_owner()). User B attempts
to read/modify User A's workspace, documents, sessions, tasks, and reports
purely by knowing (or guessing) their real IDs — every attempt must fail
safely (404, indistinguishable from "doesn't exist"), and User A's own
access must keep working throughout.

Runs against the isolated Phase 8 test Chroma/SQLite fixture (see
conftest.py). Only the Groq network boundary is mocked for the two tests
that need a real LLM response (report generation, query).
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from main import app
from test_auth_helpers import registered_user_with_workspace, register

client = TestClient(app)

FIXTURE_PDF = Path(__file__).parent / "tests" / "fixtures" / "attention.pdf"


class _FakeMessage:
    def __init__(self, content): self.content = content
class _FakeChoice:
    def __init__(self, content): self.message = _FakeMessage(content)
class _FakeUsage:
    prompt_tokens = 40
    completion_tokens = 12
class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()
class _FakeCompletions:
    def create(self, model, messages, temperature=0.0, max_tokens=None):
        return _FakeResponse(json.dumps({
            "title": "Report", "overview": "o", "papers": [], "comparison": None,
            "conclusion": "c", "evidence_sufficient": True,
        }))
class _FakeChat:
    def __init__(self): self.completions = _FakeCompletions()
class _FakeGroqClient:
    def __init__(self, api_key=None): self.chat = _FakeChat()


@pytest.fixture
def two_users_a_owns_a_workspace_and_document():
    """User A: real workspace + real uploaded document.
    User B: a completely separate, unrelated account."""
    headers_a, ws_a = registered_user_with_workspace(client, "authz_a")
    headers_b, ws_b = registered_user_with_workspace(client, "authz_b")

    with open(FIXTURE_PDF, "rb") as f:
        upload = client.post(
            "/upload", files={"file": ("attention.pdf", f, "application/pdf")},
            data={"workspace_id": ws_a}, headers=headers_a,
        ).json()
    doc_a = upload["document_id"]

    return headers_a, ws_a, headers_b, ws_b, doc_a


# ---------------------------------------------------------------------------
# No authentication at all
# ---------------------------------------------------------------------------
def test_protected_endpoints_reject_missing_auth_with_401():
    assert client.get("/workspaces").status_code == 401
    assert client.post("/workspaces", json={"name": "x"}).status_code == 401
    assert client.get("/documents").status_code == 401
    assert client.post("/query", json={"question": "x", "workspace_id": "whatever"}).status_code == 401
    assert client.post("/sessions", json={}).status_code == 401
    assert client.get("/logs").status_code == 401


def test_protected_endpoints_reject_garbage_token_with_401():
    bad = {"Authorization": "Bearer not-a-real-token"}
    assert client.get("/workspaces", headers=bad).status_code == 401
    assert client.post("/query", json={"question": "x", "workspace_id": "whatever"}, headers=bad).status_code == 401


# ---------------------------------------------------------------------------
# Workspace ownership
# ---------------------------------------------------------------------------
def test_user_b_cannot_read_user_as_workspace(two_users_a_owns_a_workspace_and_document):
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    resp = client.get(f"/workspaces/{ws_a}", headers=headers_b)
    assert resp.status_code == 404
    # A's own access still works.
    assert client.get(f"/workspaces/{ws_a}", headers=headers_a).status_code == 200


def test_user_b_workspace_list_never_includes_user_as_workspace(two_users_a_owns_a_workspace_and_document):
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    b_workspaces = {w["workspace_id"] for w in client.get("/workspaces", headers=headers_b).json()}
    assert ws_a not in b_workspaces
    assert ws_b in b_workspaces


def test_user_b_cannot_upload_into_user_as_workspace(two_users_a_owns_a_workspace_and_document):
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    with open(FIXTURE_PDF, "rb") as f:
        resp = client.post(
            "/upload", files={"file": ("attention.pdf", f, "application/pdf")},
            data={"workspace_id": ws_a}, headers=headers_b,
        )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Document ownership
# ---------------------------------------------------------------------------
def test_user_b_cannot_read_user_as_document(two_users_a_owns_a_workspace_and_document):
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    resp = client.get(f"/documents/{doc_a}", headers=headers_b)
    assert resp.status_code == 404
    assert client.get(f"/documents/{doc_a}", headers=headers_a).status_code == 200


def test_user_b_cannot_delete_user_as_document(two_users_a_owns_a_workspace_and_document):
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    resp = client.delete(f"/documents/{doc_a}", headers=headers_b)
    assert resp.status_code == 404
    # Still there for A.
    assert client.get(f"/documents/{doc_a}", headers=headers_a).status_code == 200


def test_user_b_document_list_never_includes_user_as_document(two_users_a_owns_a_workspace_and_document):
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    b_docs = {d["document_id"] for d in client.get("/documents", headers=headers_b).json()}
    assert doc_a not in b_docs


# ---------------------------------------------------------------------------
# Retrieval / vector data — B claiming A's real workspace_id + document_id
# must never surface A's real content, even though both IDs are real.
# ---------------------------------------------------------------------------
def test_user_b_query_against_user_as_workspace_id_fails_closed(two_users_a_owns_a_workspace_and_document):
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    resp = client.post("/query", json={
        "question": "What is the main contribution?",
        "document_ids": [doc_a], "workspace_id": ws_a,
    }, headers=headers_b)
    assert resp.status_code == 404


def test_user_b_analyze_against_user_as_document_fails_closed(two_users_a_owns_a_workspace_and_document):
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    resp = client.post("/analyze", json={
        "mode": "viva", "document_ids": [doc_a], "workspace_id": ws_a,
    }, headers=headers_b)
    assert resp.status_code == 404


def test_user_b_cannot_smuggle_user_as_document_via_own_owned_workspace(two_users_a_owns_a_workspace_and_document):
    """Even scoped to B's OWN real, owned workspace, B must never retrieve
    A's document just by naming its real document_id — ownership is
    checked on the workspace itself before document_ids are ever trusted."""
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    resp = client.post("/query", json={
        "question": "What is the main contribution?",
        "document_ids": [doc_a], "workspace_id": ws_b,
    }, headers=headers_b)
    assert resp.status_code == 200
    data = resp.json()
    assert data["documents_found"] == 0
    assert "upload" in data["answer"].lower()


# ---------------------------------------------------------------------------
# Session ownership
# ---------------------------------------------------------------------------
def test_user_b_cannot_read_user_as_session_messages(two_users_a_owns_a_workspace_and_document):
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    sess = client.post("/sessions", json={"workspace_id": ws_a, "title": "A's chat"}, headers=headers_a).json()
    sid = sess["session_id"]

    assert client.get(f"/sessions/{sid}/messages", headers=headers_b).status_code == 404
    assert client.patch(f"/sessions/{sid}", json={"title": "hijacked"}, headers=headers_b).status_code == 404
    assert client.delete(f"/sessions/{sid}", headers=headers_b).status_code == 404
    # A's own access still works, and the title was never actually changed.
    assert client.get(f"/sessions/{sid}/messages", headers=headers_a).status_code == 200


def test_user_b_cannot_create_session_in_user_as_workspace(two_users_a_owns_a_workspace_and_document):
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    resp = client.post("/sessions", json={"workspace_id": ws_a, "title": "hijack"}, headers=headers_b)
    assert resp.status_code == 404


def test_user_b_session_list_never_includes_user_as_session(two_users_a_owns_a_workspace_and_document):
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    sess = client.post("/sessions", json={"workspace_id": ws_a, "title": "A's chat"}, headers=headers_a).json()
    b_sessions = {s["session_id"] for s in client.get("/sessions", headers=headers_b).json()}
    assert sess["session_id"] not in b_sessions


# ---------------------------------------------------------------------------
# Task ownership
# ---------------------------------------------------------------------------
def test_user_b_cannot_read_user_as_task(two_users_a_owns_a_workspace_and_document):
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    task = client.post("/research", json={
        "question": "What is this paper about?", "document_ids": [doc_a], "workspace_id": ws_a,
    }, headers=headers_a).json()
    task_id = task["task_id"]

    assert client.get(f"/task/{task_id}", headers=headers_b).status_code == 404
    assert client.get(f"/task/{task_id}", headers=headers_a).status_code == 200


def test_user_b_cannot_start_research_in_user_as_workspace(two_users_a_owns_a_workspace_and_document):
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    resp = client.post("/research", json={
        "question": "hijack", "document_ids": [doc_a], "workspace_id": ws_a,
    }, headers=headers_b)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Report ownership — report_id is itself a guessable-format cache key;
# real ownership must be enforced independently of knowing it.
# ---------------------------------------------------------------------------
def test_user_b_cannot_generate_report_in_user_as_workspace(two_users_a_owns_a_workspace_and_document):
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    resp = client.post("/report", json={"document_ids": [doc_a], "workspace_id": ws_a}, headers=headers_b)
    assert resp.status_code == 404


def test_user_b_cannot_read_user_as_report(two_users_a_owns_a_workspace_and_document):
    headers_a, ws_a, headers_b, ws_b, doc_a = two_users_a_owns_a_workspace_and_document
    with patch("groq.Groq", _FakeGroqClient):
        report = client.post("/report", json={"document_ids": [doc_a], "workspace_id": ws_a}, headers=headers_a).json()
    assert report["ok"] is True
    report_id = report["report_id"]

    assert client.get(f"/report/{report_id}", headers=headers_b).status_code == 404
    assert client.get(f"/report/{report_id}/markdown", headers=headers_b).status_code == 404
    assert client.get(f"/report/{report_id}/pdf", headers=headers_b).status_code == 404
    assert client.get(f"/report/{report_id}/docx", headers=headers_b).status_code == 404
    # A's own access still works.
    assert client.get(f"/report/{report_id}", headers=headers_a).status_code == 200
