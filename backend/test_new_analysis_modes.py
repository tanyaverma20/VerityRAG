"""
test_new_analysis_modes.py — Evaluate Paper, Find Research Gaps, Literature
Matrix, Knowledge Graph. Same pattern as test_analysis_modes.py: real
retrieval against the isolated Phase 8 fixture, only the Groq network
boundary mocked, one physical LLM call per mode, per-paper document isolation
proven for Literature Matrix / Knowledge Graph the same way Compare is
proven in test_scoping_fixes.py.
"""
import json
import re
import uuid

import pytest
from fastapi.testclient import TestClient

import config
from ingest import get_collection
from retrieval import build_bm25_index
from database import create_session, get_session_messages, add_document
from main import app
from test_auth_helpers import registered_user_with_workspace

client = TestClient(app)

_call_count = {"n": 0}

# /analyze's PDF-grounded modes require a real authenticated,
# workspace-owning caller — see test_workspace_isolation.py for why this
# is a module-scoped fixture rather than bare module-level code.
HEADERS: dict = {}
WORKSPACE_ID: str = ""


@pytest.fixture(autouse=True, scope="module")
def _module_auth():
    global HEADERS, WORKSPACE_ID
    HEADERS, WORKSPACE_ID = registered_user_with_workspace(client, "newmode")
    yield


def _push_chunk(document_id: str, suffix: str, text: str, filename: str) -> None:
    add_document(document_id, filename, status="INDEXED", workspace_id=WORKSPACE_ID)
    col = get_collection()
    chunk_id = f"doc_{document_id}_{suffix}"
    col.add(
        documents=[text],
        ids=[chunk_id],
        metadatas=[{
            "document_id": document_id, "workspace_id": WORKSPACE_ID,
            "filename": filename, "source": filename,
            "page_number": 1, "section": "Body", "chunk_id": chunk_id,
            "parent_id": f"{document_id}_{suffix}_parent", "chunk_type": "child",
        }],
    )


class _FakeMessage:
    def __init__(self, content): self.content = content
class _FakeChoice:
    def __init__(self, content): self.message = _FakeMessage(content)
class _FakeUsage:
    prompt_tokens = 80
    completion_tokens = 40
class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeCompletions:
    def create(self, model, messages, temperature=0.0, max_tokens=None):
        _call_count["n"] += 1
        prompt = messages[0]["content"]
        # Only scan the actual EVIDENCE block for real Document IDs — the
        # prompt instructions themselves reference "--- Document ID: X ---"
        # as a format example, which would otherwise falsely match too.
        evidence_section = prompt.split("EVIDENCE:", 1)[-1]
        doc_ids = list(dict.fromkeys(re.findall(r"Document ID:\s*(\S+)", evidence_section)))

        if '"problem_clarity"' in prompt:
            return _FakeResponse(json.dumps({
                "problem_clarity": {"assessment": "Clearly framed around routing efficiency.", "support": "DIRECTLY_STATED"},
                "novelty": {"assessment": "Sparse MoE routing is presented as new.", "support": "DIRECTLY_STATED"},
                "methodology": {"assessment": "Trained on 8 GPUs, 3 days, WMT14.", "support": "DIRECTLY_STATED"},
                "experimental_design": {"assessment": "", "support": "NOT_FOUND"},
                "results": {"assessment": "Reports F1 improvements.", "support": "DIRECTLY_STATED"},
                "limitations": {"assessment": "", "support": "NOT_FOUND"},
                "reproducibility": {"assessment": "", "support": "NOT_FOUND"},
                "strengths": ["Clear routing mechanism"], "weaknesses": ["No ablation described"],
                "overall_assessment": "Solid contribution with a clearly stated method; several dimensions not covered by the evidence.",
                "evidence_sufficient": True,
            }))
        if '"gaps"' in prompt:
            return _FakeResponse(json.dumps({
                "gaps": [
                    {"gap": "No ablation study on load-balancing loss weight.", "label": "POTENTIAL_INFERRED_GAP", "evidence": "Only final config described.", "document_id": doc_ids[0] if doc_ids else ""},
                    {"gap": "English-only training data.", "label": "AUTHOR_STATED_GAP", "evidence": "Reports limitation of English-only training data.", "document_id": doc_ids[-1] if doc_ids else ""},
                ],
                "evidence_sufficient": True,
            }))
        if '"rows"' in prompt:
            rows = [{
                "document_id": d, "title": f"Paper {d}", "problem": f"Problem of {d}", "method": f"Method of {d}",
                "architecture": f"Arch of {d}", "dataset": f"Dataset of {d}", "metrics": f"Metrics of {d}",
                "results": f"Results of {d}", "limitations": f"Limitations of {d}", "research_gap": f"Gap of {d}",
            } for d in doc_ids]
            return _FakeResponse(json.dumps({"rows": rows, "evidence_sufficient": True}))
        if '"nodes"' in prompt:
            nodes = [{"id": f"{d}_paper", "label": f"Paper {d}", "type": "paper", "document_id": d} for d in doc_ids]
            nodes += [{"id": f"{d}_method", "label": f"Method of {d}", "type": "method", "document_id": d} for d in doc_ids]
            edges = [{"source": f"{d}_paper", "target": f"{d}_method", "relation": "uses", "document_id": d} for d in doc_ids]
            return _FakeResponse(json.dumps({"nodes": nodes, "edges": edges, "evidence_sufficient": True}))
        return _FakeResponse(json.dumps({"answer": "fallback", "grounded": True, "evidence_sufficient": True, "document_ids": []}))


class _FakeChat:
    def __init__(self): self.completions = _FakeCompletions()
class _FakeGroqClient:
    def __init__(self, api_key=None): self.chat = _FakeChat()


from unittest.mock import patch
_groq_patch = patch("groq.Groq", _FakeGroqClient)


@pytest.fixture(autouse=True)
def _reset():
    _call_count["n"] = 0
    _groq_patch.start()
    yield
    _groq_patch.stop()


@pytest.fixture
def two_docs():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"
    doc_a = "newmode_paper_a_" + uuid.uuid4().hex[:8]
    doc_b = "newmode_paper_b_" + uuid.uuid4().hex[:8]
    _push_chunk(doc_a, "c1", "Paper A proposes a sparse mixture-of-experts routing layer with a novel load-balancing loss.", "paperA.pdf")
    _push_chunk(doc_a, "c2", "Paper A's methodology trains on 8 GPUs for 3 days using the WMT14 dataset, reporting F1 improvements.", "paperA.pdf")
    _push_chunk(doc_b, "c1", "Paper B proposes a retrieval-augmented decoder conditioning generation on external passages.", "paperB.pdf")
    _push_chunk(doc_b, "c2", "Paper B reports 91.2 F1 on Natural Questions with a limitation of English-only training data.", "paperB.pdf")
    build_bm25_index()
    return doc_a, doc_b


# ---------------------------------------------------------------------------
def test_evaluate_paper_one_call_scoped_to_single_document(two_docs):
    doc_a, doc_b = two_docs
    resp = client.post("/analyze", json={"mode": "evaluate_paper", "document_ids": [doc_a], "workspace_id": WORKSPACE_ID}, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    ev = data["evaluation"]
    assert ev["document_id"] == doc_a
    assert ev["problem_clarity"]["support"] in ("DIRECTLY_STATED", "STRONGLY_SUPPORTED", "NOT_FOUND")
    assert ev["reproducibility"]["support"] == "NOT_FOUND", "Thin evidence must not fabricate a reproducibility assessment"
    assert data["documents_used"] == [doc_a]
    assert _call_count["n"] == 1


def test_research_gaps_one_call_labels_each_gap(two_docs):
    doc_a, doc_b = two_docs
    resp = client.post("/analyze", json={"mode": "research_gaps", "document_ids": [doc_a, doc_b], "workspace_id": WORKSPACE_ID}, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    labels = {g["label"] for g in data["gaps"]}
    assert labels.issubset({"AUTHOR_STATED_GAP", "POTENTIAL_INFERRED_GAP"})
    assert any(g["label"] == "AUTHOR_STATED_GAP" for g in data["gaps"])
    assert _call_count["n"] == 1


def test_literature_matrix_one_row_per_paper_isolated(two_docs):
    doc_a, doc_b = two_docs
    resp = client.post("/analyze", json={"mode": "literature_matrix", "document_ids": [doc_a, doc_b], "workspace_id": WORKSPACE_ID,}, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    row_doc_ids = {r["document_id"] for r in data["rows"]}
    assert row_doc_ids == {doc_a, doc_b}, "Must produce exactly one row per selected paper"
    row_a = next(r for r in data["rows"] if r["document_id"] == doc_a)
    row_b = next(r for r in data["rows"] if r["document_id"] == doc_b)
    assert doc_a in row_a["method"] and doc_b not in row_a["method"], "Each row's cells must be scoped to its own paper only"
    assert doc_b in row_b["method"] and doc_a not in row_b["method"]
    assert _call_count["n"] == 1


def test_literature_matrix_requires_two_documents(two_docs):
    doc_a, _ = two_docs
    resp = client.post("/analyze", json={"mode": "literature_matrix", "document_ids": [doc_a], "workspace_id": WORKSPACE_ID,}, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert _call_count["n"] == 0, "Must fail closed before ever calling the LLM"


def test_knowledge_graph_one_call_nodes_tagged_with_document_id(two_docs):
    doc_a, doc_b = two_docs
    resp = client.post("/analyze", json={"mode": "knowledge_graph", "document_ids": [doc_a, doc_b], "workspace_id": WORKSPACE_ID,}, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert all(n["document_id"] in (doc_a, doc_b) for n in data["nodes"])
    assert all(e["document_id"] in (doc_a, doc_b) for e in data["edges"])
    node_docs = {n["document_id"] for n in data["nodes"]}
    assert node_docs == {doc_a, doc_b}
    assert _call_count["n"] == 1


def test_knowledge_graph_single_paper(two_docs):
    doc_a, _ = two_docs
    resp = client.post("/analyze", json={"mode": "knowledge_graph", "document_ids": [doc_a], "workspace_id": WORKSPACE_ID,}, headers=HEADERS)
    data = resp.json()
    assert data["ok"] is True
    assert all(n["document_id"] == doc_a for n in data["nodes"])
    assert _call_count["n"] == 1


def test_new_modes_fail_closed_with_no_documents():
    for mode in ("evaluate_paper", "research_gaps", "literature_matrix", "knowledge_graph"):
        resp = client.post("/analyze", json={"mode": mode, "document_ids": [], "workspace_id": WORKSPACE_ID,}, headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["ok"] is False
    assert _call_count["n"] == 0


# ---------------------------------------------------------------------------
# Persistence — the SAME sessions/messages mechanism normal Q&A already
# uses (see test_analysis_persistence.py), extended to these 4 new modes
# per the expanded mode list.
# ---------------------------------------------------------------------------
def _new_session():
    sid = "newmode_sess_" + uuid.uuid4().hex[:10]
    create_session(sid, title="New conversation")
    return sid


def test_evaluate_paper_persists_evaluation(two_docs):
    doc_a, _ = two_docs
    sid = _new_session()
    resp = client.post("/analyze", json={"mode": "evaluate_paper", "document_ids": [doc_a], "session_id": sid, "workspace_id": WORKSPACE_ID,}, headers=HEADERS)
    assert resp.json()["ok"] is True
    msgs = get_session_messages(sid)
    assert len(msgs) == 1 and msgs[0]["role"] == "assistant"
    assert msgs[0]["metadata"]["frontend_role"] == "evaluate_paper"
    assert msgs[0]["metadata"]["evaluation"]["document_id"] == doc_a


def test_research_gaps_persists_gap_list(two_docs):
    doc_a, doc_b = two_docs
    sid = _new_session()
    resp = client.post("/analyze", json={"mode": "research_gaps", "document_ids": [doc_a, doc_b], "session_id": sid, "workspace_id": WORKSPACE_ID,}, headers=HEADERS)
    assert resp.json()["ok"] is True
    msgs = get_session_messages(sid)
    assert len(msgs) == 1
    assert msgs[0]["metadata"]["frontend_role"] == "research_gaps"
    assert len(msgs[0]["metadata"]["gaps"]) == 2


def test_literature_matrix_persists_rows(two_docs):
    doc_a, doc_b = two_docs
    sid = _new_session()
    resp = client.post("/analyze", json={"mode": "literature_matrix", "document_ids": [doc_a, doc_b], "session_id": sid, "workspace_id": WORKSPACE_ID,}, headers=HEADERS)
    assert resp.json()["ok"] is True
    msgs = get_session_messages(sid)
    assert len(msgs) == 1
    assert msgs[0]["metadata"]["frontend_role"] == "literature_matrix"
    assert {r["document_id"] for r in msgs[0]["metadata"]["rows"]} == {doc_a, doc_b}


def test_knowledge_graph_persists_nodes_and_edges(two_docs):
    doc_a, doc_b = two_docs
    sid = _new_session()
    resp = client.post("/analyze", json={"mode": "knowledge_graph", "document_ids": [doc_a, doc_b], "session_id": sid, "workspace_id": WORKSPACE_ID,}, headers=HEADERS)
    assert resp.json()["ok"] is True
    msgs = get_session_messages(sid)
    assert len(msgs) == 1
    assert msgs[0]["metadata"]["frontend_role"] == "knowledge_graph"
    assert msgs[0]["metadata"]["nodes"]
    assert msgs[0]["metadata"]["edges"]


def test_new_modes_persist_nothing_without_session_id(two_docs):
    doc_a, _ = two_docs
    resp = client.post("/analyze", json={"mode": "evaluate_paper", "document_ids": [doc_a], "workspace_id": WORKSPACE_ID,}, headers=HEADERS)
    assert resp.json()["ok"] is True
    # No session_id sent — nothing to look up; the call above must not raise.
