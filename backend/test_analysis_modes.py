"""
test_analysis_modes.py — /analyze (Viva, Mock Test, Explain Figure,
Recommend, Why This Design, System Design, Project Interview) and
/eval/dashboard.

Runs against the ISOLATED Phase 8 test Chroma/SQLite fixture (see
conftest.py) — never backend/chroma_store. Retrieval for PDF-grounded modes
is REAL (dense + BM25 + RRF + CrossEncoder rerank, unmocked); only the Groq
network boundary is mocked. The fake inspects the actual prompt text to
return the right JSON shape for whichever schema that prompt asked for —
proof that main.py/analysis.py are wiring the real prompts through, not a
coincidence.

Covers the "smallest necessary" verification: each mode costs exactly ONE
physical LLM call, PDF-grounded modes stay scoped to the requested
document_ids, and modes with no documents scoped fail closed (no LLM call).
"""
import json
import uuid

import pytest
from fastapi.testclient import TestClient

import config
from ingest import get_collection
from retrieval import build_bm25_index
from main import app
from database import add_document
from test_auth_helpers import registered_user_with_workspace

client = TestClient(app)

# /analyze's PDF-grounded modes now require a real authenticated,
# workspace-owning caller (see main.py's _require_workspace_owner()) —
# registered once per module, deferred to setup time (see
# test_workspace_isolation.py for why this must not be bare module-level
# code).
HEADERS: dict = {}
WORKSPACE_ID: str = ""


@pytest.fixture(autouse=True, scope="module")
def _module_auth():
    global HEADERS, WORKSPACE_ID
    HEADERS, WORKSPACE_ID = registered_user_with_workspace(client, "analyze")
    yield


def _push_chunk(document_id: str, suffix: str, text: str, filename: str) -> None:
    # Registers the document in the SQL registry under WORKSPACE_ID too
    # (not just Chroma) — _resolve_document_scope()'s workspace_id filter
    # is a real SQL-layer check (documents_in_workspace()), so a chunk
    # with no matching registry row would now be silently filtered out.
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


# ---------------------------------------------------------------------------
# Fake Groq — shape-detects from the prompt itself which JSON schema to
# return, and counts physical calls via a plain counter (TestClient's ASGI
# transport can run the request in a different OS thread than the test, so
# the contextvar-based call log isn't reliable to assert on here — see
# test_workspace_isolation.py for the same finding).
# ---------------------------------------------------------------------------
_call_count = {"n": 0}

QUESTIONS_JSON = json.dumps({
    "questions": [
        {"question": "What problem does this paper address?", "category": "motivation", "difficulty": "intermediate", "expected_answer_points": ["scaling", "efficiency"]},
        {"question": "What architecture is proposed?", "category": "architecture", "difficulty": "intermediate", "expected_answer_points": ["encoder-decoder"]},
    ],
    "grounded": True, "evidence_sufficient": True,
})

ANSWER_JSON = json.dumps({
    "answer": "This is a grounded explanation drawn from the evidence.",
    "grounded": True, "evidence_sufficient": True, "document_ids": [],
})

EVALUATION_JSON = json.dumps({
    "correctness": "partially_correct",
    "missing_points": ["Did not mention RRF fusion."],
    "technical_depth": "Reasonable but shallow.",
    "suggested_answer": "A fuller answer would mention dense+BM25+RRF+CrossEncoder.",
    "next_question": "Why is Reciprocal Rank Fusion used instead of simple score averaging?",
    "next_question_category": "RRF",
})


class _FakeMessage:
    def __init__(self, content):
        self.content = content

class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)

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
        if '"missing_points"' in prompt:
            return _FakeResponse(EVALUATION_JSON)
        if '"questions":' in prompt:
            return _FakeResponse(QUESTIONS_JSON)
        return _FakeResponse(ANSWER_JSON)

class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()

class _FakeGroqClient:
    def __init__(self, api_key=None):
        self.chat = _FakeChat()


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
    doc_a = "analyze_paper_a_" + uuid.uuid4().hex[:8]
    doc_b = "analyze_paper_b_" + uuid.uuid4().hex[:8]
    _push_chunk(doc_a, "c1", "Paper A proposes a sparse mixture-of-experts routing layer with a novel load-balancing loss.", "paperA.pdf")
    _push_chunk(doc_a, "c2", "Paper A's methodology trains on 8 GPUs for 3 days using the WMT14 dataset.", "paperA.pdf")
    _push_chunk(doc_b, "c1", "Paper B proposes a retrieval-augmented decoder conditioning generation on external passages.", "paperB.pdf")
    _push_chunk(doc_b, "c2", "Paper B reports 91.2 F1 on Natural Questions with a limitation of English-only training data.", "paperB.pdf")
    build_bm25_index()
    return doc_a, doc_b


# ---------------------------------------------------------------------------
# PDF-grounded modes
# ---------------------------------------------------------------------------
def test_viva_generates_questions_one_call_scoped(two_docs):
    doc_a, doc_b = two_docs
    resp = client.post("/analyze", json={"mode": "viva", "document_ids": [doc_a], "workspace_id": WORKSPACE_ID, "difficulty": "basic", "num_questions": 3}, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert len(data["questions"]) >= 1
    assert data["documents_used"] == [doc_a], "Viva must stay scoped to the requested document only"
    assert _call_count["n"] == 1


def test_mock_test_generates_questions_one_call(two_docs):
    doc_a, doc_b = two_docs
    resp = client.post("/analyze", json={"mode": "mock_test", "document_ids": [doc_a, doc_b], "workspace_id": WORKSPACE_ID, "difficulty": "advanced", "num_questions": 4}, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert len(data["questions"]) >= 1
    assert set(data["documents_used"]).issubset({doc_a, doc_b})
    assert _call_count["n"] == 1


def test_explain_figure_one_call_scoped(two_docs):
    doc_a, doc_b = two_docs
    resp = client.post("/analyze", json={"mode": "explain_figure", "document_ids": [doc_b], "workspace_id": WORKSPACE_ID, "figure_reference": "Figure 1"}, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "answer" in data
    assert data["documents_used"] == [doc_b]
    assert _call_count["n"] == 1


def test_recommend_one_call_covers_both_papers(two_docs):
    doc_a, doc_b = two_docs
    resp = client.post("/analyze", json={"mode": "recommend", "document_ids": [doc_a, doc_b], "workspace_id": WORKSPACE_ID}, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert set(data["documents_used"]) == {doc_a, doc_b}, "Recommendation must draw on both compared papers"
    assert _call_count["n"] == 1


def test_pdf_grounded_mode_with_no_documents_scoped_fails_closed():
    resp = client.post("/analyze", json={"mode": "viva", "document_ids": [], "workspace_id": WORKSPACE_ID}, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert "upload" in data["error"].lower()
    assert _call_count["n"] == 0, "No documents scoped must never trigger an LLM call"


# ---------------------------------------------------------------------------
# Non-PDF modes: Why This Design / System Design / Project Interview
# ---------------------------------------------------------------------------
def test_why_design_one_call_no_documents_needed():
    resp = client.post("/analyze", json={"mode": "why_design", "question": "Why RRF instead of simple score averaging?"}, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["answer"]
    assert _call_count["n"] == 1


def test_system_design_one_call():
    resp = client.post("/analyze", json={"mode": "system_design", "question": "How would you scale to millions of PDFs?"}, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["answer"]
    assert _call_count["n"] == 1


def test_design_mode_requires_a_question():
    resp = client.post("/analyze", json={"mode": "why_design"}, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert _call_count["n"] == 0


def test_project_interview_start_then_evaluate_two_calls_total(two_docs):
    # Project Interview is PDF-grounded (see test_scoping_fixes.py for the
    # dedicated "not VerityRAG" regression coverage) — document_ids required
    # here just like every other analysis mode.
    doc_a, doc_b = two_docs
    start_resp = client.post("/analyze", json={
        "mode": "project_interview_start", "document_ids": [doc_a], "workspace_id": WORKSPACE_ID,
        "difficulty": "intermediate", "topics": [],
    }, headers=HEADERS)
    assert start_resp.status_code == 200
    start_data = start_resp.json()
    assert start_data["ok"] is True
    assert start_data["question"]
    assert start_data["documents_used"] == [doc_a]

    eval_resp = client.post("/analyze", json={
        "mode": "project_interview_evaluate", "document_ids": [doc_a], "workspace_id": WORKSPACE_ID,
        "question": start_data["question"],
        "user_answer": "It uses a mixture-of-experts routing layer.",
        "difficulty": "intermediate",
        "topics": [],
    }, headers=HEADERS)
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert eval_data["ok"] is True
    assert eval_data["correctness"] in ("correct", "partially_correct", "incorrect", "unclear")
    assert eval_data["next_question"]

    assert _call_count["n"] == 2, "Start + one evaluate turn must be exactly 2 physical LLM calls total"


def test_project_interview_evaluate_requires_answer():
    resp = client.post("/analyze", json={"mode": "project_interview_evaluate", "question": "What is RRF?", "workspace_id": WORKSPACE_ID}, headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["ok"] is False
    assert _call_count["n"] == 0


def test_unknown_mode_rejected():
    resp = client.post("/analyze", json={"mode": "not_a_real_mode"}, headers=HEADERS)
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Eval dashboard — must never fabricate; only real logged/offline numbers
# ---------------------------------------------------------------------------
def test_eval_dashboard_returns_real_or_null_never_fake():
    resp = client.get("/eval/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert "live" in data and "offline_retrieval_eval" in data and "not_measured" in data
    assert "groundedness" in " ".join(data["not_measured"]).lower()
    # If live metrics are present, they must be plain numbers/None, not invented strings
    if data["live"] is not None:
        assert isinstance(data["live"]["sample_size"], int)
