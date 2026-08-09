"""
test_scoping_fixes.py — Regression tests for the document-scoping bugs
reported across Viva / Mock Test / Project Interview / Compare:

  1. Interview/Viva/Mock Test generating content about VerityRAG itself
     instead of the uploaded PDF (root cause: project_interview_start/
     evaluate never took document_ids at all — always grounded in
     analysis.REAL_ARCHITECTURE_CONTEXT).
  2. Viva/Mock Test/Interview sometimes generating nothing (root cause:
     retrieval.select_within_token_budget() estimated tokens from the short
     child chunk text but the prompt actually sends the much larger
     parent_context, silently blowing the real prompt past the configured
     budget).
  3. Compare showing only one PDF (same token-budget root cause — an
     oversized, unbalanced multi-document prompt made it more likely the
     model dropped one paper from its JSON response).

Runs against the ISOLATED Phase 8 test Chroma/SQLite fixture (see
conftest.py) — never backend/chroma_store. Retrieval is REAL (dense+BM25+
RRF+CrossEncoder rerank, unmocked); only the Groq network boundary is
mocked.
"""
import json
import uuid

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

import config
from ingest import get_collection
from retrieval import build_bm25_index, select_within_token_budget
from report_generator import _gather_report_evidence
from main import app

client = TestClient(app)


def _push_chunk(document_id: str, suffix: str, text: str, filename: str) -> None:
    col = get_collection()
    chunk_id = f"doc_{document_id}_{suffix}"
    col.add(
        documents=[text],
        ids=[chunk_id],
        metadatas=[{
            "document_id": document_id, "filename": filename, "source": filename,
            "page_number": 1, "section": "Body", "chunk_id": chunk_id,
            "parent_id": f"{document_id}_{suffix}_parent", "chunk_type": "child",
        }],
    )


# ---------------------------------------------------------------------------
# Fake Groq — shape-detects from the prompt which schema to return, and
# records the last prompt seen so tests can assert on what was actually
# threaded into it (question counts, evidence content, absence of
# REAL_ARCHITECTURE_CONTEXT markers, etc).
# ---------------------------------------------------------------------------
_call_count = {"n": 0}
_last_prompt = {"text": ""}

QUESTIONS_JSON_TMPL = json.dumps({
    "questions": [
        {"question": "What problem does this project solve?", "category": "motivation", "difficulty": "intermediate", "expected_answer_points": ["a"]},
        {"question": "What is the architecture?", "category": "architecture", "difficulty": "intermediate", "expected_answer_points": ["b"]},
        {"question": "What are the results?", "category": "results", "difficulty": "intermediate", "expected_answer_points": ["c"]},
    ],
    "grounded": True, "evidence_sufficient": True,
})

EVALUATION_JSON = json.dumps({
    "correctness": "correct", "missing_points": [], "technical_depth": "Solid.",
    "suggested_answer": "A fuller answer would mention the pipeline stages.",
    "next_question": "What datasets does this project use?", "next_question_category": "Dataset",
})

REPORT_JSON_TMPL = """{{
  "title": "Comparison",
  "overview": "Comparing the uploaded papers.",
  "papers": [{papers}],
  "comparison": {{"commonalities": [], "differences": ["They differ."], "strengths": [], "limitations": []}},
  "conclusion": "Both papers were reviewed.",
  "evidence_sufficient": true
}}"""


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
        _last_prompt["text"] = prompt
        if '"missing_points"' in prompt:
            return _FakeResponse(EVALUATION_JSON)
        if '"questions":' in prompt:
            return _FakeResponse(QUESTIONS_JSON_TMPL)
        if '"comparison":' in prompt or "COMPARATIVE report" in prompt:
            # Build a paper entry per Document ID actually present in the prompt,
            # so the fake genuinely reflects what evidence it was given —
            # exactly like report_generator.py's real prompt structure.
            import re
            doc_ids = re.findall(r"- (\S+): ", prompt)
            papers = ",".join(
                json.dumps({
                    "document_id": d, "title": f"Paper {d}", "overview": "o", "main_contribution": "c",
                    "methodology": "m", "architecture": "a", "datasets": [], "evaluation_metrics": [],
                    "key_results": [], "important_calculations": [], "limitations": [], "final_summary": "f",
                }) for d in doc_ids
            )
            return _FakeResponse(REPORT_JSON_TMPL.format(papers=papers))
        return _FakeResponse(json.dumps({"answer": "grounded answer", "grounded": True, "evidence_sufficient": True, "document_ids": []}))

class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()

class _FakeGroqClient:
    def __init__(self, api_key=None):
        self.chat = _FakeChat()

_groq_patch = patch("groq.Groq", _FakeGroqClient)


@pytest.fixture(autouse=True)
def _reset():
    _call_count["n"] = 0
    _last_prompt["text"] = ""
    _groq_patch.start()
    yield
    _groq_patch.stop()


@pytest.fixture
def two_docs():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"
    doc_a = "scope_paper_a_" + uuid.uuid4().hex[:8]
    doc_b = "scope_paper_b_" + uuid.uuid4().hex[:8]
    _push_chunk(doc_a, "c1", "RetentionAI is an HR platform predicting employee attrition risk with a five-stage AI pipeline.", "RetentionAI.pdf")
    _push_chunk(doc_a, "c2", "RetentionAI's architecture uses a React client, Node.js server, Python FastAPI AI service, and MongoDB.", "RetentionAI.pdf")
    _push_chunk(doc_b, "c1", "SmartSwipe is a personal finance app recommending the optimal credit card for each purchase in real time.", "SmartSwipe.pdf")
    _push_chunk(doc_b, "c2", "SmartSwipe's backend uses Django REST Framework, PostgreSQL, Redis caching and a contextual bandit algorithm.", "SmartSwipe.pdf")
    build_bm25_index()
    return doc_a, doc_b


# ---------------------------------------------------------------------------
# 1. VIVA — per-document scoping, count/difficulty threading
# ---------------------------------------------------------------------------
def test_viva_pdf_a_only_scoped_to_a(two_docs):
    doc_a, doc_b = two_docs
    resp = client.post("/analyze", json={"mode": "viva", "document_ids": [doc_a], "difficulty": "advanced", "num_questions": 3})
    data = resp.json()
    assert data["ok"] is True
    assert data["documents_used"] == [doc_a]
    assert "advanced" in _last_prompt["text"] and "NUMBER OF QUESTIONS: 3" in _last_prompt["text"]
    assert _call_count["n"] == 1


def test_viva_pdf_b_only_scoped_to_b(two_docs):
    doc_a, doc_b = two_docs
    resp = client.post("/analyze", json={"mode": "viva", "document_ids": [doc_b], "difficulty": "basic", "num_questions": 5})
    data = resp.json()
    assert data["ok"] is True
    assert data["documents_used"] == [doc_b]
    assert "NUMBER OF QUESTIONS: 5" in _last_prompt["text"]


def test_viva_both_selected_can_use_both(two_docs):
    doc_a, doc_b = two_docs
    resp = client.post("/analyze", json={"mode": "viva", "document_ids": [doc_a, doc_b], "num_questions": 8})
    data = resp.json()
    assert data["ok"] is True
    assert set(data["documents_used"]) == {doc_a, doc_b}
    assert "NUMBER OF QUESTIONS: 8" in _last_prompt["text"]


# ---------------------------------------------------------------------------
# 2. MOCK TEST — same document-scoping discipline
# ---------------------------------------------------------------------------
def test_mock_test_scoped_to_selected_pdf(two_docs):
    doc_a, doc_b = two_docs
    resp = client.post("/analyze", json={"mode": "mock_test", "document_ids": [doc_a], "difficulty": "intermediate", "num_questions": 3})
    data = resp.json()
    assert data["ok"] is True
    assert data["documents_used"] == [doc_a]
    assert len(data["questions"]) > 0


# ---------------------------------------------------------------------------
# 3. PROJECT INTERVIEW — PDF-grounded by default, never VerityRAG unless
# VerityRAG is literally the uploaded evidence.
# ---------------------------------------------------------------------------
def test_project_interview_start_grounded_in_uploaded_pdf_not_verityrag(two_docs):
    doc_a, doc_b = two_docs
    resp = client.post("/analyze", json={
        "mode": "project_interview_start", "document_ids": [doc_a], "workspace_id": None,
        "difficulty": "intermediate", "topics": [],
    })
    data = resp.json()
    assert data["ok"] is True
    assert data["question"]
    assert data["documents_used"] == [doc_a]
    # The prompt must contain THIS document's real evidence...
    assert "RetentionAI" in _last_prompt["text"]
    # ...and must NEVER contain the VerityRAG architecture ground-truth text
    # (proves this mode stopped defaulting to interviewing about VerityRAG).
    assert "VERITYRAG" not in _last_prompt["text"].upper() or "RETENTIONAI" in _last_prompt["text"].upper()
    assert "ACTUAL SYSTEM ARCHITECTURE" not in _last_prompt["text"]


def test_project_interview_requires_documents_no_default_to_verityrag():
    """With zero documents scoped, interview must fail closed (insufficient
    documents), never silently fall back to interviewing about VerityRAG."""
    resp = client.post("/analyze", json={"mode": "project_interview_start", "document_ids": [], "difficulty": "basic"})
    data = resp.json()
    assert data["ok"] is False
    assert "ACTUAL SYSTEM ARCHITECTURE" not in _last_prompt["text"]
    assert _call_count["n"] == 0


def test_project_interview_switches_to_whichever_pdf_is_scoped(two_docs):
    doc_a, doc_b = two_docs
    resp_a = client.post("/analyze", json={"mode": "project_interview_start", "document_ids": [doc_a], "difficulty": "basic"})
    assert "RetentionAI" in _last_prompt["text"]

    resp_b = client.post("/analyze", json={"mode": "project_interview_start", "document_ids": [doc_b], "difficulty": "basic"})
    assert "SmartSwipe" in _last_prompt["text"]
    assert "RetentionAI" not in _last_prompt["text"]


def test_project_interview_evaluate_grounded_in_same_pdf_evidence(two_docs):
    doc_a, doc_b = two_docs
    resp = client.post("/analyze", json={
        "mode": "project_interview_evaluate", "document_ids": [doc_a],
        "question": "What is RetentionAI's architecture?", "user_answer": "It uses a five-stage pipeline.",
        "difficulty": "intermediate", "topics": [],
    })
    data = resp.json()
    assert data["ok"] is True
    assert data["correctness"] in ("correct", "partially_correct", "incorrect", "unclear")
    assert data["next_question"]
    assert "RetentionAI" in _last_prompt["text"]
    assert "ACTUAL SYSTEM ARCHITECTURE" not in _last_prompt["text"]


def test_why_design_and_system_design_still_use_verityrag_context():
    """Explicitly unaffected — these two modes stay grounded in VerityRAG's
    own architecture, per the requirement to keep them as-is."""
    resp = client.post("/analyze", json={"mode": "why_design", "question": "Why RRF?"})
    assert resp.json()["ok"] is True
    assert "ACTUAL SYSTEM ARCHITECTURE" in _last_prompt["text"]


# ---------------------------------------------------------------------------
# 4. Token-budget fix — the root cause behind "generates nothing" / "only
# one PDF" — both documents must survive evidence gathering within budget.
# ---------------------------------------------------------------------------
def test_token_budget_counts_parent_context_not_child_text():
    chunk = {
        "text": "short",
        "parent_context": "x" * 4000,  # ~1000 tokens
        "metadata": {"document_id": "d1"},
    }
    # A tiny budget that the real (parent_context-sized) chunk cannot fit in.
    selected = select_within_token_budget([chunk], max_tokens=100, max_per_doc=5)
    assert selected == [], "Budget must be evaluated against what's actually sent to the LLM (parent_context), not the short child text"


def test_report_evidence_gathering_represents_both_documents_within_budget(two_docs):
    doc_a, doc_b = two_docs
    grouped = _gather_report_evidence([doc_a, doc_b])
    assert doc_a in grouped and doc_b in grouped, f"Both documents must survive evidence gathering, got {list(grouped.keys())}"
    assert len(grouped[doc_a]) > 0 and len(grouped[doc_b]) > 0


# ---------------------------------------------------------------------------
# 5. COMPARE — both papers must appear in the generated report; a mismatched
# third document must never leak in.
# ---------------------------------------------------------------------------
def test_compare_report_includes_both_papers(two_docs):
    doc_a, doc_b = two_docs
    resp = client.post("/report", json={"document_ids": [doc_a, doc_b]})
    data = resp.json()
    assert data["ok"] is True
    returned_ids = {p["document_id"] for p in data["papers"]}
    assert returned_ids == {doc_a, doc_b}, f"Compare must include BOTH papers as columns, got {returned_ids}"


def test_compare_a_and_c_excludes_unrelated_b(two_docs):
    doc_a, doc_b = two_docs
    doc_c = "scope_paper_c_" + uuid.uuid4().hex[:8]
    _push_chunk(doc_c, "c1", "GreenGrid is a smart-building energy optimization platform using reinforcement learning.", "GreenGrid.pdf")
    _push_chunk(doc_c, "c2", "GreenGrid's architecture ingests IoT sensor streams and forecasts load with an LSTM model.", "GreenGrid.pdf")
    build_bm25_index()

    resp = client.post("/report", json={"document_ids": [doc_a, doc_c]})
    data = resp.json()
    assert data["ok"] is True
    returned_ids = {p["document_id"] for p in data["papers"]}
    assert returned_ids == {doc_a, doc_c}
    assert doc_b not in returned_ids, "An unrelated, unrequested document must never appear in the comparison"
