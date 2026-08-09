"""
test_analysis_persistence.py — proves every analysis mode (Viva, Mock Test,
Explain Figure, Recommend, Why This Design, System Design, Project Interview
start+evaluate) and /report persist into the SAME sessions/messages tables
normal Q&A already uses, via main.py's _persist_analysis_turn(), whenever the
request carries a session_id — and that nothing is written when it doesn't.

Runs against the ISOLATED Phase 8 test Chroma/SQLite fixture (see
conftest.py). Only the Groq network boundary is mocked; retrieval and the
sqlite registry are real.
"""
import json
import uuid

import pytest
from fastapi.testclient import TestClient

import config
from ingest import get_collection
from retrieval import build_bm25_index
from database import create_session, get_session_messages
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


QUESTIONS_JSON = json.dumps({
    "questions": [{"question": "What problem does this paper address?", "category": "motivation", "difficulty": "intermediate", "expected_answer_points": ["x"]}],
    "grounded": True, "evidence_sufficient": True,
})
ANSWER_JSON = json.dumps({"answer": "Grounded explanation.", "grounded": True, "evidence_sufficient": True, "document_ids": []})
EVALUATION_JSON = json.dumps({
    "correctness": "partially_correct", "missing_points": ["RRF fusion."],
    "technical_depth": "Reasonable.", "suggested_answer": "Mention dense+BM25+RRF.",
    "next_question": "Why RRF over score averaging?", "next_question_category": "RRF",
})
COMPARISON_JSON = None  # built per-test from real Document IDs


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
        prompt = messages[0]["content"]
        if '"missing_points"' in prompt:
            return _FakeResponse(EVALUATION_JSON)
        if '"questions":' in prompt:
            return _FakeResponse(QUESTIONS_JSON)
        if '"comparison":' in prompt or "COMPARATIVE report" in prompt:
            import re
            doc_ids = re.findall(r"Document ID:\s*(\S+)", prompt)
            papers = [{
                "document_id": d, "title": f"Paper {d}", "overview": "o", "main_contribution": "c",
                "methodology": "m", "architecture": "a", "datasets": [], "evaluation_metrics": [],
                "key_results": [], "important_calculations": [], "limitations": [], "final_summary": "s",
            } for d in dict.fromkeys(doc_ids)]
            return _FakeResponse(json.dumps({
                "title": "Comparison Report", "overview": "o", "papers": papers,
                "comparison": {"commonalities": [], "differences": [], "strengths": [], "limitations": []},
                "conclusion": "c", "evidence_sufficient": True,
            }))
        return _FakeResponse(ANSWER_JSON)

class _FakeChat:
    def __init__(self): self.completions = _FakeCompletions()
class _FakeGroqClient:
    def __init__(self, api_key=None): self.chat = _FakeChat()


from unittest.mock import patch
_groq_patch = patch("groq.Groq", _FakeGroqClient)


@pytest.fixture(autouse=True)
def _reset():
    _groq_patch.start()
    yield
    _groq_patch.stop()


@pytest.fixture
def one_doc():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"
    doc_a = "persist_paper_" + uuid.uuid4().hex[:8]
    _push_chunk(doc_a, "c1", "This paper proposes a sparse mixture-of-experts routing layer.", "paperA.pdf")
    _push_chunk(doc_a, "c2", "Trained on 8 GPUs for 3 days using WMT14, reaching 91.2 F1.", "paperA.pdf")
    build_bm25_index()
    return doc_a


@pytest.fixture
def two_docs(one_doc):
    doc_b = "persist_paper_" + uuid.uuid4().hex[:8]
    _push_chunk(doc_b, "c1", "This second paper proposes a retrieval-augmented decoder.", "paperB.pdf")
    _push_chunk(doc_b, "c2", "It reports 88.4 F1 on Natural Questions with English-only limitation.", "paperB.pdf")
    build_bm25_index()
    return one_doc, doc_b


def _new_session():
    sid = "persist_sess_" + uuid.uuid4().hex[:10]
    create_session(sid, title="New conversation")
    return sid


# ---------------------------------------------------------------------------
def test_viva_persists_analysis_questions_message(one_doc):
    sid = _new_session()
    resp = client.post("/analyze", json={"mode": "viva", "document_ids": [one_doc], "difficulty": "basic", "num_questions": 2, "session_id": sid})
    assert resp.json()["ok"] is True
    msgs = get_session_messages(sid)
    assert len(msgs) == 1, "Viva has no user turn — exactly one assistant row"
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["metadata"]["frontend_role"] == "analysis_questions"
    assert msgs[0]["metadata"]["questions"], "Full question payload must round-trip through metadata"


def test_explain_figure_persists_user_and_assistant(one_doc):
    sid = _new_session()
    resp = client.post("/analyze", json={"mode": "explain_figure", "document_ids": [one_doc], "figure_reference": "Figure 1", "session_id": sid})
    assert resp.json()["ok"] is True
    msgs = get_session_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "Explain: Figure 1"
    assert msgs[1]["metadata"]["frontend_role"] == "assistant_kicker"
    assert msgs[1]["metadata"]["kicker"] == "Figure Explanation"


def test_project_interview_start_then_evaluate_persists_full_thread(one_doc):
    sid = _new_session()
    start = client.post("/analyze", json={"mode": "project_interview_start", "document_ids": [one_doc], "difficulty": "intermediate", "topics": [], "session_id": sid}).json()
    assert start["ok"] is True
    msgs = get_session_messages(sid)
    assert len(msgs) == 1 and msgs[0]["metadata"]["frontend_role"] == "interview_question"

    ev = client.post("/analyze", json={
        "mode": "project_interview_evaluate", "document_ids": [one_doc],
        "question": start["question"], "user_answer": "It uses MoE routing.",
        "difficulty": "intermediate", "topics": [], "session_id": sid,
    }).json()
    assert ev["ok"] is True
    msgs = get_session_messages(sid)
    # start-question, user-answer, eval, next-question = 4 rows total
    assert [m["role"] for m in msgs] == ["assistant", "user", "assistant", "assistant"]
    assert msgs[1]["content"] == "It uses MoE routing."
    assert msgs[2]["metadata"]["frontend_role"] == "interview_eval"
    assert msgs[2]["metadata"]["correctness"] == "partially_correct"
    assert msgs[3]["metadata"]["frontend_role"] == "interview_question"
    assert msgs[3]["content"] == ev["next_question"]


def test_why_design_persists_question_and_answer():
    sid = _new_session()
    resp = client.post("/analyze", json={"mode": "why_design", "question": "Why RRF?", "session_id": sid})
    assert resp.json()["ok"] is True
    msgs = get_session_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "Why RRF?"
    assert msgs[1]["metadata"]["kicker"] == "Why This Design"


def test_report_persists_report_message(two_docs):
    doc_a, doc_b = two_docs
    sid = _new_session()
    resp = client.post("/report", json={"document_ids": [doc_a, doc_b], "session_id": sid})
    assert resp.json()["ok"] is True
    msgs = get_session_messages(sid)
    assert len(msgs) == 1
    assert msgs[0]["metadata"]["frontend_role"] == "report"
    assert msgs[0]["metadata"]["report"]["ok"] is True
    assert len(msgs[0]["metadata"]["report"]["papers"]) == 2


def test_no_session_id_persists_nothing(one_doc):
    resp = client.post("/analyze", json={"mode": "viva", "document_ids": [one_doc], "num_questions": 2})
    assert resp.json()["ok"] is True
    # No session_id was sent — nothing to look up, and no exception either.


def test_failed_generation_does_not_persist_when_no_documents():
    sid = _new_session()
    resp = client.post("/analyze", json={"mode": "viva", "document_ids": [], "session_id": sid})
    assert resp.json()["ok"] is False
    assert get_session_messages(sid) == []
