"""
test_natural_qa.py — Verifies natural, arbitrary question-answering works for
ANY uploaded research PDF (not just predefined actions like "Summarize" /
"Find datasets" / "Find limitations").

Root-cause check performed before writing this file: inspected
graph/planner.py, graph/retriever.py, retrieval.py, graph/synthesizer.py,
and frontend/index.html's sendQuestion()/resolveQueryScope(). None of them
hardcode or restrict to a fixed set of question types —
  - the frontend composer sends whatever free text the user types (the
    quick-action chips are just autofill suggestions, not the only allowed
    input — see index.html's useSuggestion()/sendQuestion());
  - QueryRequest.question is an unconstrained string;
  - decompose_query_deterministic() only ever adds MORE sub-queries for a
    genuinely multi-aspect question — a normal question always passes
    through as a single, verbatim search query;
  - SYNTHESIS_PROMPT explicitly instructs the model to "adapt the structure
    to the question instead of forcing a fixed template."
So arbitrary natural Q&A already works structurally; this file exists to
prove it end-to-end against one real PDF rather than assume it.

Runs against the ISOLATED Phase 8 test Chroma/SQLite fixture (see
conftest.py) — never backend/chroma_store. Retrieval (dense + BM25 + RRF +
CrossEncoder rerank) is REAL and unmocked; only the Groq network boundary is
mocked, and the fake echoes the literal question text it received back into
its answer — proof the exact natural-language question (not a canned action
id) reached the LLM, without needing real Groq quota.
"""
import re
import uuid
from unittest.mock import patch

import pytest

import config
from ingest import get_document_id
from retrieval import build_bm25_index, get_collection
from query_transform import start_call_tracking, get_call_log, INSUFFICIENT_EVIDENCE_MESSAGE

ATTENTION_PDF = None  # resolved in a fixture below


@pytest.fixture(scope="module")
def attention_doc_id():
    from pathlib import Path
    path = Path(__file__).parent / "tests" / "fixtures" / "attention.pdf"
    return get_document_id(str(path))  # already ingested by conftest.py


# ---------------------------------------------------------------------------
# Fake Groq — echoes the question back into the answer so we can prove the
# exact natural-language text reached generation, and lets a test flip
# evidence_sufficient to simulate an honest "not in this paper" refusal.
# ---------------------------------------------------------------------------
import json

class _FakeMessage:
    def __init__(self, content):
        self.content = content

class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)

class _FakeUsage:
    prompt_tokens = 60
    completion_tokens = 25

class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()

_behavior = {"insufficient": False}

class _FakeCompletions:
    def create(self, model, messages, temperature=0.0, max_tokens=None):
        prompt = messages[0]["content"]
        q_match = re.search(r"QUESTION:\s*(.+)", prompt)
        question_seen = q_match.group(1).strip() if q_match else ""
        doc_ids = re.findall(r"Document ID:\s*(\S+)", prompt)

        if _behavior["insufficient"]:
            payload = {
                "answer": "not enough information in the evidence provided",
                "grounded": False,
                "evidence_sufficient": False,
                "document_ids": [],
            }
        else:
            payload = {
                "answer": f"[Answering: {question_seen}] Based on the evidence, here is the grounded answer.",
                "grounded": True,
                "evidence_sufficient": True,
                "document_ids": doc_ids,
            }
        return _FakeResponse(json.dumps(payload))

class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()

class _FakeGroqClient:
    def __init__(self, api_key=None):
        self.chat = _FakeChat()


@pytest.fixture(autouse=True)
def _reset():
    _behavior["insufficient"] = False
    start_call_tracking()
    yield


def _run(question: str, document_ids: list[str]):
    from graph.workflow import research_app
    initial_state = {
        "original_query": question,
        "research_type": "simple",
        "chat_history": [],
        "document_ids": document_ids,
        "structured_mode": False,
    }
    with patch("groq.Groq", _FakeGroqClient):
        return research_app.invoke(initial_state)


# ---------------------------------------------------------------------------
# Seven natural questions against ONE real uploaded PDF
# ---------------------------------------------------------------------------
NATURAL_QUESTIONS = [
    ("factual", "What is the main contribution of this paper?"),
    ("methodology", "Explain the proposed methodology."),
    ("dataset", "What datasets were used to evaluate the model?"),
    ("result", "What were the main results reported in the paper?"),
    ("explanation", "Why did the authors choose an attention-based approach instead of recurrence?"),
    ("limitation", "What are the limitations of the proposed approach?"),
]


@pytest.mark.parametrize("category,question", NATURAL_QUESTIONS)
def test_arbitrary_natural_question_answered_and_grounded(category, question, attention_doc_id):
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"
    build_bm25_index()

    final_state = _run(question, [attention_doc_id])

    # Exactly one physical LLM call, regardless of question phrasing/category.
    log = get_call_log()
    assert len(log) == 1, f"[{category}] Normal Q&A must be exactly 1 LLM call, got {len(log)}"

    # Retrieval actually ran (real dense+BM25+RRF+rerank, no mocking) and
    # returned document-scoped evidence.
    chunks = final_state.get("retrieval_results", [])
    assert chunks, f"[{category}] No evidence retrieved for a natural question against a real, ingested PDF"
    assert all(c["metadata"]["document_id"] == attention_doc_id for c in chunks), (
        f"[{category}] Retrieval must stay scoped to the uploaded PDF"
    )
    assert all("rerank_score" in c for c in chunks), f"[{category}] CrossEncoder reranking must have run"
    assert all(c.get("source_method") in ("dense", "bm25", "dense+bm25") for c in chunks), (
        f"[{category}] Hybrid dense+BM25 retrieval (fused by RRF) must have run"
    )

    # The literal question text — not a canned action id — reached synthesis.
    assert question in final_state["draft_answer"], (
        f"[{category}] The exact natural-language question must reach the LLM call verbatim"
    )
    assert final_state["grounded"] is True
    assert final_state["evidence_sufficient"] is True
    assert len(final_state["citations"]) > 0


def test_question_not_covered_by_the_pdf_gets_honest_insufficient_evidence(attention_doc_id):
    """
    Category 7: a question whose answer is genuinely absent from the paper.
    Retrieval still returns its top-k nearest chunks (that's how vector
    search works — it always returns *something*), but the model's own
    honest self-assessment (evidence_sufficient=False, exercised here via
    the fake) must produce the canonical insufficient-evidence message, not
    a fabricated answer.
    """
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"
    build_bm25_index()
    _behavior["insufficient"] = True

    question = "What was the exact carbon footprint in kilograms of training this model?"
    final_state = _run(question, [attention_doc_id])

    log = get_call_log()
    assert len(log) == 1, "Insufficient-evidence handling must still cost exactly 1 LLM call, not a retry"
    assert final_state["draft_answer"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert final_state["evidence_sufficient"] is False
    # Citations stay attached even here: the model DID run and genuinely
    # reviewed this evidence before honestly deciding it wasn't enough — that
    # is different from a failed/rate-limited generation (which shows no
    # citations at all, see graph/synthesizer.py's synthesis_succeeded gate).
    assert len(final_state["citations"]) > 0, "The evidence the model actually reviewed should stay visible"


# ---------------------------------------------------------------------------
# Two-PDF natural-question scoping: A alone -> only A, B alone -> only B,
# comparison -> both
# ---------------------------------------------------------------------------
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


def test_two_pdf_natural_question_scoping():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"

    doc_a = "qa_paper_a_" + uuid.uuid4().hex[:8]
    doc_b = "qa_paper_b_" + uuid.uuid4().hex[:8]
    _push_chunk(doc_a, "c1", "Paper A proposes a sparse mixture-of-experts routing layer for efficient scaling.", "paperA.pdf")
    _push_chunk(doc_b, "c1", "Paper B proposes a retrieval-augmented decoder that conditions generation on external passages.", "paperB.pdf")
    build_bm25_index()

    # Question about A alone -> only A's evidence
    state_a = _run("What does this paper propose?", [doc_a])
    assert state_a["retrieval_results"], "Should retrieve A's own evidence"
    assert all(c["metadata"]["document_id"] == doc_a for c in state_a["retrieval_results"])

    # Question about B alone -> only B's evidence
    state_b = _run("What does this paper propose?", [doc_b])
    assert state_b["retrieval_results"], "Should retrieve B's own evidence"
    assert all(c["metadata"]["document_id"] == doc_b for c in state_b["retrieval_results"])

    # Natural comparison question, scoped to both -> evidence from BOTH
    state_cmp = _run("Compare the approaches proposed in these two papers.", [doc_a, doc_b])
    found_docs = {c["metadata"]["document_id"] for c in state_cmp["retrieval_results"]}
    assert doc_a in found_docs and doc_b in found_docs, f"Comparison must draw evidence from both papers, got {found_docs}"
    assert found_docs.issubset({doc_a, doc_b}), f"Comparison scope leaked outside A+B: {found_docs}"
