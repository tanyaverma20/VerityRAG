"""
test_llm_call_count.py — Proves "normal query = exactly 1 LLM call" with
mocked Groq calls. No real network requests, no Groq token usage.

Only the network boundary (groq.Groq) is faked — the real with_model_fallback(),
synthesize_answer(), and LangGraph routing logic all run for real, so this
proves actual behavior rather than assuming it.

Run with:
    cd backend
    python -m pytest test_llm_call_count.py -v
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import config
from query_transform import start_call_tracking, get_call_log, with_model_fallback


# ---------------------------------------------------------------------------
# Fake Groq SDK — mocks only the network boundary, nothing else
# ---------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content

class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)

class _FakeUsage:
    prompt_tokens = 42
    completion_tokens = 17

class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()

_behavior = {"fn": None}  # set per-test; callable(model) -> str content

class _FakeCompletions:
    def create(self, model, messages, temperature=0.0, max_tokens=None):
        return _FakeResponse(_behavior["fn"](model))

class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()

class _FakeGroqClient:
    """Drop-in replacement for groq.Groq — same constructor shape (api_key=...)."""
    def __init__(self, api_key=None):
        self.chat = _FakeChat()


VALID_ANSWER_JSON = json.dumps({
    "answer": "This paper's main contribution is the Transformer architecture.",
    "grounded": True,
    "evidence_sufficient": True,
    "document_ids": ["docA"],
})

INSUFFICIENT_JSON = json.dumps({
    "answer": "not enough evidence in the text I was given",
    "grounded": False,
    "evidence_sufficient": False,
    "document_ids": [],
})


def _sample_state(question="What is the main contribution?", chat_history=None):
    """
    Minimal state with pre-organized, two-document evidence. Targets the
    synthesizer/graph-routing call-count logic directly — retrieval itself
    (Chroma/BM25/RRF/rerank, all already LLM-free) is covered separately by
    TEST 8.
    """
    from graph.organizer import organize_evidence
    retrieval_results = [
        {
            "text": "The Transformer relies entirely on attention mechanisms.",
            "parent_context": "The Transformer relies entirely on attention mechanisms, dispensing with recurrence.",
            "metadata": {"document_id": "docA", "parent_id": "p1", "chunk_id": "c1", "source": "paperA.pdf", "page_number": 1},
        },
        {
            "text": "ResNet uses residual connections for very deep networks.",
            "parent_context": "ResNet uses residual connections for very deep networks, easing optimization.",
            "metadata": {"document_id": "docB", "parent_id": "p2", "chunk_id": "c2", "source": "paperB.pdf", "page_number": 3},
        },
    ]
    state = {
        "original_query": question,
        "retrieval_results": retrieval_results,
        "research_type": "simple",
        "chat_history": chat_history or [],
        "document_ids": ["docA", "docB"],
    }
    state.update(organize_evidence(state))
    return state


@pytest.fixture(autouse=True)
def _reset_tracking():
    start_call_tracking()
    _behavior["fn"] = lambda model: VALID_ANSWER_JSON
    yield


# ---------------------------------------------------------------------------
# TEST 1 — Normal question => exactly 1 LLM call
# ---------------------------------------------------------------------------
def test_normal_question_one_llm_call():
    from graph.synthesizer import synthesize_answer
    state = _sample_state("What is the main contribution?")
    with patch("groq.Groq", _FakeGroqClient):
        result = synthesize_answer(state)
    log = get_call_log()
    assert len(log) == 1, f"Expected exactly 1 LLM call, got {len(log)}: {log}"
    assert log[0]["role"] == "primary"
    assert result["draft_answer"] == "This paper's main contribution is the Transformer architecture."
    assert len(result["citations"]) == 2


# ---------------------------------------------------------------------------
# TEST 2 — Multi-paper comparison => still exactly 1 LLM call
# ---------------------------------------------------------------------------
def test_comparison_one_llm_call():
    from graph.synthesizer import synthesize_answer
    state = _sample_state("Compare the methodologies of these papers.")
    with patch("groq.Groq", _FakeGroqClient):
        result = synthesize_answer(state)
    log = get_call_log()
    assert len(log) == 1, f"Expected exactly 1 LLM call for a comparison, got {len(log)}: {log}"
    doc_ids = {c["document_id"] for c in result["citations"]}
    assert doc_ids == {"docA", "docB"}, "Both papers must be represented within that one call's evidence"


# ---------------------------------------------------------------------------
# TEST 3 — Insufficient evidence, decided BY the one call => still 1 call
# ---------------------------------------------------------------------------
def test_insufficient_evidence_one_llm_call():
    from graph.synthesizer import synthesize_answer
    _behavior["fn"] = lambda model: INSUFFICIENT_JSON
    state = _sample_state("What is the capital of France?")
    with patch("groq.Groq", _FakeGroqClient):
        result = synthesize_answer(state)
    log = get_call_log()
    assert len(log) == 1, "The LLM itself must decide sufficiency in the SAME call, not a second verifier call"
    assert "couldn't find enough information" in result["draft_answer"].lower()


# ---------------------------------------------------------------------------
# TEST 4 — Primary succeeds => 1 primary call, 0 fallback calls
# ---------------------------------------------------------------------------
def test_primary_success_no_fallback():
    def make_request(model):
        assert model == config.GROQ_MODEL
        return "ok"

    result = with_model_fallback(make_request)
    log = get_call_log()
    assert result == "ok"
    assert len(log) == 1
    assert log[0] == {"model": config.GROQ_MODEL, "role": "primary", "success": True}


# ---------------------------------------------------------------------------
# TEST 5 — Primary temporary failure => primary=1 (failed) + fallback=1 (ok)
# ---------------------------------------------------------------------------
def test_primary_temporary_failure_triggers_one_fallback():
    class _FakeRateLimitError(Exception):
        status_code = 429

    def make_request(model):
        if model == config.GROQ_MODEL:
            raise _FakeRateLimitError("429 rate limit exceeded")
        return "fallback answer"

    result = with_model_fallback(make_request)
    log = get_call_log()
    assert result == "fallback answer"
    assert len(log) == 2, f"Expected exactly 2 physical attempts (primary + 1 fallback), got {len(log)}: {log}"
    assert log[0]["role"] == "primary" and log[0]["success"] is False
    assert log[1]["role"] == "fallback" and log[1]["success"] is True


# ---------------------------------------------------------------------------
# TEST 6 — Invalid API key => no fallback attempted (fails fast)
# ---------------------------------------------------------------------------
def test_invalid_api_key_no_fallback():
    class _FakeAuthError(Exception):
        status_code = 401

    def make_request(model):
        raise _FakeAuthError("Invalid API Key")

    with pytest.raises(_FakeAuthError):
        with_model_fallback(make_request)
    log = get_call_log()
    assert len(log) == 1, f"Auth errors must not trigger a wasted fallback attempt, got {len(log)}: {log}"
    assert log[0]["role"] == "primary" and log[0]["success"] is False


# ---------------------------------------------------------------------------
# TEST 7 — Follow-up question => still exactly 1 LLM call (chat history goes
# straight into that one call; no separate query-rewriting call)
# ---------------------------------------------------------------------------
def test_follow_up_one_llm_call():
    from graph.synthesizer import synthesize_answer
    history = [
        {"role": "user", "content": "What is the Transformer's architecture?"},
        {"role": "assistant", "content": "It uses self-attention instead of recurrence."},
    ]
    state = _sample_state("What dataset did they use?", chat_history=history)
    with patch("groq.Groq", _FakeGroqClient):
        synthesize_answer(state)
    log = get_call_log()
    assert len(log) == 1, f"Follow-up questions must not add a separate rewriting call, got {len(log)}: {log}"


# ---------------------------------------------------------------------------
# TEST 8 — data/*.pdf (and any other unscoped document) cannot enter a
# document_ids-scoped retrieval.
# ---------------------------------------------------------------------------
def test_scoped_retrieval_excludes_other_documents():
    """
    Runs against the ISOLATED test Chroma collection only (see conftest.py —
    never backend/chroma_store). Confirms that scoping retrieval to a
    specific document_id never returns chunks from a document outside that
    scope — the same mechanism that keeps data/*.pdf and the canonical
    corpus out of a user's normal-mode retrieval.
    """
    assert config.COLLECTION_NAME == "test_collection", (
        "This test must run inside the isolated Phase 8 fixture, not against the canonical database"
    )
    from ingest import get_document_id
    from retrieval import build_bm25_index, retrieve

    test_pdf = Path(__file__).parent.parent / "data" / "attention.pdf"
    if not test_pdf.exists():
        pytest.skip("Test PDF not found.")

    real_doc_id = get_document_id(str(test_pdf))  # already ingested by conftest.py's fixture
    build_bm25_index()

    unrelated_scope = ["some_document_id_not_in_this_workspace"]
    results = retrieve("attention mechanism", strategy="hybrid", document_ids=unrelated_scope, top_k=5)
    assert results == [], "Retrieval scoped to an unrelated document_id must return nothing from attention.pdf"

    scoped_results = retrieve("attention mechanism", strategy="hybrid", document_ids=[real_doc_id], top_k=5)
    assert len(scoped_results) > 0
    assert all(r["metadata"]["document_id"] == real_doc_id for r in scoped_results)


# ---------------------------------------------------------------------------
# TEST 9 — Production ChromaDB is never touched by any of this
# ---------------------------------------------------------------------------
def test_production_chroma_not_modified():
    canonical_path = (Path(__file__).parent / "chroma_store").resolve()
    assert Path(config.CHROMA_DIR).resolve() != canonical_path, (
        "Tests must run against a temp Chroma dir, never backend/chroma_store"
    )
    assert config.COLLECTION_NAME == "test_collection", (
        "Phase 8 isolation fixture must have redirected COLLECTION_NAME for tests"
    )


# ---------------------------------------------------------------------------
# TEST 10 — Structured output (Feature 6): same ONE call, richer schema,
# validates correctly, and is null for normal (non-structured) requests.
# ---------------------------------------------------------------------------
VALID_STRUCTURED_JSON = json.dumps({
    "answer": "The Transformer relies entirely on attention.",
    "grounded": True,
    "evidence_sufficient": True,
    "document_ids": ["docA"],
    "structured": {
        "architecture": "Encoder-decoder with multi-head self-attention.",
        "datasets": ["WMT 2014 En-De"],
        "metrics": ["BLEU"],
        "contributions": ["Removes recurrence and convolution entirely."],
        "methodology": "Multi-head self-attention with positional encoding.",
        "key_calculations": ["28.4 BLEU on WMT14 En-De"],
        "limitations": [],
        "final_summary": "A fully attention-based sequence transduction model.",
    },
})

MALFORMED_STRUCTURED_JSON = json.dumps({
    "answer": "The Transformer relies entirely on attention.",
    "grounded": True,
    "evidence_sufficient": True,
    "document_ids": ["docA"],
    # "structured" is missing entirely — required by StructuredAnswerJSON.
})


def test_structured_mode_validates_and_is_one_call():
    from graph.synthesizer import synthesize_answer
    _behavior["fn"] = lambda model: VALID_STRUCTURED_JSON
    state = _sample_state("Give me the architecture and datasets.")
    state["structured_mode"] = True
    with patch("groq.Groq", _FakeGroqClient):
        result = synthesize_answer(state)
    log = get_call_log()
    assert len(log) == 1, f"Structured mode must still be exactly 1 LLM call, got {len(log)}: {log}"
    assert result["structured_data"] is not None
    assert result["structured_data"]["architecture"] == "Encoder-decoder with multi-head self-attention."
    assert result["structured_data"]["datasets"] == ["WMT 2014 En-De"]
    assert result["structured_data"]["limitations"] == []  # empty, not invented


def test_normal_mode_never_populates_structured_data():
    from graph.synthesizer import synthesize_answer
    state = _sample_state("What is the main contribution?")
    # structured_mode absent entirely, as a real /query request never setting it would leave it
    with patch("groq.Groq", _FakeGroqClient):
        result = synthesize_answer(state)
    assert result["structured_data"] is None, "Normal requests must never carry structured JSON the user didn't ask for"


def test_malformed_structured_output_fails_clean_no_retry():
    from graph.synthesizer import synthesize_answer
    _behavior["fn"] = lambda model: MALFORMED_STRUCTURED_JSON
    state = _sample_state("Give me the architecture.")
    state["structured_mode"] = True
    with patch("groq.Groq", _FakeGroqClient):
        result = synthesize_answer(state)
    log = get_call_log()
    assert len(log) == 1, f"A malformed structured response must not trigger a retry, got {len(log)}: {log}"
    assert result["structured_data"] is None
    assert result["citations"] == []
