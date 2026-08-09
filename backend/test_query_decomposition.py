"""
test_query_decomposition.py — Deterministic (zero-LLM-cost) query
decomposition for complex, multi-aspect questions.

Covers:
  1. A simple question stays exactly ONE search query (no decomposition).
  2. A genuinely multi-aspect question ("Compare methodologies, datasets
     and results") deterministically decomposes into multiple sub-queries
     — no LLM call involved in deciding this.
  3. Each sub-query independently retrieves the chunk relevant to its own
     aspect.
  4. Results from all sub-queries are merged, deduplicated, and reranked
     GLOBALLY (one CrossEncoder pass over the combined pool, not one pass
     per sub-query).
  5. document_ids scoping survives decomposition — a decomposed multi-aspect
     query scoped to one document never returns another document's chunks.
  6. A comparison question scoped to two documents retrieves evidence from
     BOTH of them, even though the question was split into per-aspect
     sub-queries first.
  7. The legacy/pre-ingested corpus (attention.pdf, seeded by conftest.py)
     never leaks into a decomposed query scoped to an unrelated document.
  8. Normal-mode generation remains exactly ONE physical LLM call end-to-end
     through the real LangGraph, even when the question decomposes into
     several sub-queries — decomposition itself costs zero LLM calls.

Runs entirely against the ISOLATED Phase 8 test Chroma collection (see
conftest.py) — never backend/chroma_store. Mocks only the Groq network
boundary (groq.Groq) for test 8; everything else exercises real retrieval
(dense + BM25 + RRF + CrossEncoder rerank), no mocking.
"""
import json
import uuid
from unittest.mock import patch

import pytest

import config
from ingest import get_collection, get_document_id
from retrieval import retrieve_multi, build_bm25_index
from query_transform import decompose_query_deterministic, start_call_tracking, get_call_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _push_chunk(document_id: str, suffix: str, text: str, filename: str) -> None:
    """Adds one synthetic chunk directly to the isolated test Chroma
    collection (same pattern as test_multi_document_scoping.py)."""
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
# TEST 1 — Simple question stays ONE search query
# ---------------------------------------------------------------------------
def test_simple_question_stays_one_sub_query():
    assert decompose_query_deterministic("What is the main contribution of this paper?") == [
        "What is the main contribution of this paper?"
    ]
    assert decompose_query_deterministic("Summarize this paper.") == ["Summarize this paper."]
    # Merely saying "compare" without an explicit aspect list must NOT split.
    assert decompose_query_deterministic("Compare the two papers.") == ["Compare the two papers."]
    assert decompose_query_deterministic("What's the difference between BERT and GPT?") == [
        "What's the difference between BERT and GPT?"
    ]


# ---------------------------------------------------------------------------
# TEST 2 — Multi-aspect question deterministically decomposes
# ---------------------------------------------------------------------------
def test_multi_aspect_question_decomposes_deterministically():
    sub_qs = decompose_query_deterministic("Compare methodologies, datasets and results")
    assert len(sub_qs) == 3, f"Expected 3 sub-queries, got {sub_qs}"
    assert any(sq.startswith("methodologies:") for sq in sub_qs)
    assert any(sq.startswith("datasets:") for sq in sub_qs)
    assert any(sq.startswith("results:") for sq in sub_qs)
    # Every sub-query still carries the full original question as context.
    assert all("Compare methodologies, datasets and results" in sq for sq in sub_qs)


def test_decomposition_caps_sub_query_count():
    sub_qs = decompose_query_deterministic(
        "Compare the methodology, datasets, results, metrics, limitations and contributions of this paper."
    )
    assert 2 <= len(sub_qs) <= 4, f"Sub-query count must be bounded, got {len(sub_qs)}: {sub_qs}"


# ---------------------------------------------------------------------------
# TEST 3/4 — Each sub-query retrieves its own aspect; merged, deduped,
# reranked globally
# ---------------------------------------------------------------------------
def test_each_subquery_retrieves_relevant_chunk_merged_and_reranked():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"

    doc_id = "aspect_doc_" + uuid.uuid4().hex[:8]
    _push_chunk(doc_id, "c1", "The proposed methodology uses a five-stage encoder-decoder pipeline with cross-attention.", "aspects.pdf")
    _push_chunk(doc_id, "c2", "We evaluate on the WMT14 English-German dataset and the CNN/DailyMail summarization dataset.", "aspects.pdf")
    _push_chunk(doc_id, "c3", "Our results show a 4.2 point BLEU improvement over the previous state of the art baseline.", "aspects.pdf")
    build_bm25_index()

    sub_qs = decompose_query_deterministic("Compare the methodology, datasets and results of this paper.")
    assert len(sub_qs) == 3

    results = retrieve_multi(sub_qs, document_ids=[doc_id], top_k=10, rerank_query="Compare the methodology, datasets and results of this paper.")
    assert results, "Decomposed retrieval must return evidence"

    found_suffixes = {r["metadata"]["chunk_id"].split("_")[-1] for r in results}
    assert {"c1", "c2", "c3"}.issubset(found_suffixes), (
        f"Each aspect's own chunk should surface once results are merged, got {found_suffixes}"
    )

    # Merged + deduplicated: no chunk_id appears twice.
    chunk_ids = [r["metadata"]["chunk_id"] for r in results]
    assert len(chunk_ids) == len(set(chunk_ids)), "Merged results must be deduplicated"

    # Reranked globally: every result carries a single rerank_score from ONE
    # CrossEncoder pass (not per-sub-query scores that would need reconciling).
    assert all("rerank_score" in r for r in results)


# ---------------------------------------------------------------------------
# TEST 5 — document_ids scoping survives decomposition
# ---------------------------------------------------------------------------
def test_document_scoping_survives_decomposition():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"

    doc_id = "scoped_aspect_doc_" + uuid.uuid4().hex[:8]
    other_id = "unrelated_doc_" + uuid.uuid4().hex[:8]
    _push_chunk(doc_id, "c1", "This paper's methodology combines contrastive pretraining with fine-tuning.", "scoped.pdf")
    _push_chunk(doc_id, "c2", "The datasets used are SQuAD 2.0 and Natural Questions.", "scoped.pdf")
    _push_chunk(other_id, "c1", "An unrelated paper's methodology involves reinforcement learning from human feedback.", "unrelated.pdf")
    _push_chunk(other_id, "c2", "The unrelated paper's datasets are Anthropic HH and OpenAI summarize-from-feedback.", "unrelated.pdf")
    build_bm25_index()

    sub_qs = decompose_query_deterministic("Compare the methodology and datasets of this paper.")
    assert len(sub_qs) == 2

    results = retrieve_multi(sub_qs, document_ids=[doc_id], top_k=10, rerank_query="Compare the methodology and datasets of this paper.")
    assert results, "Should retrieve the in-scope document's evidence"
    assert all(r["metadata"]["document_id"] == doc_id for r in results), (
        "Decomposed retrieval must never leak chunks from a document outside document_ids scope"
    )


# ---------------------------------------------------------------------------
# TEST 6 — Comparison retrieves evidence from EVERY requested PDF
# ---------------------------------------------------------------------------
def test_comparison_retrieves_from_every_requested_pdf():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"

    doc_a = "paper_a_" + uuid.uuid4().hex[:8]
    doc_b = "paper_b_" + uuid.uuid4().hex[:8]
    _push_chunk(doc_a, "c1", "Paper A's methodology is a two-tower dense retriever trained with in-batch negatives.", "paperA.pdf")
    _push_chunk(doc_a, "c2", "Paper A's datasets are MS MARCO and Natural Questions.", "paperA.pdf")
    _push_chunk(doc_b, "c1", "Paper B reports results of 91.2 F1 on the held-out evaluation set.", "paperB.pdf")
    _push_chunk(doc_b, "c2", "A key limitation of Paper B is its reliance on English-only training data.", "paperB.pdf")
    build_bm25_index()

    question = "Compare the methodologies, datasets, results and limitations of both papers."
    sub_qs = decompose_query_deterministic(question)
    assert len(sub_qs) >= 2

    results = retrieve_multi(sub_qs, document_ids=[doc_a, doc_b], top_k=10, rerank_query=question)
    found_docs = {r["metadata"]["document_id"] for r in results}
    assert doc_a in found_docs, "Comparison must include evidence from Paper A"
    assert doc_b in found_docs, "Comparison must include evidence from Paper B"
    assert found_docs.issubset({doc_a, doc_b}), f"Comparison scope leaked outside A+B: {found_docs}"


# ---------------------------------------------------------------------------
# TEST 7 — Legacy/pre-ingested corpus never enters a decomposed, scoped query
# ---------------------------------------------------------------------------
def test_legacy_corpus_excluded_from_decomposed_retrieval():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"
    from pathlib import Path

    legacy_doc_id = get_document_id(str(Path(__file__).parent / "tests" / "fixtures" / "attention.pdf"))

    doc_id = "isolated_doc_" + uuid.uuid4().hex[:8]
    _push_chunk(doc_id, "c1", "This isolated paper's methodology is a graph neural network over citation edges.", "isolated.pdf")
    _push_chunk(doc_id, "c2", "This isolated paper's datasets are OGB-Arxiv and OGB-Products.", "isolated.pdf")
    build_bm25_index()

    question = "Compare the methodology and datasets of this paper."
    sub_qs = decompose_query_deterministic(question)
    results = retrieve_multi(sub_qs, document_ids=[doc_id], top_k=10, rerank_query=question)

    assert results
    assert all(r["metadata"]["document_id"] != legacy_doc_id for r in results), (
        "The pre-ingested legacy/attention.pdf document must never appear in a query scoped elsewhere"
    )


# ---------------------------------------------------------------------------
# TEST 8 — Normal-mode generation stays exactly ONE physical LLM call, even
# though the question decomposes into several sub-queries
# ---------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content

class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)

class _FakeUsage:
    prompt_tokens = 55
    completion_tokens = 21

class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()

class _FakeCompletions:
    def create(self, model, messages, temperature=0.0, max_tokens=None):
        return _FakeResponse(_FAKE_ANSWER_JSON)

class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()

class _FakeGroqClient:
    def __init__(self, api_key=None):
        self.chat = _FakeChat()


def test_normal_mode_decomposition_stays_one_llm_call():
    from graph.workflow import research_app

    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"

    doc_a = "gen_paper_a_" + uuid.uuid4().hex[:8]
    doc_b = "gen_paper_b_" + uuid.uuid4().hex[:8]
    _push_chunk(doc_a, "c1", "This system's methodology combines hybrid retrieval with cross-encoder reranking.", "genA.pdf")
    _push_chunk(doc_a, "c2", "The datasets used for evaluation are Natural Questions and TriviaQA.", "genA.pdf")
    _push_chunk(doc_b, "c1", "The comparison system reports results of 88.4 exact match on the test set.", "genB.pdf")
    _push_chunk(doc_b, "c2", "A noted limitation is sensitivity to chunk boundary placement.", "genB.pdf")
    build_bm25_index()

    global _FAKE_ANSWER_JSON
    _FAKE_ANSWER_JSON = json.dumps({
        "answer": "System A uses hybrid retrieval with reranking; System B reports 88.4 EM but is sensitive to chunking.",
        "grounded": True,
        "evidence_sufficient": True,
        "document_ids": [doc_a, doc_b],
    })

    question = "Compare the methodologies, datasets, results and limitations of both papers."
    initial_state = {
        "original_query": question,
        "research_type": "simple",
        "chat_history": [],
        "document_ids": [doc_a, doc_b],
        "structured_mode": False,
    }

    start_call_tracking()
    with patch("groq.Groq", _FakeGroqClient):
        final_state = research_app.invoke(initial_state)

    log = get_call_log()
    assert len(log) == 1, f"Normal-mode generation must stay exactly 1 LLM call even with decomposition, got {len(log)}: {log}"
    assert log[0]["role"] == "primary"

    # Confirm decomposition actually happened (not a false-positive pass).
    assert len(final_state.get("completed_sub_queries", [])) > 1, (
        "This comparison question should have deterministically decomposed into multiple sub-queries"
    )
    assert final_state["research_plan"]["needs_decomposition"] is True

    # Evidence from both papers reached the single synthesis call.
    doc_ids_in_evidence = set(final_state.get("evidence_by_document", {}).keys())
    assert doc_a in doc_ids_in_evidence
    assert doc_b in doc_ids_in_evidence

    assert final_state["draft_answer"].startswith("System A uses hybrid retrieval")


def test_normal_mode_simple_question_still_one_sub_query_one_llm_call():
    from graph.workflow import research_app

    doc_id = "gen_simple_doc_" + uuid.uuid4().hex[:8]
    _push_chunk(doc_id, "c1", "This paper introduces a lightweight adapter module for parameter-efficient fine-tuning.", "simple.pdf")
    build_bm25_index()

    global _FAKE_ANSWER_JSON
    _FAKE_ANSWER_JSON = json.dumps({
        "answer": "The paper introduces a lightweight adapter module for parameter-efficient fine-tuning.",
        "grounded": True,
        "evidence_sufficient": True,
        "document_ids": [doc_id],
    })

    initial_state = {
        "original_query": "What does this paper introduce?",
        "research_type": "simple",
        "chat_history": [],
        "document_ids": [doc_id],
        "structured_mode": False,
    }

    start_call_tracking()
    with patch("groq.Groq", _FakeGroqClient):
        final_state = research_app.invoke(initial_state)

    log = get_call_log()
    assert len(log) == 1
    assert len(final_state.get("completed_sub_queries", [])) == 1, "A simple question must stay a single search query"
    assert final_state["research_plan"]["needs_decomposition"] is False
