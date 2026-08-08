"""
test_eval.py — Phase 4 Evaluation Test Suite

At least 20 meaningful tests with real assertions.
No tests that merely check whether a function exists.

Run with:
    cd backend
    python test_eval.py
"""

import sys
import json
import traceback
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def run_test(name: str, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  PASS  {name}")
    except AssertionError as e:
        _results.append((name, False, str(e)))
        print(f"  FAIL  {name} — {e}")
    except Exception as e:
        _results.append((name, False, f"Exception: {e}"))
        print(f"  FAIL  {name} — {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))

from eval_metrics import (
    recall_at_k,
    precision_at_k,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    retrieval_metrics_suite,
    concept_coverage,
    citation_grounding_suite,
    resource_metrics,
    NOT_AVAILABLE,
)
from observability import log_query_event, read_recent_events, get_log_path


# ---------------------------------------------------------------------------
# TEST 01 — Recall@K correct calculation
# ---------------------------------------------------------------------------
def test_recall_at_k_correct():
    retrieved = ["doc_A", "doc_B", "doc_C", "doc_D"]
    relevant = ["doc_A", "doc_C", "doc_E"]
    # retrieved[:3] = ["doc_A","doc_B","doc_C"] → 2 hits / 3 relevant = 0.6667
    assert recall_at_k(retrieved, relevant, k=3) == round(2/3, 4), \
        f"Expected {round(2/3,4)}, got {recall_at_k(retrieved, relevant, k=3)}"


# ---------------------------------------------------------------------------
# TEST 02 — Recall@K returns NOT_AVAILABLE for empty relevant
# ---------------------------------------------------------------------------
def test_recall_empty_relevant():
    assert recall_at_k(["a", "b"], [], k=3) == NOT_AVAILABLE


# ---------------------------------------------------------------------------
# TEST 03 — Precision@K correct calculation
# ---------------------------------------------------------------------------
def test_precision_at_k_correct():
    retrieved = ["doc_A", "doc_B", "doc_C", "doc_D"]
    relevant = ["doc_A", "doc_C"]
    # top 3: A,B,C → 2 hits / 3 = 0.6667
    assert precision_at_k(retrieved, relevant, k=3) == round(2/3, 4)


# ---------------------------------------------------------------------------
# TEST 04 — Hit Rate@K correct calculation
# ---------------------------------------------------------------------------
def test_hit_rate_at_k():
    retrieved = ["doc_X", "doc_Y", "doc_Z"]
    relevant = ["doc_Z", "doc_W"]
    assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0
    assert hit_rate_at_k(retrieved, relevant, k=2) == 0.0  # doc_Z is at rank 3


# ---------------------------------------------------------------------------
# TEST 05 — MRR correct calculation
# ---------------------------------------------------------------------------
def test_mrr_correct():
    # first hit at rank 2 → 1/2 = 0.5
    assert mrr(["doc_A", "doc_B", "doc_C"], ["doc_B"]) == 0.5
    # first hit at rank 1 → 1.0
    assert mrr(["doc_B", "doc_A"], ["doc_B"]) == 1.0
    # no hit → 0.0
    assert mrr(["doc_X", "doc_Y"], ["doc_Z"]) == 0.0


# ---------------------------------------------------------------------------
# TEST 06 — MRR returns NOT_AVAILABLE for empty relevant
# ---------------------------------------------------------------------------
def test_mrr_empty_relevant():
    assert mrr(["doc_A"], []) == NOT_AVAILABLE


# ---------------------------------------------------------------------------
# TEST 07 — nDCG@K correct calculation (binary relevance)
# ---------------------------------------------------------------------------
def test_ndcg_at_k_correct():
    # Perfect ordering: relevant doc at rank 1 → nDCG = 1.0
    assert ndcg_at_k(["doc_A", "doc_B"], ["doc_A"], k=2) == 1.0
    # Worst in range: relevant at rank 2, not 1
    score = ndcg_at_k(["doc_B", "doc_A"], ["doc_A"], k=2)
    dcg = 1.0 / math.log2(3)   # i=1, denominator log2(1+2)=log2(3)
    idcg = 1.0 / math.log2(2)  # ideal: rank 1
    expected = round(dcg / idcg, 4)
    assert score == expected, f"Expected {expected}, got {score}"


# ---------------------------------------------------------------------------
# TEST 08 — nDCG@K returns NOT_AVAILABLE for empty relevant
# ---------------------------------------------------------------------------
def test_ndcg_empty_relevant():
    assert ndcg_at_k(["doc_A"], [], k=3) == NOT_AVAILABLE


# ---------------------------------------------------------------------------
# TEST 09 — Retrieval metrics suite returns all keys
# ---------------------------------------------------------------------------
def test_metrics_suite_keys():
    result = retrieval_metrics_suite(["d1", "d2", "d3", "d4", "d5"], ["d1", "d3"], k_values=[1, 3, 5])
    required_keys = ["recall@1", "recall@3", "recall@5",
                     "precision@1", "precision@3", "precision@5",
                     "hit_rate@1", "hit_rate@3", "hit_rate@5",
                     "ndcg@1", "ndcg@3", "ndcg@5", "mrr"]
    for k in required_keys:
        assert k in result, f"Missing key: {k}"


# ---------------------------------------------------------------------------
# TEST 10 — Document-level vs chunk-level metrics are independent
# ---------------------------------------------------------------------------
def test_doc_level_metrics():
    # Simulates: 3 relevant documents, retrieved 2 of them
    retrieved_docs = ["docA", "docC"]
    relevant_docs = ["docA", "docB", "docC"]
    r = recall_at_k(retrieved_docs, relevant_docs, k=5)
    assert r == round(2/3, 4), f"Document-level recall should be 2/3, got {r}"

    # Chunk-level independently
    retrieved_chunks = ["c1", "c2", "c9"]
    relevant_chunks = ["c1", "c5"]
    r2 = recall_at_k(retrieved_chunks, relevant_chunks, k=3)
    assert r2 == 0.5, f"Chunk-level recall should be 0.5, got {r2}"


# ---------------------------------------------------------------------------
# TEST 11 — Eval dataset schema validation
# ---------------------------------------------------------------------------
def test_eval_dataset_schema():
    eval_path = Path(__file__).parent.parent / "data" / "eval_set.json"
    assert eval_path.exists(), "eval_set.json must exist"
    data = json.loads(eval_path.read_text())
    assert len(data) >= 5, f"Eval set too small: {len(data)} questions"
    required_fields = ["question", "question_type", "expected_answer_contains",
                       "relevant_document_ids", "comparison_required"]
    for item in data:
        for f in required_fields:
            assert f in item, f"Missing field '{f}' in question: {item.get('question', '?')}"
        # relevant_document_ids must support arbitrary length (list)
        assert isinstance(item["relevant_document_ids"], list)


# ---------------------------------------------------------------------------
# TEST 12 — Eval dataset supports multi-document questions
# ---------------------------------------------------------------------------
def test_eval_dataset_multi_doc():
    eval_path = Path(__file__).parent.parent / "data" / "eval_set.json"
    data = json.loads(eval_path.read_text())
    multi_doc = [q for q in data if len(q.get("relevant_document_ids", [])) > 1]
    assert len(multi_doc) >= 3, f"Need at least 3 multi-document questions, found {len(multi_doc)}"


# ---------------------------------------------------------------------------
# TEST 13 — Concept coverage correct calculation
# ---------------------------------------------------------------------------
def test_concept_coverage_correct():
    answer = "The Transformer uses an encoder-decoder architecture with multi-head attention."
    expected = ["encoder", "decoder", "attention", "recurrent"]
    result = concept_coverage(answer, expected)
    assert result["found"] == ["encoder", "decoder", "attention"], \
        f"Unexpected found: {result['found']}"
    assert result["missing"] == ["recurrent"]
    assert result["coverage_ratio"] == round(3/4, 4)


# ---------------------------------------------------------------------------
# TEST 14 — Concept coverage returns NOT_AVAILABLE with no expected concepts
# ---------------------------------------------------------------------------
def test_concept_coverage_empty():
    result = concept_coverage("Some answer.", [])
    assert result["coverage_ratio"] == NOT_AVAILABLE


# ---------------------------------------------------------------------------
# TEST 15 — Citation Grounding Suite: Structural validity
# ---------------------------------------------------------------------------
def test_citation_grounding_structural():
    citations = [
        {"chunk_id": "doc_A_c1", "document_id": "A"},
        {"chunk_id": "", "document_id": "A"},          # INVALID (No chunk_id)
        {"chunk_id": "doc_Z_c3", "document_id": "Z"},          # DOCUMENT_MISMATCH
        {"chunk_id": "doc_A_c4", "document_id": "A"},          # NOT_RETRIEVED
    ]
    retrieved = ["doc_A_c1", "doc_Z_c3"]
    indexed = ["A"]
    verif = []
    
    res = citation_grounding_suite(citations, retrieved, indexed, verif)
    assert res["total_citations"] == 4, f"Expected 4 citations, got {res['total_citations']}"
    assert res["citation_coverage"] == NOT_AVAILABLE, f"Expected coverage NOT_AVAILABLE, got {res['citation_coverage']}"
    assert res["citation_validity"] == 0.25, f"Expected validity 0.25, got {res['citation_validity']} details: {res['details']}"
    assert res["citation_retrieval_accuracy"] == 0.25, f"Expected retrieval accuracy 0.25, got {res['citation_retrieval_accuracy']}"

# ---------------------------------------------------------------------------
# TEST 16 — Citation Grounding Suite: Verification Aggregation
# ---------------------------------------------------------------------------
def test_citation_grounding_verification():
    verif = [
        {"status": "SUPPORTED"},
        {"status": "WEAKLY_SUPPORTED"},
        {"status": "UNSUPPORTED"},
        {"status": "VERIFICATION_UNAVAILABLE"},
        {"status": "UNSUPPORTED"},
    ]
    res = citation_grounding_suite([], [], [], verif)
    assert res["total_claims"] == 5
    assert res["support_rate"] == 0.25
    assert res["weak_support_rate"] == 0.25
    assert res["unsupported_claim_rate"] == 0.50
    assert res["verification_unavailable_rate"] == 0.20


# ---------------------------------------------------------------------------
# TEST 21 — Resource metrics correct calculation
# ---------------------------------------------------------------------------
def test_resource_metrics():
    chunks = [
        {"text": "a" * 400, "metadata": {"document_id": "d1"}},
        {"text": "b" * 800, "metadata": {"document_id": "d2"}},
        {"text": "c" * 400, "metadata": {"document_id": "d1"}},
    ]
    result = resource_metrics(chunks)
    assert result["retrieved_chunks"] == 3
    assert result["contributing_documents"] == 2
    # token estimate: (400+800+400)//4 = 400
    assert result["estimated_context_tokens"] == 400


# ---------------------------------------------------------------------------
# TEST 22 — Observability logs an event and it is readable
# ---------------------------------------------------------------------------
def test_observability_logs_event():
    event = log_query_event(
        query="test observability query",
        question_type="factual",
        retrieval_strategy="hybrid",
        retrieved_chunk_count=5,
        documents_contributing=2,
        estimated_context_tokens=800,
        synthesis_attempted=True,
        verification_attempted=True,
        verification_statuses=["SUPPORTED", "WEAKLY_SUPPORTED"],
        total_latency_s=0.42,
    )
    assert event["query_preview"] == "test observability query"
    assert event["retrieval_strategy"] == "hybrid"
    assert event["retrieved_chunk_count"] == 5
    assert event["synthesis_attempted"] is True

    # Verify it was actually written to disk
    recent = read_recent_events(n=10)
    assert len(recent) > 0, "No events were written to disk"
    # The most recent event should be our test event
    last = recent[-1]
    assert last["query_preview"] == "test observability query"


# ---------------------------------------------------------------------------
# TEST 23 — Observability does NOT log API keys
# ---------------------------------------------------------------------------
def test_observability_no_secrets():
    event = log_query_event(
        query="test query",
        extra={"model_name": "llama-3.1-70b"},
    )
    event_str = json.dumps(event)
    # Should not contain anything that looks like a secret key placeholder
    assert "GROQ_API_KEY" not in event_str
    assert "api_key" not in event_str.lower().replace("query", "")


# ---------------------------------------------------------------------------
# TEST 24 — Observability failure does NOT crash the pipeline
# ---------------------------------------------------------------------------
def test_observability_failure_safe():
    from unittest.mock import patch
    import builtins
    with patch("builtins.open", side_effect=PermissionError("no write")):
        # Should not raise
        event = log_query_event(query="safe test")
    assert event["query_preview"] == "safe test"  # event dict still returned


# ---------------------------------------------------------------------------
# TEST 25 — Metrics on arbitrary-length relevant_document_ids
# ---------------------------------------------------------------------------
def test_arbitrary_length_relevant_docs():
    # 10 relevant docs — not limited to 2
    relevant = [f"doc_{i}" for i in range(10)]
    retrieved = [f"doc_{i}" for i in range(0, 20, 2)]  # even docs only
    r = recall_at_k(retrieved, relevant, k=10)
    # retrieved[:10] = doc_0, doc_2, doc_4, doc_6, doc_8, doc_10, doc_12, doc_14, doc_16, doc_18
    # relevant = doc_0..doc_9
    # hits = doc_0,doc_2,doc_4,doc_6,doc_8 = 5
    assert r == round(5/10, 4), f"Got {r}"


# ---------------------------------------------------------------------------
# TEST 26 — Baseline comparison structure
# ---------------------------------------------------------------------------
def test_baseline_comparison_structure():
    """Run dense and hybrid on a real query and compare their retrieval structure."""
    from retrieval import dense_search, retrieve, build_bm25_index
    build_bm25_index()

    dense_chunks = dense_search("attention mechanism transformer", top_k=5)
    hybrid_chunks = retrieve("attention mechanism transformer", strategy="hybrid", top_k=5)

    assert len(dense_chunks) > 0, "Dense should return chunks"
    assert len(hybrid_chunks) > 0, "Hybrid should return chunks"
    # Both must carry document_id
    for c in dense_chunks + hybrid_chunks:
        assert "metadata" in c
        assert "document_id" in c["metadata"]


# ---------------------------------------------------------------------------
# TEST 27 — Multi-document evaluation grouping
# ---------------------------------------------------------------------------
def test_multi_document_eval_grouping():
    """Simulates multi-doc retrieval and verifies document-level grouping."""
    chunks = [
        {"text": "t1", "metadata": {"document_id": "docA", "chunk_id": "c1"}},
        {"text": "t2", "metadata": {"document_id": "docB", "chunk_id": "c2"}},
        {"text": "t3", "metadata": {"document_id": "docA", "chunk_id": "c3"}},
    ]
    from run_evaluation import _doc_ids_from
    doc_ids = _doc_ids_from(chunks)
    assert "docA" in doc_ids and "docB" in doc_ids
    # Order: docA seen first
    assert doc_ids[0] == "docA"
    assert len(doc_ids) == 2  # deduplicated


# ---------------------------------------------------------------------------
# TEST 28 — Backward compatibility: retrieve() still returns expected fields
# ---------------------------------------------------------------------------
def test_backward_compat_retrieve():
    from retrieval import retrieve, build_bm25_index
    build_bm25_index()
    chunks = retrieve("transformer architecture", strategy="hybrid", top_k=3)
    assert len(chunks) > 0
    for c in chunks:
        assert "text" in c, "chunk must have text"
        assert "metadata" in c, "chunk must have metadata"
        meta = c["metadata"]
        assert "document_id" in meta, "metadata must have document_id"
        assert "source" in meta, "metadata must have source"
        assert "chunk_id" in meta, "metadata must have chunk_id"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Phase 4 Evaluation Test Suite ===\n")

    tests = [
        ("TEST 01: Recall@K correct",             test_recall_at_k_correct),
        ("TEST 02: Recall@K empty relevant",       test_recall_empty_relevant),
        ("TEST 03: Precision@K correct",           test_precision_at_k_correct),
        ("TEST 04: Hit Rate@K correct",            test_hit_rate_at_k),
        ("TEST 05: MRR correct",                   test_mrr_correct),
        ("TEST 06: MRR empty relevant",            test_mrr_empty_relevant),
        ("TEST 07: nDCG@K correct",               test_ndcg_at_k_correct),
        ("TEST 08: nDCG@K empty relevant",         test_ndcg_empty_relevant),
        ("TEST 09: Metrics suite keys",            test_metrics_suite_keys),
        ("TEST 10: Doc vs chunk metrics",          test_doc_level_metrics),
        ("TEST 11: Eval dataset schema",           test_eval_dataset_schema),
        ("TEST 12: Multi-doc questions exist",     test_eval_dataset_multi_doc),
        ("TEST 13: Concept coverage correct",      test_concept_coverage_correct),
        ("TEST 14: Concept coverage empty",        test_concept_coverage_empty),
        ("TEST 15: Citation grounding structure",  test_citation_grounding_structural),
        ("TEST 16: Citation grounding claims",     test_citation_grounding_verification),
        ("TEST 21: Resource metrics",              test_resource_metrics),
        ("TEST 22: Observability logs event",      test_observability_logs_event),
        ("TEST 23: No secrets in logs",            test_observability_no_secrets),
        ("TEST 24: Observability failure safe",    test_observability_failure_safe),
        ("TEST 25: Arbitrary-length relevant",     test_arbitrary_length_relevant_docs),
        ("TEST 26: Baseline comparison structure", test_baseline_comparison_structure),
        ("TEST 27: Multi-doc eval grouping",       test_multi_document_eval_grouping),
        ("TEST 28: Backward compat retrieve()",    test_backward_compat_retrieve),
    ]

    for name, fn in tests:
        run_test(name, fn)

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{'='*55}")
    print(f"Results: {passed} passed / {failed} failed / {len(_results)} total")
    if failed:
        print("\nFailed tests:")
        for name, ok, msg in _results:
            if not ok:
                print(f"  {name}: {msg}")
        sys.exit(1)
    else:
        print("\nAll Phase 4 evaluation tests PASSED.")

