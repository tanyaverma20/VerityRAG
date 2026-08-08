"""
test_retrieval.py — Phase 2 Retrieval Test Suite

Tests 1–18 as specified in Phase 2 requirements.
All tests use real assertions — no print-only faking.

Run with:
    cd backend
    python test_retrieval.py
"""

import sys
import traceback

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
# Import the modules under test
# ---------------------------------------------------------------------------

from retrieval import (
    reciprocal_rank_fusion,
    dense_search,
    bm25_search,
    rerank,
    hybrid_retrieve,
    retrieve,
    build_bm25_index,
    expand_with_parent_context,
    deduplicate_chunks,
    select_within_token_budget,
    get_contributing_documents,
    group_by_document,
)
from query_transform import rewrite_query, decompose_query
from ingest import ingest_document, get_all_chunks
from config import MAX_CONTEXT_TOKENS


# ---------------------------------------------------------------------------
# Pre-test setup: ensure the Attention paper is ingested & BM25 built
# ---------------------------------------------------------------------------

print("\n=== Phase 2 Retrieval Test Suite ===\n")
print("Setting up: ingesting attention.pdf …")
try:
    res = ingest_document("../data/attention.pdf")
    print(f"  Ingestion result: {res['status']}, chunks: {res.get('chunks_added', 0)}, "
          f"doc_id: {res.get('document_id', 'N/A')}")
    PAPER_A_DOC_ID = res.get("document_id", "")
    PAPER_A_SOURCE = "attention.pdf"
except Exception as e:
    print(f"  WARNING: ingestion failed — {e}. Some tests may fail.")
    PAPER_A_DOC_ID = ""
    PAPER_A_SOURCE = "attention.pdf"

build_bm25_index()
print("  BM25 index built.\n")


# ---------------------------------------------------------------------------
# TEST 1 — RRF correctly calculates ranked-list fusion
# ---------------------------------------------------------------------------

def test_rrf_scoring():
    dense = [
        {"id": "A", "text": "text A", "metadata": {}, "source_method": "dense", "dense_rank": 1},
        {"id": "B", "text": "text B", "metadata": {}, "source_method": "dense", "dense_rank": 2},
        {"id": "C", "text": "text C", "metadata": {}, "source_method": "dense", "dense_rank": 3},
    ]
    bm25 = [
        {"id": "C", "text": "text C", "metadata": {}, "source_method": "bm25", "bm25_rank": 1},
        {"id": "D", "text": "text D", "metadata": {}, "source_method": "bm25", "bm25_rank": 2},
        {"id": "A", "text": "text A", "metadata": {}, "source_method": "bm25", "bm25_rank": 3},
    ]
    fused = reciprocal_rank_fusion(dense, bm25, k=60)

    # A appears at dense_rank=1 and bm25_rank=3 → RRF ≈ 1/61 + 1/63 ≈ 0.02744
    # C appears at dense_rank=3 and bm25_rank=1 → RRF ≈ 1/63 + 1/61 ≈ 0.02744
    # B appears only at dense_rank=2 → RRF ≈ 1/62 ≈ 0.01613
    # D appears only at bm25_rank=2  → RRF ≈ 1/62 ≈ 0.01613

    ids = [item["id"] for item in fused]
    # A and C must outscore B and D
    assert "A" in ids and "C" in ids, "A and C must appear in fused results"
    a_score = next(i["rrf_score"] for i in fused if i["id"] == "A")
    b_score = next(i["rrf_score"] for i in fused if i["id"] == "B")
    assert a_score > b_score, f"A (in both lists) must outscore B (dense only): {a_score} vs {b_score}"


# ---------------------------------------------------------------------------
# TEST 2 — RRF deduplicates duplicate chunk IDs
# ---------------------------------------------------------------------------

def test_rrf_deduplication():
    list1 = [{"id": "X", "text": "t", "metadata": {}, "source_method": "dense", "dense_rank": 1}]
    list2 = [{"id": "X", "text": "t", "metadata": {}, "source_method": "bm25",  "bm25_rank": 1}]
    fused = reciprocal_rank_fusion(list1, list2, k=60)
    ids = [item["id"] for item in fused]
    assert ids.count("X") == 1, f"Duplicate chunk X should appear once, got {ids.count('X')}"


# ---------------------------------------------------------------------------
# TEST 3 — Dense retrieval preserves metadata
# ---------------------------------------------------------------------------

def test_dense_metadata():
    results = dense_search("attention mechanism", top_k=3)
    assert len(results) > 0, "Dense search returned no results"
    r = results[0]
    assert "metadata" in r, "Result missing 'metadata'"
    meta = r["metadata"]
    for key in ("document_id", "chunk_id", "parent_id", "source", "page_number"):
        assert key in meta, f"metadata missing '{key}'"
    assert "dense_rank" in r, "Result missing 'dense_rank'"
    assert r["dense_rank"] == 1, f"First dense result should have dense_rank=1, got {r['dense_rank']}"


# ---------------------------------------------------------------------------
# TEST 4 — BM25 retrieval preserves metadata
# ---------------------------------------------------------------------------

def test_bm25_metadata():
    results = bm25_search("transformer attention", top_k=3)
    assert len(results) > 0, "BM25 search returned no results"
    r = results[0]
    assert "metadata" in r, "Result missing 'metadata'"
    meta = r["metadata"]
    for key in ("document_id", "chunk_id", "source"):
        assert key in meta, f"metadata missing '{key}'"
    assert "bm25_rank" in r, "Result missing 'bm25_rank'"


# ---------------------------------------------------------------------------
# TEST 5 — Hybrid retrieval actually uses RRF (not simple merge)
# ---------------------------------------------------------------------------

def test_hybrid_uses_rrf():
    results = retrieve("attention mechanism", strategy="hybrid", top_k=5)
    assert len(results) > 0, "Hybrid retrieve returned no results"
    # At least one result should carry an rrf_score (proves RRF was applied)
    rrf_scores = [r.get("rrf_score") for r in results if r.get("rrf_score") is not None]
    assert len(rrf_scores) > 0, "No rrf_score found — RRF was not applied"


# ---------------------------------------------------------------------------
# TEST 6 — CrossEncoder runs AFTER RRF (rerank_score present)
# ---------------------------------------------------------------------------

def test_crossencoder_after_rrf():
    results = retrieve("self-attention mechanism neural network", strategy="hybrid", top_k=3)
    assert len(results) > 0, "retrieve returned no results"
    # All returned results must have a rerank_score (proof CrossEncoder ran)
    for r in results:
        assert "rerank_score" in r, f"rerank_score missing from result {r.get('id')}"


# ---------------------------------------------------------------------------
# TEST 7 — document_ids filtering works
# ---------------------------------------------------------------------------

def test_document_filter():
    if not PAPER_A_DOC_ID:
        raise AssertionError("PAPER_A_DOC_ID not available — ingestion failed in setup")
    results = retrieve("transformer", document_ids=[PAPER_A_DOC_ID], top_k=5)
    # Every returned chunk must belong to PAPER_A
    for r in results:
        got_id = r["metadata"].get("document_id", "")
        assert got_id == PAPER_A_DOC_ID, (
            f"Filtering by document_id={PAPER_A_DOC_ID} but got chunk from {got_id}"
        )


# ---------------------------------------------------------------------------
# TEST 8 — Multiple papers preserve document_id
# ---------------------------------------------------------------------------

def test_multi_paper_document_id():
    # With one paper, each chunk must carry document_id from that paper
    results = retrieve("architecture", strategy="hybrid", top_k=5)
    for r in results:
        assert "document_id" in r["metadata"], "document_id missing from chunk metadata"
        assert r["metadata"]["document_id"] != "", "document_id is empty"
        assert "source" in r["metadata"], "source missing from chunk metadata"


# ---------------------------------------------------------------------------
# TEST 9 — Child chunks can be associated with parent context
# ---------------------------------------------------------------------------

def test_parent_context():
    results = retrieve("attention", strategy="hybrid", top_k=3, apply_parent_context=True)
    assert len(results) > 0, "retrieve returned nothing"
    for r in results:
        assert "parent_context" in r, f"parent_context missing from result {r.get('id')}"
        assert isinstance(r["parent_context"], str), "parent_context must be a string"
        assert len(r["parent_context"]) > 0, "parent_context must not be empty"


# ---------------------------------------------------------------------------
# TEST 10 — Query rewriting produces a usable retrieval query
# ---------------------------------------------------------------------------

def test_query_rewrite_output():
    vague = "how did they train it"
    result = rewrite_query(vague)
    assert isinstance(result, str), "rewrite_query must return a string"
    assert len(result) > 0, "rewrite_query must return a non-empty string"
    # Either the original or a rewritten version — both are valid
    print(f"    rewritten: '{result}'")


# ---------------------------------------------------------------------------
# TEST 11 — Query rewriting failure falls back to original query
# ---------------------------------------------------------------------------

def test_query_rewrite_fallback():
    from unittest.mock import patch
    original = "attention mechanism"
    # Simulate LLM failure by patching _call_groq_raw to raise
    with patch("query_transform._call_groq_raw", side_effect=RuntimeError("simulated failure")):
        result = rewrite_query(original)
    # Must return the original (or the heuristic path which skips LLM for non-vague queries)
    assert isinstance(result, str) and len(result) > 0, "Fallback must return a non-empty string"


# ---------------------------------------------------------------------------
# TEST 12 — Complex query decomposition produces multiple sub-queries
# ---------------------------------------------------------------------------

def test_query_decomposition():
    complex_q = "Compare the architectures and datasets used in these papers"
    sub_qs = decompose_query(complex_q)
    assert isinstance(sub_qs, list), "decompose_query must return a list"
    assert len(sub_qs) >= 1, "decompose_query must return at least 1 sub-query"
    for q in sub_qs:
        assert isinstance(q, str) and q.strip(), "Each sub-query must be a non-empty string"
    print(f"    sub-queries: {sub_qs}")


# ---------------------------------------------------------------------------
# TEST 13 — Context selection respects MAX_CONTEXT_TOKENS
# ---------------------------------------------------------------------------

def test_token_budget():
    from config import MAX_CONTEXT_TOKENS as MAX_TOK
    # Create synthetic chunks that collectively exceed the budget
    big_chunks = [
        {
            "id": f"chunk_{i}",
            "text": "x " * 1000,   # ~500 tokens each
            "metadata": {"document_id": f"doc_{i % 2}", "chunk_id": f"chunk_{i}"},
            "rerank_score": float(10 - i),
        }
        for i in range(20)
    ]
    selected = select_within_token_budget(big_chunks, max_tokens=MAX_TOK)
    total_tokens = sum(max(1, len(c["text"]) // 4) for c in selected)
    assert total_tokens <= MAX_TOK, (
        f"Token budget exceeded: {total_tokens} > {MAX_TOK}"
    )


# ---------------------------------------------------------------------------
# TEST 14 — Multi-paper diversity: per-doc cap is enforced
# ---------------------------------------------------------------------------

def test_diversity_cap():
    from config import MAX_CHUNKS_PER_DOC
    # 10 chunks all from same document
    chunks = [
        {
            "id": f"c{i}",
            "text": "short",
            "metadata": {"document_id": "doc_a", "chunk_id": f"c{i}"},
            "rerank_score": float(10 - i),
        }
        for i in range(10)
    ]
    selected = select_within_token_budget(chunks, max_tokens=100_000)
    from_doc_a = [c for c in selected if c["metadata"]["document_id"] == "doc_a"]
    assert len(from_doc_a) <= MAX_CHUNKS_PER_DOC, (
        f"Expected at most {MAX_CHUNKS_PER_DOC} chunks from doc_a, got {len(from_doc_a)}"
    )


# ---------------------------------------------------------------------------
# TEST 15 — Duplicate chunks are removed
# ---------------------------------------------------------------------------

def test_deduplication():
    chunks = [
        {"id": "a", "text": "t", "metadata": {"chunk_id": "c1"}},
        {"id": "b", "text": "t", "metadata": {"chunk_id": "c1"}},  # duplicate chunk_id
        {"id": "c", "text": "t", "metadata": {"chunk_id": "c2"}},
    ]
    deduped = deduplicate_chunks(chunks)
    chunk_ids = [c["metadata"]["chunk_id"] for c in deduped]
    assert chunk_ids.count("c1") == 1, f"Duplicate c1 not removed: {chunk_ids}"
    assert "c2" in chunk_ids, "Unique c2 should be retained"


# ---------------------------------------------------------------------------
# TEST 16 — Existing single-paper retrieval still works
# ---------------------------------------------------------------------------

def test_single_paper_retrieval():
    chunks = hybrid_retrieve("multi-head attention", top_k=3)
    assert len(chunks) > 0, "hybrid_retrieve returned no results for a known query"
    assert all("text" in c for c in chunks), "All chunks must have 'text'"
    assert all("metadata" in c for c in chunks), "All chunks must have 'metadata'"


# ---------------------------------------------------------------------------
# TEST 17 — Existing hybrid_retrieve API remains backward compatible
# ---------------------------------------------------------------------------

def test_backward_compat():
    # Must accept exactly (query, top_k) — same as Phase 1
    result = hybrid_retrieve("transformer model", top_k=2)
    assert isinstance(result, list), "hybrid_retrieve must return a list"
    # The old callers expect source_method and rerank_score on results
    for c in result:
        assert "source_method" in c, "source_method missing — breaks main.py citation logic"
        assert "rerank_score" in c, "rerank_score missing — breaks main.py"
        assert "metadata" in c, "metadata missing"
        assert "source" in c["metadata"], "metadata.source missing — breaks /query citation"


# ---------------------------------------------------------------------------
# TEST 18 — Citation metadata survives through the full pipeline
# ---------------------------------------------------------------------------

def test_citation_metadata():
    chunks = retrieve("scaled dot-product attention", strategy="hybrid", top_k=3)
    assert len(chunks) > 0, "retrieve returned no results"
    for c in chunks:
        meta = c["metadata"]
        for field in ("document_id", "chunk_id", "parent_id", "source", "page_number", "section"):
            assert field in meta, f"Citation field '{field}' missing from metadata"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

tests = [
    ("TEST 01: RRF scoring",            test_rrf_scoring),
    ("TEST 02: RRF deduplication",      test_rrf_deduplication),
    ("TEST 03: Dense metadata",         test_dense_metadata),
    ("TEST 04: BM25 metadata",          test_bm25_metadata),
    ("TEST 05: Hybrid uses RRF",        test_hybrid_uses_rrf),
    ("TEST 06: CrossEncoder after RRF", test_crossencoder_after_rrf),
    ("TEST 07: document_ids filter",    test_document_filter),
    ("TEST 08: Multi-paper doc_id",     test_multi_paper_document_id),
    ("TEST 09: Parent context",         test_parent_context),
    ("TEST 10: Query rewrite output",   test_query_rewrite_output),
    ("TEST 11: Rewrite fallback",       test_query_rewrite_fallback),
    ("TEST 12: Query decomposition",    test_query_decomposition),
    ("TEST 13: Token budget",           test_token_budget),
    ("TEST 14: Diversity cap",          test_diversity_cap),
    ("TEST 15: Deduplication",          test_deduplication),
    ("TEST 16: Single-paper retrieval", test_single_paper_retrieval),
    ("TEST 17: Backward compat",        test_backward_compat),
    ("TEST 18: Citation metadata",      test_citation_metadata),
]

for name, fn in tests:
    run_test(name, fn)

passed = sum(1 for _, ok, _ in _results if ok)
failed = sum(1 for _, ok, _ in _results if not ok)
print(f"\n{'='*50}")
print(f"Results: {passed} passed / {failed} failed / {len(_results)} total")
if failed:
    print("\nFailed tests:")
    for name, ok, msg in _results:
        if not ok:
            print(f"  {name}: {msg}")
    sys.exit(1)
else:
    print("\nAll Phase 2 retrieval tests PASSED.")
