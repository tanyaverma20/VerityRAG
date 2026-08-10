"""
retrieval.py — Phase 2 Advanced Retrieval Pipeline

Architecture:
  User Query
      ↓
  dense_search()  +  bm25_search()
      ↓
  reciprocal_rank_fusion()          (replaces simple merge/deduplicate)
      ↓
  rerank()                          (CrossEncoder on RRF candidate pool)
      ↓
  expand_with_parent_context()      (attach broader page context per chunk)
      ↓
  deduplicate_chunks()
      ↓
  select_within_token_budget()      (greedy, with per-doc diversity cap)
      ↓
  Final evidence list

Public API:
  hybrid_retrieve(query, top_k)           — backward-compatible (unchanged signature)
  retrieve(query, strategy, document_ids, top_k)  — Phase 2 full interface

Why RRF?
  Dense search and BM25 return ranked lists. Simply concatenating them discards
  rank information.  RRF(d) = Σ 1/(k + rank(d)) correctly aggregates rankings
  across retrieval methods without requiring score normalisation.  k=60 is the
  standard default from the original RRF paper (Cormack et al., 2009).

Why CrossEncoder AFTER RRF?
  RRF produces a quality candidate pool from 30 items; the CrossEncoder (which
  is expensive — O(n) forward passes) then picks the best 5.  Running
  CrossEncoder before fusion would score incomparable lists.
"""

from __future__ import annotations

import contextvars
import time

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import (
    DENSE_TOP_K, BM25_TOP_K, RERANK_TOP_K,
    RERANKER_MODEL, RRF_K, MAX_CONTEXT_TOKENS, MAX_CHUNKS_PER_DOC,
)
from ingest import get_collection, get_all_chunks

# ---------------------------------------------------------------------------
# Retrieval-stage observability — real, per-request candidate/rerank counts
# for the frontend's collapsible "Retrieval Details" panel (never invented
# numbers). Mirrors query_transform.py's _call_log contextvar pattern: a
# per-request slot, safe across concurrent requests (each threadpool worker
# thread gets its own contextvar value), diagnostic-only — never influences
# retrieval behavior itself.
# ---------------------------------------------------------------------------
_retrieval_stats: contextvars.ContextVar[dict | None] = contextvars.ContextVar("_retrieval_stats", default=None)


def get_retrieval_stats() -> dict:
    """Real counts/timings from the most recent retrieve()/retrieve_multi()
    call on this request's thread: candidates_retrieved (post-RRF fusion,
    pre-rerank pool size), reranked_to (final chunk count after budget/
    truncation), and rerank_latency_s (wall-clock time of the CrossEncoder
    forward pass alone, isolated from dense/BM25/RRF/post-processing) — see
    observability.py's rerank_latency_s field. Empty dict if no retrieval
    has run yet this request."""
    stats = _retrieval_stats.get()
    return dict(stats) if stats else {}


def _record_retrieval_stats(candidates_retrieved: int, reranked_to: int, rerank_latency_s: float | None = None) -> None:
    _retrieval_stats.set({
        "candidates_retrieved": candidates_retrieved, "reranked_to": reranked_to,
        "rerank_latency_s": rerank_latency_s,
    })

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_reranker = CrossEncoder(RERANKER_MODEL)

_bm25_index = None
_bm25_corpus: list[dict] = []   # aligned with _bm25_index tokenized corpus


# ---------------------------------------------------------------------------
# BM25 index lifecycle
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index() -> None:
    """Rebuild BM25 from current Chroma contents.  Call at startup and after ingestion."""
    global _bm25_index, _bm25_corpus
    chunks = get_all_chunks()
    if not chunks:
        _bm25_index, _bm25_corpus = None, []
        return
    tokenized = [_tokenize(c["text"]) for c in chunks]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_corpus = chunks


# ---------------------------------------------------------------------------
# Individual retrieval strategies
# ---------------------------------------------------------------------------

def _scope_where_clause(document_ids: list[str] | None, workspace_id: str | None = None) -> dict | None:
    """
    Builds the Chroma `where` filter enforcing document/workspace scope
    directly at the vector layer — a second, independent enforcement point
    on top of the SQL-layer check main.py's _resolve_document_scope()
    already does (never relies on the frontend alone).

    workspace_id is OPTIONAL and additive: omitting it preserves the exact
    existing document_id-only behavior (single-tenant deployments, and
    every caller that predates workspace support, are unaffected). When a
    caller does pass it, chunks are required to carry a matching
    workspace_id in their metadata — chunks ingested before this feature
    existed (or via a test/legacy path) have no workspace_id tag and will
    correctly never match a real workspace_id filter, i.e. they're safely
    excluded rather than silently included.
    """
    clauses = []
    if document_ids:
        clauses.append({"document_id": {"$eq": document_ids[0]}} if len(document_ids) == 1 else {"document_id": {"$in": document_ids}})
    if workspace_id:
        clauses.append({"workspace_id": {"$eq": workspace_id}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def dense_search(
    query: str,
    top_k: int = DENSE_TOP_K,
    document_ids: list[str] | None = None,
    workspace_id: str | None = None,
) -> list[dict]:
    """
    Query ChromaDB with optional document_ids + workspace_id filters.

    Returns list of dicts with keys:
      id, text, metadata, source_method, dense_rank
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    n = min(top_k, collection.count())
    where = _scope_where_clause(document_ids, workspace_id)

    kwargs: dict = {"query_texts": [query], "n_results": n}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    return [
        {
            "id": _id,
            "text": doc,
            "metadata": meta,
            "source_method": "dense",
            "dense_rank": rank + 1,          # 1-based
        }
        for rank, (_id, doc, meta) in enumerate(
            zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
        )
    ]


def bm25_search(
    query: str,
    top_k: int = BM25_TOP_K,
    document_ids: list[str] | None = None,
    workspace_id: str | None = None,
) -> list[dict]:
    """
    BM25 keyword search with optional document_ids + workspace_id filters.
    Same scope semantics as dense_search's _scope_where_clause — workspace_id
    is additive/optional, never breaking a caller that omits it.

    Returns list of dicts with keys:
      id, text, metadata, source_method, bm25_rank
    """
    if _bm25_index is None or not _bm25_corpus:
        return []

    scores = _bm25_index.get_scores(_tokenize(query))
    ranked_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )

    results = []
    bm25_rank = 1
    for i in ranked_indices:
        if scores[i] <= 0:
            break
        chunk = _bm25_corpus[i]
        # Apply document_ids filter if requested
        if document_ids:
            doc_id = chunk.get("metadata", {}).get("document_id", "")
            if doc_id not in document_ids:
                continue
        # Apply workspace_id filter if requested — a chunk with no
        # workspace_id tag (legacy/pre-workspace ingest) never matches a
        # real workspace_id, so it's safely excluded rather than leaked in.
        if workspace_id:
            chunk_ws = chunk.get("metadata", {}).get("workspace_id", "")
            if chunk_ws != workspace_id:
                continue
        results.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "source_method": "bm25",
            "bm25_rank": bm25_rank,
        })
        bm25_rank += 1
        if bm25_rank > top_k:
            break

    return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *ranked_lists: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    RRF(d) = Σ_r  1 / (k + rank_r(d))

    - Documents are identified by their 'id' field.
    - All source metadata and per-list ranks are preserved.
    - rrf_score and source_method (dense/bm25/dense+bm25) are added.
    - Result is sorted by rrf_score descending.

    Reference: Cormack, Clarke & Buettcher, SIGIR 2009.
    """
    scores: dict[str, float] = {}
    merged: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for item in ranked_list:
            chunk_id = item["id"]
            rank = item.get("dense_rank") or item.get("bm25_rank") or 1
            rrf_contribution = 1.0 / (k + rank)

            if chunk_id not in scores:
                scores[chunk_id] = 0.0
                merged[chunk_id] = dict(item)  # copy
            else:
                scores[chunk_id] += rrf_contribution
                # Annotate when found by both methods
                existing_method = merged[chunk_id].get("source_method", "")
                new_method = item.get("source_method", "")
                if existing_method and new_method and existing_method != new_method:
                    merged[chunk_id]["source_method"] = "dense+bm25"
                # Preserve both ranks
                if "dense_rank" in item:
                    merged[chunk_id]["dense_rank"] = item["dense_rank"]
                if "bm25_rank" in item:
                    merged[chunk_id]["bm25_rank"] = item["bm25_rank"]

            scores[chunk_id] += rrf_contribution if chunk_id in scores and scores[chunk_id] == rrf_contribution else 0

    # Attach final RRF scores and sort
    for chunk_id, item in merged.items():
        item["rrf_score"] = round(scores[chunk_id], 6)

    return sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)


# ---------------------------------------------------------------------------
# CrossEncoder reranking  (runs AFTER RRF)
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = RERANK_TOP_K,
) -> list[dict]:
    """
    Score candidates with CrossEncoder and return top_k.

    Adds 'rerank_score' to each item.  Order is guaranteed by rerank_score DESC.
    """
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = _reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Parent context expansion
# ---------------------------------------------------------------------------

def expand_with_parent_context(chunks: list[dict]) -> list[dict]:
    """
    For each retrieved child chunk, attempt to attach its parent-page context.

    Strategy: fetch all chunks that share the same parent_id from Chroma,
    concatenate their texts in chunk_id order, and store as 'parent_context'.
    The child's own 'text' remains the precise retrieval evidence.

    This is lightweight: we only query Chroma once per unique parent_id.
    If a parent cannot be fetched (e.g. single-chunk page), parent_context
    falls back to the child text itself.
    """
    collection = get_collection()

    # Collect unique parent_ids
    parent_ids_needed: set[str] = set()
    for c in chunks:
        pid = c.get("metadata", {}).get("parent_id")
        if pid:
            parent_ids_needed.add(pid)

    # Fetch sibling chunks grouped by parent_id
    parent_texts: dict[str, str] = {}
    for parent_id in parent_ids_needed:
        try:
            results = collection.get(
                where={"parent_id": {"$eq": parent_id}},
                include=["documents", "metadatas"],
            )
            if results["ids"]:
                # Sort siblings by chunk_id for coherent reading order
                siblings = sorted(
                    zip(results["ids"], results["documents"]),
                    key=lambda x: x[0],
                )
                parent_texts[parent_id] = " ".join(text for _, text in siblings)
        except Exception:
            pass  # non-fatal; falls back to child text

    # Attach parent_context to each chunk
    for c in chunks:
        parent_id = c.get("metadata", {}).get("parent_id")
        if parent_id and parent_id in parent_texts:
            # Only attach if the parent text is meaningfully longer than the child
            full = parent_texts[parent_id]
            c["parent_context"] = full if len(full) > len(c["text"]) else c["text"]
        else:
            c["parent_context"] = c["text"]

    return chunks


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Remove duplicates by chunk_id (primary key from Phase 1B ingestion)."""
    seen: set[str] = set()
    deduped: list[dict] = []
    for c in chunks:
        chunk_id = c.get("metadata", {}).get("chunk_id") or c.get("id", "")
        if chunk_id not in seen:
            seen.add(chunk_id)
            deduped.append(c)
    return deduped


# ---------------------------------------------------------------------------
# Token-budget-aware context selection with per-document diversity
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters (conservative for academic text)."""
    return max(1, len(text) // 4)


def select_within_token_budget(
    chunks: list[dict],
    max_tokens: int = MAX_CONTEXT_TOKENS,
    max_per_doc: int = MAX_CHUNKS_PER_DOC,
) -> list[dict]:
    """
    Greedy selection that:
      1. Respects MAX_CONTEXT_TOKENS overall.
      2. Caps chunks per document_id at max_per_doc for evidence diversity.
      3. Preserves the existing rerank_score ordering (best first).

    Chunks are assumed to arrive in relevance order (highest score first).

    IMPORTANT: every caller that formats these chunks into a prompt (Q&A
    synthesis, report generation, /analyze's viva/mock-test/interview/
    explain-figure evidence) sends `parent_context` when present, not the
    shorter child `text` — expand_with_parent_context() already runs before
    this function in retrieve(), so parent_context is normally already
    attached by the time we get here. Budgeting on `text` alone was
    silently undercounting real prompt size by 2-5x (measured: a real
    2-document report prompt came out to ~7100 tokens against a 6000-token
    budget that had only counted ~2100), which could push an oversized,
    unbalanced prompt at the model and made it more likely to drop or
    truncate one document's content. Estimating from the same field that's
    actually sent keeps the budget honest.
    """
    selected: list[dict] = []
    tokens_used = 0
    doc_counts: dict[str, int] = {}

    for c in chunks:
        doc_id = c.get("metadata", {}).get("document_id", "__unknown__")
        text_tokens = _estimate_tokens(c.get("parent_context") or c["text"])

        if tokens_used + text_tokens > max_tokens:
            continue  # skip if budget exhausted
        if doc_counts.get(doc_id, 0) >= max_per_doc:
            continue  # diversity cap hit for this document

        selected.append(c)
        tokens_used += text_tokens
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    return selected


# ---------------------------------------------------------------------------
# Paper-aware evidence grouping (comparison foundation)
# ---------------------------------------------------------------------------

def group_by_document(chunks: list[dict]) -> dict[str, list[dict]]:
    """
    Group retrieved chunks by document_id.

    Returns:
        { "doc_id_abc": [chunk, chunk, ...], "doc_id_xyz": [...] }

    Used downstream to produce per-paper evidence for comparison queries.
    """
    grouped: dict[str, list[dict]] = {}
    for c in chunks:
        doc_id = c.get("metadata", {}).get("document_id", "__unknown__")
        grouped.setdefault(doc_id, []).append(c)
    return grouped


def get_contributing_documents(chunks: list[dict]) -> list[dict]:
    """
    Return a deduplicated list of documents that contributed evidence.

    Each entry: { "document_id": "...", "source": "paper_a.pdf" }
    """
    seen: set[str] = set()
    docs: list[dict] = []
    for c in chunks:
        meta = c.get("metadata", {})
        doc_id = meta.get("document_id", "")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            docs.append({"document_id": doc_id, "source": meta.get("source", "")})
    return docs


# ---------------------------------------------------------------------------
# Full Phase 2 retrieve() — the new primary entry point
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    strategy: str = "hybrid",
    document_ids: list[str] | None = None,
    top_k: int = RERANK_TOP_K,
    apply_parent_context: bool = True,
    apply_token_budget: bool = True,
    workspace_id: str | None = None,
) -> list[dict]:
    """
    Full Phase 2 retrieval pipeline.

    Args:
        query:                Original (or rewritten) retrieval query.
        strategy:             "hybrid" | "dense" | "bm25"
        document_ids:         Optional list of document_ids to restrict search.
                              None = search all indexed documents.
        top_k:                Number of final chunks to return.
        apply_parent_context: If True, expand each chunk with parent-page context.
        apply_token_budget:   If True, apply MAX_CONTEXT_TOKENS greedy selection.
        workspace_id:         Optional second, independent scope enforcement
                              point (see _scope_where_clause) — additive on
                              top of document_ids, never required.

    Returns:
        List of result dicts, each containing:
          id, text, metadata, source_method,
          dense_rank?, bm25_rank?, rrf_score?, rerank_score,
          parent_context?
    """
    # --- 1. Retrieve from each requested strategy ---
    dense_results: list[dict] = []
    bm25_results: list[dict] = []

    if strategy in ("hybrid", "dense"):
        dense_results = dense_search(query, top_k=DENSE_TOP_K, document_ids=document_ids, workspace_id=workspace_id)

    if strategy in ("hybrid", "bm25"):
        bm25_results = bm25_search(query, top_k=BM25_TOP_K, document_ids=document_ids, workspace_id=workspace_id)

    # --- 2. Fuse with RRF (or pass through for single-strategy) ---
    if strategy == "hybrid":
        candidates = reciprocal_rank_fusion(dense_results, bm25_results)
    elif strategy == "dense":
        candidates = dense_results
    else:  # bm25
        candidates = bm25_results

    if not candidates:
        return []

    # --- 3. CrossEncoder reranking on fused candidate pool ---
    _rerank_t0 = time.perf_counter()
    reranked = rerank(query, candidates, top_k=max(top_k * 3, RERANK_TOP_K * 3))
    rerank_latency_s = round(time.perf_counter() - _rerank_t0, 4)

    # --- 4. Parent context expansion ---
    if apply_parent_context:
        reranked = expand_with_parent_context(reranked)

    # --- 5. Deduplication ---
    reranked = deduplicate_chunks(reranked)

    # --- 6. Token-budget selection with per-doc diversity ---
    if apply_token_budget:
        reranked = select_within_token_budget(reranked, max_per_doc=MAX_CHUNKS_PER_DOC)

    final = reranked[:top_k]
    _record_retrieval_stats(len(candidates), len(final), rerank_latency_s)
    return final


# ---------------------------------------------------------------------------
# Multi-sub-query retrieval — for deterministically decomposed complex
# questions (see query_transform.decompose_query_deterministic).
# ---------------------------------------------------------------------------

def retrieve_multi(
    sub_queries: list[str],
    document_ids: list[str] | None = None,
    top_k: int = RERANK_TOP_K,
    apply_parent_context: bool = True,
    apply_token_budget: bool = True,
    rerank_query: str | None = None,
    workspace_id: str | None = None,
) -> list[dict]:
    """
    Retrieval for a list of sub-queries produced by deterministic
    decomposition of one complex question.

    Each sub_query gets its own dense_search()/bm25_search() pass — still
    document-scoped identically to the single-query path — but everything
    is fused with ONE Reciprocal Rank Fusion call across all sub-queries'
    ranked lists (so a chunk surfaced by multiple sub-queries correctly
    accumulates rank credit instead of being scored in isolation N times),
    then reranked ONCE, globally, against `rerank_query` (the original,
    undecomposed question — the CrossEncoder should judge relevance to what
    the user actually asked, not to a synthetic single-aspect fragment),
    then parent-context-expanded / deduplicated / token-budgeted exactly
    like retrieve() does for a single query.

    With 0 or 1 sub_queries this simply delegates to retrieve(), so the
    common case (a simple question) is byte-for-byte the existing,
    unchanged single-query behavior.
    """
    if len(sub_queries) <= 1:
        q = sub_queries[0] if sub_queries else (rerank_query or "")
        return retrieve(
            query=q,
            document_ids=document_ids,
            top_k=top_k,
            apply_parent_context=apply_parent_context,
            apply_token_budget=apply_token_budget,
            workspace_id=workspace_id,
        )

    rerank_query = rerank_query or sub_queries[0]

    # --- 1. Dense + BM25 for EACH sub-query, independently document- and
    #        workspace-scoped (same scope for every sub-query — decomposition
    #        splits the QUESTION, never the scope) ---
    ranked_lists: list[list[dict]] = []
    for sq in sub_queries:
        ranked_lists.append(dense_search(sq, top_k=DENSE_TOP_K, document_ids=document_ids, workspace_id=workspace_id))
        ranked_lists.append(bm25_search(sq, top_k=BM25_TOP_K, document_ids=document_ids, workspace_id=workspace_id))

    # --- 2. ONE global RRF fusion pass across every sub-query's results ---
    candidates = reciprocal_rank_fusion(*ranked_lists)
    if not candidates:
        return []

    # --- 3. CrossEncoder reranking ONCE, globally, against the real question ---
    _rerank_t0 = time.perf_counter()
    reranked = rerank(rerank_query, candidates, top_k=max(top_k * 3, RERANK_TOP_K * 3))
    rerank_latency_s = round(time.perf_counter() - _rerank_t0, 4)

    # --- 4. Parent context expansion ---
    if apply_parent_context:
        reranked = expand_with_parent_context(reranked)

    # --- 5. Deduplication ---
    reranked = deduplicate_chunks(reranked)

    # --- 6. Token-budget selection with per-doc diversity (ONE global budget,
    #        not one per sub-query) ---
    if apply_token_budget:
        reranked = select_within_token_budget(reranked, max_per_doc=MAX_CHUNKS_PER_DOC)

    final = reranked[:top_k]
    _record_retrieval_stats(len(candidates), len(final), rerank_latency_s)
    return final


# ---------------------------------------------------------------------------
# Backward-compatible hybrid_retrieve (unchanged signature, Phase 1 callers safe)
# ---------------------------------------------------------------------------

def hybrid_retrieve(query: str, top_k: int = RERANK_TOP_K) -> list[dict]:
    """
    Backward-compatible entry point used by main.py and eval_harness.py.

    Delegates to retrieve() with strategy="hybrid".
    Returns the same result schema as Phase 1 (plus new fields which callers ignore).
    The 'source_method' field is preserved so main.py citations still work.
    """
    return retrieve(query, strategy="hybrid", top_k=top_k)
