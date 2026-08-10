"""
retriever.py — Graph node for executing retrieval using Phase 2 pipeline.
"""
from typing import Any
import sys
import os

from .state import ResearchState

# Ensure we can import from the parent backend folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval import retrieve, retrieve_multi, deduplicate_chunks
from config import RERANK_TOP_K

def retrieve_evidence(state: ResearchState) -> dict[str, Any]:
    """
    Executes Phase 2 retrieve() for each pending sub_query, aggregating
    results into state["retrieval_results"].

    Normal mode with >1 sub_queries (deterministically decomposed by
    graph/planner.py's decompose_query_deterministic() — zero LLM cost)
    takes a dedicated global-fusion path via retrieve_multi(): dense+BM25
    for every sub-query are fused with ONE RRF pass and reranked ONCE
    against the original question, so the final evidence set and token
    budget reflect the whole question, not N independently-truncated
    fragments.

    Deep Research's adaptive multi-iteration loop (which calls this node
    repeatedly across LangGraph iterations, tracking completed_sub_queries
    so it never re-retrieves) is left on the original per-sub-query
    retrieve() loop below, unchanged.
    """
    sub_queries = state.get("sub_queries") or [state["original_query"]]
    completed_sub_queries = state.get("completed_sub_queries") or []
    document_ids = state.get("document_ids", None)
    research_type = state.get("research_type", "simple")
    workspace_id = state.get("workspace_id") or None

    # We want to keep existing chunks in the state
    all_chunks = state.get("retrieval_results") or []

    pending = [sq for sq in sub_queries if sq not in completed_sub_queries]
    newly_completed = list(completed_sub_queries)

    if pending and len(pending) > 1 and research_type != "deep":
        # Deterministically decomposed multi-aspect normal-mode question:
        # merge + globally rerank all pending sub-queries in one shot.
        try:
            chunks = retrieve_multi(
                sub_queries=pending,
                document_ids=document_ids,
                top_k=RERANK_TOP_K,
                apply_parent_context=True,
                apply_token_budget=True,
                rerank_query=state.get("original_query"),
                workspace_id=workspace_id,
            )
            all_chunks.extend(chunks)
            newly_completed.extend(pending)
        except Exception:
            import traceback
            traceback.print_exc()
    else:
        for sq in pending:
            try:
                chunks = retrieve(
                    query=sq,
                    strategy="hybrid",
                    document_ids=document_ids,
                    top_k=RERANK_TOP_K,
                    apply_parent_context=True,
                    apply_token_budget=True,
                    workspace_id=workspace_id,
                )
                all_chunks.extend(chunks)
                newly_completed.append(sq)
            except Exception as e:
                import traceback
                traceback.print_exc()
                pass # continue with other queries if one fails

    # Deduplicate chunks retrieved across different sub_queries
    unique_chunks = deduplicate_chunks(all_chunks)

    return {
        "retrieval_results": unique_chunks,
        "completed_sub_queries": newly_completed,
        "status": f"retrieved {len(unique_chunks)} chunks"
    }
