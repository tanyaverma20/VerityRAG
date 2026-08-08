"""
retriever.py — Graph node for executing retrieval using Phase 2 pipeline.
"""
from typing import Any
import sys
import os

from .state import ResearchState

# Ensure we can import from the parent backend folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval import retrieve, deduplicate_chunks
from config import RERANK_TOP_K

def retrieve_evidence(state: ResearchState) -> dict[str, Any]:
    """
    Executes Phase 2 retrieve() for each sub_query, aggregating results.
    """
    sub_queries = state.get("sub_queries") or [state["original_query"]]
    completed_sub_queries = state.get("completed_sub_queries") or []
    document_ids = state.get("document_ids", None)
    
    # We want to keep existing chunks in the state
    all_chunks = state.get("retrieval_results") or []
    
    newly_completed = list(completed_sub_queries)
    
    for sq in sub_queries:
        if sq in newly_completed:
            continue
            
        try:
            chunks = retrieve(
                query=sq,
                strategy="hybrid",
                document_ids=document_ids,
                top_k=RERANK_TOP_K,
                apply_parent_context=True,
                apply_token_budget=True,
            )
            all_chunks.extend(chunks)
            newly_completed.append(sq)
        except Exception:
            pass # continue with other queries if one fails
            
    # Deduplicate chunks retrieved across different sub_queries
    unique_chunks = deduplicate_chunks(all_chunks)
    
    return {
        "retrieval_results": unique_chunks,
        "completed_sub_queries": newly_completed,
        "status": f"retrieved {len(unique_chunks)} chunks"
    }
