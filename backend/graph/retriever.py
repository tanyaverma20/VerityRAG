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
    sub_queries = state.get("sub_queries", [state["original_query"]])
    document_ids = state.get("document_ids", None)
    
    all_chunks = []
    
    for sq in sub_queries:
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
        except Exception:
            pass # continue with other queries if one fails
            
    # Deduplicate chunks retrieved across different sub_queries
    unique_chunks = deduplicate_chunks(all_chunks)
    
    # Optionally, we could apply another token budget selection here over the merged pool,
    # but for simplicity, we pass all retrieved unique chunks down to the organizer.
    
    return {
        "retrieval_results": unique_chunks,
        "status": f"retrieved {len(unique_chunks)} chunks"
    }
