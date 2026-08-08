"""
state.py — LangGraph State Definition
"""

from typing import TypedDict, Any

class ResearchState(TypedDict):
    """
    The shared state passed through the LangGraph workflow.
    """
    # 1. Inputs
    original_query: str
    
    # 2. Planning Phase
    research_plan: dict[str, Any]  # e.g., {"mode": "single_paper", "needs_decomposition": False, "reasoning": "..."}
    sub_queries: list[str]         # Populated by decomposition if needed, or defaults to [original_query]
    document_ids: list[str]        # Optional filter, e.g., ["docA", "docB"]
    
    # 3. Retrieval Phase
    retrieval_results: list[dict]  # Flat list of chunks retrieved from Phase 2
    
    # 4. Organization Phase
    evidence_by_document: dict[str, list[dict]]  # Grouped by document, deduplicated parent context
    
    # 5. Synthesis Phase
    draft_answer: str              # The synthesized answer
    citations: list[dict]          # Structured citations mapping [DocID, ChunkID] to source details
    
    # 6. Verification Phase
    verification_results: list[dict] # Claim-level verification: claim, supporting_evidence, status
    
    # 7. Workflow Metadata
    retry_count: int               # Tracks number of synthesis retries due to unsupported claims
    status: str                    # General status or error message
