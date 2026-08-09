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
    research_type: str             # "simple" or "deep"
    session_id: str
    task_id: str
    chat_history: list[dict]       # [{"role": "user", "content": "..."}, ...]
    structured_mode: bool          # Feature 6: request structured research extraction from the SAME synthesis call
    
    # 2. Planning Phase
    research_plan: dict[str, Any]  # {"mode": "...", "comparison_dimensions": [...], etc}
    sub_queries: list[str]
    completed_sub_queries: list[str]
    document_ids: list[str]
    
    # 3. Retrieval Phase
    retrieval_results: list[dict]
    
    # 4. Organization Phase
    evidence_by_document: dict[str, list[dict]]
    
    # 5. Adaptive Research Phase (Phase 6)
    research_iterations: int
    evidence_gaps: list[str]
    adaptive_queries: list[dict]   # e.g., [{"query": "...", "reason": "...", "iteration": 1, "improved": True}]
    
    # 6. Cross-Paper Analysis (Phase 6)
    contradictions: list[dict]     # {"claim_A", "evidence_A", "claim_B", "evidence_B", "status", "context"}
    similarities: list[dict]       # {"finding", "supporting_evidence": [...]}
    differences: list[dict]        # {"finding", "supporting_evidence": [...]}
    research_gaps: list[dict]      # {"gap", "supporting_evidence": [...]}
    
    # 7. Synthesis Phase
    draft_answer: str
    claims: list[dict]             # The Claim Graph: {"claim_id", "claim_text", "claim_type", "supporting_evidence": [...], "contradicting_evidence": [...], "citations", "verification_status", "confidence"}
    confidence: str                # HIGH, MEDIUM, LOW, UNAVAILABLE
    citations: list[dict]
    grounded: bool                 # self-reported by the ONE synthesis call (normal mode); no separate verifier call
    evidence_sufficient: bool      # self-reported by the ONE synthesis call (normal mode); no separate verifier call
    self_reported_document_ids: list[str]  # auxiliary signal only; citations stay evidence-derived, not model-derived
    structured_data: dict | None   # populated only when structured_mode was requested AND synthesis succeeded
    claim_evidence_trace: list[dict]  # per-claim DIRECTLY_STATED/STRONGLY_SUPPORTED/NOT_FOUND tags; structured_mode only, never surfaced in the normal chat UI
    
    # 8. Verification Phase
    verification_results: list[dict]
    
    # 9. Workflow Metadata
    retry_count: int               # Tracks number of verification retries
    status: str                    # e.g., "RESEARCH_LIMIT_REACHED", "INSUFFICIENT_EVIDENCE", "OK"
