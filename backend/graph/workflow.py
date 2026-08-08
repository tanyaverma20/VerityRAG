"""
workflow.py — LangGraph orchestration for the Phase 3 Research Workflow
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from .state import ResearchState
from .planner import plan_research
from .retriever import retrieve_evidence
from .organizer import organize_evidence
from .synthesizer import synthesize_answer
from .verifier import verify_evidence
from .analyzer import analyze_evidence_sufficiency, generate_adaptive_queries, cross_paper_analysis, assign_confidence

MAX_RESEARCH_ITERATIONS = 3


def route_verification(state: ResearchState) -> str:
    """
    Conditional edge logic after verification.
    If any claim is UNSUPPORTED and we haven't retried yet, loop back to synthesize.
    Otherwise, proceed to END.
    """
    results = state.get("verification_results", [])
    retry_count = state.get("retry_count", 0)
    
    unsupported = any(r.get("status") == "UNSUPPORTED" for r in results)
    
    if unsupported and retry_count < 1:
        return "synthesize"
    
    return "assign_confidence"

def route_adaptive_research(state: ResearchState) -> str:
    """
    Checks if evidence is sufficient. 
    If not and iterations < MAX_RESEARCH_ITERATIONS, generate new queries.
    Otherwise proceed to cross-paper analysis.
    """
    gaps = state.get("evidence_gaps", [])
    iters = state.get("research_iterations", 1)
    
    if gaps and iters < MAX_RESEARCH_ITERATIONS:
        return "adapt_queries"
    
    return "cross_paper"

def build_research_graph() -> StateGraph:
    """
    Builds and compiles the Research Workflow StateGraph.
    """
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("plan", plan_research)
    workflow.add_node("retrieve", retrieve_evidence)
    workflow.add_node("organize", organize_evidence)
    workflow.add_node("analyze_sufficiency", analyze_evidence_sufficiency)
    workflow.add_node("adapt_queries", generate_adaptive_queries)
    workflow.add_node("cross_paper", cross_paper_analysis)
    
    # We wrap synthesize to increment the retry counter if it's called multiple times
    def synthesize_wrapper(state: ResearchState):
        current_retry = state.get("retry_count", 0)
        updates = synthesize_answer(state)
        if "draft_answer" in state and state["draft_answer"]:
            updates["retry_count"] = current_retry + 1
        else:
            updates["retry_count"] = 0
        return updates
        
    workflow.add_node("synthesize", synthesize_wrapper)
    workflow.add_node("verify", verify_evidence)
    workflow.add_node("assign_confidence", assign_confidence)
    
    # Define edges
    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "retrieve")
    workflow.add_edge("retrieve", "organize")
    workflow.add_edge("organize", "analyze_sufficiency")
    
    workflow.add_conditional_edges("analyze_sufficiency", route_adaptive_research, {
        "adapt_queries": "adapt_queries",
        "cross_paper": "cross_paper"
    })
    
    workflow.add_edge("adapt_queries", "retrieve") # Loop back to retrieve new chunks
    
    workflow.add_edge("cross_paper", "synthesize")
    workflow.add_edge("synthesize", "verify")
    
    # Conditional edge for verification retry loop
    workflow.add_conditional_edges("verify", route_verification, {
        "synthesize": "synthesize",
        "assign_confidence": "assign_confidence"
    })
    
    workflow.add_edge("assign_confidence", END)
    
    return workflow.compile()

# Global compiled graph instance
research_app = build_research_graph()
