"""
workflow.py — LangGraph orchestration for the Research Workflow

NORMAL mode (research_type != "deep") takes the short path and makes exactly
ONE LLM call total:

    plan (no LLM) -> retrieve (no LLM) -> organize (no LLM) -> synthesize
    (the ONE LLM call, self-assesses grounded/evidence_sufficient) ->
    assign_confidence (no LLM) -> END

DEEP mode keeps the full multi-step architecture (LLM-backed planner,
adaptive retrieval loop, cross-paper analysis, separate verification pass,
one synthesis retry) — but every loop in it is bounded by explicit limits
from config.py (DEEP_RESEARCH_MAX_ITERATIONS, DEEP_RESEARCH_MAX_LLM_CALLS,
DEEP_RESEARCH_MAX_RETRIES) so it can never run unbounded.
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

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DEEP_RESEARCH_MAX_ITERATIONS, DEEP_RESEARCH_MAX_LLM_CALLS, DEEP_RESEARCH_MAX_RETRIES
from query_transform import get_call_log

# Kept as a module-level name for backward compatibility with anything that
# imported it directly; sourced from config.py so it's one configurable value.
MAX_RESEARCH_ITERATIONS = DEEP_RESEARCH_MAX_ITERATIONS


def _llm_call_budget_exhausted() -> bool:
    """Hard backstop for Deep Research: even if a routing bug ever allowed
    another loop, this stops it once DEEP_RESEARCH_MAX_LLM_CALLS physical
    Groq requests have been made for this request (see query_transform's
    per-request call log)."""
    return len(get_call_log()) >= DEEP_RESEARCH_MAX_LLM_CALLS


def route_after_organize(state: ResearchState) -> str:
    """
    NORMAL mode skips straight to the single synthesis call — no sufficiency
    analysis, no adaptive retrieval loop, no cross-paper LLM pass. Those are
    Deep Research features only.
    """
    if state.get("research_type") == "deep" and not _llm_call_budget_exhausted():
        return "analyze_sufficiency"
    return "synthesize"


def route_verification(state: ResearchState) -> str:
    """
    Conditional edge logic after verification (DEEP mode only — normal mode
    never reaches this node at all).
    If any claim is UNSUPPORTED and we haven't retried yet, loop back to synthesize.
    Otherwise, proceed to END.
    """
    results = state.get("verification_results", [])
    retry_count = state.get("retry_count", 0)

    unsupported = any(r.get("status") == "UNSUPPORTED" for r in results)

    if unsupported and retry_count < DEEP_RESEARCH_MAX_RETRIES and not _llm_call_budget_exhausted():
        return "synthesize"

    return "assign_confidence"

def route_adaptive_research(state: ResearchState) -> str:
    """
    Checks if evidence is sufficient (DEEP mode only).
    If not and iterations < DEEP_RESEARCH_MAX_ITERATIONS, generate new queries.
    Otherwise proceed to cross-paper analysis.
    """
    gaps = state.get("evidence_gaps", [])
    iters = state.get("research_iterations", 1)

    if gaps and iters < DEEP_RESEARCH_MAX_ITERATIONS and not _llm_call_budget_exhausted():
        return "adapt_queries"

    return "cross_paper"

def route_after_synthesize(state: ResearchState) -> str:
    """
    NORMAL mode's single synthesis call already self-reports grounded/
    evidence_sufficient — no separate verifier LLM call needed. Only DEEP
    mode runs the additional verification pass.
    """
    if state.get("research_type") == "deep" and not _llm_call_budget_exhausted():
        return "verify"
    return "assign_confidence"

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

    # NORMAL mode -> straight to synthesize (the ONE LLM call).
    # DEEP mode -> the full sufficiency-analysis / adaptive-retrieval path.
    workflow.add_conditional_edges("organize", route_after_organize, {
        "analyze_sufficiency": "analyze_sufficiency",
        "synthesize": "synthesize",
    })

    workflow.add_conditional_edges("analyze_sufficiency", route_adaptive_research, {
        "adapt_queries": "adapt_queries",
        "cross_paper": "cross_paper"
    })

    workflow.add_edge("adapt_queries", "retrieve") # Loop back to retrieve new chunks

    workflow.add_edge("cross_paper", "synthesize")

    # NORMAL mode -> assign_confidence directly (no separate verify LLM call).
    # DEEP mode -> the separate verification pass + bounded retry loop.
    workflow.add_conditional_edges("synthesize", route_after_synthesize, {
        "verify": "verify",
        "assign_confidence": "assign_confidence",
    })

    workflow.add_conditional_edges("verify", route_verification, {
        "synthesize": "synthesize",
        "assign_confidence": "assign_confidence"
    })

    workflow.add_edge("assign_confidence", END)

    return workflow.compile()

# Global compiled graph instance
research_app = build_research_graph()
