"""
planner.py — Graph node for Query Analysis / Research Planning
"""

import json
import re
from typing import Any

from .state import ResearchState
import sys
import os

# Ensure we can import from the parent backend folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from query_transform import decompose_query, _call_groq_raw, _looks_complex
from config import GROQ_API_KEY


PLANNER_PROMPT = """You are a research planning assistant for an academic RAG system.
Given a user's research question, analyze it and output a JSON object with your research plan.

Determine:
1. "mode": One of ["single_paper", "multi_paper", "synthesis"]
   - single_paper: The question is about a specific detail likely found in one place or paper.
   - multi_paper: The question asks to compare or find information across multiple papers.
   - synthesis: The question asks for a high-level summary, broader concepts, or methodology limitations that require synthesizing evidence from multiple sources.
2. "needs_decomposition": boolean (true/false)
   - true if the question asks multiple distinct things (e.g. "What is X and what is Y?")
3. "comparison_dimensions": list of strings (optional, e.g. ["architecture", "datasets", "results", "limitations"]) if this is a multi-paper comparison question.
4. "reasoning": A brief string explaining your choice.

Return ONLY valid JSON.

Question: {question}
JSON Plan:"""

def _fallback_plan(query: str) -> dict[str, Any]:
    """Deterministic fallback if the LLM is unavailable or fails."""
    is_complex = _looks_complex(query)
    return {
        "mode": "multi_paper" if is_complex else "single_paper",
        "needs_decomposition": is_complex,
        "comparison_dimensions": ["methodology", "results"] if is_complex else [],
        "reasoning": "Fallback to heuristic planning due to LLM unavailability or failure."
    }


def plan_research(state: ResearchState) -> dict[str, Any]:
    """
    Analyzes the query and decides the research mode and decomposition.
    """
    query = state["original_query"]
    plan = None

    if GROQ_API_KEY:
        try:
            raw = _call_groq_raw(PLANNER_PROMPT.format(question=query))
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                # Validate schema
                if "mode" in parsed and "needs_decomposition" in parsed and "reasoning" in parsed:
                    plan = parsed
        except Exception:
            pass

    if plan is None:
        plan = _fallback_plan(query)
        
    research_type = state.get("research_type", "simple")
    # If the user didn't explicitly request deep but the query is very complex, we could optionally promote it.
    # For now, we respect the state's research_type, which defaults to 'simple' from the API.
    # But if the planner detects a multi_paper comparison, it's safer to promote it to deep research.
    if plan.get("mode") == "multi_paper" and research_type != "deep":
        research_type = "deep"

    # If decomposition is needed, call the Phase 2 decomposed utility
    sub_queries = [query]
    if plan.get("needs_decomposition", False):
        try:
            decomposed = decompose_query(query)
            if decomposed and len(decomposed) > 0:
                sub_queries = decomposed
        except Exception:
            pass # fallback to [query] already set

    return {
        "research_plan": plan,
        "sub_queries": sub_queries,
        "research_type": research_type,
        "status": "planned"
    }
