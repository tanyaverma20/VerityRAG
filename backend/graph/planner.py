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
from query_transform import decompose_query, _call_groq_raw, _looks_complex, decompose_query_deterministic
from config import GROQ_API_KEY


PLANNER_PROMPT = """You are a research planning assistant for an academic RAG system.
Given a user's research question and the previous chat history, analyze the context and output a JSON object with your research plan.

Context is important. If the user asks a follow-up question (e.g., "What about their training data?"), use the chat history to resolve the subject (e.g., "their" -> "Llama 2 and Mistral"). Output the fully resolved query.

Determine:
1. "resolved_query": The fully contextualized question, replacing pronouns and implicit references with explicit ones.
2. "mode": One of ["single_paper", "multi_paper", "synthesis"]
   - single_paper: The question is about a specific detail likely found in one place or paper.
   - multi_paper: The question asks to compare or find information across multiple papers.
   - synthesis: The question asks for a high-level summary, broader concepts, or methodology limitations that require synthesizing evidence from multiple sources.
3. "needs_decomposition": boolean (true/false)
   - true if the question asks multiple distinct things (e.g. "What is X and what is Y?")
4. "comparison_dimensions": list of strings (optional, e.g. ["architecture", "datasets", "results", "limitations"]) if this is a multi-paper comparison question.
5. "reasoning": A brief string explaining your choice.

Return ONLY valid JSON.

Chat History:
{chat_history}

Current Question: {question}
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


def plan_research(state: dict) -> dict:
    """
    Analyzes the query and decides the research mode and decomposition.

    NORMAL mode (research_type != "deep") makes ZERO LLM calls: mode/
    decomposition are decided heuristically, and the raw question passes
    straight to retrieval + the single synthesis call. That one call also
    receives recent chat history directly, so it resolves follow-up
    references itself ("their" -> the papers already in context) instead of
    a separate query-rewriting LLM call.

    Decomposition is also deterministic in normal mode: a genuinely
    multi-aspect question (e.g. "Compare methodologies, datasets and
    results") is split into per-aspect sub_queries by
    decompose_query_deterministic() — rule-based, not an LLM call — so each
    aspect gets its own retrieval pass before everything is fused back
    together. A simple question always collapses back to a single
    sub_query, unchanged from before.

    Only DEEP mode (explicitly requested by the user, never auto-triggered
    by this planner) uses the LLM-backed planner below.
    """
    query = state.get("latest_query") or state.get("original_query")
    research_type = state.get("research_type", "simple")

    if research_type != "deep":
        plan = _fallback_plan(query)
        plan["resolved_query"] = query
        sub_queries = decompose_query_deterministic(query)
        plan["needs_decomposition"] = len(sub_queries) > 1
        plan["reasoning"] = (
            f"Normal mode: heuristic planning only, zero LLM calls "
            f"({len(sub_queries)} sub-quer{'y' if len(sub_queries) == 1 else 'ies'})."
        )
        return {
            "original_query": query,
            "research_plan": plan,
            "sub_queries": sub_queries,
            "research_type": research_type,
            "status": "planned (heuristic, no LLM call)",
        }

    # --- DEEP MODE ONLY below this line ---
    history = state.get("chat_history", [])

    # Format chat history
    history_text = "None"
    if history:
        history_lines = []
        for msg in history[-4:]:  # Use at most the last 4 messages to avoid blowing up context
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_lines.append(f"{role.capitalize()}: {content}")
        history_text = "\n".join(history_lines)

    plan = None

    if GROQ_API_KEY:
        try:
            raw = _call_groq_raw(PLANNER_PROMPT.format(question=query, chat_history=history_text))
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                # Validate schema
                if "mode" in parsed and "needs_decomposition" in parsed and "reasoning" in parsed and "resolved_query" in parsed:
                    plan = parsed
        except Exception as e:
            print(f"LLM planner failed: {e}")
            plan = _fallback_plan(query)
            plan["resolved_query"] = query

    if plan is None:
        plan = _fallback_plan(query)
        plan["resolved_query"] = query

    resolved_query = plan.get("resolved_query", query)

    # If decomposition is needed, call the Phase 2 decomposed utility using the RESOLVED query
    sub_queries = [resolved_query]
    if plan.get("needs_decomposition", False):
        try:
            decomposed = decompose_query(resolved_query)
            if decomposed and len(decomposed) > 0:
                sub_queries = decomposed
        except Exception:
            pass # fallback to [resolved_query] already set

    return {
        "original_query": resolved_query, # Update state with resolved query for subsequent steps
        "research_plan": plan,
        "sub_queries": sub_queries,
        "research_type": research_type,
        "status": "planned"
    }
