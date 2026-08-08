"""
query_transform.py — Lightweight, LLM-optional query transformation.

Two operations are provided:
  1. rewrite_query()     — make a vague query retrieval-friendly
  2. decompose_query()   — split a complex multi-aspect question into sub-queries

Both functions are OPTIONAL helpers.  Retrieval never depends on successful
transformation; every function falls back to the original query on any error.

Design constraints (Phase 2):
  - No LangGraph, no agents, no planning loops.
  - LLM calls are only made when GROQ_API_KEY is present AND the query
    appears to need transformation.
  - All callers receive usable queries even if the LLM is unavailable.
"""

import json
import re
from config import GROQ_API_KEY


# ---------------------------------------------------------------------------
# Heuristics — avoid LLM calls for simple queries
# ---------------------------------------------------------------------------

_COMPARISON_SIGNALS = [
    "compare", "comparison", "versus", "vs", "difference between",
    "how do", "contrast", "both", "all papers", "across papers",
    "each paper", "different papers",
]

_VAGUE_SIGNALS = [
    "how did they", "what did they", "explain it", "tell me about",
    "what is it", "can you", "describe",
]


def _looks_vague(query: str) -> bool:
    q = query.lower().strip()
    return any(s in q for s in _VAGUE_SIGNALS) and len(q.split()) < 8


def _looks_complex(query: str) -> bool:
    q = query.lower().strip()
    return any(s in q for s in _COMPARISON_SIGNALS)


# ---------------------------------------------------------------------------
# LLM-backed transformation (Groq, same client as llm.py uses)
# ---------------------------------------------------------------------------

def _call_groq_raw(prompt: str) -> str:
    """Direct Groq call that returns raw text; raises on any error."""
    import time
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # smallest/fastest for transformation
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

REWRITE_PROMPT = """You are a search-query optimizer for an academic research paper RAG system.

Given a vague user question, rewrite it as a concise, keyword-rich retrieval query.

Rules:
- Return ONLY the improved query string, nothing else.
- Keep it under 30 words.
- Do not add information not implied by the original question.
- If the original query is already good, return it unchanged.

Original question: {question}
Retrieval query:"""

DECOMPOSE_PROMPT = """You are a research question decomposer for a multi-paper RAG system.

Given a complex research question, decompose it into 2–6 simpler sub-questions
that can each be answered independently by searching individual research papers.

Return ONLY a JSON array of strings. Example:
["What architecture does Paper A use?", "What datasets are used?"]

Complex question: {question}
Sub-questions JSON:"""


def rewrite_query(query: str) -> str:
    """
    Return a retrieval-friendly version of *query*.

    Uses the LLM only when:
      - GROQ_API_KEY is set, AND
      - the query looks vague (heuristic).

    Always returns a non-empty string.  Falls back to original on any error.
    """
    query = query.strip()
    if not query:
        return query

    # Skip LLM for queries that are already specific enough
    if not _looks_vague(query) or not GROQ_API_KEY:
        return query

    try:
        rewritten = _call_groq_raw(REWRITE_PROMPT.format(question=query))
        # Sanity: reject empty or implausibly long responses
        if rewritten and len(rewritten) < 300:
            return rewritten
    except Exception:
        pass

    return query  # fallback


def decompose_query(query: str) -> list[str]:
    """
    Decompose a complex question into a list of focused sub-queries.

    Uses the LLM only when:
      - GROQ_API_KEY is set, AND
      - the query looks like a comparison / multi-aspect question.

    Always returns a non-empty list.  Falls back to [original_query] on any error.
    """
    query = query.strip()
    if not query:
        return [query]

    if not _looks_complex(query) or not GROQ_API_KEY:
        return [query]

    try:
        raw = _call_groq_raw(DECOMPOSE_PROMPT.format(question=query))
        # Extract JSON array — tolerate markdown fences
        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if json_match:
            sub_qs = json.loads(json_match.group())
            if isinstance(sub_qs, list) and all(isinstance(q, str) for q in sub_qs):
                valid = [q.strip() for q in sub_qs if q.strip()]
                if valid:
                    return valid
    except Exception:
        pass

    return [query]  # fallback
