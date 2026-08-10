"""
analyzer.py — Phase 6 Analyzer node for Evidence gap detection and Cross-Paper analysis
"""
import json
from typing import Any
from .state import ResearchState

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from query_transform import _call_groq_raw
from config import GROQ_API_KEY
from json_extract import extract_json_object

SUFFICIENCY_PROMPT = """You are an evidence analyzer for an academic RAG system.
Evaluate the retrieved evidence against the research plan.
Determine if there are significant missing pieces of information required to fully answer the query.

Original Query: {query}
Research Plan: {plan}

Evidence Grouped by Document:
{evidence}

Output ONLY a JSON object with:
- "is_sufficient": boolean
- "gaps": list of strings (empty if sufficient). E.g., ["Missing training dataset details for document docB"]

JSON:"""

ADAPTIVE_QUERY_PROMPT = """You are an academic researcher. You need to find missing information.
Given the original query, the research plan, and a list of evidence gaps, generate targeted search queries to find the missing information.
DO NOT generate more than 3 queries.

Original Query: {query}
Gaps: {gaps}

Output ONLY a JSON object with:
- "queries": list of dicts: {{"query": "...", "reason": "...", "target_document": "doc_id or null"}}

JSON:"""

CROSS_PAPER_PROMPT = """You are an academic synthesizer. Analyze the evidence across multiple papers.
Identify similarities, differences, contradictions, and potential research gaps.
Contradictions should only be marked if the papers directly conflict on a fact, results, or methodology given the same context.

Original Query: {query}

Evidence:
{evidence}

Output ONLY a JSON object with:
- "similarities": list of dicts {{"finding": "...", "supporting_evidence": ["docX", "docY"]}}
- "differences": list of dicts {{"finding": "...", "supporting_evidence": ["docX", "docY"]}}
- "contradictions": list of dicts {{"claim_A": "...", "evidence_A": "docX", "claim_B": "...", "evidence_B": "docY", "status": "CONTRADICTORY", "context": "..."}}
- "research_gaps": list of dicts {{"gap": "...", "supporting_evidence": ["docX"]}}

JSON:"""

def _format_evidence(evidence_by_doc: dict) -> str:
    parts = []
    for doc_id, chunks in evidence_by_doc.items():
        parts.append(f"--- Document: {doc_id} ---")
        for c in chunks:
            parts.append(f"[{doc_id}, {c.get('chunk_id')}] {c.get('text')}")
    return "\n".join(parts)

def analyze_evidence_sufficiency(state: ResearchState) -> dict[str, Any]:
    """Analyzes if evidence is sufficient and populates evidence_gaps."""
    if state.get("research_type") == "simple":
        return {"evidence_gaps": [], "status": "sufficient"}
        
    query = state["original_query"]
    plan = state.get("research_plan", {})
    evidence_text = _format_evidence(state.get("evidence_by_document", {}))
    
    gaps = []
    
    if GROQ_API_KEY:
        try:
            prompt = SUFFICIENCY_PROMPT.format(query=query, plan=json.dumps(plan), evidence=evidence_text)
            raw = _call_groq_raw(prompt)
            parsed = extract_json_object(raw)
            if parsed:
                gaps = parsed.get("gaps", [])
        except Exception as e:
            print(f"Sufficiency analysis failed: {e}")
            pass
            
    return {"evidence_gaps": gaps}

def generate_adaptive_queries(state: ResearchState) -> dict[str, Any]:
    """Generates new targeted queries based on gaps."""
    gaps = state.get("evidence_gaps", [])
    if not gaps:
        return {"status": "skipped adaptive_queries (no gaps)"}
        
    query = state["original_query"]
    
    new_queries = []
    if GROQ_API_KEY:
        try:
            prompt = ADAPTIVE_QUERY_PROMPT.format(query=query, gaps=json.dumps(gaps))
            raw = _call_groq_raw(prompt)
            parsed = extract_json_object(raw)
            if parsed:
                new_queries = parsed.get("queries", [])
        except Exception:
            pass
            
    # Deduplicate queries
    existing = set(state.get("sub_queries", []))
    adaptive = state.get("adaptive_queries", [])
    
    added_sub_queries = []
    for q in new_queries:
        text = q.get("query")
        if text and text not in existing:
            added_sub_queries.append(text)
            existing.add(text)
            adaptive.append({
                "query": text,
                "reason": q.get("reason", "Gap detection"),
                "iteration": state.get("research_iterations", 1),
                "improved": False # Default, can be evaluated later
            })
            
    return {
        "sub_queries": state.get("sub_queries", []) + added_sub_queries,
        "adaptive_queries": adaptive,
        "research_iterations": state.get("research_iterations", 1) + 1
    }

def cross_paper_analysis(state: ResearchState) -> dict[str, Any]:
    """Analyzes cross-paper dimensions."""
    if state.get("research_type") == "simple":
        return {"status": "skipped cross_paper (simple mode)"}
        
    query = state["original_query"]
    evidence_text = _format_evidence(state.get("evidence_by_document", {}))
    
    similarities = []
    differences = []
    contradictions = []
    research_gaps = []
    
    if GROQ_API_KEY:
        try:
            prompt = CROSS_PAPER_PROMPT.format(query=query, evidence=evidence_text)
            raw = _call_groq_raw(prompt)
            parsed = extract_json_object(raw)
            if parsed:
                similarities = parsed.get("similarities", [])
                differences = parsed.get("differences", [])
                contradictions = parsed.get("contradictions", [])
                research_gaps = parsed.get("research_gaps", [])
        except Exception:
            pass
            
    return {
        "similarities": similarities or [],
        "differences": differences or [],
        "contradictions": contradictions or [],
        "research_gaps": research_gaps or []
    }

def assign_confidence(state: ResearchState) -> dict[str, Any]:
    """
    Assigns confidence based on the Phase 6 heuristic:
    - Number of independent docs
    - Verification status
    - Contradictions

    Deterministic — never calls the LLM itself.
    """
    # 1. Evidence coverage
    docs = state.get("evidence_by_document", {})
    doc_count = len(docs)

    # 2. Verification Status
    verifs = state.get("verification_results", [])

    # NORMAL mode never runs a separate verify step, so verification_results
    # is empty by design — not "verification unavailable". Use the single
    # synthesis call's own self-assessment instead of treating this the same
    # as deep mode's real verification-failure case. Deep mode (which DOES
    # populate verification_results) is unaffected and keeps the branch below.
    if not verifs and "evidence_sufficient" in state:
        evidence_sufficient = state.get("evidence_sufficient")
        grounded = state.get("grounded", evidence_sufficient)
        if doc_count == 0:
            return {"confidence": "UNAVAILABLE"}
        if not evidence_sufficient:
            return {"confidence": "LOW"}
        return {"confidence": "HIGH" if grounded else "MEDIUM"}

    if not verifs:
        return {"confidence": "UNAVAILABLE"}

    supported = sum(1 for v in verifs if v.get("status") == "SUPPORTED")
    unsupported = sum(1 for v in verifs if v.get("status") == "UNSUPPORTED")
    unavailable = sum(1 for v in verifs if v.get("status") == "VERIFICATION_UNAVAILABLE")
    
    if unavailable > 0 and len(verifs) > 0 and (unavailable / len(verifs)) > 0.5:
        return {"confidence": "UNAVAILABLE"}
        
    # 3. Contradictions
    contradictions = state.get("contradictions") or []
    major_contradictions = sum(1 for c in contradictions if c.get("status") == "CONTRADICTORY")
    
    # 4. Gaps
    gaps = len(state.get("evidence_gaps") or [])
    
    if gaps > 0 or unsupported > 0 or major_contradictions > 0 or doc_count == 0:
        return {"confidence": "LOW"}
    elif supported == len(verifs) and doc_count > 0:
        return {"confidence": "HIGH"}
    else:
        return {"confidence": "MEDIUM"}
