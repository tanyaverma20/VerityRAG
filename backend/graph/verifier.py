"""
verifier.py — Graph node for Claim-Level Evidence Verification
"""
import json
import re
import sys
import os
from typing import Any

from .state import ResearchState

# Ensure we can import from the parent backend folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from query_transform import _call_groq_raw
from config import GROQ_API_KEY


VERIFICATION_PROMPT = """You are a meticulous fact-checker. 
Analyze the provided claims (or extract them from the draft answer if none are provided).
For each claim, check if it is supported by the provided evidence.

EVIDENCE:
{evidence_text}

CLAIMS TO VERIFY:
{claims_text}

DRAFT ANSWER (for context):
{draft_answer}

INSTRUCTIONS:
Output a JSON array of verification results. Each object must have:
- "claim": The actual statement being verified.
- "supporting_evidence": The text from the evidence that supports it (or "None" if unsupported).
- "status": One of ["SUPPORTED", "WEAKLY_SUPPORTED", "UNSUPPORTED"].

Return ONLY valid JSON array.

JSON Array:"""

def verify_evidence(state: ResearchState) -> dict[str, Any]:
    """
    Evaluates claims in the draft answer against the organized evidence.
    """
    draft_answer = state.get("draft_answer", "")
    evidence_by_doc = state.get("evidence_by_document", {})
    
    if not draft_answer or not evidence_by_doc:
        return {
            "verification_results": [],
            "status": "verification skipped (no answer or evidence)"
        }
        
    # Build text representation of evidence for the verifier
    evidence_parts = []
    for doc_id, parents in evidence_by_doc.items():
        for p in parents:
            for child in p["children"]:
                evidence_parts.append(f"[{doc_id}, {child['chunk_id']}] {child['text']}")
                
    evidence_text = "\n".join(evidence_parts)
    
    claims_to_verify = state.get("claims", [])
    claims_text = "Extract from answer."
    if claims_to_verify:
        claims_text = "\n".join([f"- {c.get('claim_text', '')}" for c in claims_to_verify])
        
    prompt = VERIFICATION_PROMPT.format(evidence_text=evidence_text, claims_text=claims_text, draft_answer=draft_answer)
    
    verification_results = []
    if GROQ_API_KEY:
        try:
            raw = _call_groq_raw(prompt)
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    for item in parsed:
                        if "claim" in item and "status" in item:
                            verification_results.append(item)
        except Exception:
            pass
            
    # Fallback if verification fails
    if not verification_results:
        verification_results = [{
            "claim": "Full answer",
            "supporting_evidence": "",
            "status": "VERIFICATION_UNAVAILABLE" 
        }]
        
    # Update the claims array in the state with verification statuses if they match
    original_claims = state.get("claims") or []
    updated_claims = list(original_claims)
    for c in updated_claims:
        # Default
        c["verification_status"] = "VERIFICATION_UNAVAILABLE"
        for v in verification_results:
            if v.get("claim", "") in c.get("claim_text", "") or c.get("claim_text", "") in v.get("claim", ""):
                c["verification_status"] = v.get("status", "VERIFICATION_UNAVAILABLE")
                break
        
    return {
        "verification_results": verification_results,
        "claims": updated_claims,
        "status": f"verified {len(verification_results)} claims"
    }
