"""
synthesizer.py — Graph node for Citation-Grounded Synthesis
"""
import json
import sys
import os
from typing import Any

from .state import ResearchState

# Ensure we can import from the parent backend folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from query_transform import _call_groq_raw
from config import GROQ_API_KEY


SYNTHESIS_PROMPT = """You are a highly analytical academic research assistant.
Answer the following question using ONLY the provided evidence.

QUESTION: {question}

CROSS-PAPER INSIGHTS:
{insights_text}

EVIDENCE (Grouped by Document):
{evidence_text}

INSTRUCTIONS:
1. Synthesize the evidence across documents (if applicable) to construct a comprehensive answer.
2. For every factual claim, include an inline citation formatted exactly as [DocID, ChunkID].
3. If the evidence is insufficient to answer the question, explicitly state: "Insufficient evidence in the retrieved papers."
4. Do NOT hallucinate or include outside knowledge.
5. Do NOT invent Document IDs or Chunk IDs. Use only the ones provided.
{format_instructions}

IMPORTANT: You MUST output your response as a valid JSON object with two fields:
- "draft_answer": A string containing your fully formatted answer with inline citations.
- "claims": A list of dictionaries representing the key claims you made. Each dictionary MUST have:
  - "claim_id": A unique string ID (e.g. "claim_1")
  - "claim_text": The actual statement/claim
  - "claim_type": "FACT", "COMPARISON", "SYNTHESIS", "CONTRADICTORY", or "RESEARCH_GAP"

JSON:
"""

MULTI_PAPER_FORMAT = """
Because this is a multi-paper comparison, please structure your draft_answer clearly with the following headings if applicable:
- Executive Summary
- Comparison (across detected dimensions)
- Paper-by-Paper Findings
- Key Similarities
- Key Differences
- Contradictions / Disagreements
- Research Gaps
- Limitations
- Conclusion
"""

def synthesize_answer(state: ResearchState) -> dict[str, Any]:
    """
    Generates a draft answer using organized evidence and produces structured citations.
    """
    question = state["original_query"]
    evidence_by_doc = state.get("evidence_by_document", {})
    
    if not evidence_by_doc:
        return {
            "draft_answer": "Insufficient evidence in the retrieved papers.",
            "citations": [],
            "status": "synthesized (no evidence)"
        }
        
    # Build text representation of evidence
    evidence_parts = []
    citations_list = []
    
    for doc_id, parents in evidence_by_doc.items():
        doc_source = ""
        evidence_parts.append(f"\n--- Document ID: {doc_id} ---")
        
        for p in parents:
            context = p["parent_context"]
            evidence_parts.append(f"CONTEXT: {context}")
            
            for child in p["children"]:
                chunk_id = child["chunk_id"]
                doc_source = child.get("source", "")
                evidence_parts.append(f"  [EXACT CHUNK: {chunk_id}] {child['text']}")
                
                # Add to structured citations
                citations_list.append({
                    "document_id": doc_id,
                    "source": doc_source,
                    "chunk_id": chunk_id,
                    "page_number": child.get("page_number", ""),
                    "section": child.get("section", ""),
                })
                
    plan = state.get("research_plan", {})
    format_instructions = MULTI_PAPER_FORMAT if (state.get("research_type") == "deep" or plan.get("mode") in ["multi_paper", "synthesis"]) else ""
    
    # Inject cross-paper insights
    insights_parts = []
    if state.get("similarities"): insights_parts.append(f"Similarities: {state.get('similarities')}")
    if state.get("differences"): insights_parts.append(f"Differences: {state.get('differences')}")
    if state.get("contradictions"): insights_parts.append(f"Contradictions: {state.get('contradictions')}")
    if state.get("research_gaps"): insights_parts.append(f"Research Gaps: {state.get('research_gaps')}")
    
    insights_text = "\n".join(insights_parts) if insights_parts else "None detected."
    
    evidence_text = "\n".join(evidence_parts)
    prompt = SYNTHESIS_PROMPT.format(question=question, insights_text=insights_text, evidence_text=evidence_text, format_instructions=format_instructions)
    
    draft_answer = ""
    claims = []
    
    if GROQ_API_KEY:
        try:
            import re
            raw = _call_groq_raw(prompt)
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                draft_answer = parsed.get("draft_answer", "")
                claims = parsed.get("claims", [])
        except Exception:
            pass
            
    if not draft_answer:
        draft_answer = "Fallback Answer: Insufficient evidence in the retrieved papers (LLM unavailable or failed)."
        
    return {
        "draft_answer": draft_answer,
        "claims": claims,
        "citations": citations_list,
        "status": "synthesized"
    }
