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

EVIDENCE (Grouped by Document):
{evidence_text}

INSTRUCTIONS:
1. Synthesize the evidence across documents (if applicable) to construct a comprehensive answer.
2. For every factual claim, include an inline citation formatted exactly as [DocID, ChunkID].
3. If the evidence is insufficient to answer the question, explicitly state: "Insufficient evidence in the retrieved papers."
4. Do NOT hallucinate or include outside knowledge.
5. Do NOT invent Document IDs or Chunk IDs. Use only the ones provided.

Answer with inline citations:
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
                
    evidence_text = "\n".join(evidence_parts)
    prompt = SYNTHESIS_PROMPT.format(question=question, evidence_text=evidence_text)
    
    draft_answer = ""
    if GROQ_API_KEY:
        try:
            draft_answer = _call_groq_raw(prompt)
        except Exception:
            pass
            
    if not draft_answer:
        draft_answer = "Fallback Answer: Insufficient evidence in the retrieved papers (LLM unavailable or failed)."
        
    return {
        "draft_answer": draft_answer,
        "citations": citations_list,
        "status": "synthesized"
    }
