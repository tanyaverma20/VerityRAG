"""
synthesizer.py — Graph node for Citation-Grounded Synthesis

Output is a compact, Pydantic-validated JSON object (schemas.AnswerJSON) —
just the answer plus a couple of self-assessment booleans and the document
IDs actually drawn on. No chain-of-thought, no per-claim breakdown, no long
free-form metadata: the smaller the requested schema, the fewer output
tokens the ONE call needs to spend, and the less there is for the model to
get wrong.
"""
import re
import sys
import os
from typing import Any

from pydantic import ValidationError

from .state import ResearchState

# Ensure we can import from the parent backend folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from query_transform import (
    _call_groq_raw, is_rate_limit_error,
    INSUFFICIENT_EVIDENCE_MESSAGE, RATE_LIMIT_MESSAGE, GENERATION_FAILED_MESSAGE,
)
from config import GROQ_API_KEY
from schemas import AnswerJSON, StructuredAnswerJSON


SYNTHESIS_PROMPT = """You are VerityRAG, a research assistant.

Answer ONLY using the evidence provided below, which was retrieved from the
user's currently active uploaded documents. Do not use outside knowledge to
fill missing information. If the evidence does not contain enough
information to answer the question reliably, explicitly say so. Never
fabricate facts, numbers, results, methodologies, datasets, limitations, or
comparisons. For comparisons, distinguish the evidence belonging to each
uploaded paper. The retrieved evidence is authoritative for this answer.

CONVERSATION HISTORY (most recent last — use ONLY to resolve follow-up
references like "their" or "that paper" to what was already discussed; it is
never itself a source of facts):
{chat_history_text}

QUESTION: {question}

CROSS-PAPER INSIGHTS:
{insights_text}

EVIDENCE (Grouped by Document):
{evidence_text}

INSTRUCTIONS:
1. Write a clear, well-structured, natural-language answer. Synthesize across
   documents when applicable, and adapt the structure to the question instead
   of forcing a fixed template.
2. Base every factual statement strictly on the evidence above. Do NOT
   hallucinate, do NOT invent facts or numbers, and do NOT fill gaps using
   outside/general knowledge.
3. If the evidence is insufficient to answer the question, say exactly:
   "{insufficient_evidence_message}" — do not pad this out with a guess.
4. Do NOT include chunk IDs, document IDs, or bracketed technical citations
   (e.g. "[DocID, ChunkID]") anywhere in the answer text — write in plain
   prose. Referring to "the first paper" or a paper's likely title is fine;
   internal IDs are not.
5. Honestly self-assess this same answer — do not be generous with yourself:
   - "grounded": true only if every statement traces directly to the
     evidence above, with nothing filled in from outside knowledge.
   - "evidence_sufficient": true only if the evidence above actually
     contains enough information to answer the QUESTION reliably.
6. Keep the answer concise and useful — do not pad it, and do not include
   any chain-of-thought or explanation of your own reasoning process.
{format_instructions}

IMPORTANT: Output ONLY a compact JSON object with exactly these fields, no others:
{{
  "answer": "your natural-language answer (no technical IDs, no chain-of-thought)",
  "grounded": true or false,
  "evidence_sufficient": true or false,
  "document_ids": ["the document_id values above that this answer actually drew on"]
}}{structured_schema_suffix}

JSON:
"""

# Appended to the SAME prompt (same evidence, same ONE call — see
# Feature 6) only when the caller explicitly asked for structured research
# extraction (mode="structured"). Adds one more JSON field rather than a
# second request.
STRUCTURED_SCHEMA_SUFFIX = """

The caller has ALSO explicitly requested structured research extraction.
In ADDITION to the fields above, include a "structured" object built ONLY
from the evidence — use null (for strings) or [] (for lists) for anything
the evidence doesn't cover; never invent a value to fill a field. ALSO
include a "claims" array: break your own "answer" into its individual
factual claims, and for each one report how well the evidence above
actually supports it — "DIRECTLY_STATED" (the evidence says this near-
verbatim), "STRONGLY_SUPPORTED" (not stated outright but clearly follows
from the evidence), or "NOT_FOUND" (you included it but the evidence
above doesn't actually back it — this should be rare, since you should
avoid making such claims in the first place). List the document_id(s)
each claim traces to.
{
  "structured": {
    "architecture": "... or null",
    "datasets": ["..."],
    "metrics": ["..."],
    "contributions": ["..."],
    "methodology": "... or null",
    "key_calculations": ["specific formulas/numbers mentioned, e.g. '28.4 BLEU on WMT14 En-De'"],
    "limitations": ["..."],
    "final_summary": "... or null"
  },
  "claims": [
    {"claim": "...", "support": "DIRECTLY_STATED", "document_ids": ["..."]}
  ]
}"""

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
            "draft_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "citations": [],
            "grounded": False,
            "evidence_sufficient": False,
            "structured_data": None,
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
                    "text": child["text"],
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

    # Recent chat history, formatted directly into this ONE call so follow-up
    # questions ("What about their training data?") resolve here instead of
    # needing a separate query-rewriting LLM call.
    chat_history = state.get("chat_history") or []
    chat_history_text = "None"
    if chat_history:
        history_lines = []
        for msg in chat_history[-4:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_lines.append(f"{role.capitalize()}: {content}")
        chat_history_text = "\n".join(history_lines)

    # Structured research extraction (Feature 6) rides the SAME single call —
    # only requested when the caller explicitly opts in, so a normal
    # conversational question never pays extra output tokens for fields it
    # didn't ask for.
    structured_mode = bool(state.get("structured_mode", False))

    evidence_text = "\n".join(evidence_parts)
    prompt = SYNTHESIS_PROMPT.format(
        question=question, insights_text=insights_text, evidence_text=evidence_text,
        format_instructions=format_instructions, insufficient_evidence_message=INSUFFICIENT_EVIDENCE_MESSAGE,
        chat_history_text=chat_history_text,
        structured_schema_suffix=STRUCTURED_SCHEMA_SUFFIX if structured_mode else "",
    )

    draft_answer = ""
    grounded = False
    evidence_sufficient = False
    self_reported_document_ids: list[str] = []
    structured_data = None
    claim_evidence_trace: list[dict] = []
    synthesis_succeeded = False

    if GROQ_API_KEY:
        try:
            raw = _call_groq_raw(prompt)  # <-- the ONE LLM call for a normal query
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not match:
                raise ValueError("Model response did not contain a JSON object")
            # Pydantic validation, not ad-hoc dict.get() calls: a malformed
            # or incomplete response is treated as a synthesis failure —
            # clean fallback message below, NEVER a reason to call the LLM
            # again. with_model_fallback() already used its one allowed
            # fallback attempt inside _call_groq_raw before this point.
            schema = StructuredAnswerJSON if structured_mode else AnswerJSON
            parsed = schema.model_validate_json(match.group())
            draft_answer = parsed.answer
            grounded = parsed.grounded
            evidence_sufficient = parsed.evidence_sufficient
            self_reported_document_ids = parsed.document_ids
            if structured_mode:
                structured_data = parsed.structured.model_dump()
                claim_evidence_trace = [c.model_dump() for c in parsed.claims]
            if draft_answer:
                synthesis_succeeded = True
        except ValidationError:
            draft_answer = GENERATION_FAILED_MESSAGE
        except Exception as e:
            # with_model_fallback() already tried GROQ_FALLBACK_MODEL once
            # internally before raising — this is a genuine "both models
            # failed" (or the failure wasn't fallback-worthy, e.g. auth).
            draft_answer = RATE_LIMIT_MESSAGE if is_rate_limit_error(e) else GENERATION_FAILED_MESSAGE

    if not draft_answer:
        # No API key configured, or the model responded but not with usable/
        # valid JSON — either way, no answer was actually generated from the
        # evidence, so this is a generation failure, not "no evidence found".
        draft_answer = GENERATION_FAILED_MESSAGE
    elif synthesis_succeeded and not evidence_sufficient:
        # The model itself decided the evidence doesn't answer the question —
        # no separate verifier call needed to reach this conclusion. Keep the
        # user-facing wording consistent regardless of how the model phrased
        # its own refusal.
        draft_answer = INSUFFICIENT_EVIDENCE_MESSAGE

    # Only attach citations when the LLM actually produced an answer from
    # this evidence. If synthesis never ran (rate limit, missing key, bad/
    # invalid JSON), the retrieved chunks were never turned into anything —
    # showing them as "sources" under an apology message would misleadingly
    # imply a real answer was generated. (Citations DO stay attached when
    # the model ran but judged evidence insufficient — it genuinely
    # considered them.) Citations remain the deterministic, evidence-derived
    # list built above — never the model's own self-reported document_ids,
    # which are kept only as an auxiliary/observability signal.
    return {
        "draft_answer": draft_answer,
        "claims": [],
        "grounded": grounded,
        "evidence_sufficient": evidence_sufficient,
        "self_reported_document_ids": self_reported_document_ids,
        # None unless structured_mode was requested AND synthesis succeeded —
        # never partially-populated from a failed/invalid response.
        "structured_data": structured_data if synthesis_succeeded else None,
        # Internal claim-level evidence trace (DIRECTLY_STATED /
        # STRONGLY_SUPPORTED / NOT_FOUND) — rides along on the SAME call,
        # only populated in structured_mode, never surfaced in the normal
        # chat UI. A distinct key from "claims" (the Deep-Research verifier's
        # separate claim graph) so the two never collide.
        "claim_evidence_trace": claim_evidence_trace if synthesis_succeeded else [],
        "citations": citations_list if synthesis_succeeded else [],
        "status": "synthesized" if synthesis_succeeded else "synthesis_failed"
    }
