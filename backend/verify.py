"""
Claim verification layer.

Instead of trusting the generator's own "I'm grounded, I promise," this
module explicitly splits the answer into individual claims and checks each
one against the retrieved chunks, one at a time. Each claim gets a label
(supported / contradicted / not_found) and a confidence score.

This is deliberately a separate, auditable step rather than folded into
one giant generation prompt — that's what makes it defensible in an
interview: you can point to the exact prompt that does the checking.
"""
import json
from llm import call_llm

SPLIT_PROMPT = """Break the following answer into a list of individual factual claims.
Each claim should be a single, checkable statement (one fact per claim).
Return ONLY a JSON list of strings, nothing else.

Answer:
{answer}
"""

VERIFY_PROMPT = """You are a strict fact-checker. Given a CLAIM and a set of SOURCE PASSAGES,
decide whether the claim is supported by the passages.

Respond ONLY with JSON in this exact format:
{{"label": "supported" | "contradicted" | "not_found", "confidence": 0.0-1.0, "evidence": "short quote or paraphrase from the source, or empty string"}}

CLAIM: {claim}

SOURCE PASSAGES:
{sources}
"""


def split_into_claims(answer: str) -> list[str]:
    result = call_llm(SPLIT_PROMPT.format(answer=answer), temperature=0.0)
    try:
        claims = json.loads(result["text"])
        if isinstance(claims, list):
            return [str(c) for c in claims]
    except json.JSONDecodeError:
        pass
    # Fallback: treat each sentence as a claim if the model didn't return clean JSON
    return [s.strip() for s in answer.split(".") if len(s.strip()) > 15]


def verify_claim(claim: str, source_chunks: list[dict]) -> dict:
    sources_text = "\n---\n".join(c["text"] for c in source_chunks)
    result = call_llm(VERIFY_PROMPT.format(claim=claim, sources=sources_text), temperature=0.0)
    try:
        parsed = json.loads(result["text"])
    except json.JSONDecodeError:
        parsed = {"label": "not_found", "confidence": 0.0, "evidence": ""}
    parsed["claim"] = claim
    return parsed


def verify_answer(answer: str, source_chunks: list[dict]) -> dict:
    """Full verification pass: split -> verify each claim -> summarize."""
    claims = split_into_claims(answer)
    verified = [verify_claim(c, source_chunks) for c in claims]

    supported = sum(1 for v in verified if v["label"] == "supported")
    total = len(verified) or 1
    groundedness_score = round(supported / total, 3)

    return {
        "claims": verified,
        "groundedness_score": groundedness_score,
        "num_claims": total,
        "num_supported": supported,
    }
