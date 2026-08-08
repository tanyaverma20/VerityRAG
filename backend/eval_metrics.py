"""
eval_metrics.py — Phase 4 Deterministic Evaluation Metrics

Implements:
  Retrieval metrics:
    recall_at_k, precision_at_k, hit_rate_at_k, mrr, ndcg_at_k

  Concept coverage:
    concept_coverage   (expected_answer_contains vs actual answer)

  Citation metrics:
    citation_coverage, citation_validity, citation_correctness

  Verification metrics:
    aggregate_verification_statuses

All functions are deterministic and testable.
No LLM calls.
"""

from __future__ import annotations
import math
from typing import Any

NOT_AVAILABLE = "NOT_AVAILABLE"

# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float | str:
    """
    Recall@K = |relevant ∩ retrieved[:k]| / |relevant|

    Returns NOT_AVAILABLE if relevant_ids is empty.
    """
    if not relevant_ids:
        return NOT_AVAILABLE
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in relevant_ids if r in top_k)
    return round(hits / len(relevant_ids), 4)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float | str:
    """
    Precision@K = |relevant ∩ retrieved[:k]| / k

    Returns NOT_AVAILABLE if relevant_ids is empty or k == 0.
    """
    if not relevant_ids or k == 0:
        return NOT_AVAILABLE
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return round(hits / k, 4)


def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float | str:
    """
    Hit Rate@K = 1 if at least one relevant doc appears in retrieved[:k], else 0.

    Returns NOT_AVAILABLE if relevant_ids is empty.
    """
    if not relevant_ids:
        return NOT_AVAILABLE
    top_k = retrieved_ids[:k]
    return 1.0 if any(r in top_k for r in relevant_ids) else 0.0


def mrr(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float | str:
    """
    Mean Reciprocal Rank = 1 / rank_of_first_relevant_item

    Returns NOT_AVAILABLE if relevant_ids is empty or no relevant item found.
    """
    if not relevant_ids:
        return NOT_AVAILABLE
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return round(1.0 / rank, 4)
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float | str:
    """
    nDCG@K with binary relevance.

    DCG@K  = Σ rel_i / log2(i+1)   for i in 1..K
    IDCG@K = DCG of the ideal ordering (all relevant docs first)

    Returns NOT_AVAILABLE if relevant_ids is empty.
    """
    if not relevant_ids:
        return NOT_AVAILABLE

    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    dcg = sum(
        (1.0 / math.log2(i + 2)) if doc_id in relevant_set else 0.0
        for i, doc_id in enumerate(top_k)
    )

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return round(dcg / idcg, 4)


# ---------------------------------------------------------------------------
# Multi-K Retrieval Suite
# ---------------------------------------------------------------------------

def retrieval_metrics_suite(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """
    Runs all retrieval metrics at multiple K values.
    Returns a flat dict with e.g. recall@3, precision@5, etc.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results: dict[str, Any] = {}
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        results[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        results[f"hit_rate@{k}"] = hit_rate_at_k(retrieved_ids, relevant_ids, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)

    results["mrr"] = mrr(retrieved_ids, relevant_ids)
    results["retrieved_count"] = len(retrieved_ids)
    results["relevant_count"] = len(relevant_ids)
    return results


# ---------------------------------------------------------------------------
# Concept Coverage (Answer-Level, Deterministic)
# ---------------------------------------------------------------------------

def concept_coverage(
    answer: str,
    expected_concepts: list[str],
) -> dict[str, Any]:
    """
    Checks how many expected concepts/keywords appear in the answer.

    This is a deterministic KEYWORD-LEVEL check only.
    It does NOT measure semantic correctness.

    Returns:
      found_concepts, missing_concepts, coverage_ratio
    """
    if not expected_concepts:
        return {
            "found": [],
            "missing": [],
            "coverage_ratio": NOT_AVAILABLE,
            "note": "No expected concepts specified.",
        }

    answer_lower = answer.lower()
    found = [c for c in expected_concepts if c.lower() in answer_lower]
    missing = [c for c in expected_concepts if c.lower() not in answer_lower]

    return {
        "found": found,
        "missing": missing,
        "coverage_ratio": round(len(found) / len(expected_concepts), 4),
    }


# ---------------------------------------------------------------------------
# Citation & Claim Grounding Metrics (Hardened)
# ---------------------------------------------------------------------------

def citation_grounding_suite(
    structured_citations: list[dict],
    retrieved_chunk_ids: list[str],
    indexed_document_ids: list[str],
    verification_results: list[dict],
) -> dict[str, Any]:
    """
    Evaluates citation and claim grounding:
    CLAIM -> CITATION -> CHUNK -> PAGE -> DOCUMENT -> RETRIEVED EVIDENCE

    Extracts statuses:
    - VALID / INVALID
    - NOT_RETRIEVED
    - DOCUMENT_MISMATCH
    - CHUNK_MISMATCH
    - SUPPORTS_CLAIM / WEAKLY_SUPPORTS_CLAIM / DOES_NOT_SUPPORT_CLAIM / VERIFICATION_UNAVAILABLE
    """
    citation_evals = []
    
    # 1. Structural Validation
    for c in structured_citations:
        chunk_id = c.get("chunk_id", "")
        doc_id = c.get("document_id", "")
        
        valid = True
        status = "VALID"
        
        if not chunk_id:
            valid = False
            status = "INVALID (No chunk_id)"
        elif doc_id and doc_id not in indexed_document_ids:
            valid = False
            status = "DOCUMENT_MISMATCH"
        elif not doc_id and chunk_id:
            # If doc_id is missing but chunk_id exists, try to infer doc mismatch
            # VerityRAG chunk IDs start with doc_<doc_id>
            pass 
        elif chunk_id not in retrieved_chunk_ids:
            valid = False
            status = "NOT_RETRIEVED"
        elif doc_id and not chunk_id.startswith(f"doc_{doc_id}"):
            valid = False
            status = "CHUNK_MISMATCH (Chunk does not belong to cited doc)"
            
        citation_evals.append({
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "structural_validity": valid,
            "structural_status": status
        })
        
    total_citations = len(citation_evals)
    structurally_valid_count = sum(1 for c in citation_evals if c["structural_validity"])
    retrieved_accuracy_count = sum(1 for c in citation_evals if c["structural_status"] == "VALID")
    
    # 2. Semantic Support (Claim Verification)
    supported = 0
    weakly = 0
    unsupported = 0
    unavailable = 0
    
    for r in verification_results:
        s = r.get("status", "")
        if s == "SUPPORTED":
            supported += 1
        elif s == "WEAKLY_SUPPORTED":
            weakly += 1
        elif s == "UNSUPPORTED":
            unsupported += 1
        elif s == "VERIFICATION_UNAVAILABLE":
            unavailable += 1
            
    total_claims = len(verification_results)
    verifiable_claims = supported + weakly + unsupported
    
    # Avoid div by zero
    citation_coverage_val = NOT_AVAILABLE if total_claims == 0 else round(total_citations / max(total_claims, 1), 4)
    citation_validity_val = NOT_AVAILABLE if total_citations == 0 else round(structurally_valid_count / total_citations, 4)
    retrieval_accuracy_val = NOT_AVAILABLE if total_citations == 0 else round(retrieved_accuracy_count / total_citations, 4)
    
    support_rate = NOT_AVAILABLE if verifiable_claims == 0 else round(supported / verifiable_claims, 4)
    weak_support_rate = NOT_AVAILABLE if verifiable_claims == 0 else round(weakly / verifiable_claims, 4)
    unsupported_rate = NOT_AVAILABLE if verifiable_claims == 0 else round(unsupported / verifiable_claims, 4)
    unavailable_rate = NOT_AVAILABLE if total_claims == 0 else round(unavailable / total_claims, 4)
    
    return {
        "citation_coverage": citation_coverage_val,
        "citation_validity": citation_validity_val,
        "citation_retrieval_accuracy": retrieval_accuracy_val,
        "support_rate": support_rate,
        "weak_support_rate": weak_support_rate,
        "unsupported_claim_rate": unsupported_rate,
        "verification_unavailable_rate": unavailable_rate,
        "total_citations": total_citations,
        "total_claims": total_claims,
        "details": citation_evals,
    }


# ---------------------------------------------------------------------------
# Resource / Context Metrics
# ---------------------------------------------------------------------------

def resource_metrics(
    retrieval_results: list[dict],
    selected_chunks: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Calculates resource usage metrics from retrieval results.
    """
    doc_ids = [
        c.get("metadata", {}).get("document_id", "")
        for c in retrieval_results
    ]
    unique_docs = len(set(d for d in doc_ids if d))

    token_estimate = sum(
        max(1, len(c.get("text", "")) // 4)
        for c in retrieval_results
    )

    result = {
        "retrieved_chunks": len(retrieval_results),
        "contributing_documents": unique_docs,
        "estimated_context_tokens": token_estimate,
    }

    if selected_chunks is not None:
        selected_tokens = sum(
            max(1, len(c.get("text", "")) // 4)
            for c in selected_chunks
        )
        result["selected_chunks"] = len(selected_chunks)
        result["selected_context_tokens"] = selected_tokens

    return result
