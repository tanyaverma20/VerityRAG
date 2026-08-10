"""
groundedness_eval.py — OFFLINE, opt-in answer-quality evaluator.

NEVER wired into the normal /query request path — normal Q&A remains
exactly ONE LLM call on success, completely unchanged by this module. This
file is invoked only by a deliberate evaluation run (see evaluate_dataset()
/ the __main__ block), and its one LLM call per question reuses the SAME
structured-mode synthesis path (graph/synthesizer.py) that already exists
for opt-in structured extraction (mode="structured") — this is the SAME
call structured mode already makes when a caller opts in, not a new
second-call architecture bolted onto production.

Metrics — computed deterministically from the synthesis call's own
self-reported claim-level evidence trace (schemas.ClaimEvidence:
DIRECTLY_STATED / STRONGLY_SUPPORTED / NOT_FOUND), added earlier as an
internal structured-mode enrichment:

  - groundedness_score:     fraction of claims DIRECTLY_STATED or STRONGLY_SUPPORTED
  - unsupported_claim_rate: fraction of claims NOT_FOUND
  - evidence_coverage:      fraction of the retrieved/contributing documents
                             actually cited by at least one claim
  - answer_relevance:       deterministic keyword-overlap against
                             expected_answer_contains (eval_metrics.py —
                             no LLM call needed for this one)

Every score is either a real measured value or NOT_MEASURED — this module
never fabricates a number when there's nothing to compute it from (e.g. a
question with no expected_answer_contains, or a synthesis call that failed).

Distinguish this from RUNTIME/online metrics (LLM calls/query, latency,
cache hit rate — see observability.py, GET /eval/dashboard) — this module
is exclusively OFFLINE evaluation.
"""
from __future__ import annotations

import statistics
from typing import Any

NOT_MEASURED = "NOT_MEASURED"

_SUPPORTED_LABELS = ("DIRECTLY_STATED", "STRONGLY_SUPPORTED")


def score_claims(claims: list[dict]) -> dict[str, Any]:
    """Pure aggregation over an already-produced claim_evidence_trace list
    — no LLM call, no network I/O. Returns NOT_MEASURED fields (never 0 or
    a guess) when there are no claims to score."""
    if not claims:
        return {
            "groundedness_score": NOT_MEASURED,
            "unsupported_claim_rate": NOT_MEASURED,
            "directly_stated_rate": NOT_MEASURED,
            "strongly_supported_rate": NOT_MEASURED,
            "total_claims": 0,
        }
    total = len(claims)
    directly = sum(1 for c in claims if c.get("support") == "DIRECTLY_STATED")
    strongly = sum(1 for c in claims if c.get("support") == "STRONGLY_SUPPORTED")
    unsupported = sum(1 for c in claims if c.get("support") == "NOT_FOUND")
    return {
        "groundedness_score": round((directly + strongly) / total, 4),
        "unsupported_claim_rate": round(unsupported / total, 4),
        "directly_stated_rate": round(directly / total, 4),
        "strongly_supported_rate": round(strongly / total, 4),
        "total_claims": total,
    }


def score_evidence_coverage(claims: list[dict], contributing_document_ids: list[str]) -> float | str:
    """What fraction of the documents that actually contributed retrieved
    evidence got cited by at least one claim — a low value flags evidence
    that was retrieved but never actually used in the answer."""
    if not contributing_document_ids:
        return NOT_MEASURED
    cited_docs: set[str] = set()
    for c in claims:
        cited_docs.update(c.get("document_ids") or [])
    covered = cited_docs.intersection(contributing_document_ids)
    return round(len(covered) / len(contributing_document_ids), 4)


def score_answer_relevance(answer: str, expected_answer_contains: list[str] | None) -> dict:
    """Deterministic keyword-overlap proxy for relevance — reuses the
    existing eval_metrics.concept_coverage rather than a new metric.
    Explicitly NOT a semantic-correctness measure (see that function's own
    docstring) — a coarse, honest, zero-LLM-cost proxy only."""
    from eval_metrics import concept_coverage
    return concept_coverage(answer, expected_answer_contains or [])


def evaluate_one(question: str, document_ids: list[str], expected_answer_contains: list[str] | None = None) -> dict[str, Any]:
    """
    Makes exactly ONE structured-mode synthesis call — offline/eval-only,
    see module docstring — and scores it. Returns NOT_MEASURED fields
    (never a fabricated score) if retrieval finds nothing or synthesis
    fails; never raises up to the caller for a normal generation failure.
    """
    from graph.organizer import organize_evidence
    from graph.synthesizer import synthesize_answer
    from retrieval import retrieve, get_contributing_documents

    result_base = {"question": question, "document_ids": document_ids}

    chunks = retrieve(question, strategy="hybrid", document_ids=document_ids, top_k=5)
    if not chunks:
        return {**result_base, "error": "no evidence retrieved for this question/document scope",
                "groundedness_score": NOT_MEASURED, "unsupported_claim_rate": NOT_MEASURED,
                "evidence_coverage": NOT_MEASURED, "answer_relevance": {"coverage_ratio": NOT_MEASURED}}

    state: dict[str, Any] = {"original_query": question, "structured_mode": True}
    state.update(organize_evidence({"retrieval_results": chunks}))

    # A malformed LLM response (unparseable JSON, a network error, a
    # provider-side failure) must degrade this ONE question to NOT_MEASURED
    # rather than raising — evaluate_dataset() runs this over a whole batch,
    # and one bad response must never lose every other question's real
    # result. synthesize_answer() already has its own internal validation/
    # fallback handling (same function normal structured-mode Q&A uses),
    # but this is a deliberately defensive outer boundary for this
    # eval-only caller: never let an evaluation run crash mid-batch.
    try:
        synth = synthesize_answer(state)
    except Exception as e:
        return {**result_base, "error": f"synthesis call failed: {type(e).__name__}: {e}",
                "groundedness_score": NOT_MEASURED, "unsupported_claim_rate": NOT_MEASURED,
                "evidence_coverage": NOT_MEASURED, "answer_relevance": {"coverage_ratio": NOT_MEASURED}}

    contributing = [d["document_id"] for d in get_contributing_documents(chunks)]
    # claim_evidence_trace may be malformed (not a list, or missing) if
    # synthesis returned something unexpected — treat as "no claims" rather
    # than crash; score_claims()/score_evidence_coverage() already return
    # NOT_MEASURED for an empty list, so this degrades honestly.
    claims = synth.get("claim_evidence_trace")
    claims = claims if isinstance(claims, list) else []
    claim_scores = score_claims(claims)
    coverage = score_evidence_coverage(claims, contributing)
    relevance = score_answer_relevance(synth.get("draft_answer", "") or "", expected_answer_contains)

    return {
        **result_base,
        "answer": synth.get("draft_answer", ""),
        "grounded_self_reported": synth.get("grounded"),
        "evidence_sufficient_self_reported": synth.get("evidence_sufficient"),
        **claim_scores,
        "evidence_coverage": coverage,
        "answer_relevance": relevance,
        "contributing_documents": contributing,
        "synthesis_status": synth.get("status"),
    }


def evaluate_dataset(items: list[dict]) -> dict[str, Any]:
    """items: [{"question": ..., "document_ids": [...], "expected_answer_contains": [...]}]

    Real LLM calls: exactly len(items) — one structured-mode synthesis call
    per question, never more. Run deliberately (see evaluation/README.md),
    not as part of any automated test suite or the normal request path.
    """
    # Per-item try/except: a single item's unexpected failure (e.g. a
    # malformed retrieval call from a bad document_ids entry) degrades to
    # one NOT_MEASURED result rather than aborting the whole batch and
    # losing every other question's real, already-paid-for LLM call.
    results = []
    for it in items:
        try:
            results.append(evaluate_one(it["question"], it["document_ids"], it.get("expected_answer_contains")))
        except Exception as e:
            results.append({
                "question": it.get("question"), "document_ids": it.get("document_ids"),
                "error": f"unexpected evaluation failure: {type(e).__name__}: {e}",
                "groundedness_score": NOT_MEASURED, "unsupported_claim_rate": NOT_MEASURED,
                "evidence_coverage": NOT_MEASURED, "answer_relevance": {"coverage_ratio": NOT_MEASURED},
            })

    def _avg(key: str) -> float | str:
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return round(statistics.mean(vals), 4) if vals else NOT_MEASURED

    return {
        "dataset_size": len(items),
        "llm_calls_made": len(items),
        "avg_groundedness_score": _avg("groundedness_score"),
        "avg_unsupported_claim_rate": _avg("unsupported_claim_rate"),
        "avg_evidence_coverage": _avg("evidence_coverage"),
        "methodology": (
            "ONE structured-mode synthesis call per question (offline/eval-only — "
            "see this module's docstring; never part of the normal /query path). "
            "groundedness_score / unsupported_claim_rate are computed from the "
            "model's own self-reported claim_evidence_trace for that SAME call, "
            "not a separate verifier call. answer_relevance is a deterministic "
            "keyword-overlap proxy (eval_metrics.concept_coverage), zero LLM cost."
        ),
        "not_measured": [
            "semantic answer correctness beyond keyword overlap (would require either "
            "a reference answer + an LLM judge, or human annotation — neither is run here)",
        ],
        "per_question_results": results,
    }


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    print(
        "groundedness_eval.py is an OFFLINE evaluator that makes real Groq calls "
        "(one structured-mode synthesis call per question). It is not run "
        "automatically — pass a JSON file of {question, document_ids, "
        "expected_answer_contains} items to evaluate, e.g.:\n"
        "  python groundedness_eval.py my_eval_items.json\n"
    )
    if len(sys.argv) > 1:
        items = json.loads(Path(sys.argv[1]).read_text())
        report = evaluate_dataset(items)
        print(json.dumps(report, indent=2))
