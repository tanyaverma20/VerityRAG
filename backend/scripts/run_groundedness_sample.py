"""
scripts/run_groundedness_sample.py — deliberately invokes groundedness_eval.py
against a small, real sample of eval_set.json questions (using document_ids
confirmed still present in the real production ChromaDB) and saves the
result to data/groundedness_eval_results.json, which GET /eval/dashboard
reads if present.

This is a MANUAL, deliberately-invoked script (matches groundedness_eval.py's
own "opt-in, run deliberately" design) — makes real Groq calls, one per
question, with a short delay between calls to be considerate of rate limits.
Never run automatically as part of tests or CI.

Usage:
    cd backend
    python scripts/run_groundedness_sample.py [items_json_path] [--delay SECONDS]
"""
import json
import os
import sys
import time

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

from groundedness_eval import evaluate_one, NOT_MEASURED

OUTPUT_PATH = os.path.join(BACKEND_DIR, "..", "data", "groundedness_eval_results.json")


def main():
    items_path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else \
        os.path.join(BACKEND_DIR, "..", "evaluation", "groundedness_sample_items.json")
    delay = 3.0
    if "--delay" in sys.argv:
        delay = float(sys.argv[sys.argv.index("--delay") + 1])

    items = json.loads(open(items_path, encoding="utf-8").read())
    print(f"Running groundedness_eval on {len(items)} real questions (delay={delay}s between calls)...")

    results = []
    for i, it in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {it['question'][:70]}...")
        r = evaluate_one(it["question"], it["document_ids"], it.get("expected_answer_contains"))
        results.append(r)
        status = r.get("synthesis_status", "?")
        score = r.get("groundedness_score", NOT_MEASURED)
        print(f"    synthesis_status={status}  groundedness_score={score}")
        if i < len(items):
            time.sleep(delay)

    def _avg(key):
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else NOT_MEASURED

    n_succeeded = sum(1 for r in results if r.get("synthesis_status") not in ("synthesis_failed", None) and "error" not in r)
    n_rate_limited = sum(1 for r in results if "rate-limited" in (r.get("answer") or "").lower())

    report = {
        "dataset_size": len(items),
        "llm_calls_made": len(items),
        "questions_succeeded": n_succeeded,
        "questions_rate_limited_or_failed": len(items) - n_succeeded,
        "avg_groundedness_score": _avg("groundedness_score"),
        "avg_unsupported_claim_rate": _avg("unsupported_claim_rate"),
        "avg_evidence_coverage": _avg("evidence_coverage"),
        "methodology": (
            "ONE structured-mode synthesis call per question (offline/eval-only, "
            "backend/groundedness_eval.py), run deliberately against a small real "
            "sample of data/eval_set.json questions with document_ids confirmed "
            "present in production ChromaDB at run time. Averages are computed "
            "only over questions that actually produced a real (non-NOT_MEASURED) "
            "score — a rate-limited or failed call contributes to "
            "questions_rate_limited_or_failed, not a fabricated 0."
        ),
        "not_measured": [
            "semantic answer correctness beyond keyword overlap (would require either "
            "a reference answer + an LLM judge, or human annotation — neither is run here)",
        ],
        "per_question_results": results,
    }

    out_path = os.path.abspath(OUTPUT_PATH)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWritten to {out_path}")
    print(f"Succeeded: {n_succeeded}/{len(items)}  Rate-limited/failed: {len(items) - n_succeeded}/{len(items)}")
    print(f"avg_groundedness_score={report['avg_groundedness_score']}  "
          f"avg_unsupported_claim_rate={report['avg_unsupported_claim_rate']}  "
          f"avg_evidence_coverage={report['avg_evidence_coverage']}")


if __name__ == "__main__":
    main()
