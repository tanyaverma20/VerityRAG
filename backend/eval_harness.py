"""
Evaluation harness — run this after ingesting your papers and filling in
data/eval_set.json with real questions.

It compares three pipelines on the same question set:
  1. naive        -> LLM answers with no retrieval at all (pure hallucination risk)
  2. dense_only    -> LLM answers using only Chroma vector search
  3. hybrid_verify -> your full pipeline: BM25 + dense + rerank + claim verification

For each, it records: whether the expected keywords showed up in the answer
(a rough proxy for correctness), latency, and — for hybrid_verify — the
groundedness score from the verifier.

This is the artifact that turns "I built a RAG system" into "I built a RAG
system and measured a 23-point precision improvement over baseline." Run it,
save the output table, and put the numbers in your resume/README.
"""
import json
import time
from pathlib import Path

import pandas as pd

from retrieval import dense_search, hybrid_retrieve, build_bm25_index
from verify import verify_answer
from llm import call_llm

EVAL_SET_PATH = Path(__file__).parent.parent / "data" / "eval_set.json"

NAIVE_PROMPT = "Answer this question as best you can: {question}"
GROUNDED_PROMPT = """Answer the question using ONLY these source passages.
QUESTION: {question}
SOURCES:
{sources}
"""


def _contains_expected(answer: str, expected_keywords: list[str]) -> bool:
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords if kw)


def run_naive(question: str) -> dict:
    result = call_llm(NAIVE_PROMPT.format(question=question))
    return {"answer": result["text"], "latency": result["latency_seconds"]}


def run_dense_only(question: str) -> dict:
    chunks = dense_search(question, top_k=5)
    sources_text = "\n---\n".join(c["text"] for c in chunks)
    result = call_llm(GROUNDED_PROMPT.format(question=question, sources=sources_text))
    return {"answer": result["text"], "latency": result["latency_seconds"], "num_sources": len(chunks)}


def run_hybrid_verify(question: str) -> dict:
    chunks = hybrid_retrieve(question, top_k=5)
    sources_text = "\n---\n".join(c["text"] for c in chunks)
    result = call_llm(GROUNDED_PROMPT.format(question=question, sources=sources_text))
    verification = verify_answer(result["text"], chunks)
    return {
        "answer": result["text"],
        "latency": result["latency_seconds"],
        "num_sources": len(chunks),
        "groundedness_score": verification["groundedness_score"],
    }


def main():
    build_bm25_index()
    eval_set = json.loads(EVAL_SET_PATH.read_text())

    rows = []
    for item in eval_set:
        question = item["question"]
        expected = item.get("expected_answer_contains", [])
        print(f"Evaluating: {question}")

        for pipeline_name, fn in [
            ("naive", run_naive),
            ("dense_only", run_dense_only),
            ("hybrid_verify", run_hybrid_verify),
        ]:
            start = time.time()
            out = fn(question)
            elapsed = time.time() - start
            rows.append({
                "question": question,
                "pipeline": pipeline_name,
                "keyword_match": _contains_expected(out["answer"], expected),
                "latency_seconds": round(elapsed, 2),
                "groundedness_score": out.get("groundedness_score"),
            })

    df = pd.DataFrame(rows)
    summary = df.groupby("pipeline").agg(
        accuracy_proxy=("keyword_match", "mean"),
        avg_latency=("latency_seconds", "mean"),
        avg_groundedness=("groundedness_score", "mean"),
    )
    print("\n=== Summary ===")
    print(summary)

    out_path = Path(__file__).parent.parent / "data" / "eval_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
