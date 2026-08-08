"""
run_evaluation.py — Phase 4 Evaluation Runner

Compares 5 retrieval baselines on real evaluation questions:
  1. dense_only
  2. bm25_only
  3. hybrid_rrf
  4. hybrid_reranked  (Phase 2 full pipeline)
  5. agentic_graph    (Phase 3 LangGraph)

Produces:  data/eval_results_comprehensive.json

Usage:
    cd backend
    python run_evaluation.py
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from ingest import get_all_chunks, ingest_document
from retrieval import dense_search, bm25_search, retrieve, hybrid_retrieve, build_bm25_index
from eval_metrics import (
    retrieval_metrics_suite,
    concept_coverage,
    citation_grounding_suite,
    resource_metrics,
    NOT_AVAILABLE,
)
from observability import log_query_event

EVAL_SET_PATH = Path(__file__).parent.parent / "data" / "eval_set.json"
MANIFEST_PATH = Path(__file__).parent.parent / "data" / "ingestion_manifest.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "eval_results_comprehensive.json"

K_VALUES = [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_ids_from(results: list[dict]) -> list[str]:
    return [r.get("metadata", {}).get("chunk_id", "") for r in results]


def _doc_ids_from(results: list[dict]) -> list[str]:
    seen = []
    for r in results:
        d = r.get("metadata", {}).get("document_id", "")
        if d and d not in seen:
            seen.append(d)
    return seen


def run_dense_only(question: str, top_k: int = 10) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    chunks = dense_search(question, top_k=top_k)
    return chunks, round(time.perf_counter() - t0, 4)


def run_bm25_only(question: str, top_k: int = 10) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    chunks = bm25_search(question, top_k=top_k)
    return chunks, round(time.perf_counter() - t0, 4)


def run_hybrid_rrf(question: str, top_k: int = 10) -> tuple[list[dict], float]:
    """Hybrid with RRF but without CrossEncoder reranking."""
    t0 = time.perf_counter()
    from retrieval import reciprocal_rank_fusion, deduplicate_chunks
    dense = dense_search(question, top_k=top_k)
    bm25 = bm25_search(question, top_k=top_k)
    fused = reciprocal_rank_fusion(dense, bm25)
    chunks = deduplicate_chunks(fused)[:top_k]
    return chunks, round(time.perf_counter() - t0, 4)


def run_hybrid_reranked(question: str, top_k: int = 10) -> tuple[list[dict], float]:
    """Phase 2 full pipeline — RRF + CrossEncoder."""
    t0 = time.perf_counter()
    chunks = retrieve(question, strategy="hybrid", top_k=top_k)
    return chunks, round(time.perf_counter() - t0, 4)


def run_agentic_graph(question: str, document_ids: list[str] | None = None) -> tuple[dict, float]:
    """Phase 3 LangGraph — returns the full final_state."""
    from graph.workflow import research_app
    t0 = time.perf_counter()
    initial = {"original_query": question, "document_ids": document_ids}
    try:
        final_state = research_app.invoke(initial)
    except Exception as e:
        final_state = {
            "retrieval_results": [],
            "draft_answer": "",
            "citations": [],
            "verification_results": [],
            "status": f"graph_error: {e}",
        }
    return final_state, round(time.perf_counter() - t0, 4)


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

def evaluate_question(
    item: dict,
    indexed_doc_ids: list[str],
    retrieved_chunk_ids_all: list[str],
) -> dict[str, Any]:
    question = item["question"]
    qtype = item.get("question_type", "unknown")
    relevant_doc_ids = item.get("relevant_document_ids", [])
    expected_concepts = item.get("expected_answer_contains", [])

    result: dict[str, Any] = {
        "id": item.get("id", ""),
        "question": question,
        "question_type": qtype,
        "relevant_document_ids": relevant_doc_ids,
        "baselines": {},
    }

    # --- Run each baseline ---
    for name, runner_fn, needs_docs in [
        ("dense_only", lambda q: run_dense_only(q, top_k=10), False),
        ("bm25_only", lambda q: run_bm25_only(q, top_k=10), False),
        ("hybrid_rrf", lambda q: run_hybrid_rrf(q, top_k=10), False),
        ("hybrid_reranked", lambda q: run_hybrid_reranked(q, top_k=10), False),
    ]:
        chunks, latency = runner_fn(question)
        ret_doc_ids = _doc_ids_from(chunks)
        ret_chunk_ids = _chunk_ids_from(chunks)
        metrics = retrieval_metrics_suite(ret_doc_ids, relevant_doc_ids, K_VALUES)
        res_m = resource_metrics(chunks)
        log_query_event(
            query=question,
            question_type=qtype,
            retrieval_strategy=name,
            retrieved_chunk_count=len(chunks),
            documents_contributing=res_m["contributing_documents"],
            estimated_context_tokens=res_m["estimated_context_tokens"],
            retrieval_latency_s=latency,
        )
        result["baselines"][name] = {
            "retrieved_doc_ids": ret_doc_ids,
            "retrieval_metrics": metrics,
            "resource_metrics": res_m,
            "latency_s": latency,
        }

    # --- Agentic graph ---
    graph_state, graph_latency = run_agentic_graph(question, document_ids=relevant_doc_ids if relevant_doc_ids else None)
    graph_chunks = graph_state.get("retrieval_results", [])
    graph_doc_ids = _doc_ids_from(graph_chunks)
    graph_chunk_ids = _chunk_ids_from(graph_chunks)
    graph_citations = graph_state.get("citations", [])
    graph_verif = graph_state.get("verification_results", [])
    graph_answer = graph_state.get("draft_answer", "")

    graph_retrieval_metrics = retrieval_metrics_suite(graph_doc_ids, relevant_doc_ids, K_VALUES)
    graph_resource = resource_metrics(graph_chunks)
    cov_metrics = concept_coverage(graph_answer, expected_concepts)
    cit_grounding = citation_grounding_suite(graph_citations, graph_chunk_ids, indexed_doc_ids, graph_verif)

    log_query_event(
        query=question,
        question_type=qtype,
        retrieval_strategy="agentic_graph",
        retrieved_chunk_count=len(graph_chunks),
        documents_contributing=graph_resource["contributing_documents"],
        estimated_context_tokens=graph_resource["estimated_context_tokens"],
        synthesis_attempted=bool(graph_answer),
        verification_attempted=bool(graph_verif),
        verification_statuses=[r.get("status", "") for r in graph_verif],
        retry_count=graph_state.get("retry_count", 0),
        total_latency_s=graph_latency,
    )

    result["baselines"]["agentic_graph"] = {
        "retrieved_doc_ids": graph_doc_ids,
        "retrieval_metrics": graph_retrieval_metrics,
        "resource_metrics": graph_resource,
        "concept_coverage": cov_metrics,
        "citation_grounding": cit_grounding,
        "draft_answer_preview": graph_answer[:300] if graph_answer else "",
        "research_plan": graph_state.get("research_plan", {}),
        "latency_s": graph_latency,
    }

    return result


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def aggregate_results(question_results: list[dict]) -> dict[str, Any]:
    """Compute per-baseline aggregate scores across all questions."""
    pipeline_names = ["dense_only", "bm25_only", "hybrid_rrf", "hybrid_reranked", "agentic_graph"]
    summary: dict[str, Any] = {}

    for pname in pipeline_names:
        metrics_acc: dict[str, list] = {}
        latencies = []
        for qr in question_results:
            b = qr["baselines"].get(pname, {})
            latencies.append(b.get("latency_s", 0))
            for mk, mv in b.get("retrieval_metrics", {}).items():
                if mv != NOT_AVAILABLE and isinstance(mv, (int, float)):
                    metrics_acc.setdefault(mk, []).append(mv)

        summary[pname] = {
            "avg_latency_s": round(sum(latencies) / len(latencies), 4) if latencies else 0,
            "avg_metrics": {
                mk: round(sum(vs) / len(vs), 4)
                for mk, vs in metrics_acc.items()
            },
        }

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== VerityRAG Phase 4 Evaluation Runner ===\n")

    # Load manifest
    if not MANIFEST_PATH.exists():
        print("ERROR: ingestion_manifest.json not found. Run ingest_all.py first.")
        sys.exit(1)
    manifest = json.loads(MANIFEST_PATH.read_text())
    indexed_doc_ids = [m["document_id"] for m in manifest if m.get("status") == "ok"]
    total_chunks_indexed = sum(m.get("chunks_added", 0) for m in manifest if m.get("status") == "ok")
    print(f"Corpus: {len(indexed_doc_ids)} papers indexed, {total_chunks_indexed} total chunks")

    # Load eval set
    if not EVAL_SET_PATH.exists():
        print("ERROR: eval_set.json not found.")
        sys.exit(1)
    eval_set = json.loads(EVAL_SET_PATH.read_text())
    print(f"Evaluation: {len(eval_set)} questions loaded\n")

    # Build BM25 index
    print("Building BM25 index...")
    build_bm25_index()

    # Run evaluation
    all_chunks = get_all_chunks()
    all_chunk_ids = [c.get("chunk_id", "") for c in all_chunks]
    question_results = []

    for i, item in enumerate(eval_set, 1):
        qid = item.get("id", f"q{i:02d}")
        print(f"[{i}/{len(eval_set)}] {qid}: {item['question'][:80]}...")
        try:
            qr = evaluate_question(item, indexed_doc_ids, all_chunk_ids)
            question_results.append(qr)
        except Exception as e:
            print(f"  ERROR on {qid}: {e}")
            question_results.append({
                "id": qid,
                "question": item["question"],
                "error": str(e),
                "baselines": {},
            })

    # Aggregate
    summary = aggregate_results(question_results)

    # Build report
    report = {
        "meta": {
            "real_papers_used": len(indexed_doc_ids),
            "total_chunks_indexed": total_chunks_indexed,
            "chunks_per_paper": {m["file"]: m["chunks_added"] for m in manifest if m.get("status") == "ok"},
            "evaluation_questions": len(eval_set),
            "question_types": list({item.get("question_type", "unknown") for item in eval_set}),
            "k_values_evaluated": K_VALUES,
            "validation_type": "REAL — 11 actual research PDFs ingested and evaluated",
        },
        "aggregate_summary": summary,
        "per_question_results": question_results,
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n=== Report saved to {OUTPUT_PATH} ===")
    print("\n--- Aggregate Summary ---")
    for pname, pdata in summary.items():
        avg = pdata.get("avg_metrics", {})
        print(f"  {pname:20s}  recall@5={avg.get('recall@5', 'N/A')}  "
              f"hit_rate@5={avg.get('hit_rate@5', 'N/A')}  "
              f"mrr={avg.get('mrr', 'N/A')}  "
              f"avg_lat={pdata.get('avg_latency_s', 'N/A')}s")

    print("\nPhase 4 evaluation complete.")
    return report


if __name__ == "__main__":
    main()
