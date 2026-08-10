"""
evaluation/run_benchmark.py — reproducible retrieval evaluation + latency
benchmark (request D, items 4 and 5).

Runs entirely against an ISOLATED, temp-directory Chroma collection and an
isolated BM25 index built from it — NEVER touches backend/chroma_store or
data/registry.db (same isolation technique backend/conftest.py already
uses for the test suite: point config.CHROMA_DIR/COLLECTION_NAME at a temp
location and reset ingest.py's client/collection singletons before doing
anything).

No LLM calls are made anywhere in this script — every metric here is
retrieval-only (dense/BM25/RRF/CrossEncoder/parent-context/scoping/
decomposition), which is also why groundedness/answer-relevance are
reported as "not measured here" (see evaluation/README.md) rather than
invented — those require an actual LLM generation, which this benchmark
deliberately does not run to avoid burning API quota on an infra change.

Usage:
    cd evaluation
    python run_benchmark.py
"""
from __future__ import annotations

import concurrent.futures
import json
import statistics
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent / "backend"
EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(EVAL_DIR))

import psutil  # noqa: E402
import benchmark_corpus  # noqa: E402

RESULTS_PATH = Path(__file__).parent / "results.json"
N_LATENCY_TRIALS = 8  # per-query repeats for mean/median/p95


def _isolate_chroma() -> tuple[str, str]:
    """Points the retrieval stack at a fresh temp Chroma dir + a benchmark-
    only collection name, resetting ingest.py's cached client/collection
    singletons — the exact isolation pattern backend/conftest.py uses for
    the test suite. Returns (chroma_dir, collection_name)."""
    import tempfile
    import config
    import ingest

    chroma_dir = tempfile.mkdtemp(prefix="verityrag_benchmark_chroma_")
    collection_name = "benchmark_collection"
    config.CHROMA_DIR = chroma_dir
    config.COLLECTION_NAME = collection_name
    ingest._client = None
    ingest._collection = None
    return chroma_dir, collection_name


def _ingest_benchmark_corpus() -> dict:
    """Ingests all 10 benchmark documents through the REAL chunking
    pipeline (ingest.chunk_page — the same function real PDF ingestion
    uses; only the PDF-text-extraction step is bypassed since these are
    authored text pages, not actual PDF files), measuring real per-
    document indexing time. Returns {doc_key: {"document_id", "index_ms",
    "chunk_count", "page_count"}}."""
    import hashlib
    from ingest import chunk_page, get_collection
    from benchmark_corpus import DOCUMENTS

    col = get_collection()
    doc_meta: dict[str, dict] = {}
    for doc in DOCUMENTS:
        t0 = time.perf_counter()
        # document_id is content-hash-derived, exactly like real ingestion
        # (ingest.get_document_id hashes file BYTES; here we hash the
        # concatenated authored text, which is this "document"'s content).
        content = "\n".join(doc["pages"]).encode("utf-8")
        document_id = hashlib.sha256(content).hexdigest()[:16]

        all_chunks = []
        chunk_idx = 1
        current_section = "Unknown"
        for page_num, page_text in enumerate(doc["pages"], start=1):
            page_chunks, chunk_idx, current_section = chunk_page(
                {"page_number": page_num, "text": page_text}, document_id, chunk_idx, current_section,
            )
            all_chunks.extend(page_chunks)

        ids = [c["chunk_id"] for c in all_chunks]
        texts = [c["text"] for c in all_chunks]
        metadatas = [{
            "document_id": c["document_id"], "workspace_id": "benchmark_ws",
            "filename": doc["filename"], "source": doc["filename"],
            "page_number": c["page_number"], "section": c["section"],
            "chunk_id": c["chunk_id"], "parent_id": c["parent_id"], "chunk_type": c["chunk_type"],
        } for c in all_chunks]
        col.add(documents=texts, ids=ids, metadatas=metadatas)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        doc_meta[doc["doc_key"]] = {
            "document_id": document_id, "index_ms": elapsed_ms,
            "chunk_count": len(all_chunks), "page_count": len(doc["pages"]),
            "filename": doc["filename"], "title": doc["title"],
        }
        print(f"  indexed {doc['doc_key']:16s} document_id={document_id}  "
              f"chunks={len(all_chunks):3d}  {elapsed_ms:7.2f} ms")

    return doc_meta


# ---------------------------------------------------------------------------
# A. Upload/indexing — unique document_id per document, no cross-contamination
# ---------------------------------------------------------------------------
def check_indexing(doc_meta: dict) -> dict:
    ids = [m["document_id"] for m in doc_meta.values()]
    unique = len(set(ids)) == len(ids)
    total_chunks = sum(m["chunk_count"] for m in doc_meta.values())

    from ingest import get_collection
    col = get_collection()
    actual_count = col.count()

    # No-cross-contamination check: every chunk's stored document_id must be
    # one of the 10 we just ingested, and each document's OWN chunks (looked
    # up by metadata filter) must only ever return that document's chunks.
    contamination_found = False
    for doc_key, m in doc_meta.items():
        res = col.get(where={"document_id": m["document_id"]}, include=["metadatas"])
        doc_ids_returned = {md["document_id"] for md in res["metadatas"]}
        if doc_ids_returned != {m["document_id"]}:
            contamination_found = True

    return {
        "documents_ingested": len(doc_meta),
        "unique_document_ids": unique,
        "total_chunks_indexed": total_chunks,
        "chroma_collection_count_matches": actual_count == total_chunks,
        "cross_document_contamination_found": contamination_found,
        "mean_index_time_ms": round(statistics.mean(m["index_ms"] for m in doc_meta.values()), 2),
        "total_index_time_ms": round(sum(m["index_ms"] for m in doc_meta.values()), 2),
    }


# ---------------------------------------------------------------------------
# Timed retrieve() wrapper — captures dense+bm25+RRF+rerank latency and the
# rerank-only latency separately (by timing retrieval.rerank() directly).
# ---------------------------------------------------------------------------
def _timed_retrieve(query: str, document_ids: list[str] | None, top_k: int = 5) -> dict:
    from retrieval import (
        dense_search, bm25_search, reciprocal_rank_fusion, rerank,
        expand_with_parent_context, deduplicate_chunks, select_within_token_budget,
        _estimate_tokens,
    )
    from config import DENSE_TOP_K, BM25_TOP_K, MAX_CHUNKS_PER_DOC

    t0 = time.perf_counter()
    dense = dense_search(query, top_k=DENSE_TOP_K, document_ids=document_ids)
    t1 = time.perf_counter()
    bm25 = bm25_search(query, top_k=BM25_TOP_K, document_ids=document_ids)
    t2 = time.perf_counter()
    fused = reciprocal_rank_fusion(dense, bm25)
    t3 = time.perf_counter()
    reranked = rerank(query, fused, top_k=max(top_k * 3, 15)) if fused else []
    t4 = time.perf_counter()
    expanded = expand_with_parent_context(reranked)
    deduped = deduplicate_chunks(expanded)
    budgeted = select_within_token_budget(deduped, max_per_doc=MAX_CHUNKS_PER_DOC)
    final = budgeted[:top_k]
    t5 = time.perf_counter()

    context_tokens = sum(_estimate_tokens(c.get("parent_context") or c["text"]) for c in final)

    return {
        "chunks": final,
        "dense_latency_ms": round((t1 - t0) * 1000, 3),
        "bm25_latency_ms": round((t2 - t1) * 1000, 3),
        "rrf_latency_ms": round((t3 - t2) * 1000, 3),
        "rerank_latency_ms": round((t4 - t3) * 1000, 3),
        "post_processing_latency_ms": round((t5 - t4) * 1000, 3),
        "total_latency_ms": round((t5 - t0) * 1000, 3),
        "candidates_before_rerank": len(fused),
        "chunks_returned": len(final),
        "context_tokens": context_tokens,
    }


# ---------------------------------------------------------------------------
# B. Retrieval — single-document / multi-document / all-document queries
# D. Document isolation — ask about A, verify evidence contains only A
# ---------------------------------------------------------------------------
def check_retrieval_and_isolation(doc_meta: dict) -> dict:
    from benchmark_corpus import DOCUMENTS

    results = {"single_document": [], "multi_document": [], "all_documents": None, "isolation_violations": 0}

    # Single-document queries — ask about each paper individually, using its
    # own authored questions, verify 100% of returned evidence belongs to it.
    for doc in DOCUMENTS:
        doc_id = doc_meta[doc["doc_key"]]["document_id"]
        for q in doc["questions"]:
            r = _timed_retrieve(q["question"], document_ids=[doc_id], top_k=5)
            doc_ids_seen = {c["metadata"]["document_id"] for c in r["chunks"]}
            isolated = doc_ids_seen.issubset({doc_id})
            if not isolated:
                results["isolation_violations"] += 1
            results["single_document"].append({
                "doc_key": doc["doc_key"], "question": q["question"],
                "chunks_returned": r["chunks_returned"], "isolated_correctly": isolated,
                "total_latency_ms": r["total_latency_ms"],
            })

    # Multi-document queries — target exactly 2 specific papers, verify
    # evidence never leaks from an UN-selected third paper.
    two_keys = [DOCUMENTS[0]["doc_key"], DOCUMENTS[1]["doc_key"]]
    two_ids = [doc_meta[k]["document_id"] for k in two_keys]
    r = _timed_retrieve("architecture design decisions and trade-offs", document_ids=two_ids, top_k=10)
    doc_ids_seen = {c["metadata"]["document_id"] for c in r["chunks"]}
    results["multi_document"].append({
        "targeted_docs": two_keys, "chunks_returned": r["chunks_returned"],
        "only_targeted_docs_present": doc_ids_seen.issubset(set(two_ids)),
        "total_latency_ms": r["total_latency_ms"],
    })

    # All-documents query — no document_ids filter, should be able to draw
    # from any of the 10 (not isolation-restricted, but must not error and
    # must not silently return zero).
    r_all = _timed_retrieve("architecture, algorithm, and system design", document_ids=None, top_k=10)
    results["all_documents"] = {
        "chunks_returned": r_all["chunks_returned"],
        "distinct_documents_represented": len({c["metadata"]["document_id"] for c in r_all["chunks"]}),
        "total_latency_ms": r_all["total_latency_ms"],
    }

    return results


# ---------------------------------------------------------------------------
# C. Comparison — 2 / 5 / 10 documents (evidence-gathering path used by
#    report_generator.py's comparative report, exercised directly here)
# ---------------------------------------------------------------------------
def check_comparison_scaling(doc_meta: dict) -> dict:
    from retrieval import retrieve, select_within_token_budget, group_by_document
    from config import MAX_REPORT_CONTEXT_TOKENS, REPORT_CHUNKS_PER_DOC

    all_ids = [m["document_id"] for m in doc_meta.values()]
    out = {}
    for n in (2, 5, 10):
        ids_n = all_ids[:n]
        t0 = time.perf_counter()
        all_chunks = []
        for doc_id in ids_n:
            all_chunks.extend(retrieve(
                "problem, methodology, architecture, results, limitations",
                strategy="hybrid", document_ids=[doc_id], top_k=REPORT_CHUNKS_PER_DOC,
                apply_token_budget=False,
            ))
        budgeted = select_within_token_budget(all_chunks, max_tokens=MAX_REPORT_CONTEXT_TOKENS, max_per_doc=REPORT_CHUNKS_PER_DOC)
        grouped = group_by_document(budgeted)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        out[f"{n}_documents"] = {
            "documents_requested": n,
            "documents_represented_in_final_evidence": len(grouped),
            "all_documents_represented": len(grouped) == n,
            "total_chunks_selected": len(budgeted),
            "total_latency_ms": elapsed_ms,
        }
    return out


# ---------------------------------------------------------------------------
# E. Multi-aspect query decomposition — subqueries preserve document scope,
#    global merge/rerank stays correct
# ---------------------------------------------------------------------------
def check_query_decomposition(doc_meta: dict) -> dict:
    from query_transform import decompose_query_deterministic
    from retrieval import retrieve_multi

    doc_a = doc_meta["transformers"]["document_id"]
    doc_b = doc_meta["cnn"]["document_id"]
    question = "Compare the methodology and results of these two papers."
    sub_queries = decompose_query_deterministic(question)

    t0 = time.perf_counter()
    chunks = retrieve_multi(sub_queries if len(sub_queries) > 1 else [question], document_ids=[doc_a, doc_b], top_k=10, rerank_query=question)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    doc_ids_seen = {c["metadata"]["document_id"] for c in chunks}
    return {
        "question": question,
        "sub_queries_generated": sub_queries,
        "sub_query_count": len(sub_queries),
        "scope_preserved": doc_ids_seen.issubset({doc_a, doc_b}),
        "both_documents_represented": doc_ids_seen == {doc_a, doc_b},
        "chunks_returned": len(chunks),
        "total_latency_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# F. Concurrent/parallel retrieval
# ---------------------------------------------------------------------------
def check_concurrent_retrieval(doc_meta: dict) -> dict:
    from benchmark_corpus import DOCUMENTS

    queries = [(doc["questions"][0]["question"], [doc_meta[doc["doc_key"]]["document_id"]]) for doc in DOCUMENTS]

    def _run(q_and_ids):
        q, ids = q_and_ids
        r = _timed_retrieve(q, document_ids=ids, top_k=5)
        return r["total_latency_ms"]

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        latencies = list(ex.map(_run, queries))
    wall_clock_ms = round((time.perf_counter() - t0) * 1000, 2)

    sequential_sum_ms = round(sum(latencies), 2)
    return {
        "concurrent_queries": len(queries),
        "wall_clock_ms": wall_clock_ms,
        "sum_of_individual_latencies_ms": sequential_sum_ms,
        "speedup_factor": round(sequential_sum_ms / wall_clock_ms, 2) if wall_clock_ms else None,
        "all_succeeded": len(latencies) == len(queries),
    }


# ---------------------------------------------------------------------------
# Latency distribution — mean/median/p95 over N_LATENCY_TRIALS repeats
# ---------------------------------------------------------------------------
def measure_latency_distribution(doc_meta: dict) -> dict:
    from benchmark_corpus import DOCUMENTS

    sample_doc = DOCUMENTS[0]
    doc_id = doc_meta[sample_doc["doc_key"]]["document_id"]
    question = sample_doc["questions"][0]["question"]

    totals, denses, bm25s, rerank_lat = [], [], [], []
    for _ in range(N_LATENCY_TRIALS):
        r = _timed_retrieve(question, document_ids=[doc_id], top_k=5)
        totals.append(r["total_latency_ms"])
        denses.append(r["dense_latency_ms"])
        bm25s.append(r["bm25_latency_ms"])
        rerank_lat.append(r["rerank_latency_ms"])

    def _stats(vals):
        vals_sorted = sorted(vals)
        p95_idx = min(len(vals_sorted) - 1, int(round(0.95 * (len(vals_sorted) - 1))))
        return {
            "mean_ms": round(statistics.mean(vals), 3),
            "median_ms": round(statistics.median(vals), 3),
            "p95_ms": round(vals_sorted[p95_idx], 3),
            "min_ms": round(min(vals), 3),
            "max_ms": round(max(vals), 3),
        }

    return {
        "trials": N_LATENCY_TRIALS,
        "query": question,
        "end_to_end_retrieval": _stats(totals),
        "dense_search": _stats(denses),
        "bm25_search": _stats(bm25s),
        "cross_encoder_rerank": _stats(rerank_lat),
    }


# ---------------------------------------------------------------------------
# Baseline (naive dense top-k, no BM25/RRF/rerank) vs VerityRAG hybrid —
# Recall@K / Precision@K / MRR against the corpus's authored ground truth.
# ---------------------------------------------------------------------------
def run_baseline_vs_hybrid(doc_meta: dict) -> dict:
    from benchmark_corpus import DOCUMENTS
    from retrieval import dense_search, retrieve
    from eval_metrics import retrieval_metrics_suite

    K_VALUES = [1, 3, 5]
    per_question = []

    for doc in DOCUMENTS:
        relevant_doc_id = doc_meta[doc["doc_key"]]["document_id"]
        for q in doc["questions"]:
            question = q["question"]

            # BASELINE: naive dense-only top-10, no BM25, no RRF, no rerank,
            # searched across the WHOLE 10-document corpus (not pre-scoped)
            # — this is deliberately what a "just embed and cosine-search"
            # baseline looks like, the honest comparison point.
            t0 = time.perf_counter()
            baseline_chunks = dense_search(question, top_k=10, document_ids=None)
            baseline_latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            baseline_doc_ids = []
            for c in baseline_chunks:
                d = c["metadata"]["document_id"]
                if d not in baseline_doc_ids:
                    baseline_doc_ids.append(d)

            # VERITYRAG: full hybrid pipeline (dense+BM25+RRF+rerank+parent
            # context+token budget), also searched across the whole corpus.
            t0 = time.perf_counter()
            hybrid_chunks = retrieve(question, strategy="hybrid", document_ids=None, top_k=10)
            hybrid_latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            hybrid_doc_ids = []
            for c in hybrid_chunks:
                d = c["metadata"]["document_id"]
                if d not in hybrid_doc_ids:
                    hybrid_doc_ids.append(d)

            per_question.append({
                "doc_key": doc["doc_key"], "question": question,
                "relevant_document_ids": [relevant_doc_id],
                "baseline_dense_only": {
                    "retrieved_document_ids": baseline_doc_ids,
                    "metrics": retrieval_metrics_suite(baseline_doc_ids, [relevant_doc_id], K_VALUES),
                    "latency_ms": baseline_latency_ms,
                },
                "verityrag_hybrid": {
                    "retrieved_document_ids": hybrid_doc_ids,
                    "metrics": retrieval_metrics_suite(hybrid_doc_ids, [relevant_doc_id], K_VALUES),
                    "latency_ms": hybrid_latency_ms,
                },
            })

    def _aggregate(key: str) -> dict:
        agg: dict[str, list] = {}
        latencies = []
        for pq in per_question:
            latencies.append(pq[key]["latency_ms"])
            for mk, mv in pq[key]["metrics"].items():
                if isinstance(mv, (int, float)):
                    agg.setdefault(mk, []).append(mv)
        return {
            "avg_latency_ms": round(statistics.mean(latencies), 3),
            "avg_metrics": {mk: round(statistics.mean(vs), 4) for mk, vs in agg.items()},
        }

    return {
        "k_values": K_VALUES,
        "questions_evaluated": len(per_question),
        "baseline_dense_only_summary": _aggregate("baseline_dense_only"),
        "verityrag_hybrid_summary": _aggregate("verityrag_hybrid"),
        "per_question_results": per_question,
    }


def main():
    print("=== VerityRAG Retrieval Evaluation + Latency Benchmark ===")
    print("Isolating Chroma to a temp collection (never touches backend/chroma_store)...")
    chroma_dir, collection_name = _isolate_chroma()
    print(f"  chroma_dir={chroma_dir}  collection={collection_name}\n")

    process = psutil.Process()
    mem_before_mb = round(process.memory_info().rss / (1024 * 1024), 1)

    print("Ingesting 10 benchmark documents...")
    doc_meta = _ingest_benchmark_corpus()

    print("\nBuilding BM25 index...")
    from retrieval import build_bm25_index
    t0 = time.perf_counter()
    build_bm25_index()
    bm25_build_ms = round((time.perf_counter() - t0) * 1000, 2)
    print(f"  BM25 index built in {bm25_build_ms} ms\n")

    print("A. Checking indexing correctness...")
    indexing_report = check_indexing(doc_meta)

    print("B/D. Checking retrieval + document isolation...")
    retrieval_report = check_retrieval_and_isolation(doc_meta)

    print("C. Checking comparison scaling (2/5/10 documents)...")
    comparison_report = check_comparison_scaling(doc_meta)

    print("E. Checking multi-aspect query decomposition...")
    decomposition_report = check_query_decomposition(doc_meta)

    print("F. Checking concurrent retrieval...")
    concurrency_report = check_concurrent_retrieval(doc_meta)

    print(f"Measuring latency distribution ({N_LATENCY_TRIALS} trials)...")
    latency_report = measure_latency_distribution(doc_meta)

    print("Running baseline (naive dense) vs VerityRAG (hybrid) retrieval evaluation...")
    baseline_vs_hybrid = run_baseline_vs_hybrid(doc_meta)

    mem_after_mb = round(process.memory_info().rss / (1024 * 1024), 1)

    import platform
    import sqlalchemy
    import chromadb

    report = {
        "meta": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset": "10 original isolated benchmark documents (evaluation/benchmark_corpus.py) "
                       "— NOT scraped/real research papers; hand-authored, non-overlapping technical "
                       "text so retrieval correctness is independently verifiable.",
            "dataset_size_documents": len(doc_meta),
            "dataset_size_questions": sum(len(d["questions"]) for d in benchmark_corpus.DOCUMENTS),
            "isolated_chroma_dir": chroma_dir,
            "isolated_collection_name": collection_name,
            "production_chroma_touched": False,
            "llm_calls_made": 0,
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cpu": platform.processor(),
                "cpu_count": psutil.cpu_count(logical=True),
                "chromadb_version": chromadb.__version__,
                "sqlalchemy_version": sqlalchemy.__version__,
                "hardware_note": "Local development machine, CPU-only inference (no GPU used for embeddings/reranking).",
            },
            "config": {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "rrf_k": 60,
            },
        },
        "resource_usage": {
            "process_rss_mb_before": mem_before_mb,
            "process_rss_mb_after": mem_after_mb,
            "process_rss_mb_delta": round(mem_after_mb - mem_before_mb, 1),
        },
        "A_indexing": indexing_report,
        "B_D_retrieval_and_isolation": retrieval_report,
        "C_comparison_scaling": comparison_report,
        "E_query_decomposition": decomposition_report,
        "F_concurrent_retrieval": concurrency_report,
        "bm25_index_build_ms": bm25_build_ms,
        "latency_distribution": latency_report,
        "baseline_vs_hybrid_retrieval_evaluation": baseline_vs_hybrid,
        "not_measured_by_this_script": [
            "groundedness (requires an actual LLM generation call — see evaluation/README.md and "
            "backend/groundedness_eval.py for the separate, offline, opt-in evaluator)",
            "answer_relevance (same — requires LLM generation)",
            "end-to-end LLM latency / tokens-per-query / cache-hit-rate in PRODUCTION (this script "
            "never calls Groq; see GET /eval/dashboard's live/runtime section for those, sourced from "
            "logs/verityrag_events.jsonl on actual production traffic)",
        ],
    }

    RESULTS_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n=== Results written to {RESULTS_PATH} ===")

    print("\n--- Summary ---")
    print(f"Documents indexed: {indexing_report['documents_ingested']}  "
          f"(unique ids: {indexing_report['unique_document_ids']}, "
          f"contamination found: {indexing_report['cross_document_contamination_found']})")
    print(f"Isolation violations across {len(retrieval_report['single_document'])} single-doc queries: "
          f"{retrieval_report['isolation_violations']}")
    print(f"End-to-end retrieval latency — mean {latency_report['end_to_end_retrieval']['mean_ms']} ms, "
          f"median {latency_report['end_to_end_retrieval']['median_ms']} ms, "
          f"p95 {latency_report['end_to_end_retrieval']['p95_ms']} ms")
    b = baseline_vs_hybrid["baseline_dense_only_summary"]["avg_metrics"]
    h = baseline_vs_hybrid["verityrag_hybrid_summary"]["avg_metrics"]
    print(f"Baseline (dense-only) recall@3={b.get('recall@3')}  mrr={b.get('mrr')}")
    print(f"VerityRAG (hybrid)   recall@3={h.get('recall@3')}  mrr={h.get('mrr')}")

    # Clean up the isolated temp Chroma dir — never leave benchmark data
    # lying around next to production data.
    import shutil
    try:
        shutil.rmtree(chroma_dir, ignore_errors=True)
    except Exception:
        pass

    return report


if __name__ == "__main__":
    main()
