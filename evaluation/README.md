# VerityRAG Retrieval Evaluation & Latency Benchmark

This directory contains a **reproducible, LLM-free retrieval benchmark**
(`run_benchmark.py`) plus its most recent real, measured output
(`results.json`). It exists to answer four questions with actual numbers,
not claims: does 10-document scale work correctly, is document isolation
real, how fast is retrieval, and does the hybrid pipeline actually beat a
naive baseline.

**Every number in `results.json` is measured, not fabricated.** The script
makes zero Groq/LLM calls — every metric here is retrieval-only (dense
search, BM25, RRF, CrossEncoder reranking, parent-context expansion,
token budgeting, document/workspace scoping, deterministic query
decomposition). It runs against an **isolated temp Chroma collection**
that is created, used, and deleted entirely within one run — it never
touches `backend/chroma_store` or `data/registry.db`.

## Reproduce it yourself

```bash
cd evaluation
python run_benchmark.py
```

Takes under a minute on a laptop CPU. Overwrites `results.json`.

## Dataset

10 original, hand-authored benchmark documents (`benchmark_corpus.py`) —
**not** scraped or downloaded research papers. Each covers a distinct,
non-overlapping CS/ML topic (Transformers, CNNs, RL, DB indexing, OS
scheduling, Raft consensus, GNNs, federated learning, compiler
optimization, BM25) specifically so retrieval correctness is
independently verifiable: a query about Raft leader election should never
surface a chunk from the BM25 document. 20 authored questions (2 per
document) carry known ground-truth `relevant_document_ids`.

This corpus is intentionally small and topically distinct — good for
proving isolation/scaling/latency, but too easy to show a quality
*difference* between retrieval strategies (see "Baseline vs. Hybrid"
below, where both approaches hit the ceiling on this specific dataset).
For a retrieval-*quality* comparison on a harder, larger, more realistic
corpus, see `data/eval_results_comprehensive.json` in the repo root — a
separate, earlier evaluation against **11 real research papers and 113
real questions**, referenced explicitly below rather than re-claimed here.

## Environment (from the last real run)

| | |
|---|---|
| Python | 3.13.5 |
| Platform | Windows-11, AMD64 |
| CPU | AMD Family 25 (12 logical cores), CPU-only — no GPU used for embeddings/reranking |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| ChromaDB | 1.5.9 |
| RRF k | 60 |

## A. Indexing (10 documents)

| Metric | Value |
|---|---|
| Documents ingested | 10 |
| Unique `document_id` values | 10/10 (100%) |
| Cross-document contamination found | **No** |
| Total chunks indexed | 30 |
| Mean index time / document | 57.5 ms |
| Total index time (10 docs) | 574.5 ms |

Every document's chunks were queried back by `document_id` immediately
after ingestion and verified to contain *only* that document's own
metadata — no chunk from any other document appeared.

## B/D. Retrieval & Document Isolation

20 single-document queries (2 per benchmark document, using each
document's own authored questions) were run scoped to that document alone.

| Metric | Value |
|---|---|
| Single-document queries run | 20 |
| Isolation violations (evidence from a non-targeted document) | **0** |
| Multi-document query (2 targeted documents) — only targeted docs present | **True** |
| All-documents query (no `document_ids` filter) — distinct documents represented | 6 of 10 in the top-10 reranked results (expected — reranking picks the *most relevant* chunks across the whole corpus for a generic query, not one from every document) |

## C. Comparison Scaling (2 / 5 / 10 documents)

The same evidence-gathering path `report_generator.py`'s comparative
report uses (per-document retrieval + one global token-budgeted merge),
exercised directly:

| Documents requested | Documents represented in final evidence | Total latency |
|---|---|---|
| 2 | 2 / 2 | 121.1 ms |
| 5 | 5 / 5 | 314.5 ms |
| 10 | 10 / 10 | 572.4 ms |

Every document requested was represented in the final evidence set at
every scale tested — the per-document retrieval + global token budget
never silently dropped a paper as more were added.

## E. Multi-Aspect Query Decomposition

Question: *"Compare the methodology and results of these two papers."*
(deliberately matches the deterministic decomposition trigger pattern)

| | |
|---|---|
| Sub-queries generated | 2 (`"methodology: ..."`, `"results: ..."`) |
| Document scope preserved across all sub-queries | **True** |
| Both targeted documents represented in the final merged/reranked result | **True** |
| Total latency | 133.1 ms |

## F. Concurrent Retrieval

10 single-document queries fired concurrently via a `ThreadPoolExecutor`
(10 workers):

| | |
|---|---|
| Wall-clock time for all 10 concurrent queries | 476.5 ms |
| Sum of the same 10 queries' individual latencies | 4,422.3 ms |
| Effective speedup | **9.28x** |
| All queries succeeded | **True** |

## Latency Distribution (8 trials, single-document query)

| Stage | Mean | Median | p95 |
|---|---|---|---|
| End-to-end retrieval | 79.0 ms | 78.5 ms | 82.7 ms |
| Dense search | 16.2 ms | 15.7 ms | 18.0 ms |
| BM25 search | 0.26 ms | 0.28 ms | 0.30 ms |
| Cross-Encoder rerank | 59.0 ms | 58.9 ms | 62.0 ms |

Full breakdown (including min/max) is in `results.json` →
`latency_distribution`. Cross-Encoder reranking is the dominant cost
(~75% of end-to-end latency) — expected on CPU-only inference; dense
search and BM25 are comparatively cheap.

## Resource Usage

| | |
|---|---|
| Process RSS before benchmark | 477.6 MB |
| Process RSS after benchmark | 775.0 MB |
| Delta (embedding + reranker models loaded into memory) | 297.4 MB |

## Baseline vs. Hybrid — Retrieval Quality

**On this specific 10-document benchmark corpus**, both a naive
dense-only top-K baseline and VerityRAG's full hybrid pipeline
(dense + BM25 + RRF + CrossEncoder rerank) achieved:

| Metric (K=3) | Baseline (dense-only) | VerityRAG (hybrid) |
|---|---|---|
| Recall@3 | 1.0 | 1.0 |
| MRR | 1.0 | 1.0 |

**This is an honest result, not a data-quality bug**: with only 10
topically distinct documents and 10 candidates pulled per query, even a
naive baseline easily finds the one correct document — the corpus is too
small/easy to surface a quality delta between retrieval strategies. It
does NOT mean hybrid retrieval provides no benefit; it means *this specific
10-document isolation benchmark isn't the right instrument to measure
that* (its job was isolation/scale/latency, all confirmed above).

**The real, previously-measured baseline-vs-hybrid quality comparison**
lives in `data/eval_results_comprehensive.json` (11 real research papers,
113 real evaluation questions, `k=[1,3,5,10]`), where the improvement is
real and measured:

| Pipeline | Recall@5 | Precision@5 | MRR | Avg latency |
|---|---|---|---|---|
| dense_only (baseline) | 0.8637 | 0.3044 | 0.8835 | 27.5 ms |
| bm25_only | 0.8164 | 0.2867 | 0.7950 | 9.0 ms |
| hybrid_rrf (no rerank) | 0.8637 | 0.3044 | 0.8953 | 32.1 ms |
| **hybrid_reranked (VerityRAG)** | **0.9444** | **0.3398** | **0.9077** | 1,429.3 ms* |

\* This latency figure includes the CrossEncoder forward pass on a larger,
harder real-paper corpus and was measured on the eval-harness environment
at the time of that run — it is not directly comparable to this
directory's 10-document numbers above, which use a much smaller candidate
pool. Re-run `backend/run_evaluation.py` to reproduce it fresh; that
script is unchanged by this work.

Reading both tables together: `hybrid_reranked` improves recall@5 by
**+8.07 percentage points** (0.9444 vs 0.8637, a **+9.3% relative**
improvement) and MRR by **+0.0242** (+2.7% relative) over the dense-only
baseline, on the harder 113-question/11-paper set — measured, not
estimated.

## Not Measured By This Benchmark

- **Groundedness / hallucination rate** — requires an actual LLM
  generation call, which this script deliberately never makes (see
  `backend/groundedness_eval.py` for the separate, offline, opt-in
  evaluator that reuses the existing claim-evidence-tracing mechanism).
- **Answer relevance** — same; requires generation.
- **Production LLM calls/query, cache hit rate, real user latency** —
  these are *runtime* metrics from actual production traffic, sourced
  from `logs/verityrag_events.jsonl` and surfaced live via
  `GET /eval/dashboard`'s runtime section, not something an offline
  benchmark script can measure.

## Methodology Notes

- K values evaluated: 1, 3, 5 (baseline-vs-hybrid table above) / 1, 3, 5,
  10 (the referenced `eval_results_comprehensive.json`).
- Recall@K / Precision@K / MRR are computed by `backend/eval_metrics.py`
  (deterministic, no LLM, already covered by `backend/test_phase4.py`-style
  unit tests elsewhere in the suite).
- "Relevant" for this benchmark = the single document a question was
  authored against (`benchmark_corpus.py`), i.e. document-level relevance,
  not chunk-level.
- All timings use `time.perf_counter()`, taken immediately around the
  measured call only (model loading time is excluded from the reported
  latencies, but included in the reported total wall-clock resource-usage
  section above).
