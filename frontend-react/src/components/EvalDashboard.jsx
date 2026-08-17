import { useEffect, useState } from "react";
import * as api from "../api/client";

function Tile({ label, value }) {
  return (
    <div className="eval-tile">
      <div className="eval-tile-value">{value === null || value === undefined ? "Not measured" : String(value)}</div>
      <div className="eval-tile-label">{label}</div>
    </div>);

}

export function EvalDashboard({ onClose }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.
    getEvalDashboard().
    then(setData).
    catch((e) => setError(e instanceof Error ? e.message : "Failed to load"));
  }, []);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal eval-dashboard" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Evaluation Dashboard</h3>
          <button className="modal-close" onClick={onClose}>
            ×
          </button>
        </div>

        {error && <div className="eval-error">{error}</div>}
        {!data && !error && <div className="eval-loading">Loading…</div>}

        {data &&
        <div className="eval-sections">
            <section>
              <h4>Live (runtime, from real logged requests)</h4>
              {data.live ?
            <div className="eval-tiles">
                  <Tile label="Sample size" value={data.live.sample_size} />
                  <Tile label="Avg LLM calls/query" value={data.live.avg_llm_calls_per_query} />
                  <Tile label="Avg latency (s)" value={data.live.avg_latency_s} />
                  <Tile label="Cache hit rate" value={data.live.cache_hit_rate} />
                  <Tile label="Fallback rate" value={data.live.fallback_rate} />
                  <Tile label="Avg tokens/query" value={data.live.avg_tokens_per_query} />
                </div> :

            <p className="eval-note">{data.live_note}</p>
            }
            </section>

            <section>
              <h4>Cache</h4>
              {data.cache ?
            <div className="eval-tiles">
                  <Tile label="Backend" value={data.cache.backend} />
                  <Tile label="Answer entries" value={data.cache.answer_entries} />
                  <Tile label="Report entries" value={data.cache.report_entries} />
                  <Tile label="Hits" value={data.cache.hits} />
                  <Tile label="Misses" value={data.cache.misses} />
                  <Tile label="Hit rate" value={data.cache.hit_rate} />
                  <Tile label="Backend errors" value={data.cache.redis_errors} />
                </div> :

            <p className="eval-note">Not measured</p>
            }
            </section>

            <section>
              <h4>Offline retrieval evaluation</h4>
              {data.offline_retrieval_eval ?
            <div className="eval-tiles">
                  <Tile label="Questions evaluated" value={data.offline_retrieval_eval.questions_evaluated} />
                  <Tile label="Precision@5" value={data.offline_retrieval_eval.current_pipeline_precision_at_5} />
                  <Tile label="Recall@5" value={data.offline_retrieval_eval.current_pipeline_recall_at_5} />
                  <Tile label="MRR" value={data.offline_retrieval_eval.current_pipeline_mrr} />
                  <Tile label="Reranking Δ recall@5" value={data.offline_retrieval_eval.reranking_recall_at_5_improvement} />
                </div> :

            <p className="eval-note">Not measured</p>
            }
            </section>

            <section>
              <h4>10-document benchmark</h4>
              {data.offline_benchmark_10_document ?
            <div className="eval-tiles">
                  <Tile label="Documents" value={data.offline_benchmark_10_document.documents_benchmarked} />
                  <Tile label="Isolation violations" value={data.offline_benchmark_10_document.isolation_violations} />
                  <Tile label="Mean retrieval latency (ms)" value={data.offline_benchmark_10_document.retrieval_latency_mean_ms} />
                  <Tile label="p95 latency (ms)" value={data.offline_benchmark_10_document.retrieval_latency_p95_ms} />
                  <Tile label="Concurrent speedup" value={data.offline_benchmark_10_document.concurrent_speedup_factor} />
                </div> :

            <p className="eval-note">Not measured</p>
            }
            </section>

            <section>
              <h4>Offline groundedness evaluation</h4>
              {data.offline_groundedness_eval ?
            <div className="eval-tiles">
                  <Tile label="Dataset size" value={data.offline_groundedness_eval.dataset_size} />
                  <Tile label="Avg groundedness score" value={data.offline_groundedness_eval.avg_groundedness_score} />
                  <Tile label="Avg unsupported-claim rate" value={data.offline_groundedness_eval.avg_unsupported_claim_rate} />
                  <Tile label="Avg evidence coverage" value={data.offline_groundedness_eval.avg_evidence_coverage} />
                </div> :

            <p className="eval-note">Not measured</p>
            }
            </section>

            {data.not_measured.length > 0 &&
          <section>
                <h4>Not measured</h4>
                <ul className="eval-not-measured">
                  {data.not_measured.map((m, i) =>
              <li key={i}>{m}</li>
              )}
                </ul>
              </section>
          }
          </div>
        }
      </div>
    </div>);

}