import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as api from "../api/client";
import { EvalDashboard } from "./EvalDashboard";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("EvalDashboard", () => {
  it("shows a loading state before the fetch resolves", () => {
    vi.spyOn(api, "getEvalDashboard").mockReturnValue(new Promise(() => {})); // never resolves
    render(<EvalDashboard onClose={vi.fn()} />);
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it('renders "Not measured" for a section the backend reports as null, never fabricating a number', async () => {
    const data = {
      live: null,
      live_note: "Not enough live traffic logged yet.",
      cache: null,
      offline_retrieval_eval: null,
      offline_benchmark_10_document: null,
      offline_groundedness_eval: null,
      not_measured: ["real Tesseract OCR verification"]
    };
    vi.spyOn(api, "getEvalDashboard").mockResolvedValue(data);
    render(<EvalDashboard onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("Not enough live traffic logged yet.")).toBeInTheDocument());
    expect(screen.getAllByText("Not measured").length).toBeGreaterThan(0);
    expect(screen.getByText("real Tesseract OCR verification")).toBeInTheDocument();
  });

  it("renders real measured tiles when the backend provides actual numbers", async () => {
    const data = {
      live: null,
      live_note: "Not measured",
      cache: { backend: "redis", answer_entries: 12, report_entries: 2, hits: 8, misses: 4, hit_rate: 0.67 },
      offline_retrieval_eval: null,
      offline_benchmark_10_document: {
        documents_benchmarked: 10,
        isolation_violations: 0,
        retrieval_latency_mean_ms: 79.0,
        retrieval_latency_p95_ms: 82.7,
        concurrent_speedup_factor: 9.28
      },
      offline_groundedness_eval: null,
      not_measured: []
    };
    vi.spyOn(api, "getEvalDashboard").mockResolvedValue(data);
    render(<EvalDashboard onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("redis")).toBeInTheDocument());
    expect(screen.getByText("10")).toBeInTheDocument();
    expect(screen.getByText("79")).toBeInTheDocument();
  });

  it("shows Backend errors and Reranking delta tiles when the backend provides them", async () => {
    const data = {
      live: null,
      live_note: "Not measured",
      cache: { backend: "redis", answer_entries: 1, report_entries: 0, hits: 3, misses: 1, hit_rate: 0.75, redis_errors: 2 },
      offline_retrieval_eval: {
        questions_evaluated: 113,
        current_pipeline_precision_at_5: 0.34,
        current_pipeline_recall_at_5: 0.9444,
        current_pipeline_mrr: 0.9077,
        reranking_recall_at_5_improvement: 0.0807
      },
      offline_benchmark_10_document: null,
      offline_groundedness_eval: null,
      not_measured: []
    };
    vi.spyOn(api, "getEvalDashboard").mockResolvedValue(data);
    render(<EvalDashboard onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("Backend errors")).toBeInTheDocument());
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("Reranking Δ recall@5")).toBeInTheDocument();
    expect(screen.getByText("0.0807")).toBeInTheDocument();
  });

  it("shows an error message when the fetch fails, without crashing", async () => {
    vi.spyOn(api, "getEvalDashboard").mockRejectedValue(new Error("Backend unreachable"));
    render(<EvalDashboard onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("Backend unreachable")).toBeInTheDocument());
  });

  it("calls onClose when the × button is clicked", async () => {
    const onClose = vi.fn();
    vi.spyOn(api, "getEvalDashboard").mockResolvedValue({
      live: null,
      live_note: "Not measured",
      cache: null,
      offline_retrieval_eval: null,
      offline_benchmark_10_document: null,
      offline_groundedness_eval: null,
      not_measured: []
    });
    render(<EvalDashboard onClose={onClose} />);
    await userEvent.click(screen.getByText("×"));
    expect(onClose).toHaveBeenCalled();
  });
});