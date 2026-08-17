import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { AnalysisResultCard } from "./AnalysisResultCard";
import * as api from "../api/client";

function makeMsg(overrides) {
  return { id: "1", role: "assistant", text: "", ...overrides };
}

describe("AnalysisResultCard", () => {
  it("renders a Viva/Mock Test question set with category, difficulty, and expected points", () => {
    const msg = makeMsg({
      frontendRole: "analysis_questions",
      payload: {
        label: "Viva Questions",
        questions: [
        {
          question: "What is self-attention?",
          category: "concept",
          difficulty: "basic",
          expected_answer_points: ["Scaled dot-product", "Query/Key/Value"]
        }]

      }
    });
    render(<AnalysisResultCard message={msg} />);
    expect(screen.getByText("Viva Questions")).toBeInTheDocument();
    expect(screen.getByText("What is self-attention?")).toBeInTheDocument();
    expect(screen.getByText("concept")).toBeInTheDocument();
    expect(screen.getByText("basic")).toBeInTheDocument();
    expect(screen.getByText("Scaled dot-product")).toBeInTheDocument();
  });

  it("renders an interview question with its category tag", () => {
    const msg = makeMsg({
      frontendRole: "interview_question",
      text: "How would you scale this system?",
      payload: { category: "system_design" }
    });
    render(<AnalysisResultCard message={msg} />);
    expect(screen.getByText("Interview Question")).toBeInTheDocument();
    expect(screen.getByText("system_design")).toBeInTheDocument();
    expect(screen.getByText("How would you scale this system?")).toBeInTheDocument();
  });

  it("renders the interview-complete state when payload.ended is true", () => {
    const msg = makeMsg({ frontendRole: "interview_question", text: "Interview complete.", payload: { ended: true } });
    render(<AnalysisResultCard message={msg} />);
    expect(screen.getByText("Interview Complete")).toBeInTheDocument();
  });

  it("renders an interview evaluation with missing points, technical depth, and suggested answer", () => {
    const msg = makeMsg({
      frontendRole: "interview_eval",
      payload: {
        correctness: "PARTIALLY_CORRECT",
        missing_points: ["Mentions of masking"],
        technical_depth: "Solid but shallow on complexity.",
        suggested_answer: "Attention scales as O(n^2)."
      }
    });
    render(<AnalysisResultCard message={msg} />);
    expect(screen.getByText("PARTIALLY_CORRECT")).toBeInTheDocument();
    expect(screen.getByText("Mentions of masking")).toBeInTheDocument();
    expect(screen.getByText(/Solid but shallow/)).toBeInTheDocument();
    expect(screen.getByText(/Attention scales as/)).toBeInTheDocument();
  });

  it("renders a paper evaluation with support badges per dimension", () => {
    const msg = makeMsg({
      frontendRole: "evaluate_paper",
      payload: {
        evaluation: {
          title: "Attention Is All You Need",
          problem_clarity: { assessment: "Clearly stated.", support: "DIRECTLY_STATED" },
          novelty: { assessment: "High novelty.", support: "STRONGLY_SUPPORTED" },
          strengths: ["Simple architecture"],
          weaknesses: ["Compute-heavy"],
          overall_assessment: "Strong paper."
        }
      }
    });
    render(<AnalysisResultCard message={msg} />);
    expect(screen.getByText("Attention Is All You Need")).toBeInTheDocument();
    expect(screen.getByText("Directly stated")).toBeInTheDocument();
    expect(screen.getByText("Strongly supported")).toBeInTheDocument();
    expect(screen.getByText("Simple architecture")).toBeInTheDocument();
    expect(screen.getByText("Compute-heavy")).toBeInTheDocument();
  });

  it("falls back to plain text when an evaluate_paper message has no evaluation payload", () => {
    const msg = makeMsg({ frontendRole: "evaluate_paper", text: "Evaluation failed.", payload: {} });
    render(<AnalysisResultCard message={msg} />);
    expect(screen.getByText("Evaluation failed.")).toBeInTheDocument();
  });

  it("renders research gaps distinguishing author-stated vs inferred", () => {
    const msg = makeMsg({
      frontendRole: "research_gaps",
      payload: {
        gaps: [
        { gap: "No ablation on head count.", label: "AUTHOR_STATED_GAP", evidence: "Section 7 states...", document_id: "d1" },
        { gap: "Scaling beyond 100 layers untested.", label: "INFERRED_GAP", evidence: "", document_id: "d1" }],

        document_titles: { d1: "Attention Is All You Need" }
      }
    });
    render(<AnalysisResultCard message={msg} />);
    expect(screen.getByText("Author-stated")).toBeInTheDocument();
    expect(screen.getByText("Inferred")).toBeInTheDocument();
    expect(screen.getByText("No ablation on head count.")).toBeInTheDocument();
    expect(screen.getAllByText("Attention Is All You Need").length).toBeGreaterThan(0);
  });

  it("shows an honest empty state when no research gaps were identified", () => {
    const msg = makeMsg({ frontendRole: "research_gaps", payload: { gaps: [], document_titles: {} } });
    render(<AnalysisResultCard message={msg} />);
    expect(screen.getByText("No gaps identified from the available evidence.")).toBeInTheDocument();
  });

  it("renders a literature matrix as a real HTML table with one row per paper", () => {
    const msg = makeMsg({
      frontendRole: "literature_matrix",
      payload: {
        rows: [
        { title: "Paper A", problem: "P1", method: "M1", architecture: "", dataset: "", metrics: "", results: "", limitations: "", research_gap: "" },
        { document_id: "docB", problem: "P2", method: "M2", architecture: "", dataset: "", metrics: "", results: "", limitations: "", research_gap: "" }]

      }
    });
    render(<AnalysisResultCard message={msg} />);
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByText("Paper A")).toBeInTheDocument();
    expect(screen.getByText("docB")).toBeInTheDocument(); // falls back to document_id when no title
    expect(screen.getByText("P1")).toBeInTheDocument();
  });

  it("renders a knowledge graph as node tags plus readable edge sentences", () => {
    const msg = makeMsg({
      frontendRole: "knowledge_graph",
      payload: {
        nodes: [
        { id: "n1", label: "Self-Attention", type: "concept", document_id: "d1" },
        { id: "n2", label: "Transformer", type: "concept", document_id: "d1" }],

        edges: [{ source: "n1", target: "n2", relation: "part of" }],
        document_titles: { d1: "Attention Is All You Need" }
      }
    });
    render(<AnalysisResultCard message={msg} />);
    // "Self-Attention"/"Transformer" each appear twice: once as a node tag,
    // once inside the edge sentence's <strong> — both are real renders.
    expect(screen.getAllByText("Self-Attention").length).toBe(2);
    expect(screen.getAllByText("Transformer").length).toBe(2);
    expect(screen.getByText(/part of/)).toBeInTheDocument();
  });

  it("falls back to raw node ids in an edge sentence when a referenced node id is missing", () => {
    const msg = makeMsg({
      frontendRole: "knowledge_graph",
      payload: { nodes: [], edges: [{ source: "ghost1", target: "ghost2", relation: "relates to" }], document_titles: {} }
    });
    render(<AnalysisResultCard message={msg} />);
    expect(screen.getByText("ghost1")).toBeInTheDocument();
    expect(screen.getByText("ghost2")).toBeInTheDocument();
  });

  it("renders a comparison report with per-paper sections and format download buttons", async () => {
    // GET /report/{id}/{format} requires a real Authorization header now
    // (see main.py's report-ownership hardening), so downloads go through
    // api.downloadReport() (an authenticated fetch+blob), never a plain
    // <a href> that couldn't attach the header — this proves the button
    // calls it with the right (reportId, format), not that a URL was built.
    const downloadSpy = vi.spyOn(api, "downloadReport").mockResolvedValue();
    const msg = makeMsg({
      frontendRole: "report",
      payload: {
        report: {
          title: "Comparison: Attention vs. BERT",
          overview: "Both address sequence modeling.",
          papers: [{ title: "Attention Is All You Need", overview: "Introduces the Transformer." }],
          conclusion: "Attention is foundational to both.",
          report_id: "r1"
        }
      }
    });
    render(<AnalysisResultCard message={msg} />);
    expect(screen.getByText("Comparison: Attention vs. BERT")).toBeInTheDocument();
    expect(screen.getByText("Both address sequence modeling.")).toBeInTheDocument();
    expect(screen.getByText("Introduces the Transformer.")).toBeInTheDocument();
    const pdfButton = screen.getByText("Download PDF");
    expect(pdfButton.tagName).toBe("BUTTON");
    fireEvent.click(pdfButton);
    expect(downloadSpy).toHaveBeenCalledWith("r1", "pdf");
    downloadSpy.mockRestore();
  });

  it("falls back to plain text when a report message has no report payload", () => {
    const msg = makeMsg({ frontendRole: "report", text: "Report generation failed.", payload: {} });
    render(<AnalysisResultCard message={msg} />);
    expect(screen.getByText("Report generation failed.")).toBeInTheDocument();
  });

  it("renders the generic kicker fallback for Explain Figure / Recommend / Why Design / System Design", () => {
    const msg = makeMsg({ frontendRole: "assistant_kicker", kicker: "Figure Explanation", text: "Figure 2 shows the encoder stack." });
    render(<AnalysisResultCard message={msg} />);
    expect(screen.getByText("Figure Explanation")).toBeInTheDocument();
    expect(screen.getByText("Figure 2 shows the encoder stack.")).toBeInTheDocument();
  });

  it("renders plain text with no kicker for an unrecognized frontendRole, never crashing", () => {
    const msg = makeMsg({ frontendRole: "some_future_role", text: "Fallback content." });
    render(<AnalysisResultCard message={msg} />);
    expect(screen.getByText("Fallback content.")).toBeInTheDocument();
  });
});