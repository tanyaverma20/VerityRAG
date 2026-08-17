import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SetupModal } from "./SetupModal";

describe("SetupModal", () => {
  it("shows difficulty + number-of-questions fields for viva, not the figure/question fields", () => {
    render(<SetupModal mode="viva" onCancel={vi.fn()} onSubmit={vi.fn()} />);
    expect(screen.getByText("Viva Setup")).toBeInTheDocument();
    expect(screen.getByText("Difficulty")).toBeInTheDocument();
    expect(screen.getByText("Number of questions")).toBeInTheDocument();
    expect(screen.queryByText(/Figure\/table reference/)).not.toBeInTheDocument();
    expect(screen.queryByText("Your question")).not.toBeInTheDocument();
  });

  it("shows only difficulty (no question count) for project_interview_start", () => {
    render(<SetupModal mode="project_interview_start" onCancel={vi.fn()} onSubmit={vi.fn()} />);
    expect(screen.getByText("Project Interview Setup")).toBeInTheDocument();
    expect(screen.getByText("Difficulty")).toBeInTheDocument();
    expect(screen.queryByText("Number of questions")).not.toBeInTheDocument();
  });

  it("shows only the figure/table reference field for explain_figure", () => {
    render(<SetupModal mode="explain_figure" onCancel={vi.fn()} onSubmit={vi.fn()} />);
    expect(screen.getByText(/Figure\/table reference/)).toBeInTheDocument();
    expect(screen.queryByText("Difficulty")).not.toBeInTheDocument();
  });

  it("shows only difficulty for project_interview_start plus an optional topics picker", () => {
    render(<SetupModal mode="project_interview_start" onCancel={vi.fn()} onSubmit={vi.fn()} />);
    expect(screen.getByText(/Topics/)).toBeInTheDocument();
    expect(screen.getByText("Architecture")).toBeInTheDocument();
    expect(screen.getByText("Limitations")).toBeInTheDocument();
  });

  it("submits the selected topics for project_interview_start", async () => {
    const onSubmit = vi.fn();
    render(<SetupModal mode="project_interview_start" onCancel={vi.fn()} onSubmit={onSubmit} />);
    await userEvent.click(screen.getByText("Architecture"));
    await userEvent.click(screen.getByText("Dataset"));
    await userEvent.click(screen.getByRole("button", { name: "Start" }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ topics: ["Architecture", "Dataset"] }));
  });

  it("shows the real, fixed Why This Design? question list — not a free-text box", () => {
    render(<SetupModal mode="why_design" onCancel={vi.fn()} onSubmit={vi.fn()} />);
    expect(screen.getByText("Why ChromaDB?")).toBeInTheDocument();
    expect(screen.getByText("Why one LLM call?")).toBeInTheDocument();
    expect(screen.queryByText("Difficulty")).not.toBeInTheDocument();
  });

  it("shows the real, fixed System Design question list", () => {
    render(<SetupModal mode="system_design" onCancel={vi.fn()} onSubmit={vi.fn()} />);
    expect(screen.getByText("How would you scale to millions of PDFs?")).toBeInTheDocument();
  });

  it("clicking a Why This Design? question submits it immediately, no separate Start step", async () => {
    const onSubmit = vi.fn();
    render(<SetupModal mode="why_design" onCancel={vi.fn()} onSubmit={onSubmit} />);
    await userEvent.click(screen.getByText("Why RRF?"));
    expect(onSubmit).toHaveBeenCalledWith({ question: "Why RRF?" });
  });

  it("submits the trimmed figure reference typed by the user", async () => {
    const onSubmit = vi.fn();
    render(<SetupModal mode="explain_figure" onCancel={vi.fn()} onSubmit={onSubmit} />);
    await userEvent.type(screen.getByPlaceholderText(/Figure 2/), "  Figure 2  ");
    await userEvent.click(screen.getByRole("button", { name: "Start" }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ figureReference: "Figure 2" }));
  });

  it("cancelling the Why This Design? picker calls onCancel, not onSubmit", async () => {
    const onCancel = vi.fn();
    const onSubmit = vi.fn();
    render(<SetupModal mode="why_design" onCancel={onCancel} onSubmit={onSubmit} />);
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("calls onCancel when Cancel is clicked, not onSubmit", async () => {
    const onCancel = vi.fn();
    const onSubmit = vi.fn();
    render(<SetupModal mode="viva" onCancel={onCancel} onSubmit={onSubmit} />);
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("calls onCancel when the backdrop is clicked", async () => {
    const onCancel = vi.fn();
    const { container } = render(<SetupModal mode="viva" onCancel={onCancel} onSubmit={vi.fn()} />);
    await userEvent.click(container.querySelector(".modal-backdrop"));
    expect(onCancel).toHaveBeenCalled();
  });
});