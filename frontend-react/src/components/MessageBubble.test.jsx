import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MessageBubble } from "./MessageBubble";

describe("MessageBubble", () => {
  it("renders a pending assistant message as thinking dots, not empty text", () => {
    const msg = { id: "1", role: "assistant", text: "", pending: true };
    render(<MessageBubble message={msg} />);
    expect(screen.getByLabelText("Thinking")).toBeInTheDocument();
  });

  it("renders a plain user message", () => {
    const msg = { id: "1", role: "user", text: "What is attention?" };
    render(<MessageBubble message={msg} />);
    expect(screen.getByText("What is attention?")).toBeInTheDocument();
  });

  it("renders an assistant answer with its confidence pill", () => {
    const msg = { id: "1", role: "assistant", text: "The answer is X.", confidence: "HIGH" };
    render(<MessageBubble message={msg} />);
    expect(screen.getByText("The answer is X.")).toBeInTheDocument();
    expect(screen.getByText("HIGH")).toBeInTheDocument();
  });

  it("renders and de-duplicates citation pills by document_id + page", () => {
    const msg = {
      id: "1",
      role: "assistant",
      text: "Grounded answer.",
      confidence: "HIGH",
      citations: [
      { document_id: "docA", page_number: 3, source: "paperA.pdf" },
      { document_id: "docA", page_number: 3, source: "paperA.pdf" }, // exact duplicate
      { document_id: "docA", page_number: 4, source: "paperA.pdf" } // different page, kept
      ]
    };
    render(<MessageBubble message={msg} />);
    expect(screen.getAllByText(/paperA\.pdf/)).toHaveLength(2);
  });

  it("renders a real backend error message (e.g. rate limiting) without crashing", () => {
    const msg = {
      id: "1",
      role: "assistant",
      text: "The AI reasoning service is temporarily rate-limited.",
      error: true
    };
    render(<MessageBubble message={msg} />);
    expect(screen.getByText(/temporarily rate-limited/)).toBeInTheDocument();
  });

  it("never claims a confidence level for a message that has none", () => {
    const msg = { id: "1", role: "assistant", text: "Some answer." };
    render(<MessageBubble message={msg} />);
    expect(screen.queryByText("Grounded Synthesis")).not.toBeInTheDocument();
  });

  it("shows a Deep Research kicker instead of the default 'Grounded Synthesis' label", () => {
    const msg = { id: "1", role: "assistant", text: "The answer is X.", confidence: "HIGH", kicker: "Deep Research" };
    render(<MessageBubble message={msg} />);
    expect(screen.getByText("Deep Research")).toBeInTheDocument();
    expect(screen.queryByText("Grounded Synthesis")).not.toBeInTheDocument();
    expect(screen.getByText("HIGH")).toBeInTheDocument();
  });
});