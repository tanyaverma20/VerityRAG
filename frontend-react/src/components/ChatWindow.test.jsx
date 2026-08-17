import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ChatWindow } from "./ChatWindow";

function makeDoc(overrides = {}) {
  return {
    document_id: "doc1",
    filename: "paper.pdf",
    title: null,
    authors: null,
    year: null,
    page_count: 10,
    chunk_count: 20,
    ingestion_status: "INDEXED",
    error_message: null,
    workspace_id: "ws1",
    created_at: "",
    updated_at: "",
    ...overrides
  };
}

describe("ChatWindow", () => {
  it("shows the no-documents empty state when nothing is uploaded", () => {
    render(<ChatWindow messages={[]} documents={[]} sending={false} onSend={vi.fn()} />);
    expect(screen.getByText("Start your research workspace")).toBeInTheDocument();
  });

  it("shows the ask-anything empty state once a document is INDEXED", () => {
    render(<ChatWindow messages={[]} documents={[makeDoc()]} sending={false} onSend={vi.fn()} />);
    expect(screen.getByText("Ask anything about your papers")).toBeInTheDocument();
  });

  it("does NOT treat a still-PROCESSING document as ready to query", () => {
    render(
      <ChatWindow messages={[]} documents={[makeDoc({ ingestion_status: "PROCESSING" })]} sending={false} onSend={vi.fn()} />
    );
    expect(screen.getByText("Start your research workspace")).toBeInTheDocument();
  });

  it("submitting sends the raw question text — document scoping is resolved by the caller (see utils/scope.test.js)", async () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    const docs = [
    makeDoc({ document_id: "ready1", ingestion_status: "INDEXED" }),
    makeDoc({ document_id: "ready2", ingestion_status: "INDEXED" }),
    makeDoc({ document_id: "still_processing", ingestion_status: "PROCESSING" })];

    render(<ChatWindow messages={[]} documents={docs} sending={false} onSend={onSend} />);

    const textarea = screen.getByPlaceholderText("Ask a question about your papers…");
    await userEvent.type(textarea, "What is the contribution?");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(onSend).toHaveBeenCalledWith("What is the contribution?");
  });

  it("does not submit an empty/whitespace-only question", async () => {
    const onSend = vi.fn();
    render(<ChatWindow messages={[]} documents={[makeDoc()]} sending={false} onSend={onSend} />);
    const textarea = screen.getByPlaceholderText("Ask a question about your papers…");
    await userEvent.type(textarea, "   ");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(onSend).not.toHaveBeenCalled();
  });

  it("disables the send button while a request is in flight", () => {
    render(<ChatWindow messages={[]} documents={[makeDoc()]} sending={true} onSend={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("renders real conversation messages when present", () => {
    render(
      <ChatWindow
        messages={[
        { id: "1", role: "user", text: "Hello?" },
        { id: "2", role: "assistant", text: "Hi, grounded answer." }]
        }
        documents={[makeDoc()]}
        sending={false}
        onSend={vi.fn()} />

    );
    expect(screen.getByText("Hello?")).toBeInTheDocument();
    expect(screen.getByText("Hi, grounded answer.")).toBeInTheDocument();
    expect(screen.queryByText("Ask anything about your papers")).not.toBeInTheDocument();
  });
});