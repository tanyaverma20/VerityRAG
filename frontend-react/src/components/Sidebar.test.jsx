import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Sidebar } from "./Sidebar";

function makeDoc(overrides = {}) {
  return {
    document_id: "doc1",
    filename: "attention.pdf",
    title: null,
    ingestion_status: "INDEXED",
    ...overrides
  };
}

function baseProps(overrides = {}) {
  return {
    workspaces: [{ workspace_id: "ws1", name: "My Research", paper_count: 1 }],
    activeWorkspace: { workspace_id: "ws1", name: "My Research", paper_count: 1 },
    documents: [makeDoc()],
    sessions: [],
    activeSessionId: null,
    selectedDocIds: [],
    onToggleDocSelection: vi.fn(),
    onSwitchWorkspace: vi.fn(),
    onCreateWorkspace: vi.fn(),
    onUpload: vi.fn(),
    onDeleteDocument: vi.fn(),
    onNewChat: vi.fn(),
    onSelectSession: vi.fn(),
    onDeleteSession: vi.fn(),
    ...overrides
  };
}

describe("Sidebar", () => {
  it("shows the active workspace name and lists other workspaces in the switcher", async () => {
    render(
      <Sidebar
        {...baseProps({
          workspaces: [
          { workspace_id: "ws1", name: "My Research", paper_count: 1 },
          { workspace_id: "ws2", name: "Side Project", paper_count: 0 }]

        })} />

    );
    expect(screen.getByText("My Research")).toBeInTheDocument();
    await userEvent.click(screen.getByText("My Research"));
    expect(screen.getByText("Side Project")).toBeInTheDocument();
  });

  it("switching workspace calls onSwitchWorkspace with the picked workspace_id", async () => {
    const onSwitchWorkspace = vi.fn();
    render(
      <Sidebar
        {...baseProps({
          onSwitchWorkspace,
          workspaces: [
          { workspace_id: "ws1", name: "My Research", paper_count: 1 },
          { workspace_id: "ws2", name: "Side Project", paper_count: 0 }]

        })} />

    );
    await userEvent.click(screen.getByText("My Research"));
    await userEvent.click(screen.getByText("Side Project"));
    expect(onSwitchWorkspace).toHaveBeenCalledWith("ws2");
  });

  it("creating a new workspace calls onCreateWorkspace with the trimmed name", async () => {
    const onCreateWorkspace = vi.fn();
    render(<Sidebar {...baseProps({ onCreateWorkspace })} />);
    await userEvent.click(screen.getByText("My Research"));
    await userEvent.click(screen.getByText("+ New workspace"));
    await userEvent.type(screen.getByPlaceholderText("Workspace name"), "New Workspace");
    await userEvent.click(screen.getByText("Create"));
    expect(onCreateWorkspace).toHaveBeenCalledWith("New Workspace");
  });

  it("lists uploaded documents with their ingestion status", () => {
    render(
      <Sidebar
        {...baseProps({
          documents: [makeDoc({ document_id: "d1", filename: "a.pdf", ingestion_status: "INDEXED" }),
          makeDoc({ document_id: "d2", filename: "b.pdf", ingestion_status: "PROCESSING" })]
        })} />

    );
    expect(screen.getByText("a.pdf")).toBeInTheDocument();
    expect(screen.getByText("b.pdf")).toBeInTheDocument();
    expect(screen.getByText("INDEXED")).toBeInTheDocument();
    expect(screen.getByText("PROCESSING")).toBeInTheDocument();
  });

  it("shows the empty-papers message when no documents are uploaded", () => {
    render(<Sidebar {...baseProps({ documents: [] })} />);
    expect(screen.getByText("No papers yet — upload one to get started.")).toBeInTheDocument();
  });

  it("clicking an INDEXED document toggles its selection", async () => {
    const onToggleDocSelection = vi.fn();
    render(<Sidebar {...baseProps({ onToggleDocSelection })} />);
    await userEvent.click(screen.getByText("attention.pdf"));
    expect(onToggleDocSelection).toHaveBeenCalledWith("doc1");
  });

  it("clicking a still-PROCESSING document does not toggle selection", async () => {
    const onToggleDocSelection = vi.fn();
    render(
      <Sidebar
        {...baseProps({
          onToggleDocSelection,
          documents: [makeDoc({ ingestion_status: "PROCESSING" })]
        })} />

    );
    await userEvent.click(screen.getByText("attention.pdf"));
    expect(onToggleDocSelection).not.toHaveBeenCalled();
  });

  it("shows a 'Selected' tag on a document that is in selectedDocIds", () => {
    render(<Sidebar {...baseProps({ selectedDocIds: ["doc1"] })} />);
    expect(screen.getByText("Selected")).toBeInTheDocument();
  });

  it("removing a document calls onDeleteDocument without also toggling selection", async () => {
    const onDeleteDocument = vi.fn();
    const onToggleDocSelection = vi.fn();
    render(<Sidebar {...baseProps({ onDeleteDocument, onToggleDocSelection })} />);
    await userEvent.click(screen.getByLabelText("Remove attention.pdf"));
    expect(onDeleteDocument).toHaveBeenCalledWith("doc1");
    expect(onToggleDocSelection).not.toHaveBeenCalled();
  });

  it("uploading a PDF file calls onUpload", async () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    const { container } = render(<Sidebar {...baseProps({ onUpload })} />);
    const file = new File(["dummy"], "paper.pdf", { type: "application/pdf" });
    const input = container.querySelector('input[type="file"]');
    await userEvent.upload(input, file);
    expect(onUpload).toHaveBeenCalledWith(file);
  });

  it("lists recent chat sessions and selecting one calls onSelectSession", async () => {
    const onSelectSession = vi.fn();
    render(
      <Sidebar
        {...baseProps({
          onSelectSession,
          sessions: [{ session_id: "s1", title: "What is attention?" }]
        })} />

    );
    await userEvent.click(screen.getByText("What is attention?"));
    expect(onSelectSession).toHaveBeenCalledWith("s1");
  });

  it("deleting a session calls onDeleteSession without also selecting it", async () => {
    const onSelectSession = vi.fn();
    const onDeleteSession = vi.fn();
    render(
      <Sidebar
        {...baseProps({
          onSelectSession,
          onDeleteSession,
          sessions: [{ session_id: "s1", title: "What is attention?" }]
        })} />

    );
    await userEvent.click(screen.getByLabelText("Delete conversation"));
    expect(onDeleteSession).toHaveBeenCalledWith("s1");
    expect(onSelectSession).not.toHaveBeenCalled();
  });

  it("starting a new chat calls onNewChat", async () => {
    const onNewChat = vi.fn();
    render(<Sidebar {...baseProps({ onNewChat })} />);
    await userEvent.click(screen.getByText("+ New chat"));
    expect(onNewChat).toHaveBeenCalled();
  });
});
