import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "./client";
import { ApiError } from "./types";

function mockFetchOnce(body: unknown, init?: { ok?: boolean; status?: number }) {
  const ok = init?.ok ?? true;
  const status = init?.status ?? 200;
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok,
    status,
    statusText: ok ? "OK" : "Error",
    json: async () => body,
  }) as unknown as typeof fetch;
}

describe("api/client", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("listWorkspaces() calls GET /workspaces", async () => {
    mockFetchOnce([{ workspace_id: "w1", name: "A", created_at: "", updated_at: "", paper_count: 0, chat_count: 0 }]);
    const result = await api.listWorkspaces();
    expect(fetch).toHaveBeenCalledWith(`${api.API_BASE}/workspaces`, expect.objectContaining({}));
    expect(result).toHaveLength(1);
    expect(result[0].workspace_id).toBe("w1");
  });

  it("createWorkspace() POSTs the name as JSON", async () => {
    mockFetchOnce({ workspace_id: "w2", name: "New", created_at: "", updated_at: "", paper_count: 0, chat_count: 0 });
    await api.createWorkspace("New");
    const call = (fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(call[0]).toBe(`${api.API_BASE}/workspaces`);
    expect(call[1].method).toBe("POST");
    expect(JSON.parse(call[1].body)).toEqual({ name: "New" });
  });

  it("listDocuments() includes workspace_id in the query string when provided", async () => {
    mockFetchOnce([]);
    await api.listDocuments("ws-123");
    const call = (fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(call[0]).toBe(`${api.API_BASE}/documents?workspace_id=ws-123`);
  });

  it("listDocuments() omits the query string when no workspace_id is given", async () => {
    mockFetchOnce([]);
    await api.listDocuments();
    const call = (fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(call[0]).toBe(`${api.API_BASE}/documents`);
  });

  it("uploadDocument() sends a real multipart FormData, never JSON content-type", async () => {
    mockFetchOnce({ document_id: "d1", filename: "a.pdf", status: "ok" });
    const file = new File(["dummy pdf bytes"], "a.pdf", { type: "application/pdf" });
    await api.uploadDocument(file, "ws-1");
    const call = (fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(call[0]).toBe(`${api.API_BASE}/upload`);
    expect(call[1].body).toBeInstanceOf(FormData);
    expect(call[1].headers).toBeUndefined();
    const form = call[1].body as FormData;
    expect(form.get("workspace_id")).toBe("ws-1");
    expect((form.get("file") as File).name).toBe("a.pdf");
  });

  it("query() POSTs the full request body including workspace_id/session_id scoping", async () => {
    mockFetchOnce({ answer: "42", sources: [], structured_citations: [], documents_found: 1 });
    await api.query({ question: "What?", document_ids: ["d1"], workspace_id: "w1", session_id: "s1" });
    const call = (fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(call[0]).toBe(`${api.API_BASE}/query`);
    const body = JSON.parse(call[1].body);
    expect(body).toEqual({ question: "What?", document_ids: ["d1"], workspace_id: "w1", session_id: "s1" });
  });

  it("deleteDocument() issues a real DELETE with workspace_id scoping", async () => {
    mockFetchOnce({ status: "ok" });
    await api.deleteDocument("d1", "w1");
    const call = (fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(call[0]).toBe(`${api.API_BASE}/documents/d1?workspace_id=w1`);
    expect(call[1].method).toBe("DELETE");
  });

  it("throws ApiError with the backend's real status and detail message on a non-OK response", async () => {
    mockFetchOnce({ detail: "Document not found" }, { ok: false, status: 404 });
    await expect(api.getDocument("nope")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      message: "Document not found",
    });
  });

  it("ApiError is a real Error subclass usable with instanceof", async () => {
    mockFetchOnce({ detail: "Not found" }, { ok: false, status: 404 });
    try {
      await api.getDocument("nope");
      throw new Error("expected getDocument to throw");
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
      expect(e).toBeInstanceOf(Error);
    }
  });

  it("reportDownloadUrl() builds the correct download URL per format", () => {
    expect(api.reportDownloadUrl("r1", "pdf")).toBe(`${api.API_BASE}/report/r1/pdf`);
    expect(api.reportDownloadUrl("r1", "markdown")).toBe(`${api.API_BASE}/report/r1/markdown`);
    expect(api.reportDownloadUrl("r1", "docx")).toBe(`${api.API_BASE}/report/r1/docx`);
  });
});
