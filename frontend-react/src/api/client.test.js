import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "./client";
import { ApiError } from "./client";

function mockFetchOnce(body, init) {
  const ok = init?.ok ?? true;
  const status = init?.status ?? 200;
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok,
    status,
    statusText: ok ? "OK" : "Error",
    json: async () => body
  });
}

describe("api/client", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    localStorage.clear(); // auth token must never leak across tests
  });
  afterEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
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
    const call = fetch.mock.calls[0];
    expect(call[0]).toBe(`${api.API_BASE}/workspaces`);
    expect(call[1].method).toBe("POST");
    expect(JSON.parse(call[1].body)).toEqual({ name: "New" });
  });

  it("listDocuments() includes workspace_id in the query string when provided", async () => {
    mockFetchOnce([]);
    await api.listDocuments("ws-123");
    const call = fetch.mock.calls[0];
    expect(call[0]).toBe(`${api.API_BASE}/documents?workspace_id=ws-123`);
  });

  it("listDocuments() omits the query string when no workspace_id is given", async () => {
    mockFetchOnce([]);
    await api.listDocuments();
    const call = fetch.mock.calls[0];
    expect(call[0]).toBe(`${api.API_BASE}/documents`);
  });

  it("uploadDocument() sends a real multipart FormData, never JSON content-type", async () => {
    mockFetchOnce({ document_id: "d1", filename: "a.pdf", status: "ok" });
    const file = new File(["dummy pdf bytes"], "a.pdf", { type: "application/pdf" });
    await api.uploadDocument(file, "ws-1");
    const call = fetch.mock.calls[0];
    expect(call[0]).toBe(`${api.API_BASE}/upload`);
    expect(call[1].body).toBeInstanceOf(FormData);
    expect(call[1].headers["Content-Type"]).toBeUndefined(); // browser sets its own multipart boundary
    const form = call[1].body;
    expect(form.get("workspace_id")).toBe("ws-1");
    expect(form.get("file").name).toBe("a.pdf");
  });

  it("query() POSTs the full request body including workspace_id/session_id scoping", async () => {
    mockFetchOnce({ answer: "42", sources: [], structured_citations: [], documents_found: 1 });
    await api.query({ question: "What?", document_ids: ["d1"], workspace_id: "w1", session_id: "s1" });
    const call = fetch.mock.calls[0];
    expect(call[0]).toBe(`${api.API_BASE}/query`);
    const body = JSON.parse(call[1].body);
    expect(body).toEqual({ question: "What?", document_ids: ["d1"], workspace_id: "w1", session_id: "s1" });
  });

  it("deleteDocument() issues a real DELETE with workspace_id scoping", async () => {
    mockFetchOnce({ status: "ok" });
    await api.deleteDocument("d1", "w1");
    const call = fetch.mock.calls[0];
    expect(call[0]).toBe(`${api.API_BASE}/documents/d1?workspace_id=w1`);
    expect(call[1].method).toBe("DELETE");
  });

  it("throws ApiError with the backend's real status and detail message on a non-OK response", async () => {
    mockFetchOnce({ detail: "Document not found" }, { ok: false, status: 404 });
    await expect(api.getDocument("nope")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      message: "Document not found"
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

  // -------------------------------------------------------------------
  // Auth — real backend contract (backend/auth.py, main.py's /auth/*):
  // register/login return {user, token}; the token is persisted and
  // attached as a real Authorization header to every subsequent request.
  // -------------------------------------------------------------------
  it("no Authorization header is sent when there is no stored token", async () => {
    mockFetchOnce([]);
    await api.listWorkspaces();
    const call = fetch.mock.calls[0];
    expect(call[1].headers["Authorization"]).toBeUndefined();
  });

  it("register() POSTs email/password and persists the returned token", async () => {
    mockFetchOnce({ user: { user_id: "u1", email: "a@example.com" }, token: "tok-abc" });
    const user = await api.register("a@example.com", "hunter2hunter2");
    const call = fetch.mock.calls[0];
    expect(call[0]).toBe(`${api.API_BASE}/auth/register`);
    expect(JSON.parse(call[1].body)).toEqual({ email: "a@example.com", password: "hunter2hunter2" });
    expect(user.email).toBe("a@example.com");
    expect(api.getToken()).toBe("tok-abc");
  });

  it("login() persists the returned token, then subsequent requests carry it as a Bearer header", async () => {
    mockFetchOnce({ user: { user_id: "u1", email: "a@example.com" }, token: "tok-xyz" });
    await api.login("a@example.com", "hunter2hunter2");
    expect(api.getToken()).toBe("tok-xyz");

    mockFetchOnce([]);
    await api.listWorkspaces();
    const call = fetch.mock.calls[0];
    expect(call[1].headers["Authorization"]).toBe("Bearer tok-xyz");
  });

  it("logout() clears the stored token even if the network call fails", async () => {
    mockFetchOnce({ user: { user_id: "u1", email: "a@example.com" }, token: "tok-1" });
    await api.login("a@example.com", "hunter2hunter2");
    expect(api.getToken()).toBe("tok-1");

    globalThis.fetch = vi.fn().mockRejectedValue(new Error("network down"));
    await expect(api.logout()).rejects.toThrow();
    expect(api.getToken()).toBeNull();
  });

  it("a 401 from any endpoint clears the token and notifies onUnauthorized listeners", async () => {
    mockFetchOnce({ user: { user_id: "u1", email: "a@example.com" }, token: "tok-1" });
    await api.login("a@example.com", "hunter2hunter2");

    const listener = vi.fn();
    const unsubscribe = api.onUnauthorized(listener);
    mockFetchOnce({ detail: "Invalid or expired session." }, { ok: false, status: 401 });
    await expect(api.listWorkspaces()).rejects.toMatchObject({ status: 401 });

    expect(api.getToken()).toBeNull();
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
  });

  it("a 401 from /auth/login itself does NOT clear an unrelated existing token or fire listeners", async () => {
    // A failed login attempt is not "your session expired" — it's "this
    // login attempt was rejected" — and must not be treated the same way
    // a genuine session-expiry 401 from a protected endpoint is.
    mockFetchOnce({ user: { user_id: "u1", email: "a@example.com" }, token: "tok-1" });
    await api.login("a@example.com", "hunter2hunter2");

    const listener = vi.fn();
    const unsubscribe = api.onUnauthorized(listener);
    mockFetchOnce({ detail: "Incorrect email or password." }, { ok: false, status: 401 });
    await expect(api.login("a@example.com", "wrongpassword")).rejects.toMatchObject({ status: 401 });

    expect(api.getToken()).toBe("tok-1"); // the PRIOR session's token is untouched
    expect(listener).not.toHaveBeenCalled();
    unsubscribe();
  });

  it("getMe() calls GET /auth/me", async () => {
    mockFetchOnce({ user_id: "u1", email: "a@example.com" });
    const me = await api.getMe();
    expect(fetch.mock.calls[0][0]).toBe(`${api.API_BASE}/auth/me`);
    expect(me.email).toBe("a@example.com");
  });

  it("downloadReport() fetches with an Authorization header, then triggers a blob save", async () => {
    mockFetchOnce({ user: { user_id: "u1", email: "a@example.com" }, token: "tok-1" });
    await api.login("a@example.com", "hunter2hunter2");

    const blob = new Blob(["file bytes"], { type: "application/pdf" });
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true, status: 200, blob: async () => blob });
    const createObjectURL = vi.fn().mockReturnValue("blob:fake-url");
    const revokeObjectURL = vi.fn();
    globalThis.URL.createObjectURL = createObjectURL;
    globalThis.URL.revokeObjectURL = revokeObjectURL;

    await api.downloadReport("r1", "pdf", "my-report.pdf");

    const call = fetch.mock.calls[0];
    expect(call[0]).toBe(`${api.API_BASE}/report/r1/pdf`);
    expect(call[1].headers["Authorization"]).toBe("Bearer tok-1");
    expect(createObjectURL).toHaveBeenCalledWith(blob);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:fake-url");
  });
});