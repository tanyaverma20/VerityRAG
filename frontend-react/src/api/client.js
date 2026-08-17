// Typed API client for the VerityRAG FastAPI backend. One function per real
// endpoint (see backend/main.py) — no speculative/unused endpoints, no mock
// data. Every function does a real fetch() against API_BASE.

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8001";

// ---------------------------------------------------------------------------
// Auth token — every protected endpoint (everything except /health and
// /auth/register|login) requires a real bearer token (see backend/auth.py,
// main.py's Depends(auth.get_current_user)). Persisted in localStorage so a
// page refresh doesn't log the user out; cleared on logout or a real 401
// from the server (session expired/revoked — see the 401 handling in
// request() below).
const TOKEN_KEY = "verityrag_auth_token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

// Simple pub-sub so useAuth (or anything else) can react the moment the
// server tells us a session is no longer valid — a 401 from ANY endpoint,
// not just /auth/me, means "this token no longer works" (expired, revoked
// via logout elsewhere, or never valid), and every consumer of this client
// should fall back to the logged-out state rather than keep silently
// failing requests.
const unauthorizedListeners = new Set();

export function onUnauthorized(callback) {
  unauthorizedListeners.add(callback);
  return () => unauthorizedListeners.delete(callback);
}

async function request(path, init) {
  const isFormData = init?.body instanceof FormData;
  const headers = { ...(init?.headers ?? {}) };
  if (!isFormData) headers["Content-Type"] = "application/json";
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // response wasn't JSON — keep statusText
    }
    if (res.status === 401 && path !== "/auth/login" && path !== "/auth/register") {
      clearToken();
      for (const cb of unauthorizedListeners) cb();
    }
    throw new ApiError(detail, res.status);
  }
  return await res.json();
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------
export async function register(email, password) {
  const body = await request("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password })
  });
  setToken(body.token);
  return body.user;
}

export async function login(email, password) {
  const body = await request("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password })
  });
  setToken(body.token);
  return body.user;
}

export async function logout() {
  try {
    await request("/auth/logout", { method: "POST" });
  } finally {
    // Always clear the local token, even if the network call itself
    // failed (offline, server down) — the user's intent to log out on
    // THIS device must never be blocked by a network error.
    clearToken();
  }
}

export function getMe() {
  return request("/auth/me");
}

// ---------------------------------------------------------------------------
// Workspaces
// ---------------------------------------------------------------------------
export function listWorkspaces() {
  return request("/workspaces");
}

export function getWorkspace(workspaceId) {
  return request(`/workspaces/${workspaceId}`);
}

export function createWorkspace(name) {
  return request("/workspaces", {
    method: "POST",
    body: JSON.stringify({ name })
  });
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------
export function listDocuments(workspaceId) {
  const qs = workspaceId ? `?workspace_id=${encodeURIComponent(workspaceId)}` : "";
  return request(`/documents${qs}`);
}

export function getDocument(documentId, workspaceId) {
  const qs = workspaceId ? `?workspace_id=${encodeURIComponent(workspaceId)}` : "";
  return request(`/documents/${documentId}${qs}`);
}

export function deleteDocument(documentId, workspaceId) {
  const qs = workspaceId ? `?workspace_id=${encodeURIComponent(workspaceId)}` : "";
  return request(`/documents/${documentId}${qs}`, { method: "DELETE" });
}

export function uploadDocument(file, workspaceId) {
  const form = new FormData();
  form.append("file", file);
  if (workspaceId) form.append("workspace_id", workspaceId);
  return request("/upload", { method: "POST", body: form });
}

// ---------------------------------------------------------------------------
// Sessions (conversations)
// ---------------------------------------------------------------------------
export function listSessions(workspaceId) {
  const qs = workspaceId ? `?workspace_id=${encodeURIComponent(workspaceId)}` : "";
  return request(`/sessions${qs}`);
}

export function createSession(opts) {
  return request("/sessions", {
    method: "POST",
    body: JSON.stringify({
      collection_id: opts.collectionId ?? null,
      workspace_id: opts.workspaceId ?? null,
      title: opts.title ?? null
    })
  });
}

export function updateSessionTitle(sessionId, title) {
  return request(`/sessions/${sessionId}`, {
    method: "PATCH",
    body: JSON.stringify({ title })
  });
}

export function deleteSession(sessionId) {
  return request(`/sessions/${sessionId}`, { method: "DELETE" });
}

export function getSessionMessages(sessionId) {
  return request(`/sessions/${sessionId}/messages`);
}

// ---------------------------------------------------------------------------
// Query (normal Q&A)
// ---------------------------------------------------------------------------
export function query(body) {
  return request("/query", {
    method: "POST",
    body: JSON.stringify(body)
  });
}

// ---------------------------------------------------------------------------
// Analyze (Viva/Mock Test/Explain Figure/Recommend/etc.)
// ---------------------------------------------------------------------------
export function analyze(body) {
  return request("/analyze", {
    method: "POST",
    body: JSON.stringify(body)
  });
}

// ---------------------------------------------------------------------------
// Reports (comparative reports)
// ---------------------------------------------------------------------------
export function generateReport(body) {
  return request("/report", {
    method: "POST",
    body: JSON.stringify(body)
  });
}

export function reportDownloadUrl(reportId, format) {
  return `${API_BASE}/report/${reportId}/${format}`;
}

const REPORT_EXTENSIONS = { markdown: "md", pdf: "pdf", docx: "docx" };

// GET /report/{id}/{format} now requires a real Authorization header (see
// main.py's report-ownership hardening) — a plain <a href> download link
// can't attach custom headers, so downloading now means: fetch the bytes
// ourselves (with the token), turn them into a Blob, and trigger the save
// via a throwaway object URL + anchor click. The visible UI (a "Download
// PDF/Markdown/DOCX" button) stays exactly the same; only how the click is
// handled changes.
export async function downloadReport(reportId, format, filename) {
  const token = getToken();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const res = await fetch(reportDownloadUrl(reportId, format), { headers });
  if (!res.ok) {
    if (res.status === 401) {
      clearToken();
      for (const cb of unauthorizedListeners) cb();
    }
    throw new ApiError(res.statusText, res.status);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename || `report.${REPORT_EXTENSIONS[format] ?? format}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Evaluation dashboard
// ---------------------------------------------------------------------------
export function getEvalDashboard() {
  return request("/eval/dashboard");
}

// ---------------------------------------------------------------------------
// Deep Research (async task) — POST /research kicks off a background task,
// then poll GET /task/{id} until it's COMPLETED/FAILED.
// ---------------------------------------------------------------------------
export function startResearch(body) {
  return request("/research", {
    method: "POST",
    body: JSON.stringify(body)
  });
}

export function getTask(taskId) {
  return request(`/task/${taskId}`);
}

export { ApiError };
