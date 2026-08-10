// Typed API client for the VerityRAG FastAPI backend. One function per real
// endpoint (see backend/main.py) — no speculative/unused endpoints, no mock
// data. Every function does a real fetch() against API_BASE.

import {
  ApiError,
  type AnalyzeRequestBody,
  type AnalyzeResponse,
  type ChatMessageRecord,
  type DocumentRecord,
  type EvalDashboard,
  type QueryRequestBody,
  type QueryResponse,
  type ReportRequestBody,
  type ReportResponse,
  type SessionRecord,
  type UploadResponse,
  type Workspace,
} from "./types";

export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8001";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // response wasn't JSON — keep statusText
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// Workspaces
// ---------------------------------------------------------------------------
export function listWorkspaces(): Promise<Workspace[]> {
  return request<Workspace[]>("/workspaces");
}

export function getWorkspace(workspaceId: string): Promise<Workspace> {
  return request<Workspace>(`/workspaces/${workspaceId}`);
}

export function createWorkspace(name: string): Promise<Workspace> {
  return request<Workspace>("/workspaces", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------
export function listDocuments(workspaceId?: string): Promise<DocumentRecord[]> {
  const qs = workspaceId ? `?workspace_id=${encodeURIComponent(workspaceId)}` : "";
  return request<DocumentRecord[]>(`/documents${qs}`);
}

export function getDocument(documentId: string, workspaceId?: string): Promise<DocumentRecord> {
  const qs = workspaceId ? `?workspace_id=${encodeURIComponent(workspaceId)}` : "";
  return request<DocumentRecord>(`/documents/${documentId}${qs}`);
}

export function deleteDocument(documentId: string, workspaceId?: string): Promise<{ status: string }> {
  const qs = workspaceId ? `?workspace_id=${encodeURIComponent(workspaceId)}` : "";
  return request<{ status: string }>(`/documents/${documentId}${qs}`, { method: "DELETE" });
}

export function uploadDocument(file: File, workspaceId?: string): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  if (workspaceId) form.append("workspace_id", workspaceId);
  return request<UploadResponse>("/upload", { method: "POST", body: form });
}

// ---------------------------------------------------------------------------
// Sessions (conversations)
// ---------------------------------------------------------------------------
export function listSessions(workspaceId?: string): Promise<SessionRecord[]> {
  const qs = workspaceId ? `?workspace_id=${encodeURIComponent(workspaceId)}` : "";
  return request<SessionRecord[]>(`/sessions${qs}`);
}

export function createSession(opts: {
  collectionId?: string | null;
  workspaceId?: string | null;
  title?: string | null;
}): Promise<SessionRecord> {
  return request<SessionRecord>("/sessions", {
    method: "POST",
    body: JSON.stringify({
      collection_id: opts.collectionId ?? null,
      workspace_id: opts.workspaceId ?? null,
      title: opts.title ?? null,
    }),
  });
}

export function updateSessionTitle(sessionId: string, title: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/sessions/${sessionId}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export function deleteSession(sessionId: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/sessions/${sessionId}`, { method: "DELETE" });
}

export function getSessionMessages(sessionId: string): Promise<ChatMessageRecord[]> {
  return request<ChatMessageRecord[]>(`/sessions/${sessionId}/messages`);
}

// ---------------------------------------------------------------------------
// Query (normal Q&A)
// ---------------------------------------------------------------------------
export function query(body: QueryRequestBody): Promise<QueryResponse> {
  return request<QueryResponse>("/query", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// ---------------------------------------------------------------------------
// Analyze (Viva/Mock Test/Explain Figure/Recommend/etc.)
// ---------------------------------------------------------------------------
export function analyze(body: AnalyzeRequestBody): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>("/analyze", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// ---------------------------------------------------------------------------
// Reports (comparative reports)
// ---------------------------------------------------------------------------
export function generateReport(body: ReportRequestBody): Promise<ReportResponse> {
  return request<ReportResponse>("/report", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function reportDownloadUrl(reportId: string, format: "markdown" | "pdf" | "docx"): string {
  return `${API_BASE}/report/${reportId}/${format}`;
}

// ---------------------------------------------------------------------------
// Evaluation dashboard
// ---------------------------------------------------------------------------
export function getEvalDashboard(): Promise<EvalDashboard> {
  return request<EvalDashboard>("/eval/dashboard");
}

export { ApiError };
