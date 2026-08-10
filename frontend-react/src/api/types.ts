// Typed shapes for the VerityRAG backend API (backend/main.py). Field names
// mirror the real Pydantic models / response dicts exactly — see the
// corresponding endpoint in main.py for the source of truth on each shape.

export interface Workspace {
  workspace_id: string;
  name: string;
  created_at: string;
  updated_at: string;
  paper_count: number;
  chat_count: number;
}

export interface DocumentRecord {
  document_id: string;
  filename: string;
  title: string | null;
  authors: string | null;
  year: number | null;
  page_count: number | null;
  chunk_count: number;
  ingestion_status: "UPLOADED" | "PROCESSING" | "INDEXED" | "FAILED" | string;
  error_message: string | null;
  workspace_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface SessionRecord {
  session_id: string;
  collection_id: string | null;
  workspace_id: string | null;
  title: string | null;
  created_at: string;
  updated_at: string;
}

export interface ChatMessageRecord {
  message_id: string;
  session_id: string;
  role: "user" | "assistant" | string;
  content: string;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

export interface Citation {
  document_id: string;
  chunk_id?: string;
  parent_id?: string;
  page_number?: number | null;
  section?: string;
  text?: string;
  source?: string;
  retrieval_method?: string;
  rerank_score?: number;
  rrf_score?: number;
}

export interface ContributingDocument {
  document_id: string;
}

export interface RetrievalDetails {
  candidates_retrieved: number;
  reranked_to: number;
}

export interface QueryResponse {
  answer: string;
  sources: Citation[];
  structured_citations: Citation[];
  documents_found: number | ContributingDocument[];
  documents_used?: string[];
  retrieval_details?: RetrievalDetails;
  structured?: Record<string, unknown> | null;
  claim_evidence_trace?: unknown[];
  verification?: unknown[];
  research_type?: string;
  research_iterations?: number;
  evidence_gaps?: unknown[];
  similarities?: unknown[];
  differences?: unknown[];
  contradictions?: unknown[];
  research_gaps?: unknown[];
  claims?: unknown[];
  confidence?: "HIGH" | "MEDIUM" | "LOW" | "UNAVAILABLE" | string;
  status?: string;
  latency_s?: number;
  research_plan?: Record<string, unknown>;
}

export interface QueryRequestBody {
  question: string;
  top_k?: number;
  document_ids?: string[] | null;
  collection_id?: string | null;
  workspace_id?: string | null;
  strategy?: string;
  research_type?: "simple" | "deep";
  session_id?: string | null;
  mode?: "normal" | "comparison" | "structured";
}

export interface AnalyzeRequestBody {
  mode: string;
  document_ids?: string[] | null;
  workspace_id?: string | null;
  difficulty?: "basic" | "intermediate" | "advanced";
  topics?: string[] | null;
  num_questions?: number;
  figure_reference?: string | null;
  question?: string | null;
  user_answer?: string | null;
  session_id?: string | null;
}

export interface AnalyzeResponse {
  ok: boolean;
  mode: string;
  error?: string;
  answer?: string;
  [key: string]: unknown;
}

export interface ReportRequestBody {
  document_ids: string[];
  workspace_id?: string | null;
  session_id?: string | null;
}

export interface PaperReport {
  document_id: string;
  title: string;
  [key: string]: unknown;
}

export interface ComparisonReport {
  [key: string]: unknown;
}

export interface ReportResponse {
  ok: boolean;
  error?: string;
  report_id: string | null;
  title?: string;
  overview?: string;
  papers?: PaperReport[];
  comparison?: ComparisonReport | null;
  conclusion?: string;
  evidence_sufficient?: boolean;
  documents_found?: number;
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  status: string;
  chunks_added?: number;
  page_count?: number;
  [key: string]: unknown;
}

export interface EvalDashboard {
  live: Record<string, unknown> | null;
  live_note: string;
  cache: Record<string, unknown> | null;
  offline_retrieval_eval: Record<string, unknown> | null;
  offline_benchmark_10_document: Record<string, unknown> | null;
  offline_groundedness_eval: Record<string, unknown> | null;
  not_measured: string[];
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}
