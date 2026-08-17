import re as _re
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import uuid
import time

import auth

from ingest import ingest_document, get_collection, delete_document_from_index, backfill_orphaned_chunk_workspace_ids
from retrieval import retrieve, retrieve_multi, hybrid_retrieve, build_bm25_index, get_contributing_documents, get_retrieval_stats
from query_transform import decompose_query_deterministic
from analysis import (
    generate_pdf_questions, explain_figure, recommend_from_comparison,
    answer_design_question, start_project_interview, evaluate_interview_answer,
    evaluate_paper, find_research_gaps, build_literature_matrix, build_knowledge_graph,
)
from doc_titles import resolve_display_title
from figure_vision import persist_uploaded_pdf, get_uploaded_pdf_path, render_page_as_image_base64, vision_model_available, VISION_MODEL
from verify import verify_answer
from llm import call_llm
from database import (
    list_documents, list_documents_for_owner, get_document,
    list_collections, get_collection as db_get_collection, create_collection,
    add_document_to_collection, remove_document_from_collection, delete_collection,
    create_session, get_session, list_sessions, list_sessions_for_owner, delete_session, update_session_collection,
    add_message, get_session_messages,
    create_task, update_task_status, get_task, task_belongs_to_workspace,
    create_workspace, get_workspace, list_workspaces, documents_in_workspace,
    update_session_title, workspace_owner_id,
)
from observability import log_query_event, filter_events, read_recent_events
from query_transform import (
    NO_DOCUMENTS_MESSAGE, INSUFFICIENT_EVIDENCE_MESSAGE, RATE_LIMIT_MESSAGE,
    GENERATION_FAILED_MESSAGE, is_rate_limit_error,
    start_call_tracking, get_call_log,
)
from config import GROQ_MODEL, GROQ_FALLBACK_MODEL, REPORT_CHUNKS_PER_DOC, MAX_UPLOAD_BYTES, CORS_ALLOWED_ORIGINS
import cache
from schemas import ResearchReport, PaperReport, ComparisonReport
from report_generator import generate_report, render_report_markdown, render_report_pdf, render_report_docx
from query_transform import get_token_totals


def _new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def _call_observability_fields(request_id: str = "", active_document_count: int = 0, cache_hit: bool = False) -> dict:
    """
    Reads the physical Groq call log recorded during this request (see
    query_transform.start_call_tracking/get_call_log) and shapes it for
    log_query_event(). Internal/developer-log only — never returned to the
    frontend (item 15/17).

    llm_calls/fallback_triggered/model_name go through log_query_event's own
    named parameters (existing schema); the rest ride in `extra` rather than
    widening that function's signature further for fields that are genuinely
    supplementary diagnostics.
    """
    log = get_call_log()
    tokens = get_token_totals()
    # Real reranking-stage latency for THIS request's most recent
    # retrieve()/retrieve_multi() call (retrieval.py:get_retrieval_stats) —
    # None if no retrieval ran (e.g. a non-PDF-grounded /analyze mode).
    retrieval_stats = get_retrieval_stats()
    return {
        "llm_calls": len(log),
        "fallback_triggered": any(c["role"] == "fallback" for c in log),
        "model_name": GROQ_MODEL,
        "extra": {
            "request_id": request_id,
            "active_document_count": active_document_count,
            "fallback_model": GROQ_FALLBACK_MODEL,
            "cache_hit": cache_hit,
            "input_tokens": tokens["input_tokens"],
            "output_tokens": tokens["output_tokens"],
            "total_tokens": tokens["total_tokens"],
            "rerank_latency_s": retrieval_stats.get("rerank_latency_s"),
            "candidates_retrieved": retrieval_stats.get("candidates_retrieved"),
        },
    }


_UNCACHEABLE_ANSWERS = {RATE_LIMIT_MESSAGE, GENERATION_FAILED_MESSAGE}


def _is_cacheable_answer(answer_text: str) -> bool:
    """Never cache a transient failure — a retry a minute later should get a
    real attempt, not the same stale error forever (item 13)."""
    return bool(answer_text) and answer_text not in _UNCACHEABLE_ANSWERS


def _resolve_document_scope(document_ids: list[str] | None, collection_id: str | None, workspace_id: str | None = None) -> list[str]:
    if collection_id and not document_ids:
        col = db_get_collection(collection_id)
        if col:
            document_ids = col.get("document_ids", [])
    resolved = document_ids or []
    if workspace_id:
        # Data-layer isolation (not just UI): whatever document_ids the
        # caller asked for, only the ones that actually belong to this
        # workspace per SQLite survive. A stale/tampered client list can
        # never pull another workspace's evidence into retrieval.
        resolved = documents_in_workspace(resolved, workspace_id)
    return resolved


def _require_workspace_owner(workspace_id: str | None, user: dict) -> str:
    """The real authorization check ("Do NOT use workspace_id supplied by
    the client as the security principal"): a workspace_id is never
    trusted on its own — it must actually belong (Workspace.owner_user_id)
    to the authenticated user making the request, verified against the
    database on every call. Fails closed with 404 (never 403) so a
    mismatched/foreign/nonexistent workspace_id all look identical to the
    caller — matching every other ownership check already in this API
    (_reject_if_foreign_workspace, task_belongs_to_workspace, ...). A
    workspace with no owner at all (owner_user_id is NULL — legacy,
    created before authentication existed and not yet claimed) is never
    accessible to any authenticated user; only the very first account ever
    registered on a deployment auto-claims those once, at registration
    time (see auth.register_user / bootstrap_claim_orphaned_workspaces).
    Returns workspace_id unchanged so this can be used inline."""
    if not workspace_id:
        raise HTTPException(400, "workspace_id is required.")
    owner = workspace_owner_id(workspace_id)
    if owner is None or owner != user["user_id"]:
        raise HTTPException(404, "Workspace not found")
    return workspace_id


def _require_resource_owner(resource_workspace_id: str | None, user: dict, not_found_message: str) -> None:
    """Same ownership policy as _require_workspace_owner(), applied to a
    resource (document/session/task/report) that already carries its OWN
    workspace_id rather than accepting one from the client. A
    workspace-owned resource is only reachable by that workspace's real
    owner. A workspace-less resource (legacy data created before workspace
    support existed at all) has no ownership to enforce and stays
    reachable to any authenticated user, unchanged — the same
    backward-compatible pattern this API already uses everywhere a
    workspace_id check is optional-by-presence."""
    if not resource_workspace_id:
        return
    owner = workspace_owner_id(resource_workspace_id)
    if owner is None or owner != user["user_id"]:
        raise HTTPException(404, not_found_message)


def _safe_filename(title: str) -> str:
    s = _re.sub(r'[^A-Za-z0-9_\- ]', '', title or "report").strip().replace(" ", "_")
    return s[:60] or "report"


def _stored_to_research_report(stored: dict) -> ResearchReport:
    return ResearchReport(
        title=stored["title"], overview=stored["overview"],
        papers=[PaperReport(**p) for p in stored.get("papers", [])],
        comparison=ComparisonReport(**stored["comparison"]) if stored.get("comparison") else None,
        conclusion=stored["conclusion"], evidence_sufficient=stored.get("evidence_sufficient", True),
    )


def _persist_analysis_turn(session_id: str | None, user_text: str | None, frontend_role: str, content: str, metadata: dict) -> None:
    """
    Persists one analysis-mode turn (Viva, Mock Test, Project Interview,
    Compare, Explain Figure, Recommend, Evaluate Paper, Research Gaps,
    Literature Matrix, Knowledge Graph, ...) using the EXACT SAME
    conversation/session mechanism normal Q&A already uses (database.py's
    add_message / the sessions+messages tables) — no parallel storage, no
    new tables. `metadata["frontend_role"]` tells the frontend which
    message-rendering branch to reconstruct on reload (see
    ensureMessagesLoaded() in index.html), the same way `structured_data`/
    `structured_citations` already ride in metadata for normal answers.
    Never adds an LLM call — this is a pure DB write after generation.
    """
    if not session_id:
        return
    try:
        if user_text:
            add_message(uuid.uuid4().hex[:12], session_id, "user", user_text)
        add_message(uuid.uuid4().hex[:12], session_id, "assistant", content, metadata={"frontend_role": frontend_role, **metadata})
    except Exception as e:
        print(f"Failed to persist analysis turn: {e}")  # history is best-effort, never blocks the response


def _retrieve_for_analysis(query_text: str, document_ids: list[str], top_k: int, workspace_id: str | None = None) -> list[dict]:
    """Shared retrieval path for /analyze's PDF-grounded modes: same
    deterministic decomposition + hybrid retrieval + RRF + rerank + token
    budget as normal queries (retrieval.py) — no LLM calls, no new pipeline.
    workspace_id is an optional second, independent scope-enforcement point
    at the vector layer (see retrieval.py:_scope_where_clause) — additive on
    top of the document_ids already resolved via _resolve_document_scope."""
    sub_queries = decompose_query_deterministic(query_text)
    if len(sub_queries) > 1:
        return retrieve_multi(sub_queries, document_ids=document_ids, top_k=top_k, rerank_query=query_text, workspace_id=workspace_id)
    return retrieve(query_text, strategy="hybrid", document_ids=document_ids, top_k=top_k, workspace_id=workspace_id)


def _chunk_observability_fields(chunks: list[dict]) -> dict:
    """Real values for log_query_event()'s retrieved_chunk_count /
    estimated_context_tokens fields — previously a real, silent gap found
    during the Phase 9 observability audit: every call site in this file
    passed selected_evidence_count=len(chunks) but never these two
    top-level schema fields, so they sat permanently at their 0 default
    across ALL of main.py's real production events, misleadingly looking
    like a genuine "zero chunks retrieved" measurement even when evidence
    clearly existed (selected_evidence_count > 0 in the same event).
    retrieved_chunk_count == len(chunks) here (not a separate, fabricated
    number) because at this call layer retrieve()/retrieve_multi() already
    performs rerank + token-budget selection internally before returning —
    there's no separate "raw pre-selection" count retained at this point
    beyond what extra.candidates_retrieved (retrieval.py's own rerank-stage
    stats) already reports. estimated_context_tokens reuses the exact same
    token estimator retrieval.py's own budgeting already uses."""
    from retrieval import _estimate_tokens
    return {
        "retrieved_chunk_count": len(chunks),
        "estimated_context_tokens": _estimate_tokens(_evidence_text_from_chunks(chunks)) if chunks else 0,
    }


def _evidence_text_from_chunks(chunks: list[dict]) -> str:
    """Same evidence formatting as graph/synthesizer.py's ONE synthesis
    call, reused here so /analyze's prompts see evidence in the same
    document-grouped, parent-context-expanded shape."""
    from graph.organizer import organize_evidence
    organized = organize_evidence({"retrieval_results": chunks}).get("evidence_by_document", {})
    parts = []
    for doc_id, parents in organized.items():
        parts.append(f"\n--- Document ID: {doc_id} ---")
        for p in parents:
            parts.append(f"CONTEXT: {p['parent_context']}")
            for child in p["children"]:
                parts.append(f"  [EXACT CHUNK: {child['chunk_id']}] {child['text']}")
    return "\n".join(parts)


app = FastAPI(title="VerityRAG API")

# Security audit finding: previously allow_origins=["*"] — wide open.
# Now real authenticated requests carry a bearer token, so an
# unrestricted origin list would let any website's JavaScript make
# credentialed-looking requests on a logged-in user's behalf. Configurable
# via CORS_ALLOWED_ORIGINS (see config.py) so a real deployment sets it to
# its actual frontend origin(s); the dev default is a safe, specific list,
# never "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANSWER_PROMPT = """Answer the question using ONLY the source passages below.
If the passages don't contain enough information, say so explicitly.
Cite which passage number(s) support each part of your answer.

QUESTION: {question}

SOURCE PASSAGES:
{sources}
"""


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    # Restrict retrieval to specific document IDs
    document_ids: list[str] | None = None
    # Or restrict to a collection
    collection_id: str | None = None
    # Optional: when set, document_ids is filtered server-side to only the
    # IDs that actually belong to this workspace (see _resolve_document_scope)
    # — data-layer isolation, not just the frontend never showing other papers.
    workspace_id: str | None = None
    strategy: str = "hybrid"
    research_type: str = "simple"  # "simple" or "deep"
    session_id: str | None = None
    # "normal" | "comparison" (both go through the same scoped-retrieval path —
    # comparison is just "more than one document_id" and needs no special
    # handling) | "structured" (Feature 6: same ONE call, richer JSON — see
    # graph/synthesizer.py's structured_mode).
    mode: str = "normal"

class SessionCreateRequest(BaseModel):
    collection_id: str | None = None
    workspace_id: str | None = None
    title: str | None = None

class SessionUpdateRequest(BaseModel):
    title: str

class WorkspaceCreateRequest(BaseModel):
    name: str

class ResearchRequest(BaseModel):
    question: str
    session_id: str | None = None
    collection_id: str | None = None
    document_ids: list[str] | None = None
    research_type: str = "deep"
    workspace_id: str | None = None

class CollectionCreateRequest(BaseModel):
    name: str
    description: str | None = None
    document_ids: list[str] = []

class ReportRequest(BaseModel):
    document_ids: list[str]
    workspace_id: str | None = None
    session_id: str | None = None   # when set, persists a history record the same way /query does

class AuthRegisterRequest(BaseModel):
    email: str
    password: str

class AuthLoginRequest(BaseModel):
    email: str
    password: str

class AnalyzeRequest(BaseModel):
    # "viva" | "mock_test" | "explain_figure" | "recommend" (PDF-grounded) |
    # "why_design" | "system_design" | "project_interview_start" |
    # "project_interview_evaluate" (grounded in the real architecture, not a PDF)
    mode: str
    document_ids: list[str] | None = None
    workspace_id: str | None = None
    difficulty: str = "intermediate"      # basic | intermediate | advanced
    topics: list[str] | None = None       # focus categories (project interview / why-design)
    num_questions: int = 5
    figure_reference: str | None = None   # e.g. "Figure 2", "Table 3", free text
    question: str | None = None           # design question, or the interview question being answered
    user_answer: str | None = None        # project_interview_evaluate only
    session_id: str | None = None         # when set, persists this turn into conversation history


@app.on_event("startup")
def startup():
    build_bm25_index()
    # Chroma-side counterpart of db.repository._migrate_legacy_columns()'s
    # Postgres backfill — see backfill_orphaned_chunk_workspace_ids()'s
    # docstring. Best-effort: must never block startup.
    try:
        updated = backfill_orphaned_chunk_workspace_ids()
        if updated:
            print(f"[startup] backfilled workspace_id on {updated} legacy Chroma chunk(s)")
    except Exception as e:
        print(f"[startup] chunk workspace_id backfill check skipped: {e}")


@app.get("/health")
def health():
    collection = get_collection()
    return {"status": "ok", "chunks_indexed": collection.count()}


# --- Auth Endpoints ---
# Real registration/login/logout/me, backed entirely by auth.py + database.py
# (bcrypt-hashed passwords, opaque SHA-256-hashed bearer session tokens —
# see auth.py's module docstring for the full design rationale). Every other
# endpoint's `Depends(auth.get_current_user)` usage (wired in as ownership
# enforcement lands, see _require_workspace_owner()) traces back to these.

# Logout intentionally does NOT require Depends(auth.get_current_user) —
# logging out with an already-invalid/expired/unknown token must be a safe
# no-op (see test_auth.py::test_logout_of_an_unknown_token_does_not_raise),
# not a 401, so it uses its own permissive bearer extractor instead.
_logout_bearer = HTTPBearer(auto_error=False)


@app.post("/auth/register", status_code=201)
def api_register(req: AuthRegisterRequest):
    try:
        user, token = auth.register_user(req.email, req.password)
    except auth.AuthError as e:
        status = 409 if "already exists" in e.message else 400
        raise HTTPException(status, e.message)
    return {"user": user, "token": token}


@app.post("/auth/login")
def api_login(req: AuthLoginRequest):
    try:
        user, token = auth.login_user(req.email, req.password)
    except auth.AuthError as e:
        raise HTTPException(401, e.message)
    return {"user": user, "token": token}


@app.post("/auth/logout")
def api_logout(creds: HTTPAuthorizationCredentials | None = Depends(_logout_bearer)):
    if creds and creds.credentials:
        auth.logout_user(creds.credentials)
    return {"status": "ok"}


@app.get("/auth/me")
def api_me(user: dict = Depends(auth.get_current_user)):
    return user


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), workspace_id: str | None = Form(None), user: dict = Depends(auth.get_current_user)):
    # Real ownership check FIRST, before any file I/O — an unauthorized
    # upload attempt must never even reach disk.
    _require_workspace_owner(workspace_id, user)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported right now.")

    # Security audit finding (Phase 13): stream to the temp file in bounded
    # chunks, counting real bytes read — never trust a client-supplied
    # Content-Length header, and never buffer an unbounded upload into
    # memory or disk before checking. Aborts (and cleans up the partial
    # temp file) the moment the real byte count exceeds MAX_UPLOAD_BYTES,
    # rather than after the whole file has already been written.
    total_bytes = 0
    chunk_size = 1024 * 1024
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                tmp.close()
                Path(tmp_path).unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
            tmp.write(chunk)

    # Security audit finding (Phase 13): a client can name any file
    # `something.pdf` regardless of its real content — verify the actual
    # magic bytes before handing it to the PDF-parsing pipeline, rather
    # than relying on the filename extension alone.
    with open(tmp_path, "rb") as f:
        header = f.read(5)
    if header != b"%PDF-":
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(400, "File does not look like a real PDF (missing %PDF- header).")

    try:
        result = ingest_document(tmp_path, original_filename=file.filename, workspace_id=workspace_id)
        build_bm25_index()  # rebuild so the new doc is searchable immediately
        cache.clear_all()   # the workspace changed — stale cached answers/reports must not survive it
        # Persist the original PDF bytes (data/uploads/<document_id>.pdf) so
        # Explain Figure can later render an actual page image for it (see
        # figure_vision.py). Best-effort only — never fails the upload
        # itself if disk I/O has a problem, since text-based ingestion
        # already succeeded and is what actually matters.
        if result.get("status") == "ok" and result.get("document_id"):
            try:
                persist_uploaded_pdf(tmp_path, result["document_id"])
            except Exception as e:
                print(f"Failed to persist original PDF for visual figure rendering: {e}")
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)

# --- Workspace Endpoints ---
# Every workspace endpoint requires a real authenticated user
# (Depends(auth.get_current_user)) — a workspace_id alone is never a
# security principal (see _require_workspace_owner()'s docstring).

@app.post("/workspaces")
def api_create_workspace(req: WorkspaceCreateRequest, user: dict = Depends(auth.get_current_user)):
    name = (req.name or "").strip() or "Untitled Workspace"
    ws_id = uuid.uuid4().hex[:12]
    return create_workspace(ws_id, name, owner_user_id=user["user_id"])

@app.get("/workspaces")
def api_list_workspaces(user: dict = Depends(auth.get_current_user)):
    return list_workspaces(owner_user_id=user["user_id"])

@app.get("/workspaces/{workspace_id}")
def api_get_workspace(workspace_id: str, user: dict = Depends(auth.get_current_user)):
    _require_workspace_owner(workspace_id, user)
    return get_workspace(workspace_id)

# --- Document Endpoints ---

@app.get("/documents")
def api_list_documents(workspace_id: str | None = None, user: dict = Depends(auth.get_current_user)):
    if workspace_id:
        _require_workspace_owner(workspace_id, user)
        return list_documents(workspace_id)
    # No workspace_id given: the real authorization boundary is now every
    # workspace this user actually owns (never "every document in the
    # database" — see list_documents_for_owner()'s docstring for the
    # cross-tenant leak this closes).
    return list_documents_for_owner(user["user_id"])


@app.get("/documents/{doc_id}")
def api_get_document(doc_id: str, user: dict = Depends(auth.get_current_user)):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    # document_id is a deterministic content hash — a caller with an
    # identical file (e.g. a well-known public paper) can compute another
    # workspace's document_id without ever having uploaded it themselves,
    # so real ownership (not the client's say-so) is what's checked here.
    _require_resource_owner(doc.get("workspace_id"), user, "Document not found")
    return doc

@app.delete("/documents/{doc_id}")
def api_delete_document(doc_id: str, user: dict = Depends(auth.get_current_user)):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    _require_resource_owner(doc.get("workspace_id"), user, "Document not found")
    delete_document_from_index(doc_id)
    build_bm25_index()
    cache.clear_all()   # removed doc must immediately stop participating — no stale cached answer/report can survive it
    return {"status": "ok", "deleted": doc_id}

# --- Collection Endpoints ---

@app.get("/collections")
def api_list_collections():
    return list_collections()

@app.get("/collections/{collection_id}")
def api_get_collection(collection_id: str):
    col = db_get_collection(collection_id)
    if not col:
        raise HTTPException(404, "Collection not found")
    return col

@app.post("/collections")
def api_create_collection(req: CollectionCreateRequest):
    import uuid
    col_id = uuid.uuid4().hex[:12]
    create_collection(col_id, req.name, req.description)
    for doc_id in req.document_ids:
        add_document_to_collection(col_id, doc_id)
    return db_get_collection(col_id)

@app.delete("/collections/{collection_id}")
def api_delete_collection(collection_id: str):
    delete_collection(collection_id)
    return {"status": "ok"}

# --- Session Endpoints ---

@app.post("/sessions")
def api_create_session(req: SessionCreateRequest, user: dict = Depends(auth.get_current_user)):
    # workspace_id stays optional here (a session can be a scratch/no-
    # workspace conversation, unchanged pre-existing behavior) — but if one
    # IS given, it must really belong to this user.
    if req.workspace_id:
        _require_workspace_owner(req.workspace_id, user)
    sess_id = uuid.uuid4().hex[:12]
    return create_session(sess_id, req.collection_id, workspace_id=req.workspace_id, title=req.title)

@app.get("/sessions")
def api_list_sessions(workspace_id: str | None = None, user: dict = Depends(auth.get_current_user)):
    if workspace_id:
        _require_workspace_owner(workspace_id, user)
        return list_sessions(workspace_id)
    return list_sessions_for_owner(user["user_id"])

@app.patch("/sessions/{session_id}")
def api_update_session(session_id: str, req: SessionUpdateRequest, user: dict = Depends(auth.get_current_user)):
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    _require_resource_owner(sess.get("workspace_id"), user, "Session not found")
    update_session_title(session_id, req.title)
    return get_session(session_id)

@app.get("/sessions/{session_id}/messages")
def api_get_session_messages(session_id: str, user: dict = Depends(auth.get_current_user)):
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    _require_resource_owner(sess.get("workspace_id"), user, "Session not found")
    return get_session_messages(session_id)

@app.delete("/sessions/{session_id}")
def api_delete_session(session_id: str, user: dict = Depends(auth.get_current_user)):
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    _require_resource_owner(sess.get("workspace_id"), user, "Session not found")
    delete_session(session_id)
    return {"status": "ok"}

# --- Task Endpoints ---

@app.get("/task/{task_id}")
def api_get_task(task_id: str, user: dict = Depends(auth.get_current_user)):
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    # Isolation check (Phase 4 audit, hardened): a task created under one
    # workspace must not be readable by any other user, regardless of what
    # workspace_id a request claims — real ownership (task.workspace_id's
    # real owner_user_id), not a client-supplied value, is what's checked.
    # A task created with no workspace_id at all (legacy/no-workspace
    # callers) has no ownership to enforce and stays reachable — unchanged
    # pre-existing behavior. 404, not 403, so a mismatched request can't
    # distinguish "wrong owner" from "task doesn't exist".
    _require_resource_owner(task.get("workspace_id"), user, "Task not found")
    return task

# --- Observability Endpoints ---

@app.get("/logs")
def api_get_logs(
    task_id: str | None = None,
    session_id: str | None = None,
    event_type: str | None = None,
    limit: int = 50,
    user: dict = Depends(auth.get_current_user),
):
    # Security audit finding: observability events can include real query
    # text — gated behind authentication, and when a task_id/session_id
    # filter is given, real ownership of that specific resource is
    # required (same 404-not-403 pattern as the rest of this API) so one
    # user can't read another user's query history by guessing/enumerating
    # a task_id or session_id.
    if task_id:
        task = get_task(task_id)
        if task:
            _require_resource_owner(task.get("workspace_id"), user, "Task not found")
    if session_id:
        sess = get_session(session_id)
        if sess:
            _require_resource_owner(sess.get("workspace_id"), user, "Session not found")
    logs = filter_events(task_id=task_id, session_id=session_id, event_type=event_type, n=limit)
    return logs

def _execute_research_graph(question: str, session_id: str | None, collection_id: str | None, document_ids: list[str] | None, research_type: str, task_id: str | None = None, structured_mode: bool = False, workspace_id: str | None = None):
    from graph.workflow import research_app

    # Fresh per-request LLM call log (item 17). Call sites read it back via
    # get_call_log()/_call_observability_fields() after this function returns.
    start_call_tracking()

    target_document_ids = _resolve_document_scope(document_ids, collection_id, workspace_id)

    if not target_document_ids:
        # Grounding safety: never fall back to an unscoped search over the
        # entire Chroma corpus (which includes canonical/test papers that are
        # not part of the user's active workspace). No documents in scope
        # means there is nothing to answer from — say so, skip the LLM/graph
        # entirely (also saves tokens), and return a state-shaped dict so
        # every caller (sync /query and async /research) can use it as-is.
        return {
            "draft_answer": NO_DOCUMENTS_MESSAGE,
            "citations": [],
            "retrieval_results": [],
            "verification_results": [],
            "research_type": research_type,
            "research_iterations": 0,
            "evidence_gaps": [],
            "similarities": [],
            "differences": [],
            "contradictions": [],
            "research_gaps": [],
            "claims": [],
            "confidence": "UNAVAILABLE",
            "status": "NO_DOCUMENTS",
            "research_plan": {},
        }

    chat_history = []
    if session_id:
        chat_history = get_session_messages(session_id)
        
    initial_state = {
        "original_query": question,
        "document_ids": target_document_ids,
        "research_type": research_type,
        "session_id": session_id or "none",
        "task_id": task_id or "none",
        "chat_history": chat_history,
        "structured_mode": structured_mode,
        "workspace_id": workspace_id or "",
        # Real, reproduced bug this session: every field below is a
        # ResearchState/TypedDict schema field (graph/state.py) that no
        # node writes before analyzer.py's generate_adaptive_queries()
        # first reads it on a "deep" research run. LangGraph pre-fills
        # every schema-declared key not present in THIS initial dict as
        # None (not "absent") until some node's return value first sets
        # it — so downstream `state.get(key, default)` calls silently got
        # None instead of `default`, since dict.get()'s default only
        # applies when the key is truly absent. Confirmed live: this
        # crashed every real Deep Research task with
        # "'NoneType' object has no attribute 'append'"
        # (adaptive_queries) immediately followed by
        # "unsupported operand type(s) for +: 'NoneType' and 'int'"
        # (research_iterations) once the first bug was fixed. Seeding
        # real empty/zero defaults here — matching each field's declared
        # type — is the root-cause fix, not a per-call-site patch.
        "research_iterations": 1,
        "sub_queries": [],
        "completed_sub_queries": [],
        "evidence_gaps": [],
        "adaptive_queries": [],
        "contradictions": [],
        "similarities": [],
        "differences": [],
        "research_gaps": [],
        "retrieval_results": [],
        "evidence_by_document": {},
        "verification_results": [],
        "claims": [],
        "citations": [],
        "retry_count": 0,
    }
    
    if task_id:
        update_task_status(task_id, "RUNNING")
        
    try:
        final_state = research_app.invoke(initial_state)
    except Exception as e:
        print(f"Graph execution failed: {e}")
        if task_id:
            update_task_status(task_id, "FAILED", error_message=str(e))
        raise e
        
    return final_state

def async_research_worker(task_id: str, request: ResearchRequest):
    request_id = _new_request_id()
    try:
        start = time.time()
        final_state = _execute_research_graph(
            question=request.question,
            session_id=request.session_id,
            collection_id=request.collection_id,
            document_ids=request.document_ids,
            research_type=request.research_type,
            task_id=task_id,
            workspace_id=request.workspace_id,
        )
        
        chunks = final_state.get("retrieval_results", [])
        
        payload = {
            "answer": final_state.get("draft_answer", "Error generating answer."),
            "sources": final_state.get("citations", []),
            "structured_citations": final_state.get("citations", []),
            "documents_found": get_contributing_documents(chunks),
            "verification": final_state.get("verification_results", []),
            "research_type": final_state.get("research_type", request.research_type),
            "research_iterations": final_state.get("research_iterations", 1),
            "evidence_gaps": final_state.get("evidence_gaps", []),
            "similarities": final_state.get("similarities", []),
            "differences": final_state.get("differences", []),
            "contradictions": final_state.get("contradictions", []),
            "research_gaps": final_state.get("research_gaps", []),
            "claims": final_state.get("claims", []),
            "confidence": final_state.get("confidence", "UNAVAILABLE"),
            "status": final_state.get("status", "OK"),
            "latency_s": round(time.time() - start, 2),
            "research_plan": final_state.get("research_plan", {}),
        }
        
        if request.session_id:
            add_message(uuid.uuid4().hex[:12], request.session_id, "user", request.question)
            add_message(uuid.uuid4().hex[:12], request.session_id, "assistant", payload["answer"], metadata=payload)
            
        update_task_status(task_id, "COMPLETED", result_payload=payload)

        log_query_event(
            query=request.question,
            research_type=request.research_type,
            session_id=request.session_id,
            task_id=task_id,
            collection_id=request.collection_id,
            documents_searched=len(request.document_ids or []),
            documents_contributing=len(payload.get("documents_found") or []),
            selected_evidence_count=len(chunks),
            **_chunk_observability_fields(chunks),
            retrieval_latency_s=payload.get("latency_s", 0),
            total_latency_s=payload.get("latency_s", 0),
            research_iterations=payload.get("research_iterations", 1),
            evidence_gaps=len(payload.get("evidence_gaps", [])),
            contradiction_count=len(payload.get("contradictions", [])),
            research_gap_count=len(payload.get("research_gaps", [])),
            confidence=payload.get("confidence", "UNAVAILABLE"),
            task_status="COMPLETED",
            **_call_observability_fields(request_id, len(request.document_ids or [])),
        )
        
    except Exception as e:
        update_task_status(task_id, "FAILED", error_message=str(e))
        log_query_event(
            query=request.question,
            research_type=request.research_type,
            session_id=request.session_id,
            task_id=task_id,
            collection_id=request.collection_id,
            task_status="FAILED",
            errors=[str(e)]
        )

@app.post("/research")
def api_research(request: ResearchRequest, background_tasks: BackgroundTasks, user: dict = Depends(auth.get_current_user)):
    _require_workspace_owner(request.workspace_id, user)
    task_id = uuid.uuid4().hex[:12]
    create_task(task_id, request.session_id, workspace_id=request.workspace_id)
    background_tasks.add_task(async_research_worker, task_id, request)
    return {"task_id": task_id, "status": "PENDING"}
@app.post("/query")
def query(request: QueryRequest, user: dict = Depends(auth.get_current_user)):
    import time
    from graph.workflow import research_app

    # Real ownership check ("Do NOT use workspace_id supplied by the
    # client as the security principal"): workspace_id is now REQUIRED and
    # verified against the authenticated user before any retrieval
    # happens — closes the pre-existing gap where a caller could pass
    # someone else's document_ids directly (a deterministic content hash,
    # guessable/reusable across workspaces) and bypass workspace scoping
    # entirely by simply omitting workspace_id.
    _require_workspace_owner(request.workspace_id, user)

    start = time.time()
    request_id = _new_request_id()

    target_document_ids = _resolve_document_scope(request.document_ids, request.collection_id, request.workspace_id)
    active_document_count = len(target_document_ids)

    # Grounding safety: no documents scoped to this request means there is
    # nothing to search. Short-circuit before touching the graph/LLM at all —
    # both correct (never search the whole canonical corpus by accident) and
    # cheaper (no wasted retrieval or Groq calls).
    if not target_document_ids:
        return {
            "answer": NO_DOCUMENTS_MESSAGE,
            "sources": [],
            "structured_citations": [],
            "documents_found": 0,
            "verification": [],
        }

    # Chat history for THIS conversation, if any — resolved once here so it
    # can both (a) key the cache correctly (the same question can need a
    # different answer depending on prior turns — see cache._history_fingerprint)
    # and (b) be reused by the graph without a second lookup.
    session_id = getattr(request, "session_id", None)
    chat_history = get_session_messages(session_id) if session_id else []

    # Cache (item 13): identical question + identical active document set +
    # identical conversation context + identical retrieval config => zero LLM
    # calls, zero retrieval. Any upload/delete clears this entirely (see
    # cache.clear_all() call sites), so a stale answer can never survive a
    # workspace change.
    cached = cache.get_cached_answer(request.question, target_document_ids, request.research_type, chat_history, workspace_id=request.workspace_id)
    if cached:
        log_query_event(
            query=request.question, research_type=request.research_type,
            session_id=getattr(request, "session_id", None), documents_searched=len(target_document_ids),
            total_latency_s=round(time.time() - start, 3), task_status="CACHE_HIT",
            confidence=cached.get("confidence", "UNAVAILABLE"), llm_calls=0, model_name=GROQ_MODEL,
            extra={"request_id": request_id, "active_document_count": active_document_count, "cache_hit": True},
        )
        return {k: v for k, v in cached.items() if k != "_cached_at"}

    try:
        final_state = _execute_research_graph(
            question=request.question,
            session_id=getattr(request, "session_id", None),
            collection_id=None,  # already resolved into target_document_ids above
            document_ids=target_document_ids,
            research_type=getattr(request, "research_type", "simple"),
            structured_mode=(getattr(request, "mode", "normal") == "structured"),
            workspace_id=request.workspace_id,
        )
    except Exception as e:
        # Fallback to Phase 2 retrieval if graph fails
        print(f"Graph execution failed: {e}")
        try:
            chunks = retrieve(
                request.question,
                strategy=request.strategy,
                document_ids=request.document_ids,
                top_k=request.top_k,
            )
        except Exception as err:
            chunks = []

        if not chunks:
            return {
                "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
                "sources": [],
                "structured_citations": [],
                "documents_found": 0,
                "verification": []
            }
        sources_text = "\n---\n".join(
            f"[{i+1}] (source: {c['metadata']['source']}) {c['text']}"
            for i, c in enumerate(chunks)
        )
        generation_failed = False
        try:
            generation = call_llm(ANSWER_PROMPT.format(question=request.question, sources=sources_text))
        except Exception as e:
            generation_failed = True
            generation = {"text": RATE_LIMIT_MESSAGE if is_rate_limit_error(e) else GENERATION_FAILED_MESSAGE}

        try:
            verification = verify_answer(generation["text"], chunks) if not generation_failed else []
        except Exception:
            verification = []

        # Same rule as the primary graph path: only attach sources when an
        # answer was actually generated from them. A failed/rate-limited
        # generation must not imply "here's what I found" when nothing was
        # actually synthesized.
        sources_out = [] if generation_failed else [
            {
                "source": c["metadata"].get("source", "unknown"),
                "document_id": c["metadata"].get("document_id", ""),
                "chunk_id": c["metadata"].get("chunk_id", ""),
                "parent_id": c["metadata"].get("parent_id", ""),
                "page_number": c["metadata"].get("page_number"),
                "section": c["metadata"].get("section", ""),
                "text": c["text"],
                "retrieval_method": c.get("source_method", ""),
                "rerank_score": round(c.get("rerank_score", 0), 3),
                "rrf_score": round(c.get("rrf_score", 0), 6),
            }
            for c in chunks
        ]
        log_query_event(
            query=request.question, research_type=getattr(request, "research_type", "simple"),
            session_id=getattr(request, "session_id", None), documents_searched=len(request.document_ids or []),
            selected_evidence_count=len(chunks), total_latency_s=round(time.time() - start, 3),
            task_status="OK" if not generation_failed else "FAILED",
            errors=["graph_execution_failed_used_legacy_fallback"],
            **_call_observability_fields(request_id, active_document_count),
            **_chunk_observability_fields(chunks),
        )
        return {
            "answer": generation["text"],
            "sources": sources_out,
            "structured_citations": sources_out,
            "documents_found": 0 if generation_failed else get_contributing_documents(chunks),
            "verification": verification,
            "latency_seconds": round(time.time() - start, 3),
        }
    
    chunks = final_state.get("retrieval_results", [])
    if not chunks:
        log_query_event(
            query=request.question, research_type=getattr(request, "research_type", "simple"),
            session_id=getattr(request, "session_id", None), documents_searched=len(request.document_ids or []),
            total_latency_s=round(time.time() - start, 3), confidence="UNAVAILABLE",
            **_call_observability_fields(request_id, active_document_count),
        )
        return {
            "answer": final_state.get("draft_answer") or INSUFFICIENT_EVIDENCE_MESSAGE,
            "sources": [],
            "structured_citations": [],
            "verification": [],
            "documents_found": 0,
        }

    documents_found = get_contributing_documents(chunks)
    retrieval_stats = get_retrieval_stats()
    payload = {
        "answer": final_state.get("draft_answer") or INSUFFICIENT_EVIDENCE_MESSAGE,
        "sources": final_state.get("citations", []),
        "structured_citations": final_state.get("citations", []),
        "documents_found": documents_found,
        # Convenience alias matching the documented internal API shape
        # (documents_used: [id, ...]) — additive, doesn't replace documents_found.
        "documents_used": [d["document_id"] for d in documents_found],
        # Real (never invented) retrieval-stage counts for the frontend's
        # collapsible "Retrieval Details" panel — candidates_retrieved is the
        # post-RRF-fusion pool size before CrossEncoder reranking cut it down
        # to reranked_to (the final evidence count, == len(chunks)/len(sources)).
        "retrieval_details": {
            "candidates_retrieved": retrieval_stats.get("candidates_retrieved", len(chunks)),
            "reranked_to": retrieval_stats.get("reranked_to", len(chunks)),
        },
        # Feature 6: only non-null when mode="structured" was explicitly
        # requested AND synthesis succeeded. Never shown in the plain
        # conversational bubble unless the caller asked for it.
        "structured": final_state.get("structured_data"),
        # Same gating as "structured" above: only non-empty when
        # mode="structured" was requested AND synthesis succeeded. Internal
        # claim-level evidence trace — the normal chat UI never renders this.
        "claim_evidence_trace": final_state.get("claim_evidence_trace", []),
        "verification": final_state.get("verification_results", []),
        "research_type": final_state.get("research_type", request.research_type),
        "research_iterations": final_state.get("research_iterations", 1),
        "evidence_gaps": final_state.get("evidence_gaps", []),
        "similarities": final_state.get("similarities", []),
        "differences": final_state.get("differences", []),
        "contradictions": final_state.get("contradictions", []),
        "research_gaps": final_state.get("research_gaps", []),
        "claims": final_state.get("claims", []),
        "confidence": final_state.get("confidence", "UNAVAILABLE"),
        "status": final_state.get("status", "OK"),
        "latency_s": round(time.time() - start, 2),
        "research_plan": final_state.get("research_plan", {}),
    }
    
    if getattr(request, "session_id", None):
        add_message(uuid.uuid4().hex[:12], request.session_id, "user", request.question)
        add_message(uuid.uuid4().hex[:12], request.session_id, "assistant", payload["answer"], metadata=payload)

    # Only cache a genuine synthesized answer — never a rate-limit/generation
    # failure message (item 13: a retry later should get a real attempt).
    if _is_cacheable_answer(payload["answer"]):
        cache.set_cached_answer(request.question, target_document_ids, request.research_type, payload, chat_history, workspace_id=request.workspace_id)

    log_query_event(
        query=request.question,
        research_type=payload.get("research_type", "simple"),
        session_id=getattr(request, "session_id", None),
        collection_id=getattr(request, "collection_id", None),
        documents_searched=len(target_document_ids),
        documents_contributing=len(payload.get("documents_found") or []),
        selected_evidence_count=len(chunks),
        total_latency_s=payload.get("latency_s", 0),
        research_iterations=payload.get("research_iterations", 1),
        confidence=payload.get("confidence", "UNAVAILABLE"),
        task_status="OK",
        **_call_observability_fields(request_id, active_document_count),
        **_chunk_observability_fields(chunks),
    )

    return payload


# --- Report Endpoints ---
#
# Same shape as /query: retrieval + reranking (no LLM) -> ONE report LLM
# call -> Pydantic-validated JSON -> deterministic renderers. Cached by
# document_ids, cleared on any upload/delete (see cache.clear_all() above).

@app.post("/report")
def api_generate_report(req: ReportRequest, user: dict = Depends(auth.get_current_user)):
    if not req.document_ids:
        raise HTTPException(400, "At least one document is required to generate a report.")

    # Isolation check (hardened): report_id is itself a deterministic cache
    # key derived from document_ids/workspace_id — a client from any
    # workspace could previously generate (and read back) a comparative
    # report over document_ids it never actually uploaded, as long as it
    # could guess/learn those content-hash IDs, since NEITHER generation
    # NOR the 4 GET-by-id endpoints below checked real ownership at all.
    # workspace_id is now required and verified against the authenticated
    # user before anything else, and is stored inside the report payload
    # itself so every GET-by-id endpoint can enforce the same real
    # ownership check (see _require_resource_owner() calls below).
    _require_workspace_owner(req.workspace_id, user)
    target_document_ids = _resolve_document_scope(req.document_ids, None, req.workspace_id)
    if not target_document_ids:
        return {"ok": False, "error": NO_DOCUMENTS_MESSAGE, "report_id": None}

    request_id = _new_request_id()
    active_document_count = len(target_document_ids)
    start_call_tracking()
    start = time.time()

    cached = cache.get_cached_report(target_document_ids, workspace_id=req.workspace_id)
    if cached:
        log_query_event(
            query="[report]", documents_searched=active_document_count,
            total_latency_s=round(time.time() - start, 3), task_status="CACHE_HIT",
            llm_calls=0, model_name=GROQ_MODEL,
            extra={"request_id": request_id, "active_document_count": active_document_count, "cache_hit": True},
        )
        cached_payload = {k: v for k, v in cached.items() if k != "_cached_at"}
        _persist_analysis_turn(req.session_id, None, "report", cached_payload.get("title") or "Comparison report", {"report": cached_payload})
        return cached_payload

    result = generate_report(target_document_ids)  # <-- the ONE LLM call, internally

    if not result["ok"]:
        log_query_event(
            query="[report]", documents_searched=active_document_count,
            total_latency_s=round(time.time() - start, 3), task_status="FAILED",
            errors=[result["error"]], **_call_observability_fields(request_id, active_document_count),
        )
        return {"ok": False, "error": result["error"], "report_id": None}

    report: ResearchReport = result["report"]
    payload = {
        "ok": True,
        "report_id": None,  # filled in below once the cache key is known
        "title": report.title,
        "overview": report.overview,
        "papers": [p.model_dump() for p in report.papers],
        "comparison": report.comparison.model_dump() if report.comparison else None,
        "conclusion": report.conclusion,
        "evidence_sufficient": report.evidence_sufficient,
        "documents_found": len({c["document_id"] for c in result["citations"]}),
        # Stored (not just used to derive the cache key) so every GET-by-id
        # endpoint below can enforce real ownership on read, not just on
        # generation — see _require_resource_owner() calls below.
        "workspace_id": req.workspace_id,
    }
    report_id = cache.set_cached_report(target_document_ids, payload, workspace_id=req.workspace_id)
    stored = cache.get_report_by_id(report_id)
    stored["report_id"] = report_id  # patch in place so future cache hits already carry it

    log_query_event(
        query="[report]", documents_searched=len(req.document_ids),
        documents_contributing=payload["documents_found"],
        total_latency_s=round(time.time() - start, 3), task_status="OK",
        **_call_observability_fields(request_id, active_document_count),
    )

    final_payload = {k: v for k, v in stored.items() if k != "_cached_at"}
    _persist_analysis_turn(req.session_id, None, "report", final_payload.get("title") or "Comparison report", {"report": final_payload})
    return final_payload


def _get_owned_report_or_404(report_id: str, user: dict) -> dict:
    """Shared by all 4 report GET-by-id endpoints. report_id is itself a
    deterministic cache key (guessable-format if you already know the
    document_ids/workspace_id that produced it) — real ownership, not mere
    possession of the id, is what's checked. A stored report with no
    workspace_id (generated before this field existed — old, likely
    already TTL-expired cache entries) has no ownership to enforce and
    stays reachable, unchanged legacy behavior."""
    stored = cache.get_report_by_id(report_id)
    if not stored:
        raise HTTPException(404, "Report not found — it may have expired, or the workspace has changed since it was generated.")
    _require_resource_owner(stored.get("workspace_id"), user, "Report not found — it may have expired, or the workspace has changed since it was generated.")
    return stored


@app.get("/report/{report_id}")
def api_get_report(report_id: str, user: dict = Depends(auth.get_current_user)):
    stored = _get_owned_report_or_404(report_id, user)
    return {k: v for k, v in stored.items() if k != "_cached_at"}


@app.get("/report/{report_id}/markdown")
def api_get_report_markdown(report_id: str, user: dict = Depends(auth.get_current_user)):
    stored = _get_owned_report_or_404(report_id, user)
    report_obj = _stored_to_research_report(stored)
    md = render_report_markdown(report_obj)  # rendered from the validated JSON — no LLM call
    return Response(
        content=md, media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{_safe_filename(report_obj.title)}.md"'},
    )


@app.get("/report/{report_id}/pdf")
def api_get_report_pdf(report_id: str, user: dict = Depends(auth.get_current_user)):
    stored = _get_owned_report_or_404(report_id, user)
    report_obj = _stored_to_research_report(stored)
    pdf_bytes = render_report_pdf(report_obj)  # rendered from the validated JSON — no LLM call
    return Response(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{_safe_filename(report_obj.title)}.pdf"'},
    )


@app.get("/report/{report_id}/docx")
def api_get_report_docx(report_id: str, user: dict = Depends(auth.get_current_user)):
    stored = _get_owned_report_or_404(report_id, user)
    report_obj = _stored_to_research_report(stored)
    docx_bytes = render_report_docx(report_obj)  # rendered from the validated JSON — no LLM call
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{_safe_filename(report_obj.title)}.docx"'},
    )


# --- Analysis Endpoints (Viva / Mock Test / Explain Figure / Recommend /
# Why This Design / System Design / Project Interview) ---
#
# Same shape as everywhere else in this file: retrieval (when PDF-grounded)
# is zero LLM calls (_retrieve_for_analysis reuses the exact same hybrid
# retrieval + RRF + rerank + token-budget pipeline as /query), then exactly
# ONE Groq call via analysis.py, Pydantic-validated. No verifier call, no
# extra generation pass.

_PDF_GROUNDED_ANALYSIS_MODES = {
    "viva", "mock_test", "explain_figure", "recommend", "project_interview_start", "project_interview_evaluate",
    "evaluate_paper", "research_gaps", "literature_matrix", "knowledge_graph",
}
# "Why This Design?" / "System Design" are explicitly about VerityRAG's OWN
# architecture, not the uploaded paper — the only two modes still grounded
# in analysis.REAL_ARCHITECTURE_CONTEXT instead of retrieval.
_NON_PDF_ANALYSIS_MODES = {"why_design", "system_design"}


@app.post("/analyze")
def api_analyze(req: AnalyzeRequest, user: dict = Depends(auth.get_current_user)):
    if req.mode not in _PDF_GROUNDED_ANALYSIS_MODES and req.mode not in _NON_PDF_ANALYSIS_MODES:
        raise HTTPException(400, f"Unknown analyze mode: {req.mode}")

    # PDF-grounded modes read real, real document content, so real
    # ownership is required — same policy as /query and /report. The two
    # "own architecture" modes (why_design/system_design) touch no
    # workspace data at all and stay reachable to any authenticated user.
    if req.mode in _PDF_GROUNDED_ANALYSIS_MODES:
        _require_workspace_owner(req.workspace_id, user)

    request_id = _new_request_id()
    active_document_count = 0
    start_call_tracking()
    start = time.time()

    if req.mode in _PDF_GROUNDED_ANALYSIS_MODES:
        target_document_ids = _resolve_document_scope(req.document_ids, None, req.workspace_id)
        active_document_count = len(target_document_ids)
        if not target_document_ids:
            return {"ok": False, "mode": req.mode, "error": NO_DOCUMENTS_MESSAGE}

        if req.mode in ("viva", "mock_test"):
            query_text = "key concepts, methodology, architecture, datasets, results, contributions, limitations"
            top_k = min(15, REPORT_CHUNKS_PER_DOC * max(len(target_document_ids), 1))
            chunks = _retrieve_for_analysis(query_text, target_document_ids, top_k, workspace_id=req.workspace_id)
            if not chunks:
                return {"ok": False, "mode": req.mode, "error": INSUFFICIENT_EVIDENCE_MESSAGE}
            evidence_text = _evidence_text_from_chunks(chunks)
            mode_label = "Viva" if req.mode == "viva" else "Mock Test"
            result = generate_pdf_questions(evidence_text, mode_label, req.difficulty, max(1, min(req.num_questions, 15)))
            documents_found = get_contributing_documents(chunks)
            if result is None:
                log_query_event(query=f"[analyze:{req.mode}]", documents_searched=active_document_count, total_latency_s=round(time.time() - start, 3), task_status="FAILED", **_call_observability_fields(request_id, active_document_count))
                return {"ok": False, "mode": req.mode, "error": GENERATION_FAILED_MESSAGE}
            log_query_event(query=f"[analyze:{req.mode}]", documents_searched=active_document_count, documents_contributing=len(documents_found), selected_evidence_count=len(chunks), total_latency_s=round(time.time() - start, 3), task_status="OK", **_call_observability_fields(request_id, active_document_count), **_chunk_observability_fields(chunks))
            questions_payload = [q.model_dump() for q in result.questions]
            display_label = f"{mode_label} Questions ({req.difficulty})"
            _persist_analysis_turn(req.session_id, None, "analysis_questions", display_label, {
                "label": display_label, "questions": questions_payload, "evidence_sufficient": result.evidence_sufficient,
            })
            return {
                "ok": True, "mode": req.mode,
                "questions": questions_payload,
                "grounded": result.grounded, "evidence_sufficient": result.evidence_sufficient,
                "documents_used": [d["document_id"] for d in documents_found],
            }

        if req.mode == "explain_figure":
            ref = (req.figure_reference or "").strip()
            query_text = f"{ref} figure table caption diagram graph" if ref else "figure table graph diagram caption"
            chunks = _retrieve_for_analysis(query_text, target_document_ids, top_k=min(10, REPORT_CHUNKS_PER_DOC * max(len(target_document_ids), 1)), workspace_id=req.workspace_id)
            if not chunks:
                return {"ok": False, "mode": req.mode, "error": INSUFFICIENT_EVIDENCE_MESSAGE}
            evidence_text = _evidence_text_from_chunks(chunks)

            # Best-effort visual path: render the page image for the single
            # most relevant chunk's own document_id + page_number (chunks
            # are already reranked, so chunks[0] is the best match for this
            # figure reference) — ONLY that one document/page is ever
            # opened, never a scan across other documents. None (not an
            # error) whenever the PDF wasn't persisted, has no page number,
            # or a vision model isn't configured — explain_figure() falls
            # back to the existing text-only path cleanly in every case.
            image_b64 = None
            top_chunk_meta = chunks[0].get("metadata", {})
            top_doc_id = top_chunk_meta.get("document_id")
            top_page = top_chunk_meta.get("page_number")
            if top_doc_id and isinstance(top_page, int) and vision_model_available():
                image_b64 = render_page_as_image_base64(top_doc_id, top_page)

            result, visual_inspection_used = explain_figure(
                evidence_text, ref,
                image_base64=image_b64,
                vision_model=(VISION_MODEL if image_b64 else None),
            )
            documents_found = get_contributing_documents(chunks)
            if result is None:
                log_query_event(query="[analyze:explain_figure]", documents_searched=active_document_count, total_latency_s=round(time.time() - start, 3), task_status="FAILED", **_call_observability_fields(request_id, active_document_count))
                return {"ok": False, "mode": req.mode, "error": GENERATION_FAILED_MESSAGE}
            log_query_event(query="[analyze:explain_figure]", documents_searched=active_document_count, documents_contributing=len(documents_found), selected_evidence_count=len(chunks), total_latency_s=round(time.time() - start, 3), task_status="OK", **_call_observability_fields(request_id, active_document_count), **_chunk_observability_fields(chunks))
            answer = result.answer if result.evidence_sufficient else INSUFFICIENT_EVIDENCE_MESSAGE
            if result.evidence_sufficient:
                # Deterministic honesty caveat — appended regardless of what
                # the model itself said, so this is never dependent on
                # prompt compliance, and never claims more than actually
                # happened for THIS specific call.
                if visual_inspection_used:
                    answer = f"{answer}\n\n(Note: this explanation was generated from an actual image of the PDF page, not just extracted text.)"
                else:
                    answer = f"{answer}\n\n(Note: visual inspection was unavailable for this request; explanation is based on the paper's extracted caption/surrounding text, not a direct visual reading of the image.)"
            user_text = f"Explain: {ref}" if ref else "Explain this figure."
            _persist_analysis_turn(req.session_id, user_text, "assistant_kicker", answer, {"kicker": "Figure Explanation"})
            return {
                "ok": True, "mode": req.mode, "answer": answer,
                "grounded": result.grounded, "evidence_sufficient": result.evidence_sufficient,
                "documents_used": [d["document_id"] for d in documents_found],
            }

        if req.mode == "recommend":
            query_text = "compare methodology architecture datasets results limitations proposed approach"
            top_k = min(18, REPORT_CHUNKS_PER_DOC * max(len(target_document_ids), 1))
            chunks = _retrieve_for_analysis(query_text, target_document_ids, top_k, workspace_id=req.workspace_id)
            if not chunks:
                return {"ok": False, "mode": req.mode, "error": INSUFFICIENT_EVIDENCE_MESSAGE}
            evidence_text = _evidence_text_from_chunks(chunks)
            result = recommend_from_comparison(evidence_text)
            documents_found = get_contributing_documents(chunks)
            if result is None:
                log_query_event(query="[analyze:recommend]", documents_searched=active_document_count, total_latency_s=round(time.time() - start, 3), task_status="FAILED", **_call_observability_fields(request_id, active_document_count))
                return {"ok": False, "mode": req.mode, "error": GENERATION_FAILED_MESSAGE}
            log_query_event(query="[analyze:recommend]", documents_searched=active_document_count, documents_contributing=len(documents_found), selected_evidence_count=len(chunks), total_latency_s=round(time.time() - start, 3), task_status="OK", **_call_observability_fields(request_id, active_document_count), **_chunk_observability_fields(chunks))
            answer = result.answer if result.evidence_sufficient else INSUFFICIENT_EVIDENCE_MESSAGE
            _persist_analysis_turn(req.session_id, None, "assistant_kicker", answer, {"kicker": "Recommendation"})
            return {
                "ok": True, "mode": req.mode, "answer": answer,
                "grounded": result.grounded, "evidence_sufficient": result.evidence_sufficient,
                "documents_used": [d["document_id"] for d in documents_found],
            }

        if req.mode == "evaluate_paper":
            # Single-paper critique — if more than one document is scoped,
            # evaluate the first one only (same "pick the primary paper"
            # convention as explain_figure's single-document expectation);
            # the frontend only offers this pill when exactly one paper is
            # selected/referenced.
            eval_doc_id = target_document_ids[0]
            query_text = "problem statement, novelty, methodology, experimental design, results, limitations, reproducibility"
            top_k = min(15, REPORT_CHUNKS_PER_DOC * max(len(target_document_ids), 1))
            chunks = _retrieve_for_analysis(query_text, [eval_doc_id], top_k, workspace_id=req.workspace_id)
            if not chunks:
                return {"ok": False, "mode": req.mode, "error": INSUFFICIENT_EVIDENCE_MESSAGE}
            evidence_text = _evidence_text_from_chunks(chunks)
            result = evaluate_paper(evidence_text, eval_doc_id)
            documents_found = get_contributing_documents(chunks)
            if result is None:
                log_query_event(query="[analyze:evaluate_paper]", documents_searched=active_document_count, total_latency_s=round(time.time() - start, 3), task_status="FAILED", **_call_observability_fields(request_id, active_document_count))
                return {"ok": False, "mode": req.mode, "error": GENERATION_FAILED_MESSAGE}
            log_query_event(query="[analyze:evaluate_paper]", documents_searched=active_document_count, documents_contributing=len(documents_found), selected_evidence_count=len(chunks), total_latency_s=round(time.time() - start, 3), task_status="OK", **_call_observability_fields(request_id, active_document_count), **_chunk_observability_fields(chunks))
            payload = result.model_dump()
            # Display-only: human-readable title, never the raw document_id
            # — resolved deterministically from the filename (see doc_titles.py).
            eval_filename = next((d["source"] for d in documents_found if d["document_id"] == eval_doc_id), None)
            payload["title"] = resolve_display_title(eval_doc_id, eval_filename)
            _persist_analysis_turn(req.session_id, None, "evaluate_paper", "Paper Evaluation", {"evaluation": payload})
            return {"ok": True, "mode": req.mode, "evaluation": payload, "documents_used": [d["document_id"] for d in documents_found]}

        if req.mode == "research_gaps":
            query_text = "limitations, future work, open problems, unaddressed challenges, weaknesses, threats to validity"
            top_k = min(18, REPORT_CHUNKS_PER_DOC * max(len(target_document_ids), 1))
            chunks = _retrieve_for_analysis(query_text, target_document_ids, top_k, workspace_id=req.workspace_id)
            if not chunks:
                return {"ok": False, "mode": req.mode, "error": INSUFFICIENT_EVIDENCE_MESSAGE}
            evidence_text = _evidence_text_from_chunks(chunks)
            result = find_research_gaps(evidence_text)
            documents_found = get_contributing_documents(chunks)
            if result is None:
                log_query_event(query="[analyze:research_gaps]", documents_searched=active_document_count, total_latency_s=round(time.time() - start, 3), task_status="FAILED", **_call_observability_fields(request_id, active_document_count))
                return {"ok": False, "mode": req.mode, "error": GENERATION_FAILED_MESSAGE}
            log_query_event(query="[analyze:research_gaps]", documents_searched=active_document_count, documents_contributing=len(documents_found), selected_evidence_count=len(chunks), total_latency_s=round(time.time() - start, 3), task_status="OK", **_call_observability_fields(request_id, active_document_count), **_chunk_observability_fields(chunks))
            gaps_payload = [g.model_dump() for g in result.gaps]
            # Display-only map so the UI can show each gap's paper by a
            # human-readable title instead of its raw document_id.
            document_titles = {d["document_id"]: resolve_display_title(d["document_id"], d["source"], index=i) for i, d in enumerate(documents_found)}
            _persist_analysis_turn(req.session_id, None, "research_gaps", "Research Gaps", {"gaps": gaps_payload, "evidence_sufficient": result.evidence_sufficient, "document_titles": document_titles})
            return {"ok": True, "mode": req.mode, "gaps": gaps_payload, "evidence_sufficient": result.evidence_sufficient, "document_titles": document_titles, "documents_used": [d["document_id"] for d in documents_found]}

        if req.mode == "literature_matrix":
            if len(target_document_ids) < 2:
                return {"ok": False, "mode": req.mode, "error": "Select at least two papers to build a literature matrix."}
            query_text = "problem, method, architecture, dataset, metrics, results, limitations, research gap"
            top_k = min(18, REPORT_CHUNKS_PER_DOC * max(len(target_document_ids), 1))
            chunks = _retrieve_for_analysis(query_text, target_document_ids, top_k, workspace_id=req.workspace_id)
            if not chunks:
                return {"ok": False, "mode": req.mode, "error": INSUFFICIENT_EVIDENCE_MESSAGE}
            evidence_text = _evidence_text_from_chunks(chunks)
            result = build_literature_matrix(evidence_text)
            documents_found = get_contributing_documents(chunks)
            if result is None:
                log_query_event(query="[analyze:literature_matrix]", documents_searched=active_document_count, total_latency_s=round(time.time() - start, 3), task_status="FAILED", **_call_observability_fields(request_id, active_document_count))
                return {"ok": False, "mode": req.mode, "error": GENERATION_FAILED_MESSAGE}
            log_query_event(query="[analyze:literature_matrix]", documents_searched=active_document_count, documents_contributing=len(documents_found), selected_evidence_count=len(chunks), total_latency_s=round(time.time() - start, 3), task_status="OK", **_call_observability_fields(request_id, active_document_count), **_chunk_observability_fields(chunks))
            rows_payload = [r.model_dump() for r in result.rows]
            # Display-only: normalize every row's title — filename first,
            # then the model's own title from this SAME call, never the raw
            # document_id. document_id on each row is untouched.
            filename_by_doc = {d["document_id"]: d["source"] for d in documents_found}
            for i, row in enumerate(rows_payload):
                row["title"] = resolve_display_title(row["document_id"], filename_by_doc.get(row["document_id"]), fallback_title=row.get("title"), index=i)
            _persist_analysis_turn(req.session_id, None, "literature_matrix", "Literature Matrix", {"rows": rows_payload, "evidence_sufficient": result.evidence_sufficient})
            return {"ok": True, "mode": req.mode, "rows": rows_payload, "evidence_sufficient": result.evidence_sufficient, "documents_used": [d["document_id"] for d in documents_found]}

        if req.mode == "knowledge_graph":
            query_text = "key concepts, methodology, architecture, datasets, results, contributions, limitations, relationships"
            top_k = min(18, REPORT_CHUNKS_PER_DOC * max(len(target_document_ids), 1))
            chunks = _retrieve_for_analysis(query_text, target_document_ids, top_k, workspace_id=req.workspace_id)
            if not chunks:
                return {"ok": False, "mode": req.mode, "error": INSUFFICIENT_EVIDENCE_MESSAGE}
            evidence_text = _evidence_text_from_chunks(chunks)
            result = build_knowledge_graph(evidence_text)
            documents_found = get_contributing_documents(chunks)
            if result is None:
                log_query_event(query="[analyze:knowledge_graph]", documents_searched=active_document_count, total_latency_s=round(time.time() - start, 3), task_status="FAILED", **_call_observability_fields(request_id, active_document_count))
                return {"ok": False, "mode": req.mode, "error": GENERATION_FAILED_MESSAGE}
            log_query_event(query="[analyze:knowledge_graph]", documents_searched=active_document_count, documents_contributing=len(documents_found), selected_evidence_count=len(chunks), total_latency_s=round(time.time() - start, 3), task_status="OK", **_call_observability_fields(request_id, active_document_count), **_chunk_observability_fields(chunks))
            nodes_payload = [n.model_dump() for n in result.nodes]
            edges_payload = [e.model_dump() for e in result.edges]
            # Display-only map so node/edge document_id pills in the UI show
            # a human-readable title instead of the raw document_id.
            document_titles = {d["document_id"]: resolve_display_title(d["document_id"], d["source"], index=i) for i, d in enumerate(documents_found)}
            _persist_analysis_turn(req.session_id, None, "knowledge_graph", "Knowledge Graph", {"nodes": nodes_payload, "edges": edges_payload, "evidence_sufficient": result.evidence_sufficient, "document_titles": document_titles})
            return {"ok": True, "mode": req.mode, "nodes": nodes_payload, "edges": edges_payload, "evidence_sufficient": result.evidence_sufficient, "document_titles": document_titles, "documents_used": [d["document_id"] for d in documents_found]}

        # Project Interview — grounded in the currently selected/referenced
        # uploaded document(s) by default (NOT VerityRAG's own architecture;
        # see analysis.py's comment on start_project_interview). Same broad
        # evidence query as Viva/Mock Test; re-retrieved each turn from the
        # SAME target_document_ids the frontend keeps pinned for the whole
        # interview, so the follow-up question always stays on the same
        # uploaded document even if workspace selection changes mid-interview.
        if req.mode in ("project_interview_start", "project_interview_evaluate"):
            query_text = "key concepts, methodology, architecture, datasets, results, contributions, limitations"
            top_k = min(15, REPORT_CHUNKS_PER_DOC * max(len(target_document_ids), 1))
            chunks = _retrieve_for_analysis(query_text, target_document_ids, top_k, workspace_id=req.workspace_id)
            if not chunks:
                return {"ok": False, "mode": req.mode, "error": INSUFFICIENT_EVIDENCE_MESSAGE}
            evidence_text = _evidence_text_from_chunks(chunks)
            documents_found = get_contributing_documents(chunks)

            if req.mode == "project_interview_start":
                result = start_project_interview(evidence_text, req.difficulty, req.topics)
                if result is None:
                    log_query_event(query="[analyze:project_interview_start]", documents_searched=active_document_count, total_latency_s=round(time.time() - start, 3), task_status="FAILED", **_call_observability_fields(request_id, active_document_count))
                    return {"ok": False, "mode": req.mode, "error": GENERATION_FAILED_MESSAGE}
                log_query_event(query="[analyze:project_interview_start]", documents_searched=active_document_count, documents_contributing=len(documents_found), selected_evidence_count=len(chunks), total_latency_s=round(time.time() - start, 3), task_status="OK", **_call_observability_fields(request_id, active_document_count), **_chunk_observability_fields(chunks))
                _persist_analysis_turn(req.session_id, None, "interview_question", result.answer, {})
                return {
                    "ok": True, "mode": req.mode, "question": result.answer,
                    "documents_used": [d["document_id"] for d in documents_found],
                }

            question = (req.question or "").strip()
            user_answer = (req.user_answer or "").strip()
            if not question or not user_answer:
                return {"ok": False, "mode": req.mode, "error": "Both the question and your answer are required."}
            result = evaluate_interview_answer(evidence_text, question, user_answer, req.difficulty, req.topics)
            if result is None:
                log_query_event(query="[analyze:project_interview_evaluate]", documents_searched=active_document_count, total_latency_s=round(time.time() - start, 3), task_status="FAILED", **_call_observability_fields(request_id, active_document_count))
                return {"ok": False, "mode": req.mode, "error": GENERATION_FAILED_MESSAGE}
            log_query_event(query="[analyze:project_interview_evaluate]", documents_searched=active_document_count, documents_contributing=len(documents_found), selected_evidence_count=len(chunks), total_latency_s=round(time.time() - start, 3), task_status="OK", **_call_observability_fields(request_id, active_document_count), **_chunk_observability_fields(chunks))
            # Two persisted turns per evaluate call, matching exactly what the
            # frontend pushes into conv.messages: the evaluation card, then
            # either the next question or an "interview complete" marker —
            # only the first carries the user's answer text.
            _persist_analysis_turn(req.session_id, user_answer, "interview_eval", result.suggested_answer or result.technical_depth or "Evaluation", {
                "correctness": result.correctness, "missing_points": result.missing_points,
                "technical_depth": result.technical_depth, "suggested_answer": result.suggested_answer,
            })
            if result.next_question:
                _persist_analysis_turn(req.session_id, None, "interview_question", result.next_question, {"category": result.next_question_category})
            else:
                _persist_analysis_turn(req.session_id, None, "interview_question", "Interview complete — nice work. Click Project Interview again to start a new one.", {"ended": True})
            return {
                "ok": True, "mode": req.mode,
                "correctness": result.correctness, "missing_points": result.missing_points,
                "technical_depth": result.technical_depth, "suggested_answer": result.suggested_answer,
                "next_question": result.next_question, "next_question_category": result.next_question_category,
                "documents_used": [d["document_id"] for d in documents_found],
            }

    # --- Non-PDF modes: "Why This Design?" / "System Design" are explicitly
    # about VerityRAG's own architecture, grounded in REAL_ARCHITECTURE_CONTEXT.
    if req.mode in ("why_design", "system_design"):
        question = (req.question or "").strip()
        if not question:
            return {"ok": False, "mode": req.mode, "error": "A question is required."}
        result = answer_design_question(question, req.mode)
        if result is None:
            log_query_event(query=f"[analyze:{req.mode}]", total_latency_s=round(time.time() - start, 3), task_status="FAILED", **_call_observability_fields(request_id, 0))
            return {"ok": False, "mode": req.mode, "error": GENERATION_FAILED_MESSAGE}
        log_query_event(query=f"[analyze:{req.mode}]", total_latency_s=round(time.time() - start, 3), task_status="OK", **_call_observability_fields(request_id, 0))
        kicker = "Why This Design" if req.mode == "why_design" else "System Design"
        _persist_analysis_turn(req.session_id, question, "assistant_kicker", result.answer, {"kicker": kicker})
        return {"ok": True, "mode": req.mode, "answer": result.answer}


# --- RAG Evaluation Dashboard — ONLY real measured values, never fabricated ---

@app.get("/eval/dashboard")
def api_eval_dashboard():
    live_events = [e for e in read_recent_events(500) if e.get("task_status") in ("OK", "FAILED", "CACHE_HIT")]
    n = len(live_events)

    live_metrics = None
    if n:
        llm_calls = [e.get("llm_calls", 0) for e in live_events]
        latencies = [e["total_latency_s"] for e in live_events if e.get("total_latency_s") is not None]
        cache_hits = sum(1 for e in live_events if e.get("task_status") == "CACHE_HIT")
        tokens = [e.get("extra", {}).get("total_tokens", 0) for e in live_events if e.get("extra")]
        fallback_count = sum(1 for e in live_events if e.get("fallback_triggered"))
        live_metrics = {
            "sample_size": n,
            "avg_llm_calls_per_query": round(sum(llm_calls) / n, 3) if n else None,
            "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "cache_hit_rate": round(cache_hits / n, 3) if n else None,
            "fallback_rate": round(fallback_count / n, 3) if n else None,
            "avg_tokens_per_query": round(sum(tokens) / len(tokens), 1) if tokens else None,
        }

    offline_eval = None
    try:
        import json as _json
        eval_path = Path(__file__).parent.parent / "data" / "eval_results_comprehensive.json"
        if eval_path.exists():
            data = _json.loads(eval_path.read_text(encoding="utf-8"))
            agg = data.get("aggregate_summary", {})
            hybrid_reranked = agg.get("hybrid_reranked", {}).get("avg_metrics", {})
            hybrid_rrf = agg.get("hybrid_rrf", {}).get("avg_metrics", {})
            offline_eval = {
                "source": "data/eval_results_comprehensive.json (offline retrieval benchmark)",
                "questions_evaluated": data.get("meta", {}).get("evaluation_questions"),
                "current_pipeline_precision_at_5": hybrid_reranked.get("precision@5"),
                "current_pipeline_recall_at_5": hybrid_reranked.get("recall@5"),
                "current_pipeline_mrr": hybrid_reranked.get("mrr"),
                "reranking_recall_at_5_improvement": (
                    round(hybrid_reranked["recall@5"] - hybrid_rrf["recall@5"], 4)
                    if "recall@5" in hybrid_reranked and "recall@5" in hybrid_rrf else None
                ),
                "reranking_precision_at_5_improvement": (
                    round(hybrid_reranked["precision@5"] - hybrid_rrf["precision@5"], 4)
                    if "precision@5" in hybrid_reranked and "precision@5" in hybrid_rrf else None
                ),
            }
    except Exception:
        offline_eval = None

    # 10-document isolated benchmark (evaluation/run_benchmark.py) — indexing
    # correctness, document isolation, comparison scaling, decomposition,
    # concurrency, and latency distribution, all LLM-free and re-runnable.
    # Absent (None, not fabricated) until that script has actually been run.
    benchmark_10doc = None
    try:
        import json as _json
        bench_path = Path(__file__).parent.parent / "evaluation" / "results.json"
        if bench_path.exists():
            bdata = _json.loads(bench_path.read_text(encoding="utf-8"))
            lat = bdata.get("latency_distribution", {}).get("end_to_end_retrieval", {})
            benchmark_10doc = {
                "source": "evaluation/results.json (reproducible: python evaluation/run_benchmark.py)",
                "generated_at_utc": bdata.get("meta", {}).get("generated_at_utc"),
                "documents_benchmarked": bdata.get("meta", {}).get("dataset_size_documents"),
                "isolation_violations": bdata.get("B_D_retrieval_and_isolation", {}).get("isolation_violations"),
                "cross_document_contamination_found": bdata.get("A_indexing", {}).get("cross_document_contamination_found"),
                "retrieval_latency_mean_ms": lat.get("mean_ms"),
                "retrieval_latency_median_ms": lat.get("median_ms"),
                "retrieval_latency_p95_ms": lat.get("p95_ms"),
                "concurrent_speedup_factor": bdata.get("F_concurrent_retrieval", {}).get("speedup_factor"),
                "production_chroma_touched": bdata.get("meta", {}).get("production_chroma_touched"),
            }
    except Exception:
        benchmark_10doc = None

    # Offline groundedness/hallucination scoring (groundedness_eval.py) —
    # opt-in, real Groq calls, run deliberately and never as part of normal
    # request handling. Absent (honestly "not measured") until that script
    # has actually been run and its output saved to this path.
    groundedness = None
    try:
        import json as _json
        gnd_path = Path(__file__).parent.parent / "data" / "groundedness_eval_results.json"
        if gnd_path.exists():
            gdata = _json.loads(gnd_path.read_text(encoding="utf-8"))
            groundedness = {
                "source": "data/groundedness_eval_results.json (backend/groundedness_eval.py, offline/opt-in)",
                "dataset_size": gdata.get("dataset_size"),
                "avg_groundedness_score": gdata.get("avg_groundedness_score"),
                "avg_unsupported_claim_rate": gdata.get("avg_unsupported_claim_rate"),
                "avg_evidence_coverage": gdata.get("avg_evidence_coverage"),
            }
    except Exception:
        groundedness = None

    # Cache backend + real hit/miss counters for THIS process (cache.py) —
    # separate from the retrieval-quality benchmarks above; this is live
    # runtime cache behavior, not an offline evaluation.
    try:
        cache_stats = cache.stats()
    except Exception:
        cache_stats = None

    return {
        # --- RUNTIME (online, from real production traffic in this process) ---
        "live": live_metrics,
        "live_note": "Aggregated from logs/verityrag_events.jsonl (last up to 500 real requests). null if no queries logged yet.",
        "cache": cache_stats,
        # --- OFFLINE (reproducible benchmarks, run deliberately, not on every request) ---
        "offline_retrieval_eval": offline_eval,
        "offline_benchmark_10_document": benchmark_10doc,
        "offline_groundedness_eval": groundedness,
        "not_measured": [
            m for m in [
                None if offline_eval else "offline retrieval benchmark (data/eval_results_comprehensive.json not found — run backend/run_evaluation.py)",
                None if benchmark_10doc else "10-document benchmark (evaluation/results.json not found — run evaluation/run_benchmark.py)",
                None if groundedness else "groundedness / unsupported-claim-rate (data/groundedness_eval_results.json not found — run backend/groundedness_eval.py, an OFFLINE, opt-in, real-LLM-call script, deliberately)",
                "hallucination_rate as a single scalar (groundedness_eval.py reports unsupported_claim_rate as the closest measured proxy instead — see offline_groundedness_eval)",
                "answer_relevance beyond deterministic keyword overlap (see offline_groundedness_eval / groundedness_eval.py's answer_relevance field, which IS measured but is a coarse proxy, not semantic scoring)",
            ] if m
        ],
    }
