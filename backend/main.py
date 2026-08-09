import re as _re
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import uuid
import time

from ingest import ingest_document, get_collection, delete_document_from_index
from retrieval import retrieve, retrieve_multi, hybrid_retrieve, build_bm25_index, get_contributing_documents, get_retrieval_stats
from query_transform import decompose_query_deterministic
from analysis import (
    generate_pdf_questions, explain_figure, recommend_from_comparison,
    answer_design_question, start_project_interview, evaluate_interview_answer,
)
from verify import verify_answer
from llm import call_llm
from database import (
    list_documents, get_document,
    list_collections, get_collection as db_get_collection, create_collection,
    add_document_to_collection, remove_document_from_collection, delete_collection,
    create_session, get_session, list_sessions, delete_session, update_session_collection,
    add_message, get_session_messages,
    create_task, update_task_status, get_task,
    create_workspace, get_workspace, list_workspaces, documents_in_workspace,
    update_session_title,
)
from observability import log_query_event, filter_events, read_recent_events
from query_transform import (
    NO_DOCUMENTS_MESSAGE, INSUFFICIENT_EVIDENCE_MESSAGE, RATE_LIMIT_MESSAGE,
    GENERATION_FAILED_MESSAGE, is_rate_limit_error,
    start_call_tracking, get_call_log,
)
from config import GROQ_MODEL, GROQ_FALLBACK_MODEL, REPORT_CHUNKS_PER_DOC
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


def _retrieve_for_analysis(query_text: str, document_ids: list[str], top_k: int) -> list[dict]:
    """Shared retrieval path for /analyze's PDF-grounded modes: same
    deterministic decomposition + hybrid retrieval + RRF + rerank + token
    budget as normal queries (retrieval.py) — no LLM calls, no new pipeline."""
    sub_queries = decompose_query_deterministic(query_text)
    if len(sub_queries) > 1:
        return retrieve_multi(sub_queries, document_ids=document_ids, top_k=top_k, rerank_query=query_text)
    return retrieve(query_text, strategy="hybrid", document_ids=document_ids, top_k=top_k)


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; restrict this before deploying
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

class CollectionCreateRequest(BaseModel):
    name: str
    description: str | None = None
    document_ids: list[str] = []

class ReportRequest(BaseModel):
    document_ids: list[str]
    session_id: str | None = None   # when set, persists a history record the same way /query does

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


@app.get("/health")
def health():
    collection = get_collection()
    return {"status": "ok", "chunks_indexed": collection.count()}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), workspace_id: str | None = Form(None)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported right now.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = ingest_document(tmp_path, original_filename=file.filename, workspace_id=workspace_id)
        build_bm25_index()  # rebuild so the new doc is searchable immediately
        cache.clear_all()   # the workspace changed — stale cached answers/reports must not survive it
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)

# --- Workspace Endpoints ---

@app.post("/workspaces")
def api_create_workspace(req: WorkspaceCreateRequest):
    name = (req.name or "").strip() or "Untitled Workspace"
    ws_id = uuid.uuid4().hex[:12]
    return create_workspace(ws_id, name)

@app.get("/workspaces")
def api_list_workspaces():
    return list_workspaces()

@app.get("/workspaces/{workspace_id}")
def api_get_workspace(workspace_id: str):
    ws = get_workspace(workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")
    return ws

# --- Document Endpoints ---

@app.get("/documents")
def api_list_documents(workspace_id: str | None = None):
    return list_documents(workspace_id)

@app.get("/documents/{doc_id}")
def api_get_document(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc

@app.delete("/documents/{doc_id}")
def api_delete_document(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
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
def api_create_session(req: SessionCreateRequest):
    sess_id = uuid.uuid4().hex[:12]
    return create_session(sess_id, req.collection_id, workspace_id=req.workspace_id, title=req.title)

@app.get("/sessions")
def api_list_sessions(workspace_id: str | None = None):
    return list_sessions(workspace_id)

@app.patch("/sessions/{session_id}")
def api_update_session(session_id: str, req: SessionUpdateRequest):
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    update_session_title(session_id, req.title)
    return get_session(session_id)

@app.get("/sessions/{session_id}/messages")
def api_get_session_messages(session_id: str):
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    return get_session_messages(session_id)

@app.delete("/sessions/{session_id}")
def api_delete_session(session_id: str):
    delete_session(session_id)
    return {"status": "ok"}

# --- Task Endpoints ---

@app.get("/task/{task_id}")
def api_get_task(task_id: str):
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task

# --- Observability Endpoints ---

@app.get("/logs")
def api_get_logs(
    task_id: str | None = None,
    session_id: str | None = None,
    event_type: str | None = None,
    limit: int = 50
):
    # Safe, sanitized filtering without exposing secrets
    logs = filter_events(task_id=task_id, session_id=session_id, event_type=event_type, n=limit)
    return logs

def _execute_research_graph(question: str, session_id: str | None, collection_id: str | None, document_ids: list[str] | None, research_type: str, task_id: str | None = None, structured_mode: bool = False):
    from graph.workflow import research_app

    # Fresh per-request LLM call log (item 17). Call sites read it back via
    # get_call_log()/_call_observability_fields() after this function returns.
    start_call_tracking()

    target_document_ids = _resolve_document_scope(document_ids, collection_id)

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
            task_id=task_id
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
def api_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    task_id = uuid.uuid4().hex[:12]
    create_task(task_id, request.session_id)
    background_tasks.add_task(async_research_worker, task_id, request)
    return {"task_id": task_id, "status": "PENDING"}
@app.post("/query")
def query(request: QueryRequest):
    import time
    from graph.workflow import research_app

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
    cached = cache.get_cached_answer(request.question, target_document_ids, request.research_type, chat_history)
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
        cache.set_cached_answer(request.question, target_document_ids, request.research_type, payload, chat_history)

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
    )

    return payload


# --- Report Endpoints ---
#
# Same shape as /query: retrieval + reranking (no LLM) -> ONE report LLM
# call -> Pydantic-validated JSON -> deterministic renderers. Cached by
# document_ids, cleared on any upload/delete (see cache.clear_all() above).

@app.post("/report")
def api_generate_report(req: ReportRequest):
    if not req.document_ids:
        raise HTTPException(400, "At least one document is required to generate a report.")

    request_id = _new_request_id()
    active_document_count = len(req.document_ids)
    start_call_tracking()
    start = time.time()

    cached = cache.get_cached_report(req.document_ids)
    if cached:
        log_query_event(
            query="[report]", documents_searched=len(req.document_ids),
            total_latency_s=round(time.time() - start, 3), task_status="CACHE_HIT",
            llm_calls=0, model_name=GROQ_MODEL,
            extra={"request_id": request_id, "active_document_count": active_document_count, "cache_hit": True},
        )
        cached_payload = {k: v for k, v in cached.items() if k != "_cached_at"}
        _persist_analysis_turn(req.session_id, None, "report", cached_payload.get("title") or "Comparison report", {"report": cached_payload})
        return cached_payload

    result = generate_report(req.document_ids)  # <-- the ONE LLM call, internally

    if not result["ok"]:
        log_query_event(
            query="[report]", documents_searched=len(req.document_ids),
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
    }
    report_id = cache.set_cached_report(req.document_ids, payload)
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


@app.get("/report/{report_id}")
def api_get_report(report_id: str):
    stored = cache.get_report_by_id(report_id)
    if not stored:
        raise HTTPException(404, "Report not found — it may have expired, or the workspace has changed since it was generated.")
    return {k: v for k, v in stored.items() if k != "_cached_at"}


@app.get("/report/{report_id}/markdown")
def api_get_report_markdown(report_id: str):
    stored = cache.get_report_by_id(report_id)
    if not stored:
        raise HTTPException(404, "Report not found — it may have expired, or the workspace has changed since it was generated.")
    report_obj = _stored_to_research_report(stored)
    md = render_report_markdown(report_obj)  # rendered from the validated JSON — no LLM call
    return Response(
        content=md, media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{_safe_filename(report_obj.title)}.md"'},
    )


@app.get("/report/{report_id}/pdf")
def api_get_report_pdf(report_id: str):
    stored = cache.get_report_by_id(report_id)
    if not stored:
        raise HTTPException(404, "Report not found — it may have expired, or the workspace has changed since it was generated.")
    report_obj = _stored_to_research_report(stored)
    pdf_bytes = render_report_pdf(report_obj)  # rendered from the validated JSON — no LLM call
    return Response(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{_safe_filename(report_obj.title)}.pdf"'},
    )


@app.get("/report/{report_id}/docx")
def api_get_report_docx(report_id: str):
    stored = cache.get_report_by_id(report_id)
    if not stored:
        raise HTTPException(404, "Report not found — it may have expired, or the workspace has changed since it was generated.")
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

_PDF_GROUNDED_ANALYSIS_MODES = {"viva", "mock_test", "explain_figure", "recommend", "project_interview_start", "project_interview_evaluate"}
# "Why This Design?" / "System Design" are explicitly about VerityRAG's OWN
# architecture, not the uploaded paper — the only two modes still grounded
# in analysis.REAL_ARCHITECTURE_CONTEXT instead of retrieval.
_NON_PDF_ANALYSIS_MODES = {"why_design", "system_design"}


@app.post("/analyze")
def api_analyze(req: AnalyzeRequest):
    if req.mode not in _PDF_GROUNDED_ANALYSIS_MODES and req.mode not in _NON_PDF_ANALYSIS_MODES:
        raise HTTPException(400, f"Unknown analyze mode: {req.mode}")

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
            chunks = _retrieve_for_analysis(query_text, target_document_ids, top_k)
            if not chunks:
                return {"ok": False, "mode": req.mode, "error": INSUFFICIENT_EVIDENCE_MESSAGE}
            evidence_text = _evidence_text_from_chunks(chunks)
            mode_label = "Viva" if req.mode == "viva" else "Mock Test"
            result = generate_pdf_questions(evidence_text, mode_label, req.difficulty, max(1, min(req.num_questions, 15)))
            documents_found = get_contributing_documents(chunks)
            if result is None:
                log_query_event(query=f"[analyze:{req.mode}]", documents_searched=active_document_count, total_latency_s=round(time.time() - start, 3), task_status="FAILED", **_call_observability_fields(request_id, active_document_count))
                return {"ok": False, "mode": req.mode, "error": GENERATION_FAILED_MESSAGE}
            log_query_event(query=f"[analyze:{req.mode}]", documents_searched=active_document_count, documents_contributing=len(documents_found), selected_evidence_count=len(chunks), total_latency_s=round(time.time() - start, 3), task_status="OK", **_call_observability_fields(request_id, active_document_count))
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
            chunks = _retrieve_for_analysis(query_text, target_document_ids, top_k=min(10, REPORT_CHUNKS_PER_DOC * max(len(target_document_ids), 1)))
            if not chunks:
                return {"ok": False, "mode": req.mode, "error": INSUFFICIENT_EVIDENCE_MESSAGE}
            evidence_text = _evidence_text_from_chunks(chunks)
            result = explain_figure(evidence_text, ref)
            documents_found = get_contributing_documents(chunks)
            if result is None:
                log_query_event(query="[analyze:explain_figure]", documents_searched=active_document_count, total_latency_s=round(time.time() - start, 3), task_status="FAILED", **_call_observability_fields(request_id, active_document_count))
                return {"ok": False, "mode": req.mode, "error": GENERATION_FAILED_MESSAGE}
            log_query_event(query="[analyze:explain_figure]", documents_searched=active_document_count, documents_contributing=len(documents_found), selected_evidence_count=len(chunks), total_latency_s=round(time.time() - start, 3), task_status="OK", **_call_observability_fields(request_id, active_document_count))
            answer = result.answer if result.evidence_sufficient else INSUFFICIENT_EVIDENCE_MESSAGE
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
            chunks = _retrieve_for_analysis(query_text, target_document_ids, top_k)
            if not chunks:
                return {"ok": False, "mode": req.mode, "error": INSUFFICIENT_EVIDENCE_MESSAGE}
            evidence_text = _evidence_text_from_chunks(chunks)
            result = recommend_from_comparison(evidence_text)
            documents_found = get_contributing_documents(chunks)
            if result is None:
                log_query_event(query="[analyze:recommend]", documents_searched=active_document_count, total_latency_s=round(time.time() - start, 3), task_status="FAILED", **_call_observability_fields(request_id, active_document_count))
                return {"ok": False, "mode": req.mode, "error": GENERATION_FAILED_MESSAGE}
            log_query_event(query="[analyze:recommend]", documents_searched=active_document_count, documents_contributing=len(documents_found), selected_evidence_count=len(chunks), total_latency_s=round(time.time() - start, 3), task_status="OK", **_call_observability_fields(request_id, active_document_count))
            answer = result.answer if result.evidence_sufficient else INSUFFICIENT_EVIDENCE_MESSAGE
            _persist_analysis_turn(req.session_id, None, "assistant_kicker", answer, {"kicker": "Recommendation"})
            return {
                "ok": True, "mode": req.mode, "answer": answer,
                "grounded": result.grounded, "evidence_sufficient": result.evidence_sufficient,
                "documents_used": [d["document_id"] for d in documents_found],
            }

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
            chunks = _retrieve_for_analysis(query_text, target_document_ids, top_k)
            if not chunks:
                return {"ok": False, "mode": req.mode, "error": INSUFFICIENT_EVIDENCE_MESSAGE}
            evidence_text = _evidence_text_from_chunks(chunks)
            documents_found = get_contributing_documents(chunks)

            if req.mode == "project_interview_start":
                result = start_project_interview(evidence_text, req.difficulty, req.topics)
                if result is None:
                    log_query_event(query="[analyze:project_interview_start]", documents_searched=active_document_count, total_latency_s=round(time.time() - start, 3), task_status="FAILED", **_call_observability_fields(request_id, active_document_count))
                    return {"ok": False, "mode": req.mode, "error": GENERATION_FAILED_MESSAGE}
                log_query_event(query="[analyze:project_interview_start]", documents_searched=active_document_count, documents_contributing=len(documents_found), selected_evidence_count=len(chunks), total_latency_s=round(time.time() - start, 3), task_status="OK", **_call_observability_fields(request_id, active_document_count))
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
            log_query_event(query="[analyze:project_interview_evaluate]", documents_searched=active_document_count, documents_contributing=len(documents_found), selected_evidence_count=len(chunks), total_latency_s=round(time.time() - start, 3), task_status="OK", **_call_observability_fields(request_id, active_document_count))
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

    return {
        "live": live_metrics,
        "live_note": "Aggregated from logs/verityrag_events.jsonl (last up to 500 real requests). null if no queries logged yet.",
        "offline_retrieval_eval": offline_eval,
        "not_measured": [
            "groundedness (no automated groundedness scorer wired to production)",
            "hallucination_rate (no automated hallucination scorer wired to production)",
        ],
    }
