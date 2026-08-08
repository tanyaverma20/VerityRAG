import re as _re
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import uuid
import time

from ingest import ingest_document, get_collection, delete_document_from_index
from retrieval import retrieve, hybrid_retrieve, build_bm25_index, get_contributing_documents
from verify import verify_answer
from llm import call_llm
from database import (
    list_documents, get_document,
    list_collections, get_collection as db_get_collection, create_collection,
    add_document_to_collection, remove_document_from_collection, delete_collection,
    create_session, get_session, list_sessions, delete_session, update_session_collection,
    add_message, get_session_messages,
    create_task, update_task_status, get_task
)
from observability import log_query_event, filter_events
from query_transform import (
    NO_DOCUMENTS_MESSAGE, INSUFFICIENT_EVIDENCE_MESSAGE, RATE_LIMIT_MESSAGE,
    GENERATION_FAILED_MESSAGE, is_rate_limit_error,
    start_call_tracking, get_call_log,
)
from config import GROQ_MODEL
import cache
from schemas import ResearchReport, PaperReport, ComparisonReport
from report_generator import generate_report, render_report_markdown, render_report_pdf, render_report_docx


def _call_observability_fields() -> dict:
    """
    Reads the physical Groq call log recorded during this request (see
    query_transform.start_call_tracking/get_call_log) and shapes it for
    log_query_event(). Internal/developer-log only — never returned to the
    frontend (item 17).
    """
    log = get_call_log()
    return {
        "llm_calls": len(log),
        "fallback_triggered": any(c["role"] == "fallback" for c in log),
        "model_name": GROQ_MODEL,
    }


_UNCACHEABLE_ANSWERS = {RATE_LIMIT_MESSAGE, GENERATION_FAILED_MESSAGE}


def _is_cacheable_answer(answer_text: str) -> bool:
    """Never cache a transient failure — a retry a minute later should get a
    real attempt, not the same stale error forever (item 13)."""
    return bool(answer_text) and answer_text not in _UNCACHEABLE_ANSWERS


def _resolve_document_scope(document_ids: list[str] | None, collection_id: str | None) -> list[str]:
    if collection_id and not document_ids:
        col = db_get_collection(collection_id)
        if col:
            return col.get("document_ids", [])
    return document_ids or []


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


@app.on_event("startup")
def startup():
    build_bm25_index()


@app.get("/health")
def health():
    collection = get_collection()
    return {"status": "ok", "chunks_indexed": collection.count()}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported right now.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = ingest_document(tmp_path, original_filename=file.filename)
        build_bm25_index()  # rebuild so the new doc is searchable immediately
        cache.clear_all()   # the workspace changed — stale cached answers/reports must not survive it
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)

# --- Document Endpoints ---

@app.get("/documents")
def api_list_documents():
    return list_documents()

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
    return create_session(sess_id, req.collection_id)

@app.get("/sessions")
def api_list_sessions():
    return list_sessions()

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
            documents_contributing=payload.get("documents_found", 0),
            selected_evidence_count=len(chunks),
            retrieval_latency_s=payload.get("latency_s", 0),
            total_latency_s=payload.get("latency_s", 0),
            research_iterations=payload.get("research_iterations", 1),
            evidence_gaps=len(payload.get("evidence_gaps", [])),
            contradiction_count=len(payload.get("contradictions", [])),
            research_gap_count=len(payload.get("research_gaps", [])),
            confidence=payload.get("confidence", "UNAVAILABLE"),
            task_status="COMPLETED",
            **_call_observability_fields(),
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

    target_document_ids = _resolve_document_scope(request.document_ids, request.collection_id)

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
            **_call_observability_fields(),
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
            **_call_observability_fields(),
        )
        return {
            "answer": final_state.get("draft_answer") or INSUFFICIENT_EVIDENCE_MESSAGE,
            "sources": [],
            "structured_citations": [],
            "verification": [],
            "documents_found": 0,
        }

    documents_found = get_contributing_documents(chunks)
    payload = {
        "answer": final_state.get("draft_answer") or INSUFFICIENT_EVIDENCE_MESSAGE,
        "sources": final_state.get("citations", []),
        "structured_citations": final_state.get("citations", []),
        "documents_found": documents_found,
        # Convenience alias matching the documented internal API shape
        # (documents_used: [id, ...]) — additive, doesn't replace documents_found.
        "documents_used": [d["document_id"] for d in documents_found],
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
        documents_contributing=payload.get("documents_found", 0),
        selected_evidence_count=len(chunks),
        total_latency_s=payload.get("latency_s", 0),
        research_iterations=payload.get("research_iterations", 1),
        confidence=payload.get("confidence", "UNAVAILABLE"),
        task_status="OK",
        **_call_observability_fields(),
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

    start_call_tracking()
    start = time.time()

    cached = cache.get_cached_report(req.document_ids)
    if cached:
        log_query_event(
            query="[report]", documents_searched=len(req.document_ids),
            total_latency_s=round(time.time() - start, 3), task_status="CACHE_HIT",
            llm_calls=0, model_name=GROQ_MODEL,
        )
        return {k: v for k, v in cached.items() if k != "_cached_at"}

    result = generate_report(req.document_ids)  # <-- the ONE LLM call, internally

    if not result["ok"]:
        log_query_event(
            query="[report]", documents_searched=len(req.document_ids),
            total_latency_s=round(time.time() - start, 3), task_status="FAILED",
            errors=[result["error"]], **_call_observability_fields(),
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
        **_call_observability_fields(),
    )

    return {k: v for k, v in stored.items() if k != "_cached_at"}


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
