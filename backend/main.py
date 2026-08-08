import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingest import ingest_document, get_collection, delete_document_from_index
from retrieval import retrieve, hybrid_retrieve, build_bm25_index, get_contributing_documents
from verify import verify_answer
from llm import call_llm
from database import (
    list_documents, get_document, 
    list_collections, get_collection as db_get_collection, create_collection,
    add_document_to_collection, remove_document_from_collection, delete_collection
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

class CollectionCreateRequest(BaseModel):
    name: str
    description: str | None = None
    document_ids: list[str] = []


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
        result = ingest_document(tmp_path)
        build_bm25_index()  # rebuild so the new doc is searchable immediately
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


@app.post("/query")
def query(request: QueryRequest):
    import time
    from graph.workflow import research_app
    
    start = time.time()
    
    target_document_ids = request.document_ids
    if request.collection_id and not target_document_ids:
        col = db_get_collection(request.collection_id)
        if col:
            target_document_ids = col.get("document_ids", [])
    
    # Initialize LangGraph state
    initial_state = {
        "original_query": request.question,
        "document_ids": target_document_ids,
        "research_type": getattr(request, "research_type", "simple"),
    }
    
    # Run the graph
    try:
        final_state = research_app.invoke(initial_state)
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
                "answer": "Error executing query or no chunks found.",
                "sources": [],
                "structured_citations": [],
                "documents_found": 0,
                "verification": []
            }
        sources_text = "\n---\n".join(
            f"[{i+1}] (source: {c['metadata']['source']}) {c['text']}"
            for i, c in enumerate(chunks)
        )
        try:
            generation = call_llm(ANSWER_PROMPT.format(question=request.question, sources=sources_text))
        except Exception as e:
            generation = {"text": "Fallback Answer: LLM unavailable."}
            
        try:
            verification = verify_answer(generation["text"], chunks)
        except Exception:
            verification = []
        return {
            "answer": generation["text"],
            "sources": [
                {
                    "source": c["metadata"]["source"],
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
            ],
            "documents_found": get_contributing_documents(chunks),
            "verification": verification,
            "latency_seconds": round(time.time() - start, 3),
        }
    
    chunks = final_state.get("retrieval_results", [])
    if not chunks:
        return {
            "answer": "No documents have been ingested yet, or nothing relevant was found.",
            "sources": [],
            "structured_citations": [],
            "verification": [],
            "documents_found": 0,
        }

    return {
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
