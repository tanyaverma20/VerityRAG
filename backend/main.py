import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingest import ingest_document, get_collection
from retrieval import retrieve, hybrid_retrieve, build_bm25_index, get_contributing_documents
from verify import verify_answer
from llm import call_llm

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
    # Optional: restrict retrieval to specific document IDs (multi-paper filter)
    document_ids: list[str] | None = None
    strategy: str = "hybrid"


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


@app.post("/query")
def query(request: QueryRequest):
    import time
    from graph.workflow import research_app
    
    start = time.time()
    
    # Initialize LangGraph state
    initial_state = {
        "original_query": request.question,
        "document_ids": request.document_ids,
    }
    
    # Run the graph
    try:
        final_state = research_app.invoke(initial_state)
    except Exception as e:
        # Fallback to Phase 2 retrieval if graph fails
        print(f"Graph execution failed: {e}")
        chunks = retrieve(
            request.question,
            strategy=request.strategy,
            document_ids=request.document_ids,
            top_k=request.top_k,
        )
        if not chunks:
            return {
                "answer": "No documents have been ingested yet, or nothing relevant was found.",
                "sources": [],
                "verification": None,
            }
        sources_text = "\n---\n".join(
            f"[{i+1}] (source: {c['metadata']['source']}) {c['text']}"
            for i, c in enumerate(chunks)
        )
        generation = call_llm(ANSWER_PROMPT.format(question=request.question, sources=sources_text))
        verification = verify_answer(generation["text"], chunks)
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
            "verification": None,
        }

    return {
        "answer": final_state.get("draft_answer", ""),
        # Map original chunks for backward compatibility
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
        "structured_citations": final_state.get("citations", []),
        "documents_found": get_contributing_documents(chunks),
        "verification": final_state.get("verification_results", []),
        "latency_seconds": round(time.time() - start, 3),
        "research_plan": final_state.get("research_plan", {}),
    }
