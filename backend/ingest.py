"""
Ingestion pipeline: PDF -> text -> chunks -> embeddings -> Chroma.
Each chunk keeps its source document name and a chunk_id so later
retrieval results can be traced back to an exact passage.
"""
import hashlib
import re
from pathlib import Path

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from database import add_document, update_document_status, delete_document as db_delete_document

# Compatibility wrapper: uses sentence-transformers directly (already installed).
# Preserves the original embedding model: sentence-transformers/all-MiniLM-L6-v2
# Dimension: 384. ChromaDB requires __call__(Documents) -> Embeddings (list of lists).
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self._model.encode(list(input), convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

_embed_fn = SentenceTransformerEmbeddingFunction()
_client = None
_collection = None

def get_chroma_client():
    global _client
    if _client is None:
        from config import CHROMA_DIR
        import chromadb
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client

def get_collection():
    global _collection
    if _collection is None:
        from config import COLLECTION_NAME
        _collection = get_chroma_client().get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=_embed_fn
        )
    return _collection

def get_document_id(file_path: str) -> str:
    """Generate a deterministic document ID based on file contents."""
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def extract_pages_from_pdf(path: str) -> list[dict]:
    """Extract text from PDF while preserving page numbers (1-based)."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "page_number": i + 1,
                "text": text
            })
    return pages


def detect_section(line: str) -> str:
    """
    Lightweight deterministic section detector based on common headings.
    Returns the section name if found, else None.
    """
    line = line.strip()
    if not line:
        return None
        
    # Check for numbered headings like "1 Introduction" or "3.1 Dataset"
    numbered_match = re.match(r'^(\d+(?:\.\d+)*)\s+([A-Z].+)$', line)
    if numbered_match:
        return numbered_match.group(2).strip()
        
    # Check for common exact section names (case insensitive but usually Title Case)
    common_sections = [
        "Abstract", "Introduction", "Related Work", "Background", 
        "Method", "Methodology", "Proposed Method", "Architecture", 
        "Experiments", "Experimental Setup", "Dataset", "Results", 
        "Discussion", "Conclusion", "References"
    ]
    
    line_lower = line.lower()
    for sec in common_sections:
        if line_lower == sec.lower() or line_lower.startswith(sec.lower() + " ") or line_lower.startswith(sec.lower() + ":"):
            return sec
            
    return None


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraph-like blocks."""
    # Split by double newline or similar block separators
    blocks = re.split(r'\n\s*\n', text)
    return [b.strip() for b in blocks if b.strip()]


def chunk_page(page: dict, doc_id: str, start_chunk_idx: int, current_section: str) -> tuple[list[dict], int, str]:
    """
    Paragraph-aware chunking for a single page.
    Returns the chunks, the next chunk_idx, and the last detected section.
    """
    text = page["text"]
    page_num = page["page_number"]
    
    paragraphs = split_into_paragraphs(text)
    
    chunks = []
    current_chunk_text = ""
    chunk_idx = start_chunk_idx
    
    for para in paragraphs:
        # Try to detect a new section heading in the first line of the paragraph
        first_line = para.split('\n')[0]
        detected_sec = detect_section(first_line)
        if detected_sec:
            current_section = detected_sec
            
        # If adding this paragraph exceeds chunk size, save current chunk (if not empty)
        if len(current_chunk_text) + len(para) > CHUNK_SIZE and current_chunk_text:
            chunk_id = f"doc_{doc_id}_p{page_num:03d}_c{chunk_idx:03d}"
            parent_id = f"{doc_id}_p{page_num:03d}_parent"
            
            chunks.append({
                "document_id": doc_id,
                "parent_id": parent_id,
                "chunk_id": chunk_id,
                "text": current_chunk_text.strip(),
                "page_number": page_num,
                "section": current_section,
                "chunk_type": "child"
            })
            chunk_idx += 1
            
            # Start new chunk with overlap if possible (simplistic overlap using end of previous text)
            # A better overlap would be paragraph-based, but for now we'll just start fresh or take last N chars
            overlap_text = current_chunk_text[-CHUNK_OVERLAP:] if len(current_chunk_text) > CHUNK_OVERLAP else ""
            # Find a safe break point in overlap (like a space)
            space_idx = overlap_text.find(' ')
            if space_idx != -1:
                overlap_text = overlap_text[space_idx:]
            current_chunk_text = overlap_text.strip() + " " + para
        else:
            if current_chunk_text:
                current_chunk_text += "\n\n" + para
            else:
                current_chunk_text = para
                
        # Handle exceptionally long paragraphs that are larger than CHUNK_SIZE on their own
        while len(current_chunk_text) > CHUNK_SIZE * 1.5:
            # Force split
            split_point = current_chunk_text.rfind(' ', 0, CHUNK_SIZE)
            if split_point == -1:
                split_point = CHUNK_SIZE
                
            chunk_id = f"doc_{doc_id}_p{page_num:03d}_c{chunk_idx:03d}"
            parent_id = f"{doc_id}_p{page_num:03d}_parent"
            
            chunks.append({
                "document_id": doc_id,
                "parent_id": parent_id,
                "chunk_id": chunk_id,
                "text": current_chunk_text[:split_point].strip(),
                "page_number": page_num,
                "section": current_section,
                "chunk_type": "child"
            })
            chunk_idx += 1
            current_chunk_text = current_chunk_text[split_point:].strip()

    # Add the remaining text as the last chunk
    if current_chunk_text.strip():
        chunk_id = f"doc_{doc_id}_p{page_num:03d}_c{chunk_idx:03d}"
        parent_id = f"{doc_id}_p{page_num:03d}_parent"
        
        chunks.append({
            "document_id": doc_id,
            "parent_id": parent_id,
            "chunk_id": chunk_id,
            "text": current_chunk_text.strip(),
            "page_number": page_num,
            "section": current_section,
            "chunk_type": "child"
        })
        chunk_idx += 1

    return chunks, chunk_idx, current_section


def ingest_document(file_path: str, original_filename: str | None = None) -> dict:
    """
    Ingests a single PDF with page/section awareness and deterministic IDs.

    `original_filename` lets callers preserve the user's real filename when
    `file_path` points at a temp file (e.g. the /upload endpoint, which copies
    the upload into a NamedTemporaryFile before this runs). Defaults to the
    path's own name so bulk-ingestion scripts and tests are unaffected.
    """
    path = Path(file_path)
    display_name = original_filename or path.name
    doc_id = get_document_id(str(path))

    # 1. Register document in SQLite
    add_document(doc_id, display_name, status="PROCESSING")
    
    try:
        pages = extract_pages_from_pdf(str(path))
        if not pages:
            update_document_status(doc_id, status="FAILED", error_message="No text extracted")
            return {"filename": display_name, "chunks_added": 0, "status": "no_text_extracted"}

        all_chunks = []
        chunk_idx = 1
        current_section = "Unknown"
        
        for page in pages:
            page_chunks, chunk_idx, current_section = chunk_page(page, doc_id, chunk_idx, current_section)
            all_chunks.extend(page_chunks)

        if not all_chunks:
            update_document_status(doc_id, status="FAILED", error_message="No text extracted")
            return {"filename": display_name, "chunks_added": 0, "status": "no_text_extracted"}

        ids = [c["chunk_id"] for c in all_chunks]
        texts = [c["text"] for c in all_chunks]
        
        # Chroma metadata doesn't accept complex types or None. Must be str, int, float, or bool.
        metadatas = []
        for c in all_chunks:
            metadatas.append({
                "document_id": c["document_id"],
                "filename": display_name,
                "source": display_name,  # for backward compatibility with main.py
                "page_number": c["page_number"],
                "section": c["section"],
                "chunk_id": c["chunk_id"],
                "parent_id": c["parent_id"],
                "chunk_type": c["chunk_type"]
            })

        # Check if this document is already in ChromaDB. If it is, delete it first to avoid duplicates.
        # But wait, document_id is deterministic. If it's already indexed, it might not need re-indexing.
        # But just in case, let's delete existing chunks for this document.
        col = get_collection()
        col.delete(where={"document_id": doc_id})
        col.add(documents=texts, ids=ids, metadatas=metadatas)

        update_document_status(doc_id, status="INDEXED", chunk_count=len(all_chunks))
        return {"filename": display_name, "chunks_added": len(all_chunks), "status": "ok", "document_id": doc_id}
    
    except Exception as e:
        update_document_status(doc_id, status="FAILED", error_message=str(e))
        raise e


def get_all_chunks() -> list[dict]:
    """Pulls every stored chunk — used to build the BM25 index in retrieval.py."""
    data = get_collection().get()
    return [
        {"id": _id, "text": doc, "metadata": meta}
        for _id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]



def delete_document_from_index(document_id: str):
    """Deletes a document from ChromaDB and the SQLite registry."""
    # 1. Delete from ChromaDB
    try:
        get_collection().delete(where={"document_id": document_id})
    except Exception as e:
        print(f"Failed to delete from Chroma: {e}")
        
    # 2. Delete from SQLite Registry
    db_delete_document(document_id)
    
    # 3. Note: BM25 index must be rebuilt after this in the main API layer.
