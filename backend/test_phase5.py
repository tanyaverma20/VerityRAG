import pytest
import sqlite3
import uuid
import os
from pathlib import Path
from database import (
    get_connection, init_db, add_document, get_document, list_documents,
    delete_document, create_collection, get_collection, list_collections,
    add_document_to_collection, remove_document_from_collection
)
from ingest import ingest_document, delete_document_from_index, get_collection as get_chroma_collection
from retrieval import build_bm25_index

@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    # Clean up test data if needed, but since it's a test suite it runs in the real DB.
    # To be safe, we just use random IDs for test data.
    yield

def test_sqlite_initialization():
    conn = get_connection()
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t["name"] for t in tables]
    assert "documents" in table_names
    assert "collections" in table_names
    assert "collection_documents" in table_names
    conn.close()

def test_document_registration_and_lifecycle():
    doc_id = uuid.uuid4().hex
    # 1. Register
    add_document(doc_id, "test_file.pdf", status="UPLOADED")
    doc = get_document(doc_id)
    assert doc is not None
    assert doc["filename"] == "test_file.pdf"
    assert doc["ingestion_status"] == "UPLOADED"
    
    # 2. Update status
    from database import update_document_status
    update_document_status(doc_id, "INDEXED", 42)
    doc = get_document(doc_id)
    assert doc["ingestion_status"] == "INDEXED"
    assert doc["chunk_count"] == 42
    
    # 3. Duplicate handling (upsert)
    add_document(doc_id, "test_file_renamed.pdf", status="PROCESSING")
    doc = get_document(doc_id)
    assert doc["filename"] == "test_file_renamed.pdf"
    assert doc["ingestion_status"] == "PROCESSING"

    # 4. Deletion
    delete_document(doc_id)
    assert get_document(doc_id) is None

def test_collections_architecture():
    col_id = uuid.uuid4().hex
    doc_id_1 = uuid.uuid4().hex
    doc_id_2 = uuid.uuid4().hex
    
    add_document(doc_id_1, "paper1.pdf")
    add_document(doc_id_2, "paper2.pdf")
    
    # 1. Create collection
    create_collection(col_id, "My Test Collection")
    col = get_collection(col_id)
    assert col["name"] == "My Test Collection"
    
    # 2. Add docs
    add_document_to_collection(col_id, doc_id_1)
    add_document_to_collection(col_id, doc_id_2)
    col = get_collection(col_id)
    assert len(col["document_ids"]) == 2
    assert doc_id_1 in col["document_ids"]
    
    # 3. Remove doc
    remove_document_from_collection(col_id, doc_id_1)
    col = get_collection(col_id)
    assert len(col["document_ids"]) == 1
    
    # 4. Delete document should cascade to collection_documents
    delete_document(doc_id_2)
    col = get_collection(col_id)
    assert len(col["document_ids"]) == 0
    
    # Cleanup
    from database import delete_collection
    delete_collection(col_id)

def test_chromadb_cleanup_after_deletion():
    # Use a real file to ingest
    test_pdf_path = Path(__file__).parent.parent / "data" / "attention.pdf"
    if not test_pdf_path.exists():
        pytest.skip("Test PDF not found.")
        
    result = ingest_document(str(test_pdf_path))
    doc_id = result["document_id"]
    
    # Ensure it is in ChromaDB
    chroma_col = get_chroma_collection()
    chroma_res = chroma_col.get(where={"document_id": doc_id})
    assert len(chroma_res["ids"]) > 0
    
    # Ensure it is in SQLite
    doc = get_document(doc_id)
    assert doc is not None
    
    # Delete from index
    delete_document_from_index(doc_id)
    build_bm25_index()
    
    # Ensure removed from SQLite
    assert get_document(doc_id) is None
    
    # Ensure removed from ChromaDB
    chroma_res_after = chroma_col.get(where={"document_id": doc_id})
    assert len(chroma_res_after["ids"]) == 0
    
    # Re-ingest to restore the state for other tests
    ingest_document(str(test_pdf_path))
    build_bm25_index()

def test_api_compatibility():
    # Test that the main app can be imported without errors (syntax/dependencies)
    from main import app
    assert app.title == "VerityRAG API"
