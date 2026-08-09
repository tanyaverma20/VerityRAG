import os
import shutil
from ingest import (
    get_document_id,
    extract_pages_from_pdf,
    chunk_page,
    ingest_document,
    get_all_chunks
)
from retrieval import build_bm25_index, hybrid_retrieve

def test_same_pdf_same_id():
    path1 = "tests/fixtures/attention.pdf"
    path2 = "tests/fixtures/attention_copy.pdf"
    shutil.copy(path1, path2)
    
    id1 = get_document_id(path1)
    id2 = get_document_id(path2)
    
    os.remove(path2)
    assert id1 == id2, "TEST 1 FAILED: Same PDF contents should produce the same document_id"
    print("TEST 1 PASSED: Same PDF contents produce the same document_id")

def test_different_pdf_different_id():
    path1 = "tests/fixtures/attention.pdf"
    path2 = "tests/fixtures/dummy.pdf"
    with open(path2, "wb") as f:
        f.write(b"dummy")
        
    id1 = get_document_id(path1)
    id2 = get_document_id(path2)
    
    os.remove(path2)
    assert id1 != id2, "TEST 2 FAILED: Different PDF contents should produce different document_ids"
    print("TEST 2 PASSED: Different PDF contents produce different document_ids")

def test_chunk_schema():
    page = {"page_number": 1, "text": "This is a test paragraph for chunking."}
    chunks, next_idx, sec = chunk_page(page, "doc123", 1, "Unknown")
    
    assert len(chunks) > 0, "No chunks generated"
    c = chunks[0]
    keys = ["document_id", "parent_id", "chunk_id", "page_number", "section", "chunk_type"]
    for k in keys:
        assert k in c, f"TEST 3 FAILED: {k} missing from chunk"
    print("TEST 3 PASSED: Every generated child chunk has required fields")

def test_deterministic_chunk_ids():
    page = {"page_number": 1, "text": "This is a test paragraph for chunking."}
    chunks1, _, _ = chunk_page(page, "doc123", 1, "Unknown")
    chunks2, _, _ = chunk_page(page, "doc123", 1, "Unknown")
    
    assert chunks1[0]["chunk_id"] == chunks2[0]["chunk_id"], "TEST 4 FAILED: Chunk IDs are not deterministic"
    print("TEST 4 PASSED: Chunk IDs are deterministic")

def test_unique_chunk_ids():
    page = {"page_number": 1, "text": "A" * 1500} # Forces multiple chunks
    chunks, _, _ = chunk_page(page, "doc123", 1, "Unknown")
    
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "TEST 5 FAILED: Chunk IDs are not unique"
    print("TEST 5 PASSED: Chunk IDs are unique within a document")

def test_preserve_page_numbers():
    pages = extract_pages_from_pdf("tests/fixtures/attention.pdf")
    assert len(pages) > 1, "PDF should have multiple pages"
    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2
    print("TEST 6 PASSED: A PDF with multiple pages preserves page numbers correctly")

def test_unrecognized_sections():
    page = {"page_number": 1, "text": "Just some random text without any headings."}
    chunks, _, _ = chunk_page(page, "doc123", 1, "Unknown")
    assert chunks[0]["section"] == "Unknown", "TEST 7 FAILED: Unrecognized section did not fall back to 'Unknown'"
    print("TEST 7 PASSED: Missing/unrecognized section headings result in section='Unknown'")

def test_long_paragraph_split():
    long_text = "A " * 1000  # 2000 chars
    page = {"page_number": 1, "text": long_text}
    chunks, _, _ = chunk_page(page, "doc123", 1, "Unknown")
    assert len(chunks) > 1, "TEST 8 FAILED: Long paragraph wasn't split"
    print("TEST 8 PASSED: Very long paragraphs are safely split")

def test_chroma_insertion():
    res = ingest_document("tests/fixtures/attention.pdf")
    assert res["chunks_added"] > 0, "Ingestion should succeed and add chunks"
    
    chunks = get_all_chunks()
    assert len(chunks) > 0, "get_all_chunks() should return at least one chunk"
    
    c = chunks[0]
    assert "metadata" in c, "Chunk should have metadata"
    meta = c["metadata"]
    
    keys = ["document_id", "parent_id", "chunk_id", "page_number", "section", "chunk_type", "source"]
    for k in keys:
        assert k in meta, f"Metadata is missing {k}"
        
    print("TEST 9 PASSED: ChromaDB accepts the generated documents and metadata correctly")

def test_retrieval():
    # Build BM25 index which depends on Chroma content
    build_bm25_index()
    
    # Use retrieve() which supports document_ids filtering
    from retrieval import retrieve
    results = retrieve(
        "attention mechanism", 
        strategy="hybrid",
        top_k=2, 
        document_ids=[get_document_id("tests/fixtures/attention.pdf")]
    )
    assert len(results) > 0, "Retrieval returned zero results"
    
    res = results[0]
    assert "text" in res, "Result missing text"
    assert "metadata" in res, "Result missing metadata"
    
    meta = res["metadata"]
    assert "chunk_id" in meta, "Result metadata missing chunk_id"
    assert "document_id" in meta, "Result metadata missing document_id"
    assert "source" in meta, "Result metadata missing source"
    
    # Assert evidence can be traced to the newly indexed document
    assert meta["source"] == "attention.pdf", "Retrieved chunk did not originate from the newly indexed document"
    print("TEST 10 PASSED: Existing retrieval can still read the newly indexed chunks via hybrid_retrieve")

def run_all_tests():
    try:
        test_same_pdf_same_id()
        test_different_pdf_different_id()
        test_chunk_schema()
        test_deterministic_chunk_ids()
        test_unique_chunk_ids()
        test_preserve_page_numbers()
        test_unrecognized_sections()
        test_long_paragraph_split()
        test_chroma_insertion()
        test_retrieval()
        print("\nALL TESTS COMPLETED SUCCESSFULLY!")
    except Exception as e:
        import traceback
        print(f"Test failed with error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
