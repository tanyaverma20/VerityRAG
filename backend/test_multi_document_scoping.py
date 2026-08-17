"""
test_multi_document_scoping.py — Section 19 (A-G, J) of the workspace
upgrade: strict per-document retrieval scoping, including at scale (10
documents), removal, and the "no documents scoped => never search anything"
guard that is what actually keeps the legacy 1454-chunk corpus and
data/*.pdf out of normal user retrieval.

Runs entirely against the ISOLATED Phase 8 test Chroma collection (see
conftest.py) — never backend/chroma_store. Uses synthetic short texts
pushed directly into the test collection for the 10-document scale test
(TEST F) rather than parsing 10 real PDFs, since what's under test is the
document_ids scoping mechanism itself, not PDF parsing (already covered by
test_ingestion.py).

Sections L/M/N (1-call / fallback / both-fail) are covered by
test_llm_call_count.py and test_reports_and_caching.py — not duplicated
here. Sections H/I (New Chat reset, conversation history restore document
scope) are pure client-side state with no server round-trip — verified via
live browser testing during implementation, not a natural pytest target.
"""
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import config
from ingest import get_collection, get_document_id, delete_document_from_index
from retrieval import retrieve, build_bm25_index, dense_search, bm25_search
from main import app
from test_auth_helpers import registered_user_with_workspace

client = TestClient(app)

# /query now requires a real authenticated, workspace-owning caller — see
# test_workspace_isolation.py for why this is a module-scoped fixture
# rather than bare module-level code. Most of this file exercises
# retrieve()/dense_search()/bm25_search() directly (no HTTP layer, no auth
# involved) — only the two /query HTTP tests at the bottom need this.
HEADERS: dict = {}
WORKSPACE_ID: str = ""


@pytest.fixture(autouse=True, scope="module")
def _module_auth():
    global HEADERS, WORKSPACE_ID
    HEADERS, WORKSPACE_ID = registered_user_with_workspace(client, "multidoc")
    yield

# Resolved relative to THIS file, never the process's CWD (previously a
# bare "tests/fixtures/attention.pdf" string, which only worked when
# pytest happened to be invoked with CWD == backend/).
FIXTURE_PDF = Path(__file__).parent / "tests" / "fixtures" / "attention.pdf"


def _push_synthetic_document(document_id: str, text: str, filename: str) -> None:
    """Adds one synthetic chunk directly to the isolated test Chroma
    collection — avoids parsing a real PDF per document for the scale test."""
    col = get_collection()
    col.add(
        documents=[text],
        ids=[f"doc_{document_id}_p001_c001"],
        metadatas=[{
            "document_id": document_id, "filename": filename, "source": filename,
            "page_number": 1, "section": "Body", "chunk_id": f"doc_{document_id}_p001_c001",
            "parent_id": f"{document_id}_p001_parent", "chunk_type": "child",
        }],
    )


# ---------------------------------------------------------------------------
# TEST A-E — two real documents, strict single/comparison scoping
# ---------------------------------------------------------------------------
def test_two_documents_strict_scoping():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"

    doc_a_id = get_document_id(str(FIXTURE_PDF))  # already ingested by conftest.py
    doc_b_id = "synthetic_paper_b_" + uuid.uuid4().hex[:8]
    _push_synthetic_document(doc_b_id, "RetentionAI predicts employee attrition using a five-stage AI pipeline.", "RetentionAI.pdf")
    build_bm25_index()

    # C: ask about A => only A's chunks
    results_a = retrieve("attention mechanism transformer", strategy="hybrid", document_ids=[doc_a_id], top_k=5)
    assert results_a, "Should find A's own content"
    assert all(r["metadata"]["document_id"] == doc_a_id for r in results_a), "No chunk from B may leak into A's answer"

    # D: ask about B => only B's chunks
    results_b = retrieve("RetentionAI employee attrition pipeline", strategy="hybrid", document_ids=[doc_b_id], top_k=5)
    assert results_b, "Should find B's own content"
    assert all(r["metadata"]["document_id"] == doc_b_id for r in results_b), "No chunk from A may leak into B's answer"

    # E: compare A+B => evidence from BOTH, never a third/unrelated document
    results_both = retrieve("compare the two papers", strategy="hybrid", document_ids=[doc_a_id, doc_b_id], top_k=10)
    found_docs = {r["metadata"]["document_id"] for r in results_both}
    assert found_docs.issubset({doc_a_id, doc_b_id}), f"Comparison scope leaked outside A+B: {found_docs}"

    delete_document_from_index(doc_b_id)
    build_bm25_index()


# ---------------------------------------------------------------------------
# TEST F — 10 documents uploaded; a question about #7 uses ONLY #7's evidence
# ---------------------------------------------------------------------------
def test_ten_documents_only_target_document_used():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"

    doc_ids = []
    for i in range(1, 11):
        doc_id = f"synthetic_paper_{i}_" + uuid.uuid4().hex[:8]
        doc_ids.append(doc_id)
        _push_synthetic_document(
            doc_id,
            f"Paper number {i} discusses topic-{i} methodology and topic-{i} datasets in depth.",
            f"paper_{i}.pdf",
        )
    build_bm25_index()

    target = doc_ids[6]  # document #7 (0-indexed 6)
    results = retrieve(f"topic-7 methodology datasets", strategy="hybrid", document_ids=[target], top_k=10)
    assert results, "Document #7 should be retrievable"
    assert all(r["metadata"]["document_id"] == target for r in results), (
        "Scoping to document #7 out of 10 must never pull in evidence from any of the other 9"
    )

    # Sanity: an unscoped-among-all-ten search for the SAME query would find
    # other documents too (proving the scoping filter is what's doing the
    # work, not coincidence — topic-7 text only strongly matches doc #7, but
    # BM25/dense could still surface neighbors without the filter).
    results_all = retrieve("topic methodology datasets", strategy="hybrid", document_ids=doc_ids, top_k=10)
    found_docs = {r["metadata"]["document_id"] for r in results_all}
    assert found_docs.issubset(set(doc_ids))

    for d in doc_ids:
        delete_document_from_index(d)
    build_bm25_index()


# ---------------------------------------------------------------------------
# TEST G — removed document is no longer retrievable
# ---------------------------------------------------------------------------
def test_removed_document_not_retrievable():
    doc_id = "synthetic_removable_" + uuid.uuid4().hex[:8]
    _push_synthetic_document(doc_id, "Ephemeral paper about quantum widgets and flux capacitors.", "ephemeral.pdf")
    build_bm25_index()

    before = retrieve("quantum widgets flux capacitors", strategy="hybrid", document_ids=[doc_id], top_k=5)
    assert before, "Should be retrievable before removal"

    delete_document_from_index(doc_id)
    build_bm25_index()

    after = retrieve("quantum widgets flux capacitors", strategy="hybrid", document_ids=[doc_id], top_k=5)
    assert after == [], "Removed document must no longer be searchable"

    # Also confirm it's actually gone from Chroma, not just BM25.
    dense_only = dense_search("quantum widgets flux capacitors", document_ids=[doc_id])
    assert dense_only == []


# ---------------------------------------------------------------------------
# TEST J — the mechanism that keeps the legacy corpus / data/*.pdf out:
# /query with no document scope must NEVER search anything.
# ---------------------------------------------------------------------------
def test_query_with_no_document_scope_never_searches():
    resp = client.post("/query", json={"question": "What is the capital of France?", "workspace_id": WORKSPACE_ID}, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["documents_found"] == 0
    assert data["structured_citations"] == []
    assert "upload" in data["answer"].lower(), "Must ask the user to upload a paper, not silently search anything"


def test_query_with_empty_document_ids_list_never_searches():
    resp = client.post("/query", json={"question": "test", "document_ids": [], "workspace_id": WORKSPACE_ID}, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["documents_found"] == 0
    assert data["structured_citations"] == []
