"""
test_chunk_workspace_backfill.py — regression test for a real bug found
during this session's live browser verification pass: a document whose
Postgres row already carries a real workspace_id (e.g. backfilled by
db.repository._migrate_legacy_columns()) can still have Chroma chunks
tagged workspace_id="" if it was ingested before workspace scoping
existed. retrieval._scope_where_clause() deliberately treats an empty
chunk workspace_id as never matching a real workspace filter (fail-closed
isolation) — which silently makes such a document permanently
unretrievable (documents_found=0 on every real query) once its Postgres
row is backfilled, unless its Chroma metadata is backfilled too.

Confirmed live against the real running app: a genuine pre-existing
document (bdfaa68d8984f0dc, "attention.pdf") backfilled into the
'default' workspace returned documents_found=0 on
"What problem is this paper trying to solve?" because its 49 Chroma
chunks still read workspace_id="". ingest.backfill_orphaned_chunk_workspace_ids()
fixes it; this file proves the fix with a reproduced (smaller-scale)
version of the exact scenario, entirely inside the isolated test fixture
— it never touches real production data.
"""
import uuid
from unittest.mock import patch

import config
from ingest import get_collection, backfill_orphaned_chunk_workspace_ids
from retrieval import retrieve, build_bm25_index


def _push_legacy_chunk(document_id: str, suffix: str, text: str, filename: str, workspace_id: str = "") -> str:
    """Pushes a chunk the way pre-workspace-scoping ingestion did — an
    empty workspace_id tag, exactly like the real orphaned document's
    chunks were found to have."""
    col = get_collection()
    chunk_id = f"doc_{document_id}_{suffix}"
    col.add(
        documents=[text],
        ids=[chunk_id],
        metadatas=[{
            "document_id": document_id, "workspace_id": workspace_id,
            "filename": filename, "source": filename,
            "page_number": 1, "section": "Body", "chunk_id": chunk_id,
            "parent_id": f"{document_id}_{suffix}_parent", "chunk_type": "child",
        }],
    )
    return chunk_id


def _fresh_id(prefix: str) -> str:
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def test_reproduces_the_real_bug_orphaned_chunks_are_unretrievable_before_backfill():
    doc_id = _fresh_id("doc")
    _push_legacy_chunk(doc_id, "c1", "This paper studies sparse routing in mixture-of-experts models.", "legacy.pdf", workspace_id="")
    build_bm25_index()

    # Exactly the real bug: workspace-scoped retrieval finds nothing for a
    # document whose Postgres row says it belongs to "default" but whose
    # chunk metadata still says workspace_id="".
    chunks = retrieve("sparse routing mixture-of-experts", document_ids=[doc_id], top_k=10, workspace_id="default", apply_token_budget=False)
    assert chunks == []


def test_backfill_updates_stale_chunks_and_retrieval_succeeds_afterward():
    doc_id = _fresh_id("doc")
    chunk_id = _push_legacy_chunk(doc_id, "c1", "This paper studies sparse routing in mixture-of-experts models.", "legacy.pdf", workspace_id="")
    build_bm25_index()

    fake_doc_row = {"document_id": doc_id, "workspace_id": "default"}
    with patch("database.list_documents", return_value=[fake_doc_row]):
        updated_count = backfill_orphaned_chunk_workspace_ids()

    assert updated_count == 1

    # The chunk's own workspace_id metadata is now correct...
    col = get_collection()
    result = col.get(ids=[chunk_id], include=["metadatas"])
    assert result["metadatas"][0]["workspace_id"] == "default"

    # ...and workspace-scoped retrieval now actually finds it — the exact
    # fix for the real bug reproduced above.
    build_bm25_index()
    chunks = retrieve("sparse routing mixture-of-experts", document_ids=[doc_id], top_k=10, workspace_id="default", apply_token_budget=False)
    assert len(chunks) > 0
    assert all(c["metadata"]["document_id"] == doc_id for c in chunks)


def test_backfill_never_touches_a_chunk_already_correctly_tagged():
    doc_id = _fresh_id("doc")
    chunk_id = _push_legacy_chunk(doc_id, "c1", "Already-correct chunk.", "correct.pdf", workspace_id="ws-already-correct")
    build_bm25_index()

    fake_doc_row = {"document_id": doc_id, "workspace_id": "ws-already-correct"}
    with patch("database.list_documents", return_value=[fake_doc_row]):
        updated_count = backfill_orphaned_chunk_workspace_ids()

    assert updated_count == 0
    col = get_collection()
    result = col.get(ids=[chunk_id], include=["metadatas"])
    assert result["metadatas"][0]["workspace_id"] == "ws-already-correct"


def test_backfill_is_idempotent_second_run_is_a_no_op():
    doc_id = _fresh_id("doc")
    _push_legacy_chunk(doc_id, "c1", "Idempotency check chunk.", "idem.pdf", workspace_id="")
    build_bm25_index()

    fake_doc_row = {"document_id": doc_id, "workspace_id": "default"}
    with patch("database.list_documents", return_value=[fake_doc_row]):
        first_run = backfill_orphaned_chunk_workspace_ids()
        second_run = backfill_orphaned_chunk_workspace_ids()

    assert first_run == 1
    assert second_run == 0


def test_backfill_skips_documents_with_no_workspace_id_never_invents_one():
    doc_id = _fresh_id("doc")
    _push_legacy_chunk(doc_id, "c1", "Still-unassigned chunk.", "unassigned.pdf", workspace_id="")

    fake_doc_row = {"document_id": doc_id, "workspace_id": None}
    with patch("database.list_documents", return_value=[fake_doc_row]):
        updated_count = backfill_orphaned_chunk_workspace_ids()

    assert updated_count == 0
    col = get_collection()
    chunk_id = f"doc_{doc_id}_c1"
    result = col.get(ids=[chunk_id], include=["metadatas"])
    assert result["metadatas"][0]["workspace_id"] == ""
