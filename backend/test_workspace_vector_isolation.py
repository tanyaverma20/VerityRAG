"""
test_workspace_vector_isolation.py — workspace-safe vector isolation
(request D, item 13.D). Every indexed chunk carries workspace_id alongside
document_id/chunk_id/parent_id; retrieval enforces workspace_id AND
document_id scope directly at the Chroma layer (retrieval.py:
_scope_where_clause), not just via the pre-existing SQL-layer check
(_resolve_document_scope / documents_in_workspace) — defense in depth, not
a replacement for it. Runs entirely against the isolated Phase 8 fixture.

NOTE: this is workspace-level isolation, not user authentication — this
repo has no auth layer, and none is invented here (see the module's final
test, which documents that boundary explicitly).
"""
import uuid

import pytest

import config
from ingest import get_collection
from retrieval import retrieve, dense_search, bm25_search, build_bm25_index


def _push_chunk(document_id: str, workspace_id: str, suffix: str, text: str, filename: str) -> None:
    col = get_collection()
    chunk_id = f"doc_{document_id}_{suffix}"
    col.add(
        documents=[text],
        ids=[chunk_id],
        metadatas=[{
            "document_id": document_id, "workspace_id": workspace_id or "",
            "filename": filename, "source": filename,
            "page_number": 1, "section": "Body", "chunk_id": chunk_id,
            "parent_id": f"{document_id}_{suffix}_parent", "chunk_type": "child",
        }],
    )


def _fresh_id(prefix: str) -> str:
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@pytest.fixture
def two_workspace_docs():
    ws_a = _fresh_id("ws")
    ws_b = _fresh_id("ws")
    doc_a = _fresh_id("doc")
    doc_b = _fresh_id("doc")
    _push_chunk(doc_a, ws_a, "c1", "Workspace A's paper discusses sparse mixture-of-experts routing.", "paperA.pdf")
    _push_chunk(doc_b, ws_b, "c1", "Workspace B's paper discusses retrieval-augmented decoding.", "paperB.pdf")
    build_bm25_index()
    return ws_a, ws_b, doc_a, doc_b


# ---------------------------------------------------------------------------
# 1 & 2. Workspace A -> Document A retrieval; Workspace B -> Document B retrieval
# ---------------------------------------------------------------------------
def test_workspace_a_retrieves_only_its_own_document(two_workspace_docs):
    ws_a, ws_b, doc_a, doc_b = two_workspace_docs
    chunks = retrieve("mixture-of-experts routing", document_ids=[doc_a, doc_b], top_k=10, workspace_id=ws_a, apply_token_budget=False)
    doc_ids_seen = {c["metadata"]["document_id"] for c in chunks}
    assert doc_ids_seen.issubset({doc_a}), f"Workspace A must never see Workspace B's chunks, got {doc_ids_seen}"


def test_workspace_b_retrieves_only_its_own_document(two_workspace_docs):
    ws_a, ws_b, doc_a, doc_b = two_workspace_docs
    chunks = retrieve("retrieval-augmented decoding", document_ids=[doc_a, doc_b], top_k=10, workspace_id=ws_b, apply_token_budget=False)
    doc_ids_seen = {c["metadata"]["document_id"] for c in chunks}
    assert doc_ids_seen.issubset({doc_b}), f"Workspace B must never see Workspace A's chunks, got {doc_ids_seen}"


# ---------------------------------------------------------------------------
# 3. Workspace A querying "all documents" (no document_ids filter) -> only
#    Workspace A's documents, never a leak from Workspace B.
# ---------------------------------------------------------------------------
def test_workspace_scoped_query_with_no_document_ids_still_excludes_other_workspaces(two_workspace_docs):
    ws_a, ws_b, doc_a, doc_b = two_workspace_docs
    chunks = retrieve("paper", document_ids=None, top_k=20, workspace_id=ws_a, apply_token_budget=False)
    doc_ids_seen = {c["metadata"]["document_id"] for c in chunks}
    assert doc_b not in doc_ids_seen, "An unscoped-by-document-id query must still never cross into another workspace"


# ---------------------------------------------------------------------------
# 4. Workspace A explicitly requesting Document B -> empty (rejected)
# ---------------------------------------------------------------------------
def test_explicitly_requesting_a_foreign_workspaces_document_returns_nothing(two_workspace_docs):
    ws_a, ws_b, doc_a, doc_b = two_workspace_docs
    chunks = retrieve("retrieval-augmented decoding", document_ids=[doc_b], top_k=10, workspace_id=ws_a, apply_token_budget=False)
    assert chunks == [], "Explicitly asking for a document belonging to a different workspace must return nothing"


# ---------------------------------------------------------------------------
# 5. Same filename across different workspaces -> no collision (document_id
#    is content-hash-derived and independent of filename/workspace anyway,
#    but this proves the metadata itself doesn't conflate them).
# ---------------------------------------------------------------------------
def test_same_filename_in_two_workspaces_does_not_collide():
    ws_a = _fresh_id("ws")
    ws_b = _fresh_id("ws")
    doc_a = _fresh_id("doc")
    doc_b = _fresh_id("doc")
    _push_chunk(doc_a, ws_a, "c1", "Version from workspace A mentions gradient boosting.", "shared_name.pdf")
    _push_chunk(doc_b, ws_b, "c1", "Version from workspace B mentions convolutional networks.", "shared_name.pdf")
    build_bm25_index()

    chunks_a = retrieve("gradient boosting convolutional", document_ids=[doc_a, doc_b], top_k=10, workspace_id=ws_a, apply_token_budget=False)
    ids_a = {c["metadata"]["document_id"] for c in chunks_a}
    assert ids_a.issubset({doc_a})

    chunks_b = retrieve("gradient boosting convolutional", document_ids=[doc_a, doc_b], top_k=10, workspace_id=ws_b, apply_token_budget=False)
    ids_b = {c["metadata"]["document_id"] for c in chunks_b}
    assert ids_b.issubset({doc_b})


# ---------------------------------------------------------------------------
# 6. Stale document_ids (never actually ingested) -> safely filtered, no crash
# ---------------------------------------------------------------------------
def test_stale_document_ids_are_safely_filtered_not_erroring(two_workspace_docs):
    ws_a, ws_b, doc_a, doc_b = two_workspace_docs
    stale_id = "never_ingested_" + uuid.uuid4().hex[:8]
    chunks = retrieve("paper", document_ids=[doc_a, stale_id], top_k=10, workspace_id=ws_a, apply_token_budget=False)
    doc_ids_seen = {c["metadata"]["document_id"] for c in chunks}
    assert stale_id not in doc_ids_seen
    assert doc_ids_seen.issubset({doc_a})


# ---------------------------------------------------------------------------
# 7. Legacy chunks with no workspace_id tag -> never automatically retrieved
#    by a workspace-scoped query (they simply don't match any real
#    workspace_id, so they're excluded rather than leaked in).
# ---------------------------------------------------------------------------
def test_legacy_chunks_without_workspace_id_are_excluded_from_workspace_scoped_queries():
    ws_a = _fresh_id("ws")
    legacy_doc = _fresh_id("doc")
    _push_chunk(legacy_doc, "", "c1", "A legacy chunk ingested before workspace support existed.", "legacy.pdf")
    build_bm25_index()

    chunks = retrieve("legacy chunk ingested", document_ids=[legacy_doc], top_k=10, workspace_id=ws_a, apply_token_budget=False)
    assert chunks == [], "A legacy/untagged chunk must never surface in a workspace-scoped query"

    # But it's still reachable when NOT workspace-scoped (backward compatible
    # — this is the exact pre-existing single-tenant behavior, unaffected).
    unscoped_chunks = retrieve("legacy chunk ingested", document_ids=[legacy_doc], top_k=10, workspace_id=None, apply_token_budget=False)
    assert any(c["metadata"]["document_id"] == legacy_doc for c in unscoped_chunks)


# ---------------------------------------------------------------------------
# Unit-level: dense_search / bm25_search / _scope_where_clause directly
# ---------------------------------------------------------------------------
def test_dense_search_applies_workspace_filter(two_workspace_docs):
    ws_a, ws_b, doc_a, doc_b = two_workspace_docs
    results = dense_search("mixture-of-experts routing", top_k=10, document_ids=[doc_a, doc_b], workspace_id=ws_a)
    assert all(r["metadata"]["document_id"] == doc_a for r in results)


def test_bm25_search_applies_workspace_filter(two_workspace_docs):
    ws_a, ws_b, doc_a, doc_b = two_workspace_docs
    results = bm25_search("paper", top_k=10, document_ids=[doc_a, doc_b], workspace_id=ws_b)
    assert all(r["metadata"]["document_id"] == doc_b for r in results)


def test_workspace_id_is_optional_and_backward_compatible(two_workspace_docs):
    # Omitting workspace_id entirely must reproduce the exact original
    # (pre-this-feature) behavior: document_ids alone govern scope.
    ws_a, ws_b, doc_a, doc_b = two_workspace_docs
    results = retrieve("paper", document_ids=[doc_a, doc_b], top_k=10, apply_token_budget=False)
    doc_ids_seen = {c["metadata"]["document_id"] for c in results}
    assert doc_ids_seen == {doc_a, doc_b}, "Without workspace_id, both documents must remain reachable (unchanged legacy behavior)"


# ---------------------------------------------------------------------------
# Documented boundary: workspace isolation vs. user authentication
# ---------------------------------------------------------------------------
def test_no_user_authentication_layer_exists_this_is_workspace_isolation_only():
    """
    This repository has no user-authentication system (no login, no user
    table, no session-to-identity binding) — workspace_id is a plain
    client-supplied scope identifier, not a security principal. This test
    exists to document that boundary explicitly rather than let it be
    silently assumed: workspace-level data isolation is implemented and
    tested above; authenticating WHICH human is allowed to use a given
    workspace_id is a separate, unimplemented concern.
    """
    import inspect
    import database
    source_names = {name for name, _ in inspect.getmembers(database)}
    assert not any("user" in n.lower() or "auth" in n.lower() or "login" in n.lower() for n in source_names), (
        "If a user/auth table appears, this test (and the README's documented boundary) must be updated deliberately."
    )
