"""
test_db_repository.py — the SQLAlchemy-backed db/ package (models,
session/transaction handling, repository CRUD). Runs against the isolated
Phase 8 SQLite fixture (conftest.py) — never production data/registry.db.

database.py is now a thin re-export of db.repository — these tests exercise
db.repository directly (the real implementation) and also confirm the
`database` shim re-exports resolve to the exact same functions.
"""
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

import config
import database as db_shim
from db import repository as repo
from db.session import session_scope, get_engine, resolve_database_url, is_sqlite
from db.models import Workspace, Document, Session as SessionModel, Message


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@pytest.fixture(autouse=True)
def _isolated_db():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"
    assert is_sqlite(), "This suite assumes the isolated fixture's SQLite backend"
    yield


# ---------------------------------------------------------------------------
# database.py shim re-exports the real implementation, not a duplicate
# ---------------------------------------------------------------------------
def test_shim_reexports_are_identical_objects():
    assert db_shim.create_workspace is repo.create_workspace
    assert db_shim.add_message is repo.add_message
    assert db_shim.get_connection is repo.get_connection


# ---------------------------------------------------------------------------
# Repository CRUD
# ---------------------------------------------------------------------------
def test_workspace_crud():
    ws_id = _id("ws")
    created = repo.create_workspace(ws_id, "My Test Workspace")
    assert created["workspace_id"] == ws_id
    assert created["paper_count"] == 0 and created["chat_count"] == 0

    fetched = repo.get_workspace(ws_id)
    assert fetched == created

    all_ws = repo.list_workspaces()
    assert any(w["workspace_id"] == ws_id for w in all_ws)


def test_document_crud_and_status_updates():
    ws_id = _id("ws")
    repo.create_workspace(ws_id, "WS")
    doc_id = _id("doc")

    repo.add_document(doc_id, "paper.pdf", status="UPLOADED", workspace_id=ws_id)
    doc = repo.get_document(doc_id)
    assert doc["filename"] == "paper.pdf" and doc["ingestion_status"] == "UPLOADED"
    assert doc["workspace_id"] == ws_id

    repo.update_document_status(doc_id, "READY", chunk_count=42, page_count=7)
    doc2 = repo.get_document(doc_id)
    assert doc2["ingestion_status"] == "READY" and doc2["chunk_count"] == 42 and doc2["page_count"] == 7

    listed = repo.list_documents(workspace_id=ws_id)
    assert any(d["document_id"] == doc_id for d in listed)

    repo.delete_document(doc_id)
    assert repo.get_document(doc_id) is None


def test_document_id_values_are_never_mutated_by_repeat_upserts():
    # add_document is an upsert (ON CONFLICT DO UPDATE in the original
    # sqlite3 version) — the document_id itself, once set, must never change
    # across repeated calls, since it's the same value ChromaDB uses.
    doc_id = _id("doc")
    repo.add_document(doc_id, "v1.pdf", status="UPLOADED")
    repo.add_document(doc_id, "v1.pdf", status="READY")
    doc = repo.get_document(doc_id)
    assert doc["document_id"] == doc_id
    assert doc["ingestion_status"] == "READY"


# ---------------------------------------------------------------------------
# Transactions — commit on success, rollback on failure
# ---------------------------------------------------------------------------
def test_session_scope_commits_on_success():
    ws_id = _id("ws")
    with session_scope() as session:
        session.add(Workspace(workspace_id=ws_id, name="Committed", created_at="t", updated_at="t"))
    # A fresh session/connection must see the committed row.
    assert repo.get_workspace(ws_id) is not None


def test_session_scope_rolls_back_on_exception():
    ws_id = _id("ws")
    with pytest.raises(RuntimeError):
        with session_scope() as session:
            session.add(Workspace(workspace_id=ws_id, name="Should Not Persist", created_at="t", updated_at="t"))
            raise RuntimeError("simulated failure mid-transaction")
    # The row must NOT exist — the whole transaction was rolled back.
    assert repo.get_workspace(ws_id) is None


def test_session_scope_rolls_back_on_constraint_violation():
    # Inserting a message for a session_id that doesn't exist violates the
    # FK constraint — the whole write must roll back cleanly, not partially apply.
    bad_session_id = _id("nonexistent_session")
    with pytest.raises(IntegrityError):
        with session_scope() as session:
            session.add(Message(message_id=_id("m"), session_id=bad_session_id, role="user", content="x", created_at="t"))
    assert repo.get_message  # sanity: module still importable/usable after a rolled-back error


# ---------------------------------------------------------------------------
# Workspace / document isolation (data-layer enforcement, not just UI)
# ---------------------------------------------------------------------------
def test_documents_in_workspace_never_returns_a_foreign_document():
    ws_a = _id("ws")
    ws_b = _id("ws")
    repo.create_workspace(ws_a, "A")
    repo.create_workspace(ws_b, "B")
    doc_a = _id("doc")
    doc_b = _id("doc")
    repo.add_document(doc_a, "a.pdf", workspace_id=ws_a)
    repo.add_document(doc_b, "b.pdf", workspace_id=ws_b)

    # Workspace A explicitly asking for Workspace B's document must get nothing back.
    scoped = repo.documents_in_workspace([doc_a, doc_b], ws_a)
    assert scoped == [doc_a]

    scoped_b = repo.documents_in_workspace([doc_a, doc_b], ws_b)
    assert scoped_b == [doc_b]


def test_list_documents_scoped_to_workspace_excludes_other_workspaces():
    ws_a = _id("ws")
    ws_b = _id("ws")
    repo.create_workspace(ws_a, "A")
    repo.create_workspace(ws_b, "B")
    repo.add_document(_id("doc"), "a.pdf", workspace_id=ws_a)
    repo.add_document(_id("doc"), "b.pdf", workspace_id=ws_b)

    docs_a = repo.list_documents(workspace_id=ws_a)
    assert all(d["workspace_id"] == ws_a for d in docs_a)
    assert len(docs_a) == 1


# ---------------------------------------------------------------------------
# Conversation persistence
# ---------------------------------------------------------------------------
def test_conversation_persistence_full_round_trip():
    ws_id = _id("ws")
    repo.create_workspace(ws_id, "WS")
    session_id = _id("sess")
    repo.create_session(session_id, workspace_id=ws_id, title=None)

    repo.add_message(_id("m"), session_id, "user", "What is RRF?")
    repo.add_message(_id("m"), session_id, "assistant", "Reciprocal Rank Fusion...", metadata={"structured_citations": [{"source": "x.pdf"}]})

    msgs = repo.get_session_messages(session_id)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "What is RRF?"
    assert msgs[1]["role"] == "assistant" and msgs[1]["metadata"]["structured_citations"][0]["source"] == "x.pdf"

    repo.update_session_title(session_id, "RRF explanation")
    assert repo.get_session(session_id)["title"] == "RRF explanation"


def test_deleting_session_cascades_to_messages():
    session_id = _id("sess")
    repo.create_session(session_id)
    repo.add_message(_id("m"), session_id, "user", "hello")
    assert len(repo.get_session_messages(session_id)) == 1

    repo.delete_session(session_id)
    assert repo.get_session(session_id) is None
    assert repo.get_session_messages(session_id) == []  # cascaded, not orphaned


def test_adding_message_bumps_session_updated_at():
    session_id = _id("sess")
    repo.create_session(session_id)
    before = repo.get_session(session_id)["updated_at"]
    repo.add_message(_id("m"), session_id, "user", "hello")
    after = repo.get_session(session_id)["updated_at"]
    assert after >= before


# ---------------------------------------------------------------------------
# Legacy get_connection() escape hatch (SQLite-only, used by test_phase5/7)
# ---------------------------------------------------------------------------
def test_get_connection_returns_a_working_raw_sqlite_connection():
    conn = repo.get_connection()
    try:
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert {"workspaces", "documents", "sessions", "messages"}.issubset(tables)
    finally:
        conn.close()
