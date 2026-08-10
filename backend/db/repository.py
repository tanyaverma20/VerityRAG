"""
db/repository.py — CRUD repository layer, SQLAlchemy-backed.

Every function here has the EXACT name, signature, and return shape
(plain dict / list[dict], same keys) that database.py's raw-sqlite3
version already had, so main.py / ingest.py / every test file that does
`from database import ...` keeps working completely unchanged — see
database.py, which now just re-exports this module.

Transaction handling: every write goes through db.session.session_scope(),
which commits on success and rolls back + re-raises on any exception —
callers never manage commits themselves, and a failure never leaves a
half-written row.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional, List

from sqlalchemy import select, func, delete as sa_delete
from sqlalchemy.exc import SQLAlchemyError

from db.session import session_scope, get_engine, init_schema, is_sqlite
from db.models import Workspace, Document, Collection, CollectionDocument, Session as SessionModel, Message, Task


def _now() -> str:
    return datetime.utcnow().isoformat()


# ---------------------------------------------------------------------------
# Schema init + legacy-migration safety net
# ---------------------------------------------------------------------------
def _migrate_legacy_columns() -> None:
    """Additive-only, mirrors the original database.py's _migrate_schema:
    adds workspace_id/title columns to a pre-existing SQLite registry.db
    that predates workspace support, then backfills orphaned rows into one
    auto-created 'My Research' workspace. Never drops or rewrites a table.
    A no-op (each ALTER is individually swallowed if the column already
    exists) against every database this repo has actually shipped with."""
    engine = get_engine()
    with engine.connect() as conn:
        for stmt in (
            "ALTER TABLE documents ADD COLUMN workspace_id VARCHAR",
            "ALTER TABLE sessions ADD COLUMN workspace_id VARCHAR",
            "ALTER TABLE sessions ADD COLUMN title VARCHAR",
            "ALTER TABLE tasks ADD COLUMN workspace_id VARCHAR",
        ):
            try:
                conn.exec_driver_sql(stmt)
                conn.commit()
            except SQLAlchemyError:
                conn.rollback()  # column already exists — expected on every non-fresh DB

    with session_scope() as session:
        orphaned_docs = session.execute(select(func.count()).select_from(Document).where(Document.workspace_id.is_(None))).scalar_one()
        orphaned_sessions = session.execute(select(func.count()).select_from(SessionModel).where(SessionModel.workspace_id.is_(None))).scalar_one()
        if orphaned_docs or orphaned_sessions:
            now = _now()
            default_ws = session.get(Workspace, "default")
            if default_ws is None:
                session.add(Workspace(workspace_id="default", name="My Research", created_at=now, updated_at=now))
                # session_scope()'s sessionmaker is configured with
                # autoflush=False (deliberately, elsewhere, for isolation-
                # fixture correctness) — so the INSERT for this new
                # Workspace row is only PENDING in the ORM session here,
                # not yet sent to the database. The two statements below
                # are raw Core-style bulk UPDATEs (session.execute(), not
                # ORM object mutation), which do NOT trigger or wait on
                # autoflush. Without an explicit flush, they would run
                # BEFORE the workspace INSERT reaches the database,
                # violating documents_workspace_id_fkey /
                # sessions_workspace_id_fkey on any real foreign-key-
                # enforcing backend (confirmed as a real, actively-
                # occurring failure against live PostgreSQL — silently
                # swallowed by init_db()'s best-effort try/except, so the
                # backfill just never happened). Flushing here makes the
                # INSERT visible to the UPDATEs that follow, in the same
                # transaction.
                session.flush()
            session.execute(Document.__table__.update().where(Document.workspace_id.is_(None)).values(workspace_id="default"))
            session.execute(SessionModel.__table__.update().where(SessionModel.workspace_id.is_(None)).values(workspace_id="default"))


def init_db() -> None:
    init_schema()
    try:
        _migrate_legacy_columns()
    except SQLAlchemyError as e:
        # Best-effort safety net only — a fresh database (no pre-existing
        # rows) has nothing to migrate, and this must never block startup.
        print(f"[db] legacy-column migration check skipped: {e}")


def get_connection():
    """Legacy escape hatch retained ONLY for the handful of existing tests
    that assert against raw sqlite_master / PRAGMA output directly
    (test_phase5.py, test_phase7.py). Always SQLite — this was never a
    Postgres-compatible call in the original database.py either. Raises
    clearly if the configured backend isn't SQLite, rather than silently
    connecting to the wrong database."""
    import sqlite3
    from pathlib import Path
    url = None
    from db.session import resolve_database_url
    url = resolve_database_url()
    if not is_sqlite(url):
        raise RuntimeError(
            "get_connection() is a SQLite-only legacy helper (used by a couple of "
            "raw-schema tests) and DATABASE_URL is configured for a non-SQLite backend. "
            "Use the repository functions instead of a raw connection in production/Postgres mode."
        )
    db_path = Path(url.replace("sqlite:///", "", 1))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ---------------------------------------------------------------------------
# Serialization helpers — plain dicts, same keys the original returned.
# ---------------------------------------------------------------------------
def _doc_dict(d: Document) -> dict:
    return {
        "document_id": d.document_id, "filename": d.filename, "title": d.title,
        "authors": d.authors, "year": d.year, "page_count": d.page_count,
        "chunk_count": d.chunk_count, "ingestion_status": d.ingestion_status,
        "error_message": d.error_message, "workspace_id": d.workspace_id,
        "created_at": d.created_at, "updated_at": d.updated_at,
    }


def _ws_dict(w: Workspace, session) -> dict:
    paper_count = session.execute(select(func.count()).select_from(Document).where(Document.workspace_id == w.workspace_id)).scalar_one()
    chat_count = session.execute(select(func.count()).select_from(SessionModel).where(SessionModel.workspace_id == w.workspace_id)).scalar_one()
    return {
        "workspace_id": w.workspace_id, "name": w.name,
        "created_at": w.created_at, "updated_at": w.updated_at,
        "paper_count": paper_count, "chat_count": chat_count,
    }


def _collection_dict(c: Collection, session) -> dict:
    doc_ids = session.execute(select(CollectionDocument.document_id).where(CollectionDocument.collection_id == c.collection_id)).scalars().all()
    return {
        "collection_id": c.collection_id, "name": c.name, "description": c.description,
        "created_at": c.created_at, "document_ids": list(doc_ids),
    }


def _session_dict(s: SessionModel) -> dict:
    return {
        "session_id": s.session_id, "collection_id": s.collection_id, "workspace_id": s.workspace_id,
        "title": s.title, "created_at": s.created_at, "updated_at": s.updated_at,
    }


def _message_dict(m: Message) -> dict:
    return {
        "message_id": m.message_id, "session_id": m.session_id, "role": m.role,
        "content": m.content, "metadata": json.loads(m.metadata_json) if m.metadata_json else None,
        "created_at": m.created_at,
    }


def _task_dict(t: Task) -> dict:
    d = {
        "task_id": t.task_id, "session_id": t.session_id, "workspace_id": t.workspace_id, "status": t.status,
        "created_at": t.created_at, "started_at": t.started_at, "completed_at": t.completed_at,
        "error_message": t.error_message, "result_payload": t.result_payload,
        "progress_metadata": t.progress_metadata,
    }
    for key in ("result_payload", "progress_metadata"):
        if d[key] and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except Exception:
                pass
    return d


# ---------------------------------------------------------------------------
# Document operations
# ---------------------------------------------------------------------------
def add_document(document_id: str, filename: str, status: str = "UPLOADED", workspace_id: Optional[str] = None) -> dict:
    now = _now()
    with session_scope() as session:
        existing = session.get(Document, document_id)
        if existing:
            existing.filename = filename
            existing.ingestion_status = status
            if workspace_id:
                existing.workspace_id = workspace_id
            existing.updated_at = now
        else:
            session.add(Document(
                document_id=document_id, filename=filename, ingestion_status=status,
                workspace_id=workspace_id, chunk_count=0, created_at=now, updated_at=now,
            ))
        if workspace_id:
            ws = session.get(Workspace, workspace_id)
            if ws:
                ws.updated_at = now
    return get_document(document_id)


def update_document_status(document_id: str, status: str, chunk_count: int = 0, error_message: Optional[str] = None, page_count: Optional[int] = None) -> None:
    now = _now()
    with session_scope() as session:
        doc = session.get(Document, document_id)
        if not doc:
            return
        doc.ingestion_status = status
        doc.chunk_count = chunk_count
        doc.error_message = error_message
        doc.updated_at = now
        if page_count is not None:
            doc.page_count = page_count


def get_document(document_id: str) -> Optional[dict]:
    with session_scope() as session:
        doc = session.get(Document, document_id)
        return _doc_dict(doc) if doc else None


def list_documents(workspace_id: Optional[str] = None) -> List[dict]:
    with session_scope() as session:
        stmt = select(Document).order_by(Document.created_at.desc())
        if workspace_id:
            stmt = stmt.where(Document.workspace_id == workspace_id)
        return [_doc_dict(d) for d in session.execute(stmt).scalars().all()]


def documents_in_workspace(document_ids: List[str], workspace_id: str) -> List[str]:
    """Data-layer isolation check: never trusts the caller's document_ids
    list at face value — filters it down to only IDs that actually belong
    to workspace_id."""
    if not document_ids or not workspace_id:
        return []
    with session_scope() as session:
        valid = set(session.execute(
            select(Document.document_id).where(Document.workspace_id == workspace_id, Document.document_id.in_(document_ids))
        ).scalars().all())
    return [d for d in document_ids if d in valid]


def delete_document(document_id: str) -> None:
    with session_scope() as session:
        session.execute(sa_delete(Document).where(Document.document_id == document_id))


# ---------------------------------------------------------------------------
# Workspace operations
# ---------------------------------------------------------------------------
def create_workspace(workspace_id: str, name: str) -> dict:
    now = _now()
    with session_scope() as session:
        session.add(Workspace(workspace_id=workspace_id, name=name, created_at=now, updated_at=now))
    return get_workspace(workspace_id)


def get_workspace(workspace_id: str) -> Optional[dict]:
    with session_scope() as session:
        ws = session.get(Workspace, workspace_id)
        return _ws_dict(ws, session) if ws else None


def list_workspaces() -> List[dict]:
    with session_scope() as session:
        rows = session.execute(select(Workspace).order_by(Workspace.updated_at.desc())).scalars().all()
        return [_ws_dict(w, session) for w in rows]


def touch_workspace(workspace_id: str, conn=None) -> None:
    # `conn` kept as an accepted (unused) parameter for call-site
    # compatibility with the original signature; the session-per-call
    # transaction boundary already makes the separate-connection
    # optimization it existed for unnecessary here.
    now = _now()
    with session_scope() as session:
        ws = session.get(Workspace, workspace_id)
        if ws:
            ws.updated_at = now


# ---------------------------------------------------------------------------
# Collection operations
# ---------------------------------------------------------------------------
def create_collection(collection_id: str, name: str, description: Optional[str] = None) -> dict:
    now = _now()
    with session_scope() as session:
        session.add(Collection(collection_id=collection_id, name=name, description=description, created_at=now))
    return get_collection(collection_id)


def get_collection(collection_id: str) -> Optional[dict]:
    with session_scope() as session:
        c = session.get(Collection, collection_id)
        return _collection_dict(c, session) if c else None


def list_collections() -> List[dict]:
    with session_scope() as session:
        rows = session.execute(select(Collection).order_by(Collection.created_at.desc())).scalars().all()
        return [_collection_dict(c, session) for c in rows]


def add_document_to_collection(collection_id: str, document_id: str) -> None:
    with session_scope() as session:
        existing = session.get(CollectionDocument, (collection_id, document_id))
        if not existing:
            session.add(CollectionDocument(collection_id=collection_id, document_id=document_id))


def remove_document_from_collection(collection_id: str, document_id: str) -> None:
    with session_scope() as session:
        session.execute(sa_delete(CollectionDocument).where(
            CollectionDocument.collection_id == collection_id, CollectionDocument.document_id == document_id,
        ))


def delete_collection(collection_id: str) -> None:
    with session_scope() as session:
        session.execute(sa_delete(Collection).where(Collection.collection_id == collection_id))


# ---------------------------------------------------------------------------
# Session operations
# ---------------------------------------------------------------------------
def create_session(session_id: str, collection_id: Optional[str] = None, workspace_id: Optional[str] = None, title: Optional[str] = None) -> dict:
    now = _now()
    with session_scope() as session:
        session.add(SessionModel(
            session_id=session_id, collection_id=collection_id, workspace_id=workspace_id,
            title=title, created_at=now, updated_at=now,
        ))
        if workspace_id:
            ws = session.get(Workspace, workspace_id)
            if ws:
                ws.updated_at = now
    return get_session(session_id)


def get_session(session_id: str) -> Optional[dict]:
    with session_scope() as session:
        s = session.get(SessionModel, session_id)
        return _session_dict(s) if s else None


def list_sessions(workspace_id: Optional[str] = None) -> List[dict]:
    with session_scope() as session:
        stmt = select(SessionModel).order_by(SessionModel.updated_at.desc())
        if workspace_id:
            stmt = stmt.where(SessionModel.workspace_id == workspace_id)
        return [_session_dict(s) for s in session.execute(stmt).scalars().all()]


def update_session_title(session_id: str, title: str) -> None:
    now = _now()
    with session_scope() as session:
        s = session.get(SessionModel, session_id)
        if s:
            s.title = title
            s.updated_at = now


def delete_session(session_id: str) -> None:
    with session_scope() as session:
        session.execute(sa_delete(SessionModel).where(SessionModel.session_id == session_id))


def update_session_collection(session_id: str, collection_id: Optional[str]) -> None:
    now = _now()
    with session_scope() as session:
        s = session.get(SessionModel, session_id)
        if s:
            s.collection_id = collection_id
            s.updated_at = now


# ---------------------------------------------------------------------------
# Message operations
# ---------------------------------------------------------------------------
def add_message(message_id: str, session_id: str, role: str, content: str, metadata: Optional[dict] = None) -> dict:
    now = _now()
    meta_str = json.dumps(metadata) if metadata else None
    with session_scope() as session:
        session.add(Message(
            message_id=message_id, session_id=session_id, role=role, content=content,
            metadata_json=meta_str, created_at=now,
        ))
        s = session.get(SessionModel, session_id)
        if s:
            s.updated_at = now
    return get_message(message_id)


def get_message(message_id: str) -> Optional[dict]:
    with session_scope() as session:
        m = session.get(Message, message_id)
        return _message_dict(m) if m else None


def get_session_messages(session_id: str) -> List[dict]:
    with session_scope() as session:
        rows = session.execute(
            select(Message).where(Message.session_id == session_id).order_by(Message.created_at.asc())
        ).scalars().all()
        return [_message_dict(m) for m in rows]


# ---------------------------------------------------------------------------
# Task operations
# ---------------------------------------------------------------------------
def create_task(task_id: str, session_id: Optional[str] = None, workspace_id: Optional[str] = None) -> dict:
    now = _now()
    with session_scope() as session:
        session.add(Task(task_id=task_id, session_id=session_id, workspace_id=workspace_id, status="PENDING", created_at=now))
    return get_task(task_id)


def task_belongs_to_workspace(task_id: str, workspace_id: str) -> bool:
    """Data-layer isolation check for GET /task/{task_id}: mirrors
    documents_in_workspace()'s "never trust the caller" pattern. A task
    created without a workspace_id (legacy/no-workspace callers) has no
    ownership to check, so it stays reachable — unchanged pre-existing
    behavior — exactly like a document/session with no workspace_id."""
    task = get_task(task_id)
    if not task:
        return False
    if not task.get("workspace_id"):
        return True
    return task["workspace_id"] == workspace_id


def update_task_status(task_id: str, status: str, result_payload: Optional[dict] = None, error_message: Optional[str] = None, progress_metadata: Optional[dict] = None) -> None:
    now = _now()
    with session_scope() as session:
        t = session.get(Task, task_id)
        if not t:
            return
        if status == "RUNNING" and not t.started_at:
            t.started_at = now
        elif status in ("COMPLETED", "FAILED") and not t.completed_at:
            t.completed_at = now
        t.status = status
        if result_payload is not None:
            t.result_payload = json.dumps(result_payload)
        if progress_metadata is not None:
            t.progress_metadata = json.dumps(progress_metadata)
        if error_message is not None:
            t.error_message = error_message


def get_task(task_id: str) -> Optional[dict]:
    with session_scope() as session:
        t = session.get(Task, task_id)
        return _task_dict(t) if t else None
