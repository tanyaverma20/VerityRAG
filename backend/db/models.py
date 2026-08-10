"""
db/models.py — SQLAlchemy ORM models mirroring the existing SQLite schema
(see the original CREATE TABLE statements previously in database.py).

Column names, types, and relationships are unchanged from the original
schema so existing document_id / session_id / etc. values keep working
identically whether the backing store is SQLite (dev/test) or PostgreSQL
(production, when DATABASE_URL is set). No vector data lives here — that's
ChromaDB's job (backend/ingest.py, backend/retrieval.py).

Indexes are added on every column that's actually filtered/joined on in
repository.py (workspace_id, document_id-adjacent foreign keys, session_id)
— these were implicit table scans under raw sqlite3; PostgreSQL in
particular benefits from them under real concurrent load.
"""
from __future__ import annotations

from sqlalchemy import (
    Column, String, Integer, Text, ForeignKey, Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Workspace(Base):
    __tablename__ = "workspaces"

    workspace_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)

    documents = relationship("Document", back_populates="workspace", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="workspace", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    document_id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    title = Column(String, nullable=True)
    authors = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    page_count = Column(Integer, nullable=True)
    chunk_count = Column(Integer, default=0)
    ingestion_status = Column(String, nullable=False)
    error_message = Column(Text, nullable=True)
    workspace_id = Column(String, ForeignKey("workspaces.workspace_id", ondelete="CASCADE"), nullable=True)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)

    workspace = relationship("Workspace", back_populates="documents")

    __table_args__ = (
        Index("ix_documents_workspace_id", "workspace_id"),
        Index("ix_documents_created_at", "created_at"),
    )


class Collection(Base):
    __tablename__ = "collections"

    collection_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(String, nullable=False)

    documents = relationship("CollectionDocument", back_populates="collection", cascade="all, delete-orphan")


class CollectionDocument(Base):
    __tablename__ = "collection_documents"

    collection_id = Column(String, ForeignKey("collections.collection_id", ondelete="CASCADE"), primary_key=True)
    document_id = Column(String, ForeignKey("documents.document_id", ondelete="CASCADE"), primary_key=True)

    collection = relationship("Collection", back_populates="documents")

    __table_args__ = (
        Index("ix_collection_documents_document_id", "document_id"),
    )


class Session(Base):
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True)
    collection_id = Column(String, ForeignKey("collections.collection_id", ondelete="SET NULL"), nullable=True)
    workspace_id = Column(String, ForeignKey("workspaces.workspace_id", ondelete="CASCADE"), nullable=True)
    title = Column(String, nullable=True)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)

    workspace = relationship("Workspace", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_sessions_workspace_id", "workspace_id"),
        Index("ix_sessions_updated_at", "updated_at"),
    )


class Message(Base):
    __tablename__ = "messages"

    message_id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    metadata_json = Column("metadata", Text, nullable=True)
    created_at = Column(String, nullable=False)

    session = relationship("Session", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_session_id", "session_id"),
        Index("ix_messages_created_at", "created_at"),
    )


class Task(Base):
    __tablename__ = "tasks"

    task_id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=True)
    # workspace_id is deliberately NOT a ForeignKey (unlike Document/Session)
    # — a task must remain readable/gettable even if its workspace is later
    # deleted, matching how it already survives session deletion (session_id
    # is nullable here too). Nullable + no enforcement when omitted, exactly
    # like every other workspace_id column in this schema, so pre-existing
    # single-tenant callers are unaffected. Added specifically to close a
    # real gap found during the Phase 4 isolation audit: GET /task/{task_id}
    # previously had no workspace check at all, so any client that learned
    # a task_id (e.g. one they'd seen for their own request earlier) could
    # read the full result payload — including citations/evidence — of a
    # DIFFERENT workspace's background research task.
    workspace_id = Column(String, nullable=True)
    status = Column(String, nullable=False)
    created_at = Column(String, nullable=False)
    started_at = Column(String, nullable=True)
    completed_at = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    result_payload = Column(Text, nullable=True)
    progress_metadata = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_tasks_session_id", "session_id"),
        Index("ix_tasks_workspace_id", "workspace_id"),
    )
