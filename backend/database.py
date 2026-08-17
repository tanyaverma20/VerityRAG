"""
database.py — compatibility shim.

All persistence logic now lives in db/ (SQLAlchemy ORM models + a
repository layer, dual SQLite/PostgreSQL — see db/session.py,
db/repository.py). This module exists ONLY so every existing
`from database import ...` call site (main.py, ingest.py, every test file)
keeps working completely unchanged.

Backend selection: PostgreSQL when DATABASE_URL is set (production),
otherwise the same SQLite file this module has always defaulted to
(config.REGISTRY_DB_PATH) — see db/session.py:resolve_database_url().
Vector embeddings are never stored here; ChromaDB remains solely
responsible for vector storage/retrieval.
"""
from db.repository import (
    get_connection, init_db,
    add_document, update_document_status, get_document, list_documents,
    list_documents_for_owner,
    documents_in_workspace, delete_document,
    create_workspace, get_workspace, list_workspaces, touch_workspace,
    create_collection, get_collection, list_collections,
    add_document_to_collection, remove_document_from_collection, delete_collection,
    create_session, get_session, list_sessions, list_sessions_for_owner, update_session_title,
    delete_session, update_session_collection,
    add_message, get_message, get_session_messages,
    create_task, update_task_status, get_task, task_belongs_to_workspace,
    create_user, get_user_by_id, get_user_by_email, count_users,
    bootstrap_claim_orphaned_workspaces,
    create_user_session, get_user_by_token_hash, delete_user_session, delete_expired_user_sessions,
    workspace_owner_id,
)

__all__ = [
    "get_connection", "init_db",
    "add_document", "update_document_status", "get_document", "list_documents",
    "list_documents_for_owner",
    "documents_in_workspace", "delete_document",
    "create_workspace", "get_workspace", "list_workspaces", "touch_workspace",
    "create_collection", "get_collection", "list_collections",
    "add_document_to_collection", "remove_document_from_collection", "delete_collection",
    "create_session", "get_session", "list_sessions", "list_sessions_for_owner", "update_session_title",
    "delete_session", "update_session_collection",
    "add_message", "get_message", "get_session_messages",
    "create_task", "update_task_status", "get_task", "task_belongs_to_workspace",
    "create_user", "get_user_by_id", "get_user_by_email", "count_users",
    "bootstrap_claim_orphaned_workspaces",
    "create_user_session", "get_user_by_token_hash", "delete_user_session", "delete_expired_user_sessions",
    "workspace_owner_id",
]

# Initialize the schema when this module loads — identical to the original
# database.py's behavior. init_db() is additive-only (CREATE TABLE IF NOT
# EXISTS equivalent), so this is always a safe no-op against a database
# that already has real production data.
init_db()
