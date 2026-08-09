import sqlite3
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, List, Dict, Any

def get_connection():
    from config import REGISTRY_DB_PATH
    db_path = Path(REGISTRY_DB_PATH)
    # Ensure data directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def _migrate_schema(conn) -> None:
    """
    Additive-only migrations for a pre-existing registry.db (never drops or
    rewrites a table — this DB already holds real historical documents/
    sessions/messages from prior live testing and must not lose any of it).

    Adds workspace_id (documents, sessions) and title (sessions) columns if
    an older schema is missing them, then backfills every pre-existing NULL
    workspace_id to one auto-created "My Research" workspace so nothing that
    already existed silently disappears from the new workspace-scoped UI.
    """
    doc_cols = {r["name"] for r in conn.execute("PRAGMA table_info(documents)")}
    if "workspace_id" not in doc_cols:
        conn.execute("ALTER TABLE documents ADD COLUMN workspace_id TEXT")

    session_cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
    if "workspace_id" not in session_cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN workspace_id TEXT")
    if "title" not in session_cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT")

    orphaned = conn.execute(
        "SELECT COUNT(*) c FROM documents WHERE workspace_id IS NULL "
        "UNION ALL SELECT COUNT(*) c FROM sessions WHERE workspace_id IS NULL"
    ).fetchall()
    if any(row["c"] > 0 for row in orphaned):
        now = _now()
        default_ws_id = "default"
        conn.execute("""
            INSERT OR IGNORE INTO workspaces (workspace_id, name, created_at, updated_at)
            VALUES (?, 'My Research', ?, ?)
        """, (default_ws_id, now, now))
        conn.execute("UPDATE documents SET workspace_id = ? WHERE workspace_id IS NULL", (default_ws_id,))
        conn.execute("UPDATE sessions SET workspace_id = ? WHERE workspace_id IS NULL", (default_ws_id,))


def init_db():
    conn = get_connection()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS workspaces (
                workspace_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                title TEXT,
                authors TEXT,
                year INTEGER,
                page_count INTEGER,
                chunk_count INTEGER DEFAULT 0,
                ingestion_status TEXT NOT NULL,
                error_message TEXT,
                workspace_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS collections (
                collection_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS collection_documents (
                collection_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                PRIMARY KEY (collection_id, document_id),
                FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE CASCADE,
                FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                collection_id TEXT,
                workspace_id TEXT,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                session_id TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                result_payload TEXT,
                progress_metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
        """)
        _migrate_schema(conn)
    conn.close()

def _now() -> str:
    return datetime.utcnow().isoformat()

# --- Document Operations ---

def add_document(document_id: str, filename: str, status: str = "UPLOADED", workspace_id: Optional[str] = None) -> dict:
    conn = get_connection()
    now = _now()
    with conn:
        conn.execute("""
            INSERT INTO documents (document_id, filename, ingestion_status, workspace_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                filename = excluded.filename,
                ingestion_status = excluded.ingestion_status,
                workspace_id = COALESCE(excluded.workspace_id, documents.workspace_id),
                updated_at = excluded.updated_at
        """, (document_id, filename, status, workspace_id, now, now))
        if workspace_id:
            touch_workspace(workspace_id, conn=conn)
    conn.close()
    return get_document(document_id)

def update_document_status(document_id: str, status: str, chunk_count: int = 0, error_message: Optional[str] = None, page_count: Optional[int] = None):
    conn = get_connection()
    now = _now()
    with conn:
        if page_count is not None:
            conn.execute("""
                UPDATE documents
                SET ingestion_status = ?, chunk_count = ?, page_count = ?, error_message = ?, updated_at = ?
                WHERE document_id = ?
            """, (status, chunk_count, page_count, error_message, now, document_id))
        else:
            conn.execute("""
                UPDATE documents
                SET ingestion_status = ?, chunk_count = ?, error_message = ?, updated_at = ?
                WHERE document_id = ?
            """, (status, chunk_count, error_message, now, document_id))
    conn.close()

def get_document(document_id: str) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def list_documents(workspace_id: Optional[str] = None) -> List[dict]:
    conn = get_connection()
    if workspace_id:
        rows = conn.execute(
            "SELECT * FROM documents WHERE workspace_id = ? ORDER BY created_at DESC", (workspace_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def documents_in_workspace(document_ids: List[str], workspace_id: str) -> List[str]:
    """Filters document_ids down to only those that actually belong to
    workspace_id — the data-layer isolation check behind /query's
    workspace_id parameter (never trusts the caller's list at face value)."""
    if not document_ids or not workspace_id:
        return []
    conn = get_connection()
    placeholders = ",".join("?" for _ in document_ids)
    rows = conn.execute(
        f"SELECT document_id FROM documents WHERE workspace_id = ? AND document_id IN ({placeholders})",
        (workspace_id, *document_ids),
    ).fetchall()
    conn.close()
    valid = {r["document_id"] for r in rows}
    return [d for d in document_ids if d in valid]

def delete_document(document_id: str):
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
    conn.close()

# --- Workspace Operations ---
#
# A workspace is a container for a set of documents + conversations. RAG
# isolation itself still runs on the existing document_id-scoped Chroma
# filtering (retrieval.py) — nothing about the vector store changes. What's
# new is that the *list* of document_ids a request is allowed to use is now
# resolved from a workspace-filtered SQLite query (list_documents/
# documents_in_workspace above) instead of being trusted from the client
# unconditionally, so workspace isolation is enforced at the data layer, not
# just by the frontend never showing the wrong papers.

def create_workspace(workspace_id: str, name: str) -> dict:
    conn = get_connection()
    now = _now()
    with conn:
        conn.execute("""
            INSERT INTO workspaces (workspace_id, name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (workspace_id, name, now, now))
    conn.close()
    return get_workspace(workspace_id)


def get_workspace(workspace_id: str) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,)).fetchone()
    if not row:
        conn.close()
        return None
    d = dict(row)
    d["paper_count"] = conn.execute(
        "SELECT COUNT(*) c FROM documents WHERE workspace_id = ?", (workspace_id,)
    ).fetchone()["c"]
    d["chat_count"] = conn.execute(
        "SELECT COUNT(*) c FROM sessions WHERE workspace_id = ?", (workspace_id,)
    ).fetchone()["c"]
    conn.close()
    return d


def list_workspaces() -> List[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM workspaces ORDER BY updated_at DESC").fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["paper_count"] = conn.execute(
            "SELECT COUNT(*) c FROM documents WHERE workspace_id = ?", (d["workspace_id"],)
        ).fetchone()["c"]
        d["chat_count"] = conn.execute(
            "SELECT COUNT(*) c FROM sessions WHERE workspace_id = ?", (d["workspace_id"],)
        ).fetchone()["c"]
        out.append(d)
    conn.close()
    return out


def touch_workspace(workspace_id: str, conn=None) -> None:
    """Bumps updated_at — called on upload/new-chat so the sidebar can order
    workspaces by recent activity. Accepts an existing connection so callers
    already inside a transaction (e.g. add_document) don't open a second one."""
    now = _now()
    if conn is not None:
        conn.execute("UPDATE workspaces SET updated_at = ? WHERE workspace_id = ?", (now, workspace_id))
        return
    own_conn = get_connection()
    with own_conn:
        own_conn.execute("UPDATE workspaces SET updated_at = ? WHERE workspace_id = ?", (now, workspace_id))
    own_conn.close()


# --- Collection Operations ---

def create_collection(collection_id: str, name: str, description: Optional[str] = None) -> dict:
    conn = get_connection()
    now = _now()
    with conn:
        conn.execute("""
            INSERT INTO collections (collection_id, name, description, created_at)
            VALUES (?, ?, ?, ?)
        """, (collection_id, name, description, now))
    conn.close()
    return get_collection(collection_id)

def get_collection(collection_id: str) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM collections WHERE collection_id = ?", (collection_id,)).fetchone()
    if not row:
        conn.close()
        return None
    
    col_dict = dict(row)
    doc_rows = conn.execute("SELECT document_id FROM collection_documents WHERE collection_id = ?", (collection_id,)).fetchall()
    col_dict["document_ids"] = [r["document_id"] for r in doc_rows]
    conn.close()
    return col_dict

def list_collections() -> List[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM collections ORDER BY created_at DESC").fetchall()
    collections = []
    for row in rows:
        col_dict = dict(row)
        doc_rows = conn.execute("SELECT document_id FROM collection_documents WHERE collection_id = ?", (col_dict["collection_id"],)).fetchall()
        col_dict["document_ids"] = [r["document_id"] for r in doc_rows]
        collections.append(col_dict)
    conn.close()
    return collections

def add_document_to_collection(collection_id: str, document_id: str):
    conn = get_connection()
    with conn:
        # Ignore if it already exists
        conn.execute("""
            INSERT OR IGNORE INTO collection_documents (collection_id, document_id)
            VALUES (?, ?)
        """, (collection_id, document_id))
    conn.close()

def remove_document_from_collection(collection_id: str, document_id: str):
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM collection_documents WHERE collection_id = ? AND document_id = ?", (collection_id, document_id))
    conn.close()

def delete_collection(collection_id: str):
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM collections WHERE collection_id = ?", (collection_id,))
    conn.close()

# --- Session Operations ---

def create_session(session_id: str, collection_id: Optional[str] = None, workspace_id: Optional[str] = None, title: Optional[str] = None) -> dict:
    conn = get_connection()
    now = _now()
    with conn:
        conn.execute("""
            INSERT INTO sessions (session_id, collection_id, workspace_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, collection_id, workspace_id, title, now, now))
        if workspace_id:
            touch_workspace(workspace_id, conn=conn)
    conn.close()
    return get_session(session_id)

def get_session(session_id: str) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def list_sessions(workspace_id: Optional[str] = None) -> List[dict]:
    conn = get_connection()
    if workspace_id:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE workspace_id = ? ORDER BY updated_at DESC", (workspace_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]

def update_session_title(session_id: str, title: str) -> None:
    conn = get_connection()
    now = _now()
    with conn:
        conn.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?", (title, now, session_id))
    conn.close()

def delete_session(session_id: str):
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.close()

def update_session_collection(session_id: str, collection_id: Optional[str]):
    conn = get_connection()
    now = _now()
    with conn:
        conn.execute("UPDATE sessions SET collection_id = ?, updated_at = ? WHERE session_id = ?", (collection_id, now, session_id))
    conn.close()

# --- Message Operations ---

def add_message(message_id: str, session_id: str, role: str, content: str, metadata: Optional[dict] = None) -> dict:
    conn = get_connection()
    now = _now()
    meta_str = json.dumps(metadata) if metadata else None
    with conn:
        conn.execute("""
            INSERT INTO messages (message_id, session_id, role, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (message_id, session_id, role, content, meta_str, now))
        conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
    conn.close()
    
    # fetch the inserted message
    return get_message(message_id)

def get_message(message_id: str) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["metadata"] = json.loads(d["metadata"]) if d.get("metadata") else None
    return d

def get_session_messages(session_id: str) -> List[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,)).fetchall()
    conn.close()
    msgs = []
    for r in rows:
        d = dict(r)
        d["metadata"] = json.loads(d["metadata"]) if d.get("metadata") else None
        msgs.append(d)
    return msgs

# --- Task Operations ---

def create_task(task_id: str, session_id: Optional[str] = None) -> dict:
    conn = get_connection()
    now = _now()
    with conn:
        conn.execute("""
            INSERT INTO tasks (task_id, session_id, status, created_at)
            VALUES (?, ?, 'PENDING', ?)
        """, (task_id, session_id, now))
    conn.close()
    return get_task(task_id)

def update_task_status(task_id: str, status: str, result_payload: Optional[dict] = None, error_message: Optional[str] = None, progress_metadata: Optional[dict] = None):
    conn = get_connection()
    now = _now()
    
    # Get existing task to retain fields if needed, and to figure out dates
    task = get_task(task_id)
    if not task:
        return
        
    started_at = task.get("started_at")
    completed_at = task.get("completed_at")
    
    if status == 'RUNNING' and not started_at:
        started_at = now
    elif status in ['COMPLETED', 'FAILED'] and not completed_at:
        completed_at = now
        
    payload_str = json.dumps(result_payload) if result_payload else task.get("result_payload", None)
    if isinstance(payload_str, dict):
        payload_str = json.dumps(payload_str) # safety
        
    prog_str = json.dumps(progress_metadata) if progress_metadata else task.get("progress_metadata", None)
    if isinstance(prog_str, dict):
        prog_str = json.dumps(prog_str)
        
    err_str = error_message if error_message is not None else task.get("error_message")

    with conn:
        conn.execute("""
            UPDATE tasks 
            SET status = ?, started_at = ?, completed_at = ?, error_message = ?, result_payload = ?, progress_metadata = ?
            WHERE task_id = ?
        """, (status, started_at, completed_at, err_str, payload_str, prog_str, task_id))
    conn.close()

def get_task(task_id: str) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    if d.get("result_payload") and isinstance(d["result_payload"], str):
        try:
            d["result_payload"] = json.loads(d["result_payload"])
        except:
            pass
    if d.get("progress_metadata") and isinstance(d["progress_metadata"], str):
        try:
            d["progress_metadata"] = json.loads(d["progress_metadata"])
        except:
            pass
    return d

# Initialize DB when module loads
init_db()
