import sqlite3
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, List, Dict, Any

DB_PATH = Path(__file__).parent.parent / "data" / "registry.db"

def get_connection():
    # Ensure data directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_connection()
    with conn:
        conn.executescript("""
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
        """)
    conn.close()

def _now() -> str:
    return datetime.utcnow().isoformat()

# --- Document Operations ---

def add_document(document_id: str, filename: str, status: str = "UPLOADED") -> dict:
    conn = get_connection()
    now = _now()
    with conn:
        conn.execute("""
            INSERT INTO documents (document_id, filename, ingestion_status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                filename = excluded.filename,
                ingestion_status = excluded.ingestion_status,
                updated_at = excluded.updated_at
        """, (document_id, filename, status, now, now))
    conn.close()
    return get_document(document_id)

def update_document_status(document_id: str, status: str, chunk_count: int = 0, error_message: Optional[str] = None):
    conn = get_connection()
    now = _now()
    with conn:
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

def list_documents() -> List[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_document(document_id: str):
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
    conn.close()

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

# Initialize DB when module loads
init_db()
