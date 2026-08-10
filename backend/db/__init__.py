"""
db/ — the SQLAlchemy-backed persistence layer.

Structured application data (workspaces, documents, collections, sessions,
messages, tasks) lives here. Vector embeddings are NOT stored here — that
remains ChromaDB's responsibility entirely (see backend/ingest.py,
backend/retrieval.py); this package never touches backend/chroma_store.

- models.py     ORM models (one class per existing table, same columns).
- session.py    Engine/session factory. DATABASE_URL (if set) selects
                PostgreSQL in production; otherwise falls back to the same
                SQLite file backend/database.py has always used
                (config.REGISTRY_DB_PATH), so local dev and the existing
                test suite are unaffected unless DATABASE_URL is configured.
- repository.py CRUD functions with the EXACT names/signatures/return
                shapes database.py already had, so every existing call
                site (`from database import ...`) keeps working unchanged
                — database.py is now a thin re-export of this package.
"""
