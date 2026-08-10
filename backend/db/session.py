"""
db/session.py — engine/session factory.

Resolution order for the database URL (read FRESH on every call, never
cached at import time — this is what lets test fixtures that override
config.REGISTRY_DB_PATH *after* this module was first imported still take
effect, exactly like the original database.py's local `from config import
REGISTRY_DB_PATH` inside get_connection()):

  1. DATABASE_URL env var, if set — this is how production selects
     PostgreSQL. Example: postgresql+psycopg2://user:pass@host:5432/dbname
  2. Otherwise, sqlite:///<config.REGISTRY_DB_PATH> — the exact same
     SQLite file database.py has always defaulted to, so local dev and the
     existing isolated test fixture (conftest.py) are unaffected unless
     DATABASE_URL is explicitly configured.

The engine is cached and only rebuilt if the resolved URL actually changes
(e.g. a test fixture pointing REGISTRY_DB_PATH at a fresh temp file) —
production gets real connection pooling; tests get correct isolation.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session as SASession
from sqlalchemy.pool import StaticPool

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None
_resolved_url: str | None = None


def resolve_database_url() -> str:
    # `config` is imported unconditionally (not just in the sqlite-fallback
    # branch) because importing it is also what triggers python-dotenv's
    # load_dotenv() as a module-level side effect. If a caller reaches this
    # function before anything else in the process has imported `config`
    # (e.g. a standalone script that imports db.session/db.repository
    # directly, rather than going through main.py's import chain where
    # `ingest`/`config` are always imported first), DATABASE_URL from a
    # real .env file would not be in os.environ yet on this very first
    # call — silently resolving to SQLite instead of the intended
    # PostgreSQL URL. Importing config first removes that ordering
    # dependency entirely; it's cheap (module-cached after the first call)
    # and always reflects the CURRENT config.REGISTRY_DB_PATH value either way.
    import config
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    return f"sqlite:///{config.REGISTRY_DB_PATH}"


def is_sqlite(url: str | None = None) -> bool:
    return (url or resolve_database_url()).startswith("sqlite")


def get_engine() -> Engine:
    global _engine, _SessionLocal, _resolved_url
    url = resolve_database_url()
    if _engine is None or _resolved_url != url:
        if _engine is not None:
            _engine.dispose()
        if is_sqlite(url):
            connect_args = {"check_same_thread": False}
            kwargs = {"connect_args": connect_args}
            if ":memory:" in url:
                kwargs["poolclass"] = StaticPool
            _engine = create_engine(url, **kwargs)

            @event.listens_for(_engine, "connect")
            def _sqlite_pragma(dbapi_conn, _rec):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys = ON")
                cur.close()
        else:
            # Production PostgreSQL: real connection pooling +
            # pool_pre_ping so a dropped/stale connection is detected and
            # transparently replaced rather than surfacing as a query
            # error, and pool_recycle avoids the DB server's own idle
            # connection timeout killing a pooled connection under us.
            _engine = create_engine(
                url,
                pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
                max_overflow=int(os.getenv("DB_POOL_MAX_OVERFLOW", "10")),
                pool_pre_ping=True,
                pool_recycle=1800,
            )
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False)
        _resolved_url = url
    return _engine


def get_sessionmaker() -> sessionmaker:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[SASession]:
    """Transaction boundary used by every repository function: commits on
    success, rolls back and re-raises on any exception. Callers never
    manage commit/rollback themselves."""
    SessionLocal = get_sessionmaker()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_schema() -> None:
    """Creates every table if it doesn't already exist — additive-only,
    exactly like database.py's original `CREATE TABLE IF NOT EXISTS`
    behavior, so calling this against a database that already has real
    production data is always a safe no-op."""
    from db.models import Base
    Base.metadata.create_all(get_engine())
