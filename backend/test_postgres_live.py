"""
test_postgres_live.py — real PostgreSQL integration tests (Phase 2).

These run against an ACTUAL PostgreSQL server (not a mock), exercising the
exact code paths main.py uses in production: db.session.resolve_database_url()
/ get_engine() connection pooling, and db.repository's real CRUD +
transaction + cascade behavior.

conftest.py's isolated_test_env fixture (session-scoped, autouse) removes
DATABASE_URL from os.environ for the whole pytest session so the rest of
the suite never touches a real database by accident. This module reads the
ORIGINAL value straight out of the .env file via python-dotenv's
dotenv_values() (which does NOT mutate os.environ, unlike load_dotenv()),
completely independent of whatever conftest.py did — then temporarily
re-points db.session at the real database for the duration of each test
in this file only, restoring the SQLite-isolated state again afterward so
no other test in the suite is affected.

Auto-skips the whole module (not a failure) when no real, reachable
PostgreSQL server is configured — e.g. on a machine/CI runner that only has
the SQLite dev fallback. This mirrors how the rest of the suite already
treats infra-dependent tests as "skipped", never "failed", when the
external prerequisite genuinely isn't there.
"""
import os
import uuid
from datetime import datetime

import pytest
from dotenv import dotenv_values, find_dotenv

import db.session as db_session

_dotenv_path = find_dotenv(usecwd=True)
_REAL_DATABASE_URL = (dotenv_values(_dotenv_path).get("DATABASE_URL") if _dotenv_path else None) or None


def _postgres_reachable(url: str | None) -> bool:
    if not url or not url.startswith("postgresql"):
        return False
    try:
        from sqlalchemy import create_engine, text
        eng = create_engine(url, connect_args={"connect_timeout": 2})
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(_REAL_DATABASE_URL),
    reason="No reachable real PostgreSQL server configured via DATABASE_URL in .env — "
           "these are live integration tests, not mocked, and are skipped rather than "
           "faked when the real infrastructure prerequisite isn't available.",
)


@pytest.fixture()
def real_pg_engine():
    """Temporarily points db.session at the REAL PostgreSQL database for one
    test, then restores the SQLite-isolated state conftest.py set up, so
    this is the only file in the suite that ever touches real Postgres."""
    saved = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = _REAL_DATABASE_URL
    if db_session._engine is not None:
        db_session._engine.dispose()
    db_session._engine = None
    db_session._SessionLocal = None
    db_session._resolved_url = None

    resolved = db_session.resolve_database_url()
    assert resolved.startswith("postgresql"), f"Expected the real Postgres URL, got {resolved!r}"

    # database.py's module-level init_db() (schema create + the additive
    # ALTER-column safety net in db.repository._migrate_legacy_columns())
    # only ever runs ONCE, at that module's first import in this process —
    # which happened before this fixture swapped the engine to point at
    # the real database, so it ran against the SQLite-isolated engine
    # active at that time, never against this real Postgres connection.
    # Re-running it here (additive-only, a safe no-op on a database that's
    # already current) is what keeps a real, pre-existing Postgres database
    # schema-current for new columns added after it was first created —
    # confirmed as a real, reproducible gap: a fresh `owner_user_id` column
    # added to the Workspace model was missing from real Postgres until
    # this call was added.
    import database as _database_module
    _database_module.init_db()

    yield db_session.get_engine()

    if db_session._engine is not None:
        db_session._engine.dispose()
    if saved is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = saved
    db_session._engine = None
    db_session._SessionLocal = None
    db_session._resolved_url = None
    # Re-establish the isolated SQLite engine conftest.py's fixture expects
    # to still be in effect for every other test in the suite.
    back = db_session.resolve_database_url()
    assert back.startswith("sqlite:///"), "Test isolation must be restored after a live-Postgres test"


def test_engine_dialect_and_pooling(real_pg_engine):
    assert real_pg_engine.dialect.name == "postgresql"
    pool = real_pg_engine.pool
    assert type(pool).__name__ == "QueuePool", "Production must use real connection pooling, not NullPool/StaticPool"
    assert pool.size() == int(os.getenv("DB_POOL_SIZE", "5"))
    assert pool._pre_ping is True, "pool_pre_ping must be enabled so a dropped connection is detected, not surfaced as a query error"


def test_live_select_roundtrip(real_pg_engine):
    from sqlalchemy import text
    with real_pg_engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1


def test_real_crud_and_isolation_helper(real_pg_engine):
    from db import repository as repo

    ws_id = "live_test_ws_" + uuid.uuid4().hex[:8]
    doc_id = "live_test_doc_" + uuid.uuid4().hex[:8]
    try:
        ws = repo.create_workspace(ws_id, "Live Postgres Test Workspace")
        assert ws["workspace_id"] == ws_id

        doc = repo.add_document(doc_id, "live_test.pdf", status="UPLOADED", workspace_id=ws_id)
        assert doc["workspace_id"] == ws_id

        repo.update_document_status(doc_id, "INDEXED", chunk_count=7)
        assert repo.get_document(doc_id)["chunk_count"] == 7

        valid = repo.documents_in_workspace([doc_id, "does_not_exist"], ws_id)
        assert valid == [doc_id], "documents_in_workspace() must never trust unowned IDs at face value"
    finally:
        from sqlalchemy import delete as sa_delete
        from db.models import Workspace
        with db_session.session_scope() as session:
            session.execute(sa_delete(Workspace).where(Workspace.workspace_id == ws_id))


def test_transaction_rolls_back_on_fk_violation(real_pg_engine):
    from db.models import Message

    orphan_id = "live_test_orphan_" + uuid.uuid4().hex[:8]
    raised = False
    try:
        with db_session.session_scope() as session:
            session.add(Message(
                message_id=orphan_id,
                session_id="live_test_nonexistent_session_" + uuid.uuid4().hex[:8],
                role="user", content="must never commit",
                created_at=datetime.utcnow().isoformat(),
            ))
    except Exception:
        raised = True
    assert raised, "An FK-violating write must raise, not fail silently"

    from db import repository as repo
    assert repo.get_message(orphan_id) is None, "session_scope() must roll back on failure — nothing partial may survive"


def test_cascade_delete_two_hops(real_pg_engine):
    """workspace -> document / session -> message / task, via the real
    DB-level ON DELETE CASCADE constraints (repository.py deletes use bulk
    sa_delete(), which bypasses SQLAlchemy ORM-level cascade — so this is
    what actually proves cascading works in production)."""
    from db import repository as repo
    from db.models import Workspace
    from sqlalchemy import delete as sa_delete

    ws_id = "live_test_cascade_ws_" + uuid.uuid4().hex[:8]
    doc_id = "live_test_cascade_doc_" + uuid.uuid4().hex[:8]
    sess_id = "live_test_cascade_sess_" + uuid.uuid4().hex[:8]
    msg_id = "live_test_cascade_msg_" + uuid.uuid4().hex[:8]
    task_id = "live_test_cascade_task_" + uuid.uuid4().hex[:8]

    repo.create_workspace(ws_id, "Cascade Test")
    repo.add_document(doc_id, "cascade.pdf", workspace_id=ws_id)
    repo.create_session(sess_id, workspace_id=ws_id)
    repo.add_message(msg_id, sess_id, "user", "hi")
    repo.create_task(task_id, session_id=sess_id)

    with db_session.session_scope() as session:
        session.execute(sa_delete(Workspace).where(Workspace.workspace_id == ws_id))

    assert repo.get_document(doc_id) is None
    assert repo.get_session(sess_id) is None
    assert repo.get_message(msg_id) is None, "2-hop cascade: message must go when its session's workspace is deleted"
    assert repo.get_task(task_id) is None, "2-hop cascade: task must go when its session's workspace is deleted"


def test_alembic_head_matches_live_schema(real_pg_engine):
    """Confirms the live database is actually at the migration head this
    codebase expects — catches drift between db/models.py and what's
    really been migrated onto the server."""
    from sqlalchemy import text
    with real_pg_engine.connect() as conn:
        row = conn.execute(text("SELECT version_num FROM alembic_version")).fetchone()
    assert row is not None, "alembic_version table must exist and have exactly one row"

    from alembic.config import Config
    from alembic.script import ScriptDirectory
    cfg = Config(os.path.join(os.path.dirname(__file__), "alembic.ini"))
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    assert row[0] in heads, f"Live DB is at revision {row[0]!r} but the codebase's migration head(s) are {heads!r}"
