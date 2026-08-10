import os
import tempfile
import shutil
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Test isolation, structured in two parts:
#
#   1. pytest_configure() — runs BEFORE pytest collects a single test file.
#      This is where the actual environment isolation (CHROMA_DIR,
#      COLLECTION_NAME, REGISTRY_DB_PATH, neutralizing DATABASE_URL, engine/
#      client singleton resets) happens. This placement is load-bearing, not
#      cosmetic: pytest's collection phase IMPORTS every test_*.py module,
#      and a module can contain top-level (module-scope) side-effecting code
#      that runs the instant it's imported — before ANY fixture, including a
#      session-scoped autouse one, has a chance to run (fixtures only run
#      during the later SETUP phase, strictly after collection completes).
#      Two test files (test_graph.py, test_retrieval.py) used to do exactly
#      this — a bare module-level `ingest_document(...)` call — and it was
#      confirmed as a real, reproducible bug: depending on which other test
#      files happened to be collected alongside them, that call could run
#      before isolation was established and silently write into the REAL
#      production ChromaDB and PostgreSQL database. Both files were fixed to
#      do their ingestion inside an autouse fixture instead (which correctly
#      defers execution to the SETUP phase). This pytest_configure() move is
#      the belt to that fix's suspenders: even if some future test file
#      reintroduces a bare module-level side effect, isolation is now
#      already in place before collection ever starts, so it's structurally
#      impossible for it to reach real production infrastructure.
#
#   2. isolated_test_env() — a session-scoped autouse fixture that does the
#      rest of the one-time setup that's fine to happen after collection
#      (pre-ingesting the shared test fixture PDF) and handles teardown
#      (temp directory cleanup, restoring DATABASE_URL for the developer's
#      shell). It reuses the paths pytest_configure() already established.
# ---------------------------------------------------------------------------

_temp_chroma: str | None = None
_temp_db_dir: str | None = None
_temp_db_path: str | None = None
_saved_database_url: str | None = None


def pytest_configure(config):
    global _temp_chroma, _temp_db_dir, _temp_db_path, _saved_database_url

    _temp_chroma = tempfile.mkdtemp(prefix="verityrag_chroma_test_")
    _temp_db_dir = tempfile.mkdtemp(prefix="verityrag_db_test_")
    _temp_db_path = os.path.join(_temp_db_dir, "test_registry.db")

    # Import config FIRST, before touching os.environ at all. config.py
    # calls python-dotenv's load_dotenv() at module level, which only ever
    # runs on this, its FIRST import in the process (later imports reuse the
    # cached module and do NOT re-run load_dotenv()). If DATABASE_URL were
    # popped from os.environ before config's first import happened, the
    # first import of config ANYWHERE later (e.g. this function's own next
    # line, or any test module during collection) would trigger
    # load_dotenv() and — since DATABASE_URL would be absent from os.environ
    # at that point — silently RE-ADD the real DATABASE_URL from the actual
    # .env file, undoing the pop below without any error. Confirmed as a
    # real, reproducible bug. Importing config here, first, before any
    # os.environ mutation, removes the ordering dependency entirely.
    import config

    os.environ["CHROMA_DIR"] = _temp_chroma
    os.environ["COLLECTION_NAME"] = "test_collection"
    os.environ["REGISTRY_DB_PATH"] = _temp_db_path

    # CRITICAL: db/session.py's resolve_database_url() checks DATABASE_URL
    # FIRST, before ever looking at REGISTRY_DB_PATH — so once a real
    # DATABASE_URL is configured in the environment (e.g. a real local
    # PostgreSQL instance for production use), every test in this suite
    # would silently run against that REAL database instead of the isolated
    # SQLite file set up above, unless this is explicitly neutralized here.
    # Confirmed as a real, live bug more than once during stabilization (a
    # test run wrote genuine rows into a real local PostgreSQL 'verityrag'
    # database) — this removal is the fix. Saved/restored around the whole
    # pytest process so a developer's shell environment is never
    # permanently altered by running the test suite.
    _saved_database_url = os.environ.pop("DATABASE_URL", None)

    config.CHROMA_DIR = _temp_chroma
    config.COLLECTION_NAME = "test_collection"
    config.REGISTRY_DB_PATH = _temp_db_path

    # Force reset of singletons in ingest.py, in case anything already
    # imported it (unlikely this early, but robust rather than order-
    # dependent).
    import ingest
    ingest._client = None
    ingest._collection = None

    # Force reset of the SQLAlchemy engine singleton in db/session.py too —
    # a cached Postgres-bound engine must not survive into the test session
    # now that DATABASE_URL has been removed above.
    import db.session as _db_session
    if _db_session._engine is not None:
        _db_session._engine.dispose()
    _db_session._engine = None
    _db_session._SessionLocal = None
    _db_session._resolved_url = None

    # Initialize the new temporary SQLite database.
    from database import init_db
    init_db()

    # Hard safety net: if resolve_database_url() ever returns anything other
    # than the isolated SQLite path here (e.g. a future change reintroduces
    # this class of bug), fail loudly and immediately — before a single test
    # is even collected — rather than silently letting any test run against
    # a real database.
    resolved = _db_session.resolve_database_url()
    assert resolved.startswith("sqlite:///"), (
        f"Test isolation broken: resolve_database_url() returned {resolved!r} instead of "
        f"the isolated SQLite path — refusing to run the suite against a non-test database."
    )
    assert _temp_db_path.replace("\\", "/") in resolved.replace("\\", "/"), (
        "Test isolation broken: resolved SQLite path does not match this fixture's temp path."
    )


def pytest_unconfigure(config):
    # Runs after the whole pytest process is done — final cleanup that
    # doesn't depend on any fixture having run (mirrors pytest_configure()
    # being collection's counterpart rather than a fixture).
    global _saved_database_url
    if _temp_chroma:
        try:
            shutil.rmtree(_temp_chroma, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Failed to cleanup temp Chroma directory: {e}")
    if _temp_db_dir:
        try:
            shutil.rmtree(_temp_db_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Failed to cleanup temp DB directory: {e}")
    if _saved_database_url is not None:
        os.environ["DATABASE_URL"] = _saved_database_url


@pytest.fixture(autouse=True, scope="session")
def isolated_test_env():
    """Runs once per session, during the SETUP phase (after
    pytest_configure() has already established isolation and after
    collection has finished). Pre-populates the isolated store with the
    shared test fixture PDF so retrieval-dependent tests have something to
    find."""
    from ingest import ingest_document
    from retrieval import build_bm25_index
    try:
        # Dedicated test fixture, not the data/ folder — a copy of the same
        # file (content-hash document_id is identical either way).
        test_pdf = os.path.join(Path(__file__).parent, "tests", "fixtures", "attention.pdf")
        if os.path.exists(test_pdf):
            ingest_document(test_pdf)
            build_bm25_index()
    except Exception as e:
        print(f"Warning: Failed to pre-ingest test data: {e}")

    yield
