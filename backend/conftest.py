import os
import tempfile
import shutil
import pytest
from pathlib import Path

# This fixture runs before any test in the suite.
# It sets up temporary directories for Chroma and SQLite,
# overrides the environment variables so config.py picks them up,
# and cleans up after the tests finish.
@pytest.fixture(autouse=True, scope="session")
def isolated_test_env():
    temp_chroma = tempfile.mkdtemp(prefix="verityrag_chroma_test_")
    temp_db_dir = tempfile.mkdtemp(prefix="verityrag_db_test_")
    temp_db_path = os.path.join(temp_db_dir, "test_registry.db")
    
    os.environ["CHROMA_DIR"] = temp_chroma
    os.environ["COLLECTION_NAME"] = "test_collection"
    os.environ["REGISTRY_DB_PATH"] = temp_db_path
    
    import config
    config.CHROMA_DIR = temp_chroma
    config.COLLECTION_NAME = "test_collection"
    config.REGISTRY_DB_PATH = temp_db_path
    
    # Force reset of singletons in ingest.py
    import ingest
    ingest._client = None
    ingest._collection = None
    
    # Initialize the new temporary SQLite database
    from database import init_db
    init_db()
    
    # Pre-populate the test environment with a document so that retrieval tests pass
    from ingest import ingest_document
    from retrieval import build_bm25_index
    try:
        # Dedicated test fixture, not the data/ folder (item 21) — a copy of
        # the same file (content-hash document_id is identical either way).
        test_pdf = os.path.join(Path(__file__).parent, "tests", "fixtures", "attention.pdf")
        if os.path.exists(test_pdf):
            ingest_document(test_pdf)
            build_bm25_index()
    except Exception as e:
        print(f"Warning: Failed to pre-ingest test data: {e}")
    
    yield
    
    # Teardown: delete temporary directories
    try:
        shutil.rmtree(temp_chroma)
        shutil.rmtree(temp_db_dir)
    except Exception as e:
        print(f"Warning: Failed to cleanup temp directories: {e}")
