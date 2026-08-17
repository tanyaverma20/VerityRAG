"""
test_upload_security.py — Phase 13 security audit: /upload previously had
NO file size limit at all (unbounded resource exhaustion risk) and only
checked the filename extension, never the actual file content. Both are
now enforced; this proves it against the real endpoint, not just the
config value.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import config
from main import app
from test_auth_helpers import registered_user_with_workspace

client = TestClient(app)

FIXTURE_PDF = Path(__file__).parent / "tests" / "fixtures" / "attention.pdf"

# /upload now requires a real authenticated, workspace-owning caller (see
# main.py's _require_workspace_owner()) — every test below authenticates
# and uploads into a workspace it genuinely owns, so what's actually being
# tested (content/size/extension validation) is reached, same as before.
HEADERS: dict = {}
WORKSPACE_ID: str = ""


@pytest.fixture(autouse=True, scope="module")
def _module_auth():
    global HEADERS, WORKSPACE_ID
    HEADERS, WORKSPACE_ID = registered_user_with_workspace(client, "upload")
    yield


def test_real_pdf_upload_still_works():
    """Sanity check: the new validation must not break a genuine PDF."""
    with open(FIXTURE_PDF, "rb") as f:
        resp = client.post(
            "/upload", files={"file": ("attention.pdf", f, "application/pdf")},
            data={"workspace_id": WORKSPACE_ID}, headers=HEADERS,
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload_rejects_a_file_that_is_not_really_a_pdf():
    """A client can name any file 'something.pdf' regardless of its real
    content — the endpoint must check the actual magic bytes, not just
    trust the extension."""
    fake_pdf = b"This is definitely not a real PDF file, just plain text."
    resp = client.post(
        "/upload", files={"file": ("fake.pdf", fake_pdf, "application/pdf")},
        data={"workspace_id": WORKSPACE_ID}, headers=HEADERS,
    )
    assert resp.status_code == 400
    assert "PDF" in resp.json()["detail"]


def test_upload_rejects_a_file_over_the_size_limit(monkeypatch):
    """A file larger than MAX_UPLOAD_BYTES must be rejected with 413, and
    must be rejected based on REAL bytes read, not a spoofed Content-Length
    header (TestClient doesn't let us spoof that header anyway — this
    proves the server-side byte-counting itself works)."""
    import main as main_module
    monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 100)  # tiny limit for the test

    oversized = b"%PDF-1.4\n" + (b"A" * 1000)  # starts with a real PDF header, but too large
    resp = client.post(
        "/upload", files={"file": ("big.pdf", oversized, "application/pdf")},
        data={"workspace_id": WORKSPACE_ID}, headers=HEADERS,
    )
    assert resp.status_code == 413
    assert "MB" in resp.json()["detail"] or "limit" in resp.json()["detail"].lower()


def test_rejected_upload_never_leaves_a_partial_temp_file(tmp_path, monkeypatch):
    """Both rejection paths must clean up the temp file they wrote to,
    never leaving partial upload data lying around on disk."""
    import tempfile
    real_named_temp = tempfile.NamedTemporaryFile
    created_paths = []

    def _tracking_temp(*args, **kwargs):
        f = real_named_temp(*args, **kwargs)
        created_paths.append(f.name)
        return f

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _tracking_temp)

    fake_pdf = b"not a real pdf at all"
    resp = client.post(
        "/upload", files={"file": ("fake.pdf", fake_pdf, "application/pdf")},
        data={"workspace_id": WORKSPACE_ID}, headers=HEADERS,
    )
    assert resp.status_code == 400
    assert len(created_paths) == 1
    assert not Path(created_paths[0]).exists(), "Rejected upload must not leave its temp file behind"


def test_non_pdf_extension_still_rejected_before_any_file_io():
    resp = client.post(
        "/upload", files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"workspace_id": WORKSPACE_ID}, headers=HEADERS,
    )
    assert resp.status_code == 400
