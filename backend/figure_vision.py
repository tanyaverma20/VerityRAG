"""
figure_vision.py — page-image rendering for Explain Figure, using PyMuPDF
(already a dependency; no new system binary required) to render a specific
PDF page as an image, plus the honest gate for whether a vision-capable
model is actually configured to look at it.

HONESTY CONTRACT (matches the explicit product requirement):
  - Rendering a page to a PNG image is real and happens here, whenever the
    originally-uploaded PDF was persisted (see main.py's /upload ->
    persist_uploaded_pdf()) and the requested page exists.
  - Whether that image is then actually sent to and read by a
    vision-capable model is a SEPARATE fact, gated by GROQ_VISION_MODEL.
    If that env var is unset (no vision model provisioned on the account —
    true for this deployment today), vision_model_available() returns
    False, and callers (analysis.py:explain_figure) MUST fall back to the
    existing text/caption-based explanation and say so explicitly. This
    module never claims visual understanding on its own — it only ever
    reports whether the prerequisites for it are actually met.

document_id scoping: render_page_as_image_base64() only ever opens the ONE
PDF file matching the requested document_id (data/uploads/<document_id>.pdf)
— it never scans or reads any other document.
"""
from __future__ import annotations

import base64
import os
import shutil
from pathlib import Path

# Import config FIRST, before reading any os.getenv() at module level.
# config.py calls python-dotenv's load_dotenv() at module level, which only
# ever runs on this, its FIRST import in the process. If this module were
# imported before anything else had imported config (e.g. a standalone
# script, or a different future import order in main.py), reading
# GROQ_VISION_MODEL directly from os.getenv() here would silently bind
# VISION_MODEL to "" for the rest of the process's lifetime — even with a
# real value configured in .env — since this is a module-level constant,
# never re-evaluated later. Confirmed as a real, reproducible bug (same
# class as the one already fixed in db/session.py:resolve_database_url()):
# in the real running app this happened to work by ordering luck only
# (main.py imports `ingest`, which imports `config`, before it imports
# figure_vision), but a standalone script that imports figure_vision first
# hit exactly this failure. Importing config here removes the ordering
# dependency entirely.
import config  # noqa: F401

UPLOADS_DIR = Path(__file__).parent.parent / "data" / "uploads"

# Empty by default — no vision-capable model is assumed to exist. Only set
# this if a real vision-capable model id is actually available on the
# configured Groq account (verify via client.models.list() before setting
# it — do not guess a model name).
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "").strip()


def vision_model_available() -> bool:
    """True only when a real vision model id has been explicitly
    configured. False (the honest default) means Explain Figure must use
    its existing text/caption-based path and say so."""
    return bool(VISION_MODEL)


def persist_uploaded_pdf(tmp_path: str, document_id: str) -> Path:
    """Copies an uploaded PDF's bytes to a stable, document_id-addressed
    location so a later Explain Figure call can render a page from it.
    Called once, right after successful ingestion (main.py's /upload) —
    never re-ingests, never touches Chroma/SQLite, purely a file copy."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / f"{document_id}.pdf"
    shutil.copyfile(tmp_path, dest)
    return dest


def get_uploaded_pdf_path(document_id: str) -> Path | None:
    """None (not an error) when this document's original PDF was never
    persisted — e.g. it was ingested before this feature existed, or was
    ingested via a script (backend/ingest_all.py) rather than /upload."""
    p = UPLOADS_DIR / f"{document_id}.pdf"
    return p if p.exists() else None


def render_page_as_image_base64(document_id: str, page_number: int, zoom: float = 1.0) -> str | None:
    """
    Renders ONE page of THIS document's originally-uploaded PDF as a PNG,
    base64-encoded (data-URL-ready). Returns None — never raises — when:
      - the PDF was never persisted for this document_id,
      - the requested page is out of range,
      - PyMuPDF isn't installed, or
      - any rendering error occurs.
    A None return means the caller must fall back to the text-based
    explanation honestly, not silently retry or guess.

    Default zoom lowered from 2.0 to 1.0 (real, reproduced live-browser
    finding this session): a zoom=2.0 render of a typical academic PDF page
    pushed the vision request to ~9242 tokens, over this account's real
    Groq TPM cap for the configured vision model (8000 tokens/minute,
    on_demand tier — confirmed via the API's own 413 rate_limit_exceeded
    response), causing every live vision call to be rejected before the
    model ever saw the image and silently degrade to the honest text
    fallback. Direct measurement (this session) showed the model's image
    tokenizer does NOT cost scale smoothly with pixel count — 1.5x zoom
    still hit the identical 9242-token rejection as 2.0x, but 1.0x (and a
    tested 0.8x) both dropped below the cap and produced a REAL successful
    vision read of the page (confirmed: the model correctly named and
    described the page's actual diagram content, not a generic guess).
    1.0x was chosen over 0.8x as the new default for the extra legibility
    margin on dense academic figures/tables. This does not touch the
    honest-fallback path itself, which still fires correctly on any
    genuine failure (no PDF persisted, model still unavailable, or the
    account is rate-limited for an unrelated reason).
    """
    pdf_path = get_uploaded_pdf_path(document_id)
    if not pdf_path:
        return None
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None

    try:
        doc = fitz.open(str(pdf_path))
        try:
            if not (1 <= page_number <= doc.page_count):
                return None
            page = doc.load_page(page_number - 1)  # fitz pages are 0-indexed
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            png_bytes = pix.tobytes("png")
            return base64.b64encode(png_bytes).decode("ascii")
        finally:
            doc.close()
    except Exception:
        return None
