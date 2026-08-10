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


def render_page_as_image_base64(document_id: str, page_number: int, zoom: float = 2.0) -> str | None:
    """
    Renders ONE page of THIS document's originally-uploaded PDF as a PNG,
    base64-encoded (data-URL-ready). Returns None — never raises — when:
      - the PDF was never persisted for this document_id,
      - the requested page is out of range,
      - PyMuPDF isn't installed, or
      - any rendering error occurs.
    A None return means the caller must fall back to the text-based
    explanation honestly, not silently retry or guess.
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
