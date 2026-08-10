"""
ocr_fallback.py — OCR fallback for scanned/image-only PDFs.

Only ever invoked for a page where normal PDF text extraction (pypdf, in
ingest.py) came back empty/insufficient — a deterministic, measured
condition (see is_text_insufficient()), never a heuristic guess. A normal,
already-text-extractable PDF never triggers OCR at all; this is strictly a
fallback, not a replacement for the existing extraction path.

Requires the Tesseract OCR system binary to be installed separately — this
app never installs, bundles, or assumes it. If it isn't available,
ingestion proceeds exactly as it always has (pages with insufficient text
are simply left as-is) and reports that clearly in the returned status
dict, rather than silently proceeding as if OCR had run, or crashing.

OCR output enters the EXACT SAME chunk -> embed -> Dense+BM25+RRF+
CrossEncoder retrieval pipeline as normal pypdf-extracted text — this
module only ever produces plain page text; it never touches chunking,
embeddings, or retrieval.
"""
from __future__ import annotations

import io
import os

OCR_ENABLED = os.getenv("OCR_ENABLED", "true").strip().lower() not in ("false", "0", "no")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()

# Deterministic "insufficient text" threshold — a page pypdf extracted
# fewer than this many non-whitespace characters from is treated as
# effectively textless (a scanned/image page), not "just a short page".
MIN_CHARS_PER_PAGE = 20


def is_text_insufficient(text: str | None) -> bool:
    return len((text or "").strip()) < MIN_CHARS_PER_PAGE


def _configure_tesseract() -> bool:
    """True only if pytesseract AND the real Tesseract system binary are
    both actually usable. Never raises — a missing OCR engine is a normal,
    expected condition in most environments, not an application error."""
    if not OCR_ENABLED:
        return False
    try:
        import pytesseract
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_available() -> bool:
    return _configure_tesseract()


def ocr_page_image(pdf_path: str, page_number_1_indexed: int, zoom: float = 2.0) -> str | None:
    """Renders one PDF page to an image (PyMuPDF, already a dependency)
    and runs Tesseract OCR on it. Returns None — never raises — on any
    failure, including OCR simply not being installed, so callers always
    have a clean fallback path (keep the original, possibly empty, text)."""
    if not _configure_tesseract():
        return None
    try:
        import fitz
        import pytesseract
        from PIL import Image

        doc = fitz.open(pdf_path)
        try:
            if not (1 <= page_number_1_indexed <= doc.page_count):
                return None
            page = doc.load_page(page_number_1_indexed - 1)  # fitz is 0-indexed
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            return text.strip() or None
        finally:
            doc.close()
    except Exception as e:
        print(f"[ocr_fallback] OCR failed for page {page_number_1_indexed}: {e}")
        return None


def extract_pages_with_ocr_fallback(pdf_path: str, pages: list[dict]) -> tuple[list[dict], dict]:
    """
    `pages`: the [{"page_number", "text"}] list from ingest.py's
    extract_pages_from_pdf() — including pages with empty/insufficient
    text (that function no longer filters them out; see its own docstring).

    For every page whose pypdf-extracted text is insufficient, attempts
    OCR and replaces JUST that page's text if OCR produced something
    usable. A page that already had sufficient real text is never touched
    — OCR only ever fills a genuine gap, never overrides normal
    extraction. document_id and page numbers are untouched throughout.

    Returns (updated_pages, status) — status is real, structured
    observability (how many pages needed OCR, how many succeeded/failed,
    whether the OCR engine was even available), never silent.
    """
    status = {
        "ocr_engine_available": _configure_tesseract(),
        "pages_total": len(pages),
        "pages_needing_ocr": 0,
        "pages_ocr_succeeded": 0,
        "pages_ocr_failed": 0,
        "note": None,
    }

    insufficient = [p for p in pages if is_text_insufficient(p.get("text"))]
    status["pages_needing_ocr"] = len(insufficient)

    if not insufficient:
        return pages, status

    if not status["ocr_engine_available"]:
        status["note"] = (
            f"{len(insufficient)} of {len(pages)} page(s) had insufficient extracted text, but the "
            f"Tesseract OCR engine is not installed in this environment — those pages were left with "
            f"their original (possibly empty) text rather than fabricating content. Install Tesseract "
            f"and set OCR_ENABLED=true / TESSERACT_CMD to enable OCR."
        )
        return pages, status

    updated: list[dict] = []
    for p in pages:
        if is_text_insufficient(p.get("text")):
            ocr_text = ocr_page_image(pdf_path, p["page_number"])
            if ocr_text and not is_text_insufficient(ocr_text):
                updated.append({"page_number": p["page_number"], "text": ocr_text})
                status["pages_ocr_succeeded"] += 1
            else:
                updated.append(p)  # OCR didn't help either — keep original, don't fabricate
                status["pages_ocr_failed"] += 1
        else:
            updated.append(p)

    if status["pages_ocr_failed"]:
        status["note"] = f"{status['pages_ocr_failed']} page(s) had insufficient text and OCR did not recover usable content for them."

    return updated, status
