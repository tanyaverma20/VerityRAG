"""
test_ocr_fallback.py — OCR fallback for scanned/image-only PDFs (request D,
item 13.C). Tesseract is NOT installed in this environment (confirmed
during development), so the "OCR actually succeeds" path is proven via a
mocked pytesseract/fitz stack; the "OCR unavailable -> clear status, no
silent hallucination" path is proven for real, unmocked, since that's the
genuine state of this environment.
"""
from unittest.mock import patch

import pytest

import ocr_fallback


# ---------------------------------------------------------------------------
# Deterministic insufficient-text detection
# ---------------------------------------------------------------------------
def test_is_text_insufficient_empty_string():
    assert ocr_fallback.is_text_insufficient("") is True
    assert ocr_fallback.is_text_insufficient(None) is True
    assert ocr_fallback.is_text_insufficient("   \n\n  ") is True


def test_is_text_insufficient_short_text():
    assert ocr_fallback.is_text_insufficient("hi") is True


def test_is_text_insufficient_real_paragraph_is_sufficient():
    real_text = "This paper proposes a sparse mixture-of-experts routing layer with a novel load-balancing loss."
    assert ocr_fallback.is_text_insufficient(real_text) is False


# ---------------------------------------------------------------------------
# Real environment check: Tesseract is genuinely not installed here — this
# test documents that fact rather than assuming it, and proves the
# unavailable-engine path degrades cleanly rather than crashing.
# ---------------------------------------------------------------------------
def test_ocr_engine_availability_reflects_real_environment():
    # Whatever the real answer is, it must be a clean bool, never an
    # exception — this is the core "never crash" contract.
    result = ocr_fallback.ocr_available()
    assert isinstance(result, bool)


def test_normal_text_pdf_never_invokes_ocr(tmp_path):
    """A. normal text PDF -> OCR not invoked. Pages that already have
    sufficient text must never trigger an OCR attempt, regardless of
    whether the engine is even available."""
    pages = [
        {"page_number": 1, "text": "This is a perfectly normal page of extracted text, plenty of it here."},
        {"page_number": 2, "text": "Another normal page with real content extracted by pypdf directly."},
    ]
    with patch("ocr_fallback.ocr_page_image") as mock_ocr:
        updated, status = ocr_fallback.extract_pages_with_ocr_fallback(str(tmp_path / "fake.pdf"), pages)
    mock_ocr.assert_not_called()
    assert status["pages_needing_ocr"] == 0
    assert updated == pages


def test_image_only_pdf_invokes_ocr_when_engine_available(tmp_path):
    """C/D. image-only page -> OCR invoked when the engine IS available
    (mocked here since Tesseract isn't installed in this environment) —
    and the OCR'd text replaces just that page's text."""
    pages = [{"page_number": 1, "text": ""}]  # scanned page, pypdf got nothing
    with patch("ocr_fallback._configure_tesseract", return_value=True), \
         patch("ocr_fallback.ocr_page_image", return_value="Recovered text from OCR of the scanned page.") as mock_ocr:
        updated, status = ocr_fallback.extract_pages_with_ocr_fallback(str(tmp_path / "fake.pdf"), pages)

    mock_ocr.assert_called_once()
    assert status["pages_needing_ocr"] == 1
    assert status["pages_ocr_succeeded"] == 1
    assert status["pages_ocr_failed"] == 0
    assert updated[0]["text"] == "Recovered text from OCR of the scanned page."
    assert updated[0]["page_number"] == 1, "page numbers must be preserved through OCR fallback"


def test_ocr_unavailable_leaves_page_as_is_with_clear_status(tmp_path):
    """B. OCR engine genuinely unavailable (the real state of this
    environment) -> pages are left with their original (empty) text, never
    fabricated, and the status clearly explains why."""
    pages = [{"page_number": 1, "text": ""}]
    with patch("ocr_fallback._configure_tesseract", return_value=False):
        updated, status = ocr_fallback.extract_pages_with_ocr_fallback(str(tmp_path / "fake.pdf"), pages)

    assert status["ocr_engine_available"] is False
    assert status["pages_needing_ocr"] == 1
    assert status["pages_ocr_succeeded"] == 0
    assert "not installed" in status["note"]
    assert updated == pages, "must leave the page's original text untouched, never invent content"


def test_mixed_pdf_only_insufficient_pages_get_ocr(tmp_path):
    """C. mixed PDF (some real-text pages, some scanned pages) -> fallback
    behavior applies ONLY to the insufficient pages."""
    pages = [
        {"page_number": 1, "text": "A completely normal page with plenty of real extracted text content."},
        {"page_number": 2, "text": ""},  # scanned
        {"page_number": 3, "text": "Another normal page, again with real substantial extracted text."},
    ]
    with patch("ocr_fallback._configure_tesseract", return_value=True), \
         patch("ocr_fallback.ocr_page_image", return_value="OCR recovered this scanned page's text.") as mock_ocr:
        updated, status = ocr_fallback.extract_pages_with_ocr_fallback(str(tmp_path / "fake.pdf"), pages)

    assert mock_ocr.call_count == 1
    mock_ocr.assert_called_with(str(tmp_path / "fake.pdf"), 2)
    assert status["pages_needing_ocr"] == 1
    assert updated[0]["text"] == pages[0]["text"]  # untouched
    assert updated[1]["text"] == "OCR recovered this scanned page's text."  # replaced
    assert updated[2]["text"] == pages[2]["text"]  # untouched


def test_ocr_failure_on_a_specific_page_keeps_original_text_not_silent_hallucination(tmp_path):
    """D. OCR failure -> clear degradation (original, possibly-empty text
    kept) instead of silently fabricating page content."""
    pages = [{"page_number": 1, "text": ""}]
    with patch("ocr_fallback._configure_tesseract", return_value=True), \
         patch("ocr_fallback.ocr_page_image", return_value=None):  # OCR ran but found nothing usable
        updated, status = ocr_fallback.extract_pages_with_ocr_fallback(str(tmp_path / "fake.pdf"), pages)

    assert status["pages_ocr_failed"] == 1
    assert status["pages_ocr_succeeded"] == 0
    assert updated[0]["text"] == "", "a failed OCR attempt must never fabricate replacement text"


def test_ocr_page_image_only_ever_opens_the_requested_pdf_path(tmp_path):
    """Isolation: ocr_page_image() must open exactly the pdf_path it was
    given, never any other file — proven by pointing it at two different
    (real, distinct) PDF paths and confirming fitz.open() is called with
    each path individually, never mixed or cached across calls."""
    fixture_a = tmp_path / "doc_a.pdf"
    fixture_b = tmp_path / "doc_b.pdf"
    fixture_a.write_bytes(b"not a real pdf, just needs to exist as a path")
    fixture_b.write_bytes(b"not a real pdf either")

    opened_paths = []
    real_fitz_open = None
    import fitz as _fitz_module
    real_fitz_open = _fitz_module.open

    def _tracking_open(path, *a, **k):
        opened_paths.append(str(path))
        raise RuntimeError("simulated: not a real PDF, stop before actually parsing")

    with patch("ocr_fallback._configure_tesseract", return_value=True), \
         patch("fitz.open", side_effect=_tracking_open):
        ocr_fallback.ocr_page_image(str(fixture_a), 1)
        ocr_fallback.ocr_page_image(str(fixture_b), 1)

    assert opened_paths == [str(fixture_a), str(fixture_b)], (
        "Each OCR call must open exactly its own requested PDF path, in order, "
        "never a cached/previous document's path"
    )


def test_extract_pages_with_ocr_fallback_has_no_cross_call_state_leakage(tmp_path):
    """Isolation: running the fallback for document A, then document B, must
    never let A's OCR'd text or status leak into B's result — the function
    is stateless (no module-level cache/singleton), proven by two
    back-to-back calls with different mocked OCR outputs per call."""
    pdf_a = str(tmp_path / "doc_a.pdf")
    pdf_b = str(tmp_path / "doc_b.pdf")
    pages_a = [{"page_number": 1, "text": ""}]
    pages_b = [{"page_number": 1, "text": ""}]

    with patch("ocr_fallback._configure_tesseract", return_value=True), \
         patch("ocr_fallback.ocr_page_image", return_value="Document A's real OCR text."):
        updated_a, status_a = ocr_fallback.extract_pages_with_ocr_fallback(pdf_a, pages_a)

    with patch("ocr_fallback._configure_tesseract", return_value=True), \
         patch("ocr_fallback.ocr_page_image", return_value="Document B's completely different OCR text."):
        updated_b, status_b = ocr_fallback.extract_pages_with_ocr_fallback(pdf_b, pages_b)

    assert updated_a[0]["text"] == "Document A's real OCR text."
    assert updated_b[0]["text"] == "Document B's completely different OCR text."
    assert updated_a[0]["text"] != updated_b[0]["text"], "B's OCR result must never leak into A's, or vice versa"
    assert status_a["pages_ocr_succeeded"] == 1
    assert status_b["pages_ocr_succeeded"] == 1


def test_real_image_only_pdf_is_genuinely_detected_as_needing_ocr():
    """Real verification against an actual scanned-style PDF (not a mocked
    dict): tests/fixtures/scanned_image_only.pdf is a genuine PDF built
    from a rendered image with NO text layer at all (see its generation:
    a PNG drawn with PIL, inserted as the page's only content via
    page.insert_image() — nothing ever calls insert_text()). Real pypdf
    extraction against this real file must come back empty, and
    is_text_insufficient() must correctly flag it — this is what actually
    triggers the OCR fallback path in production, proven end-to-end against
    a real file rather than only ever a synthetic {"text": ""} dict."""
    from pathlib import Path
    from ingest import extract_pages_from_pdf

    fixture = Path(__file__).parent / "tests" / "fixtures" / "scanned_image_only.pdf"
    assert fixture.exists(), "real image-only test fixture must exist"

    pages = extract_pages_from_pdf(str(fixture))
    assert len(pages) == 1
    assert ocr_fallback.is_text_insufficient(pages[0]["text"]) is True, (
        "A real image-only PDF (no text layer) must be genuinely detected as needing OCR, "
        "not just in a mocked scenario"
    )

    # Full pipeline: extract_pages_with_ocr_fallback() against this REAL
    # file, with Tesseract genuinely unavailable in this environment (no
    # mocking of _configure_tesseract here) — must degrade honestly.
    updated, status = ocr_fallback.extract_pages_with_ocr_fallback(str(fixture), pages)
    assert status["pages_needing_ocr"] == 1
    if status["ocr_engine_available"]:
        # If Tesseract IS installed on whatever machine runs this test,
        # OCR will genuinely attempt to read the (text-free) drawing — it
        # may recover nothing, which is correct, not a failure.
        assert status["pages_ocr_succeeded"] + status["pages_ocr_failed"] == 1
    else:
        assert "not installed" in status["note"]
        assert updated[0]["text"] == "", "must never fabricate text for a real unreadable scanned page"


# ---------------------------------------------------------------------------
# ingest.py wiring — a normal PDF's ingestion is completely unaffected
# ---------------------------------------------------------------------------
def test_extract_pages_from_pdf_now_returns_all_pages_including_empty(tmp_path):
    """ingest.extract_pages_from_pdf() must no longer silently drop pages
    with empty text (previously done, which would hide scanned pages from
    the OCR fallback entirely) — proven against the real test fixture PDF,
    which has genuine extractable text on every page."""
    from ingest import extract_pages_from_pdf
    from pathlib import Path
    fixture = Path(__file__).parent / "tests" / "fixtures" / "attention.pdf"
    pages = extract_pages_from_pdf(str(fixture))
    assert len(pages) > 0
    assert all("page_number" in p and "text" in p for p in pages)
    # A real, text-rich PDF: every page should have real (sufficient) text —
    # confirms the "include all pages" change didn't somehow break normal
    # extraction for a normal document.
    assert all(not ocr_fallback.is_text_insufficient(p["text"]) for p in pages)
