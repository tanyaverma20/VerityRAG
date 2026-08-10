"""
test_figure_vision.py — figure_vision.py (page/figure image rendering) and
analysis.explain_figure()'s honest vision/text fallback (request D, item
13.B). Uses the real test PDF fixture (backend/tests/fixtures/attention.pdf)
for actual PyMuPDF rendering — no mocking of the rendering itself, since
it's zero-LLM-cost, deterministic, and worth proving for real.
"""
import shutil
from pathlib import Path

import pytest

import figure_vision


@pytest.fixture
def persisted_test_pdf():
    """Persists the real test fixture PDF under a throwaway document_id, in
    the SAME data/uploads/ location the real /upload endpoint writes to,
    and cleans it up afterward."""
    fixture_path = Path(__file__).parent / "tests" / "fixtures" / "attention.pdf"
    assert fixture_path.exists(), "test fixture PDF must exist"
    document_id = "figvision_test_doc"
    dest = figure_vision.persist_uploaded_pdf(str(fixture_path), document_id)
    yield document_id
    dest.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Page/figure extraction — real PyMuPDF rendering
# ---------------------------------------------------------------------------
def test_render_page_as_image_returns_real_png_bytes(persisted_test_pdf):
    result = figure_vision.render_page_as_image_base64(persisted_test_pdf, page_number=1)
    assert result is not None
    import base64
    png_bytes = base64.b64decode(result)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n", "must be a real PNG file (correct magic bytes)"
    assert len(png_bytes) > 1000, "a rendered page image should be a substantial, non-trivial file"


def test_render_page_out_of_range_returns_none_not_an_error(persisted_test_pdf):
    assert figure_vision.render_page_as_image_base64(persisted_test_pdf, page_number=99999) is None
    assert figure_vision.render_page_as_image_base64(persisted_test_pdf, page_number=0) is None


def test_render_page_image_metadata_scales_with_zoom(persisted_test_pdf):
    """A real check on the rendered image's own metadata (dimensions), not
    just that bytes came back — a higher zoom factor must produce a
    proportionally larger image, proving the zoom parameter genuinely
    controls the PyMuPDF render matrix rather than being ignored."""
    import base64
    import io
    from PIL import Image

    low = figure_vision.render_page_as_image_base64(persisted_test_pdf, page_number=1, zoom=1.0)
    high = figure_vision.render_page_as_image_base64(persisted_test_pdf, page_number=1, zoom=3.0)
    assert low is not None and high is not None

    low_img = Image.open(io.BytesIO(base64.b64decode(low)))
    high_img = Image.open(io.BytesIO(base64.b64decode(high)))
    assert low_img.format == "PNG" and high_img.format == "PNG"
    assert high_img.width > low_img.width and high_img.height > low_img.height, (
        f"3x zoom ({high_img.size}) must render larger than 1x zoom ({low_img.size})"
    )
    # Roughly proportional (matrix-scaled), not coincidentally different.
    assert abs((high_img.width / low_img.width) - 3.0) < 0.05


# ---------------------------------------------------------------------------
# Document isolation — only ever opens the ONE requested document_id's PDF
# ---------------------------------------------------------------------------
def test_get_uploaded_pdf_path_is_scoped_to_exactly_one_document_id(persisted_test_pdf):
    other_doc_id = "some_other_document_never_persisted"
    assert figure_vision.get_uploaded_pdf_path(other_doc_id) is None
    assert figure_vision.get_uploaded_pdf_path(persisted_test_pdf) is not None


def test_rendering_a_different_document_id_never_touches_the_persisted_one(persisted_test_pdf):
    # A document_id that was never persisted must render nothing, even
    # though a DIFFERENT document's PDF is sitting right next to where it
    # would be — no accidental cross-document fallback.
    assert figure_vision.render_page_as_image_base64("totally_unrelated_doc_id", page_number=1) is None


# ---------------------------------------------------------------------------
# Missing visual input / vision-model-unavailable — honest fallback
# ---------------------------------------------------------------------------
def test_missing_pdf_file_returns_none():
    assert figure_vision.render_page_as_image_base64("never_uploaded_at_all", page_number=1) is None


def test_vision_model_available_reflects_env_var(monkeypatch):
    monkeypatch.setattr(figure_vision, "VISION_MODEL", "")
    assert figure_vision.vision_model_available() is False
    monkeypatch.setattr(figure_vision, "VISION_MODEL", "some-real-vision-model-id")
    assert figure_vision.vision_model_available() is True


# ---------------------------------------------------------------------------
# analysis.explain_figure() — honest visual_inspection_used flag
# ---------------------------------------------------------------------------
def test_explain_figure_without_image_never_claims_visual_inspection():
    from unittest.mock import patch
    import json

    fake_answer = json.dumps({"answer": "Text-based explanation.", "grounded": True, "evidence_sufficient": True, "document_ids": []})

    class _FakeMessage:
        def __init__(self, content): self.content = content
    class _FakeChoice:
        def __init__(self, content): self.message = _FakeMessage(content)
    class _FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
    class _FakeResponse:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]
            self.usage = _FakeUsage()
    class _FakeCompletions:
        def create(self, **kwargs):
            return _FakeResponse(fake_answer)
    class _FakeChat:
        def __init__(self): self.completions = _FakeCompletions()
    class _FakeGroqClient:
        def __init__(self, api_key=None): self.chat = _FakeChat()

    from analysis import explain_figure
    with patch("groq.Groq", _FakeGroqClient):
        result, visual_used = explain_figure("some evidence text", "Figure 1", image_base64=None, vision_model=None)

    assert visual_used is False
    assert result is not None
    assert result.answer == "Text-based explanation."


def test_explain_figure_with_image_and_vision_model_reports_visual_inspection_used():
    from unittest.mock import patch
    import json

    fake_answer = json.dumps({"answer": "I can see a routing diagram.", "grounded": True, "evidence_sufficient": True, "document_ids": []})

    class _FakeMessage:
        def __init__(self, content): self.content = content
    class _FakeChoice:
        def __init__(self, content): self.message = _FakeMessage(content)
    class _FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
    class _FakeResponse:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]
            self.usage = _FakeUsage()
    class _FakeCompletions:
        def create(self, **kwargs):
            # Confirm the multimodal message shape was actually used.
            content = kwargs["messages"][0]["content"]
            assert isinstance(content, list)
            assert any(block.get("type") == "image_url" for block in content)
            return _FakeResponse(fake_answer)
    class _FakeChat:
        def __init__(self): self.completions = _FakeCompletions()
    class _FakeGroqClient:
        def __init__(self, api_key=None): self.chat = _FakeChat()

    from analysis import explain_figure
    with patch("groq.Groq", _FakeGroqClient):
        result, visual_used = explain_figure(
            "some evidence text", "Figure 1",
            image_base64="ZmFrZWJhc2U2NA==", vision_model="fake-vision-model",
        )

    assert visual_used is True
    assert result is not None
    assert result.answer == "I can see a routing diagram."


def test_explain_figure_vision_call_returns_malformed_json_falls_back_to_text():
    """Distinct from a vision call raising (network/provider failure,
    already covered): here the vision call SUCCEEDS and returns a string,
    but it's not valid JSON (or doesn't validate against AnswerJSON) — a
    real, plausible failure mode for any LLM call. explain_figure() must
    parse-and-validate defensively and fall through to a real text-only
    call, not crash and not silently claim visual_used=True with garbage."""
    from unittest.mock import patch
    import json

    fake_text_answer = json.dumps({"answer": "Fallback text explanation.", "grounded": True, "evidence_sufficient": True, "document_ids": []})
    call_log = []

    class _FakeMessage:
        def __init__(self, content): self.content = content
    class _FakeChoice:
        def __init__(self, content): self.message = _FakeMessage(content)
    class _FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
    class _FakeResponse:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]
            self.usage = _FakeUsage()
    class _FakeCompletions:
        def create(self, **kwargs):
            content = kwargs["messages"][0]["content"]
            is_vision_call = isinstance(content, list)
            call_log.append(is_vision_call)
            if is_vision_call:
                # A real, syntactically-broken response — not an exception,
                # not valid JSON at all.
                return _FakeResponse("Sure! I can see the figure shows a diagram with {broken json here")
            return _FakeResponse(fake_text_answer)
    class _FakeChat:
        def __init__(self): self.completions = _FakeCompletions()
    class _FakeGroqClient:
        def __init__(self, api_key=None): self.chat = _FakeChat()

    from analysis import explain_figure
    with patch("groq.Groq", _FakeGroqClient):
        result, visual_used = explain_figure(
            "some evidence text", "Figure 1",
            image_base64="ZmFrZWJhc2U2NA==", vision_model="fake-vision-model",
        )

    assert call_log == [True, False], "must attempt vision first, then fall back to a real text-only call"
    assert visual_used is False, "malformed vision output must never be reported as a successful visual inspection"
    assert result is not None
    assert result.answer == "Fallback text explanation."


def test_explain_figure_vision_call_returns_valid_json_but_wrong_schema_falls_back():
    """The vision call returns syntactically valid JSON, but missing
    required AnswerJSON fields — Pydantic validation must reject it and
    fall through, not crash on a KeyError/AttributeError downstream."""
    from unittest.mock import patch
    import json

    fake_text_answer = json.dumps({"answer": "Fallback text explanation.", "grounded": True, "evidence_sufficient": True, "document_ids": []})
    call_log = []

    class _FakeMessage:
        def __init__(self, content): self.content = content
    class _FakeChoice:
        def __init__(self, content): self.message = _FakeMessage(content)
    class _FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
    class _FakeResponse:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]
            self.usage = _FakeUsage()
    class _FakeCompletions:
        def create(self, **kwargs):
            content = kwargs["messages"][0]["content"]
            is_vision_call = isinstance(content, list)
            call_log.append(is_vision_call)
            if is_vision_call:
                # Valid JSON, wrong shape entirely (not AnswerJSON's fields).
                return _FakeResponse(json.dumps({"description": "a diagram", "confidence": 0.9}))
            return _FakeResponse(fake_text_answer)
    class _FakeChat:
        def __init__(self): self.completions = _FakeCompletions()
    class _FakeGroqClient:
        def __init__(self, api_key=None): self.chat = _FakeChat()

    from analysis import explain_figure
    with patch("groq.Groq", _FakeGroqClient):
        result, visual_used = explain_figure(
            "some evidence text", "Figure 1",
            image_base64="ZmFrZWJhc2U2NA==", vision_model="fake-vision-model",
        )

    assert call_log == [True, False]
    assert visual_used is False
    assert result is not None
    assert result.answer == "Fallback text explanation."


def test_explain_figure_vision_call_failure_falls_back_to_text_honestly():
    """If the vision call fails for any reason, explain_figure() must fall
    back to the text-only path and report visual_used=False — never a
    partial/broken result claiming visual inspection that didn't happen."""
    from unittest.mock import patch
    import json

    fake_text_answer = json.dumps({"answer": "Fallback text explanation.", "grounded": True, "evidence_sufficient": True, "document_ids": []})
    call_log = []

    class _FakeMessage:
        def __init__(self, content): self.content = content
    class _FakeChoice:
        def __init__(self, content): self.message = _FakeMessage(content)
    class _FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
    class _FakeResponse:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]
            self.usage = _FakeUsage()
    class _FakeCompletions:
        def create(self, **kwargs):
            content = kwargs["messages"][0]["content"]
            is_vision_call = isinstance(content, list)
            call_log.append(is_vision_call)
            if is_vision_call:
                raise ConnectionError("simulated vision model outage")
            return _FakeResponse(fake_text_answer)
    class _FakeChat:
        def __init__(self): self.completions = _FakeCompletions()
    class _FakeGroqClient:
        def __init__(self, api_key=None): self.chat = _FakeChat()

    from analysis import explain_figure
    with patch("groq.Groq", _FakeGroqClient):
        result, visual_used = explain_figure(
            "some evidence text", "Figure 1",
            image_base64="ZmFrZWJhc2U2NA==", vision_model="fake-vision-model",
        )

    assert call_log == [True, False], "must attempt vision first, then fall back to a real text-only call"
    assert visual_used is False
    assert result is not None
    assert result.answer == "Fallback text explanation."


# ---------------------------------------------------------------------------
# End-to-end via /analyze: page selection (which page gets rendered) and
# the vision-model gate, exercised through the real HTTP endpoint.
# ---------------------------------------------------------------------------
def test_analyze_explain_figure_never_attempts_rendering_when_no_vision_model_configured(monkeypatch):
    """With GROQ_VISION_MODEL unset (this deployment's real, current state),
    the /analyze explain_figure path must never even attempt to render a
    page image — confirms the gate in main.py (`vision_model_available()`)
    actually short-circuits BEFORE calling render_page_as_image_base64(),
    not just that explain_figure() itself degrades gracefully if it were
    called with no image."""
    import json
    from pathlib import Path
    from unittest.mock import patch
    from fastapi.testclient import TestClient

    import main
    from main import app

    monkeypatch.setattr(main, "VISION_MODEL", "")
    monkeypatch.setattr("figure_vision.VISION_MODEL", "")

    render_calls = []
    original_render = main.render_page_as_image_base64
    def _tracking_render(*a, **k):
        render_calls.append((a, k))
        return original_render(*a, **k)
    monkeypatch.setattr(main, "render_page_as_image_base64", _tracking_render)

    client = TestClient(app)
    fixture_pdf = Path(__file__).parent / "tests" / "fixtures" / "attention.pdf"
    with open(fixture_pdf, "rb") as f:
        upload = client.post("/upload", files={"file": ("attention.pdf", f, "application/pdf")}).json()
    doc_id = upload["document_id"]

    fake_answer = json.dumps({"answer": "Text-only explanation.", "grounded": True, "evidence_sufficient": True, "document_ids": [doc_id]})
    class _FakeMessage:
        def __init__(self, content): self.content = content
    class _FakeChoice:
        def __init__(self, content): self.message = _FakeMessage(content)
    class _FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
    class _FakeResponse:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]
            self.usage = _FakeUsage()
    class _FakeCompletions:
        def create(self, **kwargs):
            assert isinstance(kwargs["messages"][0]["content"], str), "must never be a multimodal call when no vision model is configured"
            return _FakeResponse(fake_answer)
    class _FakeChat:
        def __init__(self): self.completions = _FakeCompletions()
    class _FakeGroqClient:
        def __init__(self, api_key=None): self.chat = _FakeChat()

    with patch("groq.Groq", _FakeGroqClient):
        resp = client.post("/analyze", json={
            "mode": "explain_figure", "document_ids": [doc_id], "figure_reference": "Figure 1",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert render_calls == [], "render_page_as_image_base64() must never be called when no vision model is configured"
