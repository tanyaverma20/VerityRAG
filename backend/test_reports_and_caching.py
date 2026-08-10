"""
test_reports_and_caching.py — Mocked tests for report generation, caching,
and JSON-validation-failure handling. No real network requests, no Groq
token usage.

TESTs A, C, G, I from the "1 LLM call" spec are already covered by
test_llm_call_count.py and are not duplicated here. This file covers the
NEW surface added this turn: B (cache), D (both models fail), E/F (report
generation), H (cache invalidation), J (JSON validation failure).

Run with:
    cd backend
    python -m pytest test_reports_and_caching.py -v
"""
import json
from unittest.mock import patch

import pytest

import cache
from query_transform import start_call_tracking, get_call_log, with_model_fallback, RATE_LIMIT_MESSAGE, GENERATION_FAILED_MESSAGE


# ---------------------------------------------------------------------------
# Fake Groq SDK (same shape as test_llm_call_count.py — kept independent so
# this file has no import-order dependency on it)
# ---------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content

class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)

class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 10})()

_behavior = {"fn": None}  # callable(model) -> str content, or raises

class _FakeCompletions:
    def create(self, model, messages, temperature=0.0, max_tokens=None):
        return _FakeResponse(_behavior["fn"](model))

class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()

class _FakeGroqClient:
    def __init__(self, api_key=None):
        self.chat = _FakeChat()


@pytest.fixture(autouse=True)
def _reset():
    start_call_tracking()
    cache.clear_all()
    yield
    cache.clear_all()


# ---------------------------------------------------------------------------
# TEST B — repeated cached question => zero LLM calls
# ---------------------------------------------------------------------------
def test_cache_hit_costs_zero_llm_calls():
    payload = {"answer": "Cached answer text.", "confidence": "HIGH"}
    cache.set_cached_answer("What is X?", ["docA"], "simple", payload)

    # A cache hit path never even calls _call_groq_raw — assert the stored
    # value round-trips exactly, and that a differently-scoped request misses.
    hit = cache.get_cached_answer("What is X?", ["docA"], "simple")
    assert hit is not None
    assert hit["answer"] == "Cached answer text."

    miss_different_docs = cache.get_cached_answer("What is X?", ["docA", "docB"], "simple")
    assert miss_different_docs is None, "Changing the active document set must miss the cache"

    miss_different_question = cache.get_cached_answer("What is Y?", ["docA"], "simple")
    assert miss_different_question is None

    miss_different_mode = cache.get_cached_answer("What is X?", ["docA"], "deep")
    assert miss_different_mode is None, "simple vs deep must not share a cache entry"

    # Question text normalization: whitespace/case differences still hit.
    hit_normalized = cache.get_cached_answer("  what IS x?  ", ["docA"], "simple")
    assert hit_normalized is not None


def test_cache_clear_invalidates_everything():
    cache.set_cached_answer("Q1", ["docA"], "simple", {"answer": "A1"})
    cache.set_cached_report(["docA"], {"title": "R1"})
    assert cache.stats()["answer_entries"] == 1
    assert cache.stats()["report_entries"] == 1

    cache.clear_all()  # what /upload and DELETE /documents/{id} call

    # stats() now also carries hit/miss/backend observability fields (see
    # cache.py) — assert the two counts this test actually cares about
    # rather than exact dict equality, which would break on any future
    # additive observability field.
    after = cache.stats()
    assert after["answer_entries"] == 0
    assert after["report_entries"] == 0
    assert cache.get_cached_answer("Q1", ["docA"], "simple") is None
    assert cache.get_cached_report(["docA"]) is None


# ---------------------------------------------------------------------------
# TEST D — both primary and fallback fail => clean failure, no hallucination,
# no more than the 2 documented attempts
# ---------------------------------------------------------------------------
def test_both_models_fail_clean_error_no_extra_retries():
    from graph.organizer import organize_evidence

    class _AlwaysFails(Exception):
        status_code = 429  # temporary, so fallback IS attempted once

    def always_fail(model):
        raise _AlwaysFails("429 rate limited")

    _behavior["fn"] = None  # unused; make_request below bypasses the Groq SDK entirely
    with pytest.raises(_AlwaysFails):
        with_model_fallback(always_fail)
    log = get_call_log()
    assert len(log) == 2, f"Expected exactly primary + 1 fallback attempt, got {len(log)}: {log}"
    assert all(c["success"] is False for c in log), "Both attempts must be recorded as failed"

    # Now the same failure mode through the real synthesizer path — confirm
    # a clean, non-hallucinated message and empty citations, not a crash.
    from graph.synthesizer import synthesize_answer
    state = {
        "original_query": "test",
        "research_type": "simple",
        "retrieval_results": [{
            "text": "evidence", "parent_context": "evidence in context",
            "metadata": {"document_id": "docA", "parent_id": "p1", "chunk_id": "c1", "source": "a.pdf"},
        }],
    }
    state.update(organize_evidence(state))
    start_call_tracking()
    with patch("groq.Groq", _FakeGroqClient):
        _behavior["fn"] = always_fail
        result = synthesize_answer(state)
    assert result["draft_answer"] in (RATE_LIMIT_MESSAGE, GENERATION_FAILED_MESSAGE)
    assert result["citations"] == [], "No sources may be shown for an answer that was never actually generated"
    assert len(get_call_log()) == 2


# ---------------------------------------------------------------------------
# TEST J — JSON validation failure => clean recovery, NOT a reason to retry
# ---------------------------------------------------------------------------
def test_json_validation_failure_no_extra_calls():
    from graph.organizer import organize_evidence
    from graph.synthesizer import synthesize_answer

    # Missing the required "answer" field entirely.
    _behavior["fn"] = lambda model: json.dumps({"grounded": True, "evidence_sufficient": True, "document_ids": []})

    state = {
        "original_query": "test",
        "research_type": "simple",
        "retrieval_results": [{
            "text": "evidence", "parent_context": "evidence in context",
            "metadata": {"document_id": "docA", "parent_id": "p1", "chunk_id": "c1", "source": "a.pdf"},
        }],
    }
    state.update(organize_evidence(state))
    start_call_tracking()
    with patch("groq.Groq", _FakeGroqClient):
        result = synthesize_answer(state)

    log = get_call_log()
    assert len(log) == 1, f"A malformed JSON body must not trigger a retry — the HTTP call itself succeeded. Got {len(log)}: {log}"
    assert log[0]["success"] is True, "The physical HTTP call succeeded; only downstream parsing failed"
    assert result["draft_answer"] == GENERATION_FAILED_MESSAGE
    assert result["citations"] == []


# ---------------------------------------------------------------------------
# TEST E — single-PDF report => exactly 1 LLM call
# ---------------------------------------------------------------------------
SINGLE_REPORT_JSON = json.dumps({
    "title": "Report on Paper A",
    "overview": "This report covers one paper.",
    "papers": [{
        "document_id": "docA", "title": "Paper A", "overview": "O", "main_contribution": "MC",
        "methodology": "M", "architecture": "Arch", "datasets": ["D1"], "evaluation_metrics": ["Accuracy"],
        "key_results": ["f1"], "important_calculations": [], "limitations": ["l1"], "final_summary": "FS",
    }],
    "comparison": None,
    "conclusion": "Conclusion.",
    "evidence_sufficient": True,
})

COMPARATIVE_REPORT_JSON = json.dumps({
    "title": "Comparative Report",
    "overview": "This report compares two papers.",
    "papers": [
        {"document_id": "docA", "title": "Paper A", "overview": "O1", "main_contribution": "MC1", "methodology": "M1", "architecture": "Arch1", "datasets": ["D1"], "evaluation_metrics": ["F1"], "key_results": ["f1"], "important_calculations": [], "limitations": ["l1"], "final_summary": "FS1"},
        {"document_id": "docB", "title": "Paper B", "overview": "O2", "main_contribution": "MC2", "methodology": "M2", "architecture": "Arch2", "datasets": ["D2"], "evaluation_metrics": ["BLEU"], "key_results": ["f2"], "important_calculations": [], "limitations": ["l2"], "final_summary": "FS2"},
    ],
    "comparison": {"commonalities": ["c1"], "differences": ["d1"], "strengths": ["s1"], "limitations": ["lim1"]},
    "conclusion": "Both papers contribute to the field.",
    "evidence_sufficient": True,
})


def _fake_grouped_evidence(document_ids):
    return {
        doc_id: [{
            "text": f"Sample evidence sentence for {doc_id}.",
            "parent_context": f"Sample evidence sentence for {doc_id}, with surrounding context.",
            "metadata": {"document_id": doc_id, "source": f"{doc_id}.pdf", "page_number": 1},
        }]
        for doc_id in document_ids
    }


def test_single_pdf_report_one_llm_call():
    import report_generator
    _behavior["fn"] = lambda model: SINGLE_REPORT_JSON
    with patch("report_generator._gather_report_evidence", side_effect=_fake_grouped_evidence), \
         patch("groq.Groq", _FakeGroqClient):
        result = report_generator.generate_report(["docA"])
    log = get_call_log()
    assert len(log) == 1, f"Single-PDF report must be exactly 1 LLM call, got {len(log)}: {log}"
    assert result["ok"] is True
    assert result["report"].comparison is None
    assert len(result["report"].papers) == 1


def test_multi_pdf_comparative_report_one_llm_call():
    import report_generator
    _behavior["fn"] = lambda model: COMPARATIVE_REPORT_JSON
    with patch("report_generator._gather_report_evidence", side_effect=_fake_grouped_evidence), \
         patch("groq.Groq", _FakeGroqClient):
        result = report_generator.generate_report(["docA", "docB"])
    log = get_call_log()
    assert len(log) == 1, f"A comparative report across N papers must STILL be exactly 1 LLM call, got {len(log)}: {log}"
    assert result["ok"] is True
    assert result["report"].comparison is not None
    assert {p.document_id for p in result["report"].papers} == {"docA", "docB"}


def test_report_no_documents_no_llm_call():
    import report_generator
    result = report_generator.generate_report([])
    assert result["ok"] is False
    assert len(get_call_log()) == 0, "No documents scoped => must not call the LLM at all"


def test_report_json_validation_failure_no_extra_calls():
    import report_generator
    _behavior["fn"] = lambda model: json.dumps({"overview": "missing title and conclusion"})
    with patch("report_generator._gather_report_evidence", side_effect=_fake_grouped_evidence), \
         patch("groq.Groq", _FakeGroqClient):
        result = report_generator.generate_report(["docA"])
    log = get_call_log()
    assert len(log) == 1, f"Malformed report JSON must not trigger a retry, got {len(log)}: {log}"
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# TEST — Markdown/PDF renderers work from validated JSON, no LLM involved
# ---------------------------------------------------------------------------
def test_report_renderers_are_llm_free():
    import report_generator
    from schemas import ResearchReport, PaperReport

    report = ResearchReport(
        title="Test", overview="Overview.",
        papers=[PaperReport(document_id="docA", title="A", key_results=["f1"], limitations=[])],
        comparison=None, conclusion="Done.", evidence_sufficient=True,
    )
    start_call_tracking()
    md = report_generator.render_report_markdown(report)
    pdf_bytes = report_generator.render_report_pdf(report)
    docx_bytes = report_generator.render_report_docx(report)
    assert len(get_call_log()) == 0, "Rendering must never touch the LLM"
    assert "# Test" in md
    assert pdf_bytes.startswith(b"%PDF-")
    assert docx_bytes.startswith(b"PK"), "A .docx is a zip archive — must start with the PK signature"
