"""
test_groundedness_eval.py — groundedness_eval.py's deterministic scoring
functions (pure, no LLM) plus a mocked end-to-end evaluate_one() proving
exactly ONE Groq call happens per question and the scores are computed
from the model's own claim_evidence_trace, not invented.
"""
import json

import pytest

from groundedness_eval import score_claims, score_evidence_coverage, score_answer_relevance, evaluate_one, NOT_MEASURED


# ---------------------------------------------------------------------------
# Pure scoring functions — no LLM, no retrieval
# ---------------------------------------------------------------------------
def test_score_claims_empty_list_is_not_measured():
    result = score_claims([])
    assert result["groundedness_score"] == NOT_MEASURED
    assert result["unsupported_claim_rate"] == NOT_MEASURED
    assert result["total_claims"] == 0


def test_score_claims_all_supported():
    claims = [
        {"claim": "a", "support": "DIRECTLY_STATED"},
        {"claim": "b", "support": "STRONGLY_SUPPORTED"},
    ]
    result = score_claims(claims)
    assert result["groundedness_score"] == 1.0
    assert result["unsupported_claim_rate"] == 0.0
    assert result["total_claims"] == 2


def test_score_claims_mixed_support():
    claims = [
        {"claim": "a", "support": "DIRECTLY_STATED"},
        {"claim": "b", "support": "NOT_FOUND"},
        {"claim": "c", "support": "NOT_FOUND"},
        {"claim": "d", "support": "STRONGLY_SUPPORTED"},
    ]
    result = score_claims(claims)
    assert result["groundedness_score"] == 0.5   # 2 of 4 supported
    assert result["unsupported_claim_rate"] == 0.5
    assert result["directly_stated_rate"] == 0.25
    assert result["strongly_supported_rate"] == 0.25


def test_score_evidence_coverage_no_contributing_docs_is_not_measured():
    assert score_evidence_coverage([{"document_ids": ["doc1"]}], []) == NOT_MEASURED


def test_score_evidence_coverage_partial():
    claims = [{"claim": "a", "document_ids": ["doc1"]}]
    result = score_evidence_coverage(claims, ["doc1", "doc2"])
    assert result == 0.5  # only doc1 was cited out of 2 contributing documents


def test_score_evidence_coverage_full():
    claims = [{"document_ids": ["doc1"]}, {"document_ids": ["doc2"]}]
    assert score_evidence_coverage(claims, ["doc1", "doc2"]) == 1.0


def test_score_answer_relevance_uses_concept_coverage():
    result = score_answer_relevance("The model uses RRF and BM25.", ["RRF", "BM25", "reranking"])
    assert result["coverage_ratio"] == pytest.approx(2 / 3, abs=1e-4)
    assert "reranking" in result["missing"]


def test_score_answer_relevance_no_expected_concepts_is_not_measured():
    # eval_metrics.concept_coverage (reused as-is) uses its own sentinel,
    # "NOT_AVAILABLE" — not this module's NOT_MEASURED; asserting the exact
    # reused value rather than assuming it matches this module's constant.
    result = score_answer_relevance("Some answer.", None)
    assert result["coverage_ratio"] == "NOT_AVAILABLE"


# ---------------------------------------------------------------------------
# evaluate_one() — mocked Groq, real retrieval against isolated fixture data
# ---------------------------------------------------------------------------
class _FakeMessage:
    def __init__(self, content): self.content = content
class _FakeChoice:
    def __init__(self, content): self.message = _FakeMessage(content)
class _FakeUsage:
    prompt_tokens = 80
    completion_tokens = 40
class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


def test_evaluate_one_makes_exactly_one_llm_call_and_scores_real_claims():
    call_count = {"n": 0}

    fake_answer = json.dumps({
        "answer": "The Transformer uses self-attention.", "grounded": True, "evidence_sufficient": True,
        "document_ids": ["bdfaa68d8984f0dc"],
        "structured": {"architecture": "Transformer", "datasets": [], "metrics": [], "contributions": [],
                        "methodology": None, "key_calculations": [], "limitations": [], "final_summary": None},
        "claims": [
            {"claim": "The Transformer uses self-attention.", "support": "DIRECTLY_STATED", "document_ids": ["bdfaa68d8984f0dc"]},
        ],
    })

    class _FakeCompletions:
        def create(self, model, messages, temperature=0.0, max_tokens=None):
            call_count["n"] += 1
            return _FakeResponse(fake_answer)

    class _FakeChat:
        def __init__(self): self.completions = _FakeCompletions()
    class _FakeGroqClient:
        def __init__(self, api_key=None): self.chat = _FakeChat()

    from unittest.mock import patch
    with patch("groq.Groq", _FakeGroqClient):
        # "bdfaa68d8984f0dc" is attention.pdf's real content-hash id,
        # pre-ingested into the isolated Phase 8 test fixture by conftest.py.
        result = evaluate_one(
            "What architecture does this paper use?",
            document_ids=["bdfaa68d8984f0dc"],
            expected_answer_contains=["self-attention"],
        )

    assert call_count["n"] == 1, "evaluate_one() must make exactly ONE LLM call, never a separate verifier call"
    assert result["groundedness_score"] == 1.0
    assert result["unsupported_claim_rate"] == 0.0
    assert result["evidence_coverage"] == 1.0
    assert result["answer_relevance"]["coverage_ratio"] == 1.0


def test_evaluate_one_no_evidence_never_calls_llm():
    call_count = {"n": 0}

    class _FakeCompletions:
        def create(self, *a, **k):
            call_count["n"] += 1
            raise AssertionError("must not be called when there's no evidence")

    class _FakeChat:
        def __init__(self): self.completions = _FakeCompletions()
    class _FakeGroqClient:
        def __init__(self, api_key=None): self.chat = _FakeChat()

    from unittest.mock import patch
    with patch("groq.Groq", _FakeGroqClient):
        result = evaluate_one("Anything", document_ids=["nonexistent_doc_id_entirely"])

    assert call_count["n"] == 0
    assert result["groundedness_score"] == NOT_MEASURED
    assert "error" in result


def test_evaluate_one_survives_a_real_provider_failure_via_synthesizes_own_fallback():
    """synthesize_answer() already has its own internal error handling
    (same function normal structured-mode Q&A uses) — a raw provider
    failure (connection error) never actually propagates out of it, it
    degrades to a graceful "couldn't generate a reliable answer" fallback
    state instead. evaluate_one() must score that fallback honestly (no
    claims means NOT_MEASURED), not crash and not fabricate a score."""
    class _FakeCompletions:
        def create(self, *a, **k):
            raise ConnectionError("simulated: provider unreachable")
    class _FakeChat:
        def __init__(self): self.completions = _FakeCompletions()
    class _FakeGroqClient:
        def __init__(self, api_key=None): self.chat = _FakeChat()

    from unittest.mock import patch
    with patch("groq.Groq", _FakeGroqClient):
        result = evaluate_one(
            "What architecture does this paper use?",
            document_ids=["bdfaa68d8984f0dc"],
        )
    assert result["groundedness_score"] == NOT_MEASURED
    assert result["grounded_self_reported"] is False


def test_evaluate_one_handles_synthesize_answer_itself_raising():
    """Defense-in-depth: even if synthesize_answer() were to raise directly
    (bypassing its own internal handling — e.g. a future regression, or a
    truly unexpected error type its own except clause doesn't cover),
    evaluate_one()'s own try/except must still degrade this ONE question to
    NOT_MEASURED rather than propagate and abort a whole batch run."""
    from unittest.mock import patch

    # evaluate_one() does `from graph.synthesizer import synthesize_answer`
    # as a LOCAL import inside the function, re-resolved on every call — so
    # patching the attribute on its source module (not a module-level name
    # in groundedness_eval itself, which doesn't exist) is what actually
    # takes effect here.
    with patch("graph.synthesizer.synthesize_answer", side_effect=RuntimeError("simulated: totally unexpected failure")):
        result = evaluate_one("What architecture does this paper use?", document_ids=["bdfaa68d8984f0dc"])

    assert result["groundedness_score"] == NOT_MEASURED
    assert "error" in result
    assert "RuntimeError" in result["error"]


def test_evaluate_one_handles_malformed_claim_evidence_trace():
    """claim_evidence_trace present but not a list (e.g. synthesis returned
    something unexpected) must be treated as no claims, not crash."""
    fake_answer = json.dumps({
        "answer": "Some answer.", "grounded": True, "evidence_sufficient": True,
        "document_ids": ["bdfaa68d8984f0dc"],
        "structured": {"architecture": None, "datasets": [], "metrics": [], "contributions": [],
                        "methodology": None, "key_calculations": [], "limitations": [], "final_summary": None},
        "claims": "not_a_list_this_is_malformed",
    })

    class _FakeCompletions:
        def create(self, model, messages, temperature=0.0, max_tokens=None):
            return _FakeResponse(fake_answer)
    class _FakeChat:
        def __init__(self): self.completions = _FakeCompletions()
    class _FakeGroqClient:
        def __init__(self, api_key=None): self.chat = _FakeChat()

    from unittest.mock import patch
    with patch("groq.Groq", _FakeGroqClient):
        result = evaluate_one("What architecture does this paper use?", document_ids=["bdfaa68d8984f0dc"])

    # Must not raise, and must degrade to "no claims" rather than crash.
    assert result["groundedness_score"] == NOT_MEASURED
    assert result["total_claims"] == 0


def test_evaluate_dataset_one_bad_item_does_not_lose_other_results():
    """A single item that raises inside evaluate_one() must not take down
    the whole batch — every other item's real result must still come back."""
    from unittest.mock import patch
    import groundedness_eval as ge

    good_answer = json.dumps({
        "answer": "The Transformer uses self-attention.", "grounded": True, "evidence_sufficient": True,
        "document_ids": ["bdfaa68d8984f0dc"],
        "structured": {"architecture": "Transformer", "datasets": [], "metrics": [], "contributions": [],
                        "methodology": None, "key_calculations": [], "limitations": [], "final_summary": None},
        "claims": [{"claim": "The Transformer uses self-attention.", "support": "DIRECTLY_STATED", "document_ids": ["bdfaa68d8984f0dc"]}],
    })

    call_n = {"n": 0}
    def _fake_evaluate_one(question, document_ids, expected_answer_contains=None):
        call_n["n"] += 1
        if call_n["n"] == 1:
            raise RuntimeError("simulated unexpected failure on the first item")
        return {"question": question, "document_ids": document_ids, "groundedness_score": 1.0}

    with patch.object(ge, "evaluate_one", side_effect=_fake_evaluate_one):
        report = ge.evaluate_dataset([
            {"question": "q1 (will fail)", "document_ids": ["doc_a"]},
            {"question": "q2 (should still succeed)", "document_ids": ["doc_b"]},
        ])

    assert report["dataset_size"] == 2
    assert len(report["per_question_results"]) == 2, "The failing item must not drop the whole batch"
    assert "error" in report["per_question_results"][0]
    assert report["per_question_results"][0]["groundedness_score"] == NOT_MEASURED
    assert report["per_question_results"][1]["groundedness_score"] == 1.0, "The second (healthy) item must still succeed"


def test_this_module_is_never_imported_by_main_or_query_transform():
    """Documents the offline-only boundary: neither the FastAPI app nor the
    hot-path Groq wrapper import this evaluator, so it can never be reached
    by a normal /query request. main.py's eval dashboard MAY reference the
    module BY NAME in a string (e.g. describing where an optional results
    file came from) without importing/calling it — only an actual `import`
    statement would put this module on the request path."""
    import re
    import main
    import query_transform
    with open(main.__file__, encoding="utf-8") as f:
        main_src = f.read()
    with open(query_transform.__file__, encoding="utf-8") as f:
        qt_src = f.read()
    import_pattern = re.compile(r"^\s*(from|import)\s+groundedness_eval\b", re.MULTILINE)
    assert not import_pattern.search(main_src), "main.py must never import groundedness_eval"
    assert not import_pattern.search(qt_src), "query_transform.py must never import groundedness_eval"
