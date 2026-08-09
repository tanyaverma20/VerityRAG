"""
test_observability_and_budget.py — the remaining items from this turn's
"15 differentiators" testing list not already covered elsewhere:

  N. Context token budget is actually enforced (not just configured).
  O. Cross-encoder reranking makes ZERO LLM calls.
  P. Production Chroma is untouched (sanity re-check alongside the above).

Plus coverage for this turn's new observability fields (request_id,
active_document_count, cache_hit, input/output/total tokens) — item 15.

All mocked/deterministic where an LLM would otherwise be involved; the
reranker itself is a local cross-encoder model, not a Groq call, so no
mocking is even needed for O — this test proves that directly.
"""
import json
from unittest.mock import patch

import pytest

import config
from query_transform import start_call_tracking, get_call_log, get_token_totals


# ---------------------------------------------------------------------------
# TEST N — context token budget is actually enforced
# ---------------------------------------------------------------------------
def test_context_token_budget_is_enforced():
    from retrieval import select_within_token_budget, _estimate_tokens

    # Each chunk is ~400 chars => ~100 estimated tokens. 10 chunks would be
    # ~1000 tokens; cap the budget well below that and confirm the selection
    # actually stops early rather than including everything regardless.
    chunks = [
        {
            "text": ("Evidence chunk number %d. " % i) * 20,  # long enough to matter
            "metadata": {"document_id": f"doc_{i % 3}", "chunk_id": f"c{i}"},
        }
        for i in range(10)
    ]
    budget = 250  # deliberately small
    selected = select_within_token_budget(chunks, max_tokens=budget, max_per_doc=10)

    total_tokens = sum(_estimate_tokens(c["text"]) for c in selected)
    assert total_tokens <= budget, f"Selected evidence ({total_tokens} tokens) exceeded the configured budget ({budget})"
    assert len(selected) < len(chunks), "A real budget must actually exclude some candidates, not pass everything through"


def test_context_token_budget_respects_per_document_diversity_cap():
    from retrieval import select_within_token_budget

    # 5 chunks all from the SAME document — max_per_doc=2 must cap it to 2
    # even though the token budget alone would allow more.
    chunks = [
        {"text": "short evidence", "metadata": {"document_id": "docA", "chunk_id": f"c{i}"}}
        for i in range(5)
    ]
    selected = select_within_token_budget(chunks, max_tokens=100_000, max_per_doc=2)
    assert len(selected) == 2, "MAX_CHUNKS_PER_DOC must cap per-document representation regardless of remaining token budget"


def test_max_answer_tokens_is_configurable():
    """Item 6: MAX_ANSWER_TOKENS must be a real, overridable config value,
    not a hardcoded magic number buried in the call site."""
    assert hasattr(config, "MAX_ANSWER_TOKENS")
    assert isinstance(config.MAX_ANSWER_TOKENS, int) and config.MAX_ANSWER_TOKENS > 0


# ---------------------------------------------------------------------------
# TEST O — cross-encoder reranking makes ZERO LLM calls
# ---------------------------------------------------------------------------
def test_reranking_makes_zero_llm_calls():
    from retrieval import rerank

    start_call_tracking()
    candidates = [
        {"text": "The Transformer uses self-attention.", "metadata": {"document_id": "docA", "chunk_id": "c1"}},
        {"text": "BERT uses masked language modeling.", "metadata": {"document_id": "docB", "chunk_id": "c2"}},
        {"text": "ResNet uses residual connections.", "metadata": {"document_id": "docC", "chunk_id": "c3"}},
    ]
    results = rerank("What architecture uses attention?", candidates, top_k=2)

    assert len(get_call_log()) == 0, "The cross-encoder reranker must never make a Groq/LLM API call — it's a local model"
    assert len(results) == 2
    assert all("rerank_score" in r for r in results)
    # The query is about attention — the Transformer chunk should score highest.
    assert results[0]["metadata"]["document_id"] == "docA"


# ---------------------------------------------------------------------------
# Observability (item 15): request_id, active_document_count, cache_hit,
# input/output/total tokens actually populate the logged event.
# ---------------------------------------------------------------------------
class _FakeMessage:
    def __init__(self, content):
        self.content = content

class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)

class _FakeUsage:
    prompt_tokens = 123
    completion_tokens = 45

class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()

class _FakeCompletions:
    def create(self, model, messages, temperature=0.0, max_tokens=None):
        return _FakeResponse(json.dumps({
            "answer": "Grounded answer.", "grounded": True, "evidence_sufficient": True, "document_ids": ["docA"],
        }))

class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()

class _FakeGroqClient:
    def __init__(self, api_key=None):
        self.chat = _FakeChat()


def test_token_totals_reflect_real_usage():
    from graph.organizer import organize_evidence
    from graph.synthesizer import synthesize_answer

    start_call_tracking()
    state = {
        "original_query": "test",
        "research_type": "simple",
        "retrieval_results": [{
            "text": "evidence", "parent_context": "evidence in context",
            "metadata": {"document_id": "docA", "parent_id": "p1", "chunk_id": "c1", "source": "a.pdf"},
        }],
    }
    state.update(organize_evidence(state))
    with patch("groq.Groq", _FakeGroqClient):
        synthesize_answer(state)

    totals = get_token_totals()
    assert totals["input_tokens"] == 123
    assert totals["output_tokens"] == 45
    assert totals["total_tokens"] == 168


def test_query_endpoint_logs_observability_extras(tmp_path, monkeypatch):
    """Confirms /query's logged event actually carries request_id,
    active_document_count, and cache_hit — not just that the fields exist
    in the function signature."""
    from fastapi.testclient import TestClient
    import main as main_module
    import observability

    # Redirect the event log to a scratch file so we don't touch the real one.
    fake_log = tmp_path / "events.jsonl"
    monkeypatch.setattr(observability, "_LOG_FILE", fake_log)

    client = TestClient(main_module.app)
    resp = client.post("/query", json={"question": "test"})  # no document_ids => NO_DOCUMENTS short-circuit, 0 LLM calls
    assert resp.status_code == 200
    # The NO_DOCUMENTS short-circuit in /query returns before any log_query_event
    # call today — this is expected and fine (nothing to log for a no-op).
    # What matters here is that a SCOPED request does produce the fields; we
    # verify that shape directly via the cache-hit path instead, which is
    # deterministic and doesn't require a real LLM call.
    import cache
    cache.clear_all()
    cache.set_cached_answer("cached question", ["docA"], "simple", {"answer": "Cached.", "confidence": "HIGH"})
    resp2 = client.post("/query", json={"question": "cached question", "document_ids": ["docA"]})
    assert resp2.status_code == 200
    assert resp2.json()["answer"] == "Cached."

    assert fake_log.exists()
    lines = fake_log.read_text(encoding="utf-8").strip().splitlines()
    last_event = json.loads(lines[-1])
    assert last_event["task_status"] == "CACHE_HIT"
    assert last_event["llm_calls"] == 0
    assert last_event["extra"]["cache_hit"] is True
    assert last_event["extra"]["active_document_count"] == 1
    assert last_event["extra"]["request_id"]  # non-empty
    cache.clear_all()


# ---------------------------------------------------------------------------
# TEST P — production Chroma untouched by anything in this file
# ---------------------------------------------------------------------------
def test_production_chroma_still_untouched():
    from pathlib import Path
    canonical_path = (Path(__file__).parent / "chroma_store").resolve()
    assert Path(config.CHROMA_DIR).resolve() != canonical_path
    assert config.COLLECTION_NAME == "test_collection"
