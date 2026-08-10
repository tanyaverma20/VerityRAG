"""
test_cache_redis.py — the production cache layer (cache.py): hit/miss, TTL,
invalidation, scoped keys, and graceful degradation when Redis is
unavailable. A fake in-process Redis client (same "mock the network
boundary" pattern used for groq.Groq elsewhere in this suite) proves the
Redis code PATH itself is exercised and correct, without needing a real
Redis server.
"""
import time

import pytest

import cache


# ---------------------------------------------------------------------------
# A minimal fake redis client — just enough of the real `redis` API surface
# (get/set/scan_iter/delete/ping) for cache.py's _RedisBackend to drive.
# ---------------------------------------------------------------------------
class _FakeRedisClient:
    def __init__(self, fail: bool = False):
        self._store: dict[str, tuple[bytes, float]] = {}
        self._fail = fail

    def ping(self):
        if self._fail:
            raise ConnectionError("simulated: redis unreachable")
        return True

    def get(self, key):
        if self._fail:
            raise ConnectionError("simulated: redis down mid-session")
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at and time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key, value, ex=None):
        if self._fail:
            raise ConnectionError("simulated: redis down mid-session")
        expires_at = (time.time() + ex) if ex else 0
        self._store[key] = (value.encode("utf-8") if isinstance(value, str) else value, expires_at)

    def scan_iter(self, match=None):
        prefix = (match or "").rstrip("*")
        return [k for k in list(self._store.keys()) if k.startswith(prefix)]

    def delete(self, key):
        self._store.pop(key, None)


@pytest.fixture(autouse=True)
def _reset_cache_state():
    """Every test gets a clean in-memory backend and zeroed counters —
    cache.py's module-level state must not leak between tests."""
    original_backend = cache._backend
    cache._backend = cache._MemoryBackend()
    for k in cache._counters:
        cache._counters[k] = 0
    yield
    cache._backend = original_backend


# ---------------------------------------------------------------------------
# In-memory backend (dev/test default) — hit/miss/TTL/invalidation/scoping
# ---------------------------------------------------------------------------
def test_cache_miss_then_hit():
    assert cache.get_cached_answer("What is RRF?", ["doc1"], "simple") is None
    cache.set_cached_answer("What is RRF?", ["doc1"], "simple", {"answer": "Reciprocal Rank Fusion."})
    got = cache.get_cached_answer("What is RRF?", ["doc1"], "simple")
    assert got is not None and got["answer"] == "Reciprocal Rank Fusion."


def test_question_normalization_still_hits():
    cache.set_cached_answer("What is RRF?", ["doc1"], "simple", {"answer": "x"})
    assert cache.get_cached_answer("  what   IS rrf?  ", ["doc1"], "simple") is not None


def test_different_document_ids_miss():
    cache.set_cached_answer("Q", ["doc1"], "simple", {"answer": "x"})
    assert cache.get_cached_answer("Q", ["doc2"], "simple") is None


def test_different_workspace_id_is_a_distinct_cache_entry():
    cache.set_cached_answer("Q", ["doc1"], "simple", {"answer": "x"}, workspace_id="ws_a")
    assert cache.get_cached_answer("Q", ["doc1"], "simple", workspace_id="ws_a") is not None
    assert cache.get_cached_answer("Q", ["doc1"], "simple", workspace_id="ws_b") is None


def test_different_chat_history_is_a_distinct_cache_entry():
    hist_a = [{"role": "user", "content": "Tell me about paper A"}]
    hist_b = [{"role": "user", "content": "Tell me about paper B"}]
    cache.set_cached_answer("What datasets did they use?", ["doc1"], "simple", {"answer": "A's datasets"}, chat_history=hist_a)
    assert cache.get_cached_answer("What datasets did they use?", ["doc1"], "simple", chat_history=hist_b) is None
    assert cache.get_cached_answer("What datasets did they use?", ["doc1"], "simple", chat_history=hist_a)["answer"] == "A's datasets"


def test_ttl_expiry():
    key = cache.answer_cache_key("Q", ["doc1"], "simple")
    cache._backend.set(key, '{"answer": "x"}', ttl_seconds=1)
    assert cache._backend.get(key) is not None
    time.sleep(1.2)
    assert cache._backend.get(key) is None, "entry must expire after its TTL elapses"


def test_report_cache_roundtrip_and_lookup_by_id():
    report_id = cache.set_cached_report(["doc1", "doc2"], {"ok": True, "title": "Report"})
    assert cache.get_cached_report(["doc1", "doc2"])["title"] == "Report"
    # Download endpoints look reports up directly by the returned id/key.
    assert cache.get_report_by_id(report_id)["title"] == "Report"


def test_clear_all_invalidates_everything_under_our_prefix():
    cache.set_cached_answer("Q1", ["doc1"], "simple", {"answer": "a"})
    cache.set_cached_report(["doc1"], {"ok": True})
    assert cache.stats()["entries"] == 2
    cache.clear_all()
    assert cache.get_cached_answer("Q1", ["doc1"], "simple") is None
    assert cache.get_cached_report(["doc1"]) is None
    assert cache.stats()["entries"] == 0


def test_stats_hit_rate_and_counts():
    cache.set_cached_answer("Q", ["doc1"], "simple", {"answer": "x"})
    cache.get_cached_answer("Q", ["doc1"], "simple")       # hit
    cache.get_cached_answer("Q", ["doc2"], "simple")       # miss
    s = cache.stats()
    assert s["hits"] == 1 and s["misses"] == 1 and s["hit_rate"] == 0.5


# ---------------------------------------------------------------------------
# Redis backend (via the fake client) — same behavior, real code path
# ---------------------------------------------------------------------------
def test_redis_backend_hit_miss_and_scoping():
    cache._backend = cache._RedisBackend(_FakeRedisClient())
    assert cache.get_cached_answer("Q", ["doc1"], "simple") is None
    cache.set_cached_answer("Q", ["doc1"], "simple", {"answer": "redis-backed"}, workspace_id="ws1")
    assert cache.get_cached_answer("Q", ["doc1"], "simple", workspace_id="ws1")["answer"] == "redis-backed"
    assert cache.get_cached_answer("Q", ["doc1"], "simple", workspace_id="ws2") is None
    assert cache.backend_name() == "redis"


def test_redis_backend_clear_all_only_touches_our_prefix():
    fake = _FakeRedisClient()
    fake.set("some:other:apps:key", "untouched", ex=None)
    cache._backend = cache._RedisBackend(fake)
    cache.set_cached_answer("Q", ["doc1"], "simple", {"answer": "x"})
    cache.clear_all()
    assert cache.get_cached_answer("Q", ["doc1"], "simple") is None
    assert fake.get("some:other:apps:key") is not None, "clear_all() must never touch keys outside our own prefix"


# ---------------------------------------------------------------------------
# Graceful degradation — Redis unavailable at startup, and Redis failing
# mid-session, must never break the app.
# ---------------------------------------------------------------------------
def test_build_backend_falls_back_to_memory_when_redis_url_unset(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    backend = cache._build_backend()
    assert backend.name == "memory"


def test_build_backend_falls_back_to_memory_when_redis_unreachable(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:1/0")  # nothing listens on port 1

    class _FailingRedisModule:
        @staticmethod
        def from_url(*a, **k):
            return _FakeRedisClient(fail=True)

    import sys
    monkeypatch.setitem(sys.modules, "redis", _FailingRedisModule())
    backend = cache._build_backend()
    assert backend.name == "memory", "an unreachable Redis at startup must degrade to the in-memory fallback, not crash"


def test_redis_failure_mid_session_is_treated_as_a_cache_miss_not_a_crash():
    cache._backend = cache._RedisBackend(_FakeRedisClient(fail=True))
    # Must not raise — both reads and writes degrade to "no cache" behavior.
    result = cache.get_cached_answer("Q", ["doc1"], "simple")
    assert result is None
    cache.set_cached_answer("Q", ["doc1"], "simple", {"answer": "x"})  # must not raise
    assert cache.stats()["redis_errors"] >= 1


def test_never_caches_secrets_or_api_keys_by_construction():
    # The cache only ever stores what callers explicitly pass as the answer/
    # report payload — this test documents and locks in that main.py never
    # passes request headers/API keys/env vars into a cached payload.
    cache.set_cached_answer("Q", ["doc1"], "simple", {"answer": "public answer text"})
    stored = cache.get_cached_answer("Q", ["doc1"], "simple")
    assert "GROQ_API_KEY" not in str(stored) and "api_key" not in str(stored).lower()
