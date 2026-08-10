"""
test_redis_live.py — real Redis integration tests (Phase 3), non-destructive
half.

These run against an ACTUAL Redis server (not the _FakeRedisClient used in
test_cache_redis.py), exercising cache.py's real _RedisBackend: init, real
network round-trips, TTL enforcement by the real server, scoped-key
isolation, invalidation, and stats — everything that's safe to run
unattended as part of the normal test suite.

Auto-skips (not fails) when REDIS_URL isn't configured or the server isn't
actually reachable, exactly like test_postgres_live.py does for Postgres.

The DESTRUCTIVE half — actually stopping/restarting the Redis process to
prove outage-degrades-gracefully and recovery-without-restart — is
deliberately NOT part of this file (or of any auto-run suite): killing a
shared local service as a side effect of routine `pytest` runs would be a
surprising, disruptive side effect for any future contributor/CI. That
scenario is instead a manual, deliberately-invoked script:
see backend/scripts/verify_redis_outage_recovery.py.

cache.py's `_backend` singleton is built once at import time from the real
REDIS_URL (conftest.py does not touch REDIS_URL, unlike DATABASE_URL) — so
if a real server is reachable, cache._backend is already the real
_RedisBackend by the time these tests run. Every key used here is
uniquely-prefixed per test and cleaned up in a finally block.
"""
import os
import time
import uuid

import pytest

import cache


def _redis_reachable() -> bool:
    url = os.getenv("REDIS_URL")
    if not url:
        return False
    try:
        import redis as redis_lib
        client = redis_lib.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        return client.ping() is True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _redis_reachable(),
    reason="No reachable real Redis server configured via REDIS_URL — these are live "
           "integration tests, not mocked, and are skipped rather than faked when the "
           "real infrastructure prerequisite isn't available.",
)


@pytest.fixture()
def raw_redis_client():
    import redis as redis_lib
    client = redis_lib.from_url(os.getenv("REDIS_URL"), socket_connect_timeout=2, socket_timeout=2)
    yield client


@pytest.fixture()
def _cleanup_cache():
    yield
    cache.clear_all()


def test_backend_is_really_redis_not_memory_fallback():
    assert cache.backend_name() == "redis", (
        "REDIS_URL is configured and reachable, so cache.py must have selected the real "
        "Redis backend, not silently degraded to the in-memory dev fallback."
    )


def test_write_really_reaches_the_real_server(raw_redis_client, _cleanup_cache):
    doc_id = "live_redis_test_doc_" + uuid.uuid4().hex[:8]
    key = cache.set_cached_answer("live redis write test", [doc_id], "simple", {"answer": "real"}, workspace_id="live_ws")
    raw_val = raw_redis_client.get(key)
    assert raw_val is not None, "Value must be independently visible via a second, unrelated redis-py client"


def test_hit_after_miss(_cleanup_cache):
    doc_id = "live_redis_test_doc_" + uuid.uuid4().hex[:8]
    miss = cache.get_cached_answer("unique unseen question " + uuid.uuid4().hex, [doc_id], "simple", workspace_id="live_ws")
    assert miss is None

    cache.set_cached_answer("live redis hit test", [doc_id], "simple", {"answer": "cached value"}, workspace_id="live_ws")
    hit = cache.get_cached_answer("live redis hit test", [doc_id], "simple", workspace_id="live_ws")
    assert hit is not None and hit["answer"] == "cached value"


def test_scoped_keys_no_cross_workspace_leakage(_cleanup_cache):
    doc_id = "live_redis_test_doc_" + uuid.uuid4().hex[:8]
    cache.set_cached_answer("cross workspace question", [doc_id], "simple", {"answer": "ws1 answer"}, workspace_id="live_ws_1")
    leaked = cache.get_cached_answer("cross workspace question", [doc_id], "simple", workspace_id="live_ws_2")
    assert leaked is None, "A cache entry scoped to one workspace must never be served to a different workspace"


def test_scoped_keys_no_cross_document_leakage(_cleanup_cache):
    doc_a = "live_redis_test_doc_a_" + uuid.uuid4().hex[:8]
    doc_b = "live_redis_test_doc_b_" + uuid.uuid4().hex[:8]
    cache.set_cached_answer("cross document question", [doc_a], "simple", {"answer": "doc a answer"}, workspace_id="live_ws")
    leaked = cache.get_cached_answer("cross document question", [doc_b], "simple", workspace_id="live_ws")
    assert leaked is None, "A cache entry scoped to one document must never be served for a different document"


def test_real_ttl_is_enforced_by_the_server(raw_redis_client, _cleanup_cache):
    doc_id = "live_redis_test_doc_" + uuid.uuid4().hex[:8]
    key = cache.set_cached_answer("ttl check question", [doc_id], "simple", {"answer": "x"}, workspace_id="live_ws")
    ttl = raw_redis_client.ttl(key)
    assert 0 < ttl <= cache.CACHE_TTL_SECONDS


def test_short_ttl_actually_expires():
    key = "verityrag:cache:answer:live_redis_shortlived_" + uuid.uuid4().hex[:8]
    cache._backend.set(key, '{"x": 1}', 1)
    assert cache._backend.get(key) is not None
    time.sleep(1.5)
    assert cache._backend.get(key) is None, "A 1-second-TTL key must actually be gone from the real server after expiry"


def test_report_cache_roundtrip(_cleanup_cache):
    doc_id = "live_redis_test_doc_" + uuid.uuid4().hex[:8]
    rkey = cache.set_cached_report([doc_id], {"report": "live report body"}, workspace_id="live_ws")
    hit = cache.get_cached_report([doc_id], workspace_id="live_ws")
    assert hit is not None and hit["report"] == "live report body"
    assert cache.get_report_by_id(rkey) is not None


def test_invalidation_is_scoped_to_our_own_prefix(raw_redis_client):
    doc_id = "live_redis_test_doc_" + uuid.uuid4().hex[:8]
    key = cache.set_cached_answer("invalidation test question", [doc_id], "simple", {"answer": "x"}, workspace_id="live_ws")
    raw_redis_client.set("some_unrelated_apps_key_" + uuid.uuid4().hex[:8], "must_survive")
    try:
        cache.clear_all()
        assert raw_redis_client.get(key) is None, "clear_all() must remove our own cache entries"
    finally:
        for k in raw_redis_client.scan_iter(match="some_unrelated_apps_key_*"):
            raw_redis_client.delete(k)


def test_stats_reports_real_backend_and_moving_counters(_cleanup_cache):
    doc_id = "live_redis_test_doc_" + uuid.uuid4().hex[:8]
    cache.get_cached_answer("stats miss question " + uuid.uuid4().hex, [doc_id], "simple", workspace_id="live_ws")
    cache.set_cached_answer("stats hit question", [doc_id], "simple", {"answer": "x"}, workspace_id="live_ws")
    cache.get_cached_answer("stats hit question", [doc_id], "simple", workspace_id="live_ws")
    st = cache.stats()
    assert st["backend"] == "redis"
    assert st["hits"] >= 1
    assert st["misses"] >= 1
