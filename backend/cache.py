"""
cache.py — production cache for normal answers and generated reports.

Backend selection (never touches the retrieval algorithm itself, only
whether a request needs to run it at all):

  - REDIS_URL set AND reachable  -> Redis (production).
  - otherwise                     -> in-memory dict (dev/test fallback,
                                      the original behavior this module
                                      always had).

Any Redis error at any point (connection refused, timeout, mid-session
outage) is caught per-call and treated as a cache miss for that call —
the app always continues correctly via a normal (uncached) generation,
it never raises up into a request handler. See _redis_call().

Cache key = the request content + the active workspace_id + document_ids +
mode/research_type + the retrieval config that could change what evidence
gets selected. Since document_ids are content-hash-derived (see ingest.py),
any real change to a document's content already produces a different
document_id, so including document_ids in the key already captures "the
documents changed"; workspace_id is included too so the key is explicit
about scope even in the (currently impossible, since document_ids are
content hashes) case of two workspaces somehow sharing an id.

Correctness over hit-rate: ANY document upload or removal clears the whole
cache (see clear_all()) rather than trying to surgically invalidate just
the affected entries — the safer, simpler choice for a personal/small-team
scale tool. clear_all() only ever deletes keys under this app's own
"verityrag:cache:" prefix, so it's safe to point at a Redis instance shared
with other tools.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from config import DENSE_TOP_K, BM25_TOP_K, RERANK_TOP_K, MAX_CONTEXT_TOKENS, MAX_CHUNKS_PER_DOC

CACHE_KEY_PREFIX = "verityrag:cache:"
_ANSWER_PREFIX = f"{CACHE_KEY_PREFIX}answer:"
_REPORT_PREFIX = f"{CACHE_KEY_PREFIX}report:"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(24 * 3600)))  # 24h default; 0 = never expire

# Bumped whenever retrieval tuning changes at runtime (not currently
# mutable, but keeps the key honest if that ever changes) so a config change
# can't silently serve stale-config answers.
_RETRIEVAL_CONFIG_FINGERPRINT = (DENSE_TOP_K, BM25_TOP_K, RERANK_TOP_K, MAX_CONTEXT_TOKENS, MAX_CHUNKS_PER_DOC)


# ---------------------------------------------------------------------------
# Observability — hit/miss/error counters, never exposed to the normal chat
# UI, readable via stats() the same way the rest of this module already was.
# ---------------------------------------------------------------------------
_counters = {"hits": 0, "misses": 0, "sets": 0, "redis_errors": 0}


# ---------------------------------------------------------------------------
# Backend: in-memory (dev/test fallback — the module's original behavior)
# ---------------------------------------------------------------------------
class _MemoryBackend:
    name = "memory"

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, float]] = {}  # key -> (json_value, expires_at or 0)

    def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at and time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        expires_at = (time.time() + ttl_seconds) if ttl_seconds else 0
        self._store[key] = (value, expires_at)

    def clear_prefix(self, prefix: str) -> None:
        for k in [k for k in self._store if k.startswith(prefix)]:
            self._store.pop(k, None)

    def count_prefix(self, prefix: str) -> int:
        return sum(1 for k in self._store if k.startswith(prefix))


# ---------------------------------------------------------------------------
# Backend: Redis (production)
# ---------------------------------------------------------------------------
class _RedisBackend:
    name = "redis"

    def __init__(self, client) -> None:
        self._client = client

    def get(self, key: str) -> str | None:
        val = self._client.get(key)
        return val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else val

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        if ttl_seconds:
            self._client.set(key, value, ex=ttl_seconds)
        else:
            self._client.set(key, value)

    def clear_prefix(self, prefix: str) -> None:
        # SCAN, not KEYS — never blocks a shared Redis instance, and only
        # ever touches keys under our own namespace.
        for k in self._client.scan_iter(match=f"{prefix}*"):
            self._client.delete(k)

    def count_prefix(self, prefix: str) -> int:
        return sum(1 for _ in self._client.scan_iter(match=f"{prefix}*"))


def _build_backend():
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        print("[cache] REDIS_URL not set — using in-memory cache (dev/test mode).")
        return _MemoryBackend()
    try:
        import redis as redis_lib
        client = redis_lib.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        print("[cache] Connected to Redis — production cache active.")
        return _RedisBackend(client)
    except Exception as e:
        print(f"[cache] Redis unavailable ({type(e).__name__}: {e}) — falling back to in-memory cache.")
        return _MemoryBackend()


_backend = _build_backend()


def _redis_call(fn, *, on_error):
    """Every backend call goes through here so a Redis outage mid-session
    degrades to a cache miss for that one call (never raises into a request
    handler, never crashes the app) — the in-memory backend's own calls
    never raise in the first place, so this is a no-op wrapper for it."""
    try:
        return fn()
    except Exception as e:
        _counters["redis_errors"] += 1
        print(f"[cache] backend error, treating as cache miss: {type(e).__name__}: {e}")
        return on_error


def _make_key(namespace_prefix: str, *parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{namespace_prefix}{digest}"


def _normalize_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


def backend_name() -> str:
    return _backend.name


# ---------------------------------------------------------------------------
# Normal answers
# ---------------------------------------------------------------------------

def _history_fingerprint(chat_history: list[dict] | None) -> list[tuple]:
    """
    The same question can need a DIFFERENT answer depending on prior
    conversation turns (e.g. "What datasets did they use?" means something
    different depending on which paper "they" resolves to). Without this,
    two different conversations asking the identical raw question text
    against the identical documents would wrongly share a cache entry once
    chat history became a real, live-used feature — this must match the
    same recent-history window the synthesis prompt actually sees.
    """
    return [(m.get("role"), m.get("content")) for m in (chat_history or [])[-4:]]


def answer_cache_key(question: str, document_ids: list[str], research_type: str, chat_history: list[dict] | None = None, workspace_id: str | None = None) -> str:
    return _make_key(
        _ANSWER_PREFIX, _normalize_question(question), sorted(document_ids), research_type,
        _history_fingerprint(chat_history), _RETRIEVAL_CONFIG_FINGERPRINT, workspace_id,
    )


def get_cached_answer(question: str, document_ids: list[str], research_type: str, chat_history: list[dict] | None = None, workspace_id: str | None = None) -> dict | None:
    key = answer_cache_key(question, document_ids, research_type, chat_history, workspace_id)
    raw = _redis_call(lambda: _backend.get(key), on_error=None)
    if raw is None:
        _counters["misses"] += 1
        return None
    _counters["hits"] += 1
    return json.loads(raw)


def set_cached_answer(question: str, document_ids: list[str], research_type: str, payload: dict, chat_history: list[dict] | None = None, workspace_id: str | None = None) -> str:
    key = answer_cache_key(question, document_ids, research_type, chat_history, workspace_id)
    value = {**payload, "_cached_at": time.time()}
    _redis_call(lambda: _backend.set(key, json.dumps(value, default=str), CACHE_TTL_SECONDS), on_error=None)
    _counters["sets"] += 1
    return key


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def report_cache_key(document_ids: list[str], workspace_id: str | None = None) -> str:
    return _make_key(_REPORT_PREFIX, sorted(document_ids), _RETRIEVAL_CONFIG_FINGERPRINT, workspace_id)


def get_cached_report(document_ids: list[str], workspace_id: str | None = None) -> dict | None:
    key = report_cache_key(document_ids, workspace_id)
    raw = _redis_call(lambda: _backend.get(key), on_error=None)
    if raw is None:
        _counters["misses"] += 1
        return None
    _counters["hits"] += 1
    return json.loads(raw)


def set_cached_report(document_ids: list[str], payload: dict, workspace_id: str | None = None) -> str:
    key = report_cache_key(document_ids, workspace_id)
    value = {**payload, "_cached_at": time.time()}
    _redis_call(lambda: _backend.set(key, json.dumps(value, default=str), CACHE_TTL_SECONDS), on_error=None)
    _counters["sets"] += 1
    return key


def get_report_by_id(report_id: str) -> dict | None:
    """Report download endpoints look reports up by their cache key directly."""
    raw = _redis_call(lambda: _backend.get(report_id), on_error=None)
    return json.loads(raw) if raw is not None else None


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------

def clear_all() -> None:
    """Call on any document upload or removal — see module docstring. Only
    ever deletes keys under this app's own CACHE_KEY_PREFIX."""
    _redis_call(lambda: _backend.clear_prefix(CACHE_KEY_PREFIX), on_error=None)


def stats() -> dict:
    """Diagnostic only — never exposed to the normal UI. Includes which
    backend is actually serving cache traffic right now (useful for
    confirming Redis is really in play in production, not silently
    degraded to the in-memory fallback)."""
    answer_entries = _redis_call(lambda: _backend.count_prefix(_ANSWER_PREFIX), on_error=0)
    report_entries = _redis_call(lambda: _backend.count_prefix(_REPORT_PREFIX), on_error=0)
    total_lookups = _counters["hits"] + _counters["misses"]
    return {
        "backend": _backend.name,
        "answer_entries": answer_entries,
        "report_entries": report_entries,
        "entries": answer_entries + report_entries,
        "hits": _counters["hits"],
        "misses": _counters["misses"],
        "sets": _counters["sets"],
        "hit_rate": round(_counters["hits"] / total_lookups, 3) if total_lookups else None,
        "redis_errors": _counters["redis_errors"],
    }
