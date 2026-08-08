"""
cache.py — In-memory cache for normal answers and generated reports (item 13).

Deliberately simple: this is a single-process local app, not a distributed
service, so a process-lifetime in-memory dict is the right amount of
engineering — no Redis, no DB table, no TTL machinery.

Cache key = the request content + the active document_ids + the retrieval
config that could change what evidence gets selected. Since document_ids are
content-hash-derived (see ingest.py), any real change to a document's
content already produces a different document_id, so including document_ids
in the key already captures "the documents changed."

Correctness over hit-rate: ANY document upload or removal clears the whole
cache rather than trying to surgically invalidate just the affected entries.
For a personal-scale research tool this is the safer, simpler choice — it
guarantees "never return cached results after the relevant uploaded
documents have changed" without a reverse-index to get subtly wrong.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from config import DENSE_TOP_K, BM25_TOP_K, RERANK_TOP_K, MAX_CONTEXT_TOKENS, MAX_CHUNKS_PER_DOC

_answer_cache: dict[str, dict[str, Any]] = {}
_report_cache: dict[str, dict[str, Any]] = {}

# Bumped whenever retrieval tuning changes at runtime (not currently
# mutable, but keeps the key honest if that ever changes) so a config change
# can't silently serve stale-config answers.
_RETRIEVAL_CONFIG_FINGERPRINT = (DENSE_TOP_K, BM25_TOP_K, RERANK_TOP_K, MAX_CONTEXT_TOKENS, MAX_CHUNKS_PER_DOC)


def _make_key(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _normalize_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


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


def answer_cache_key(question: str, document_ids: list[str], research_type: str, chat_history: list[dict] | None = None) -> str:
    return _make_key(
        "answer", _normalize_question(question), sorted(document_ids), research_type,
        _history_fingerprint(chat_history), _RETRIEVAL_CONFIG_FINGERPRINT,
    )


def get_cached_answer(question: str, document_ids: list[str], research_type: str, chat_history: list[dict] | None = None) -> dict | None:
    return _answer_cache.get(answer_cache_key(question, document_ids, research_type, chat_history))


def set_cached_answer(question: str, document_ids: list[str], research_type: str, payload: dict, chat_history: list[dict] | None = None) -> str:
    key = answer_cache_key(question, document_ids, research_type, chat_history)
    _answer_cache[key] = {**payload, "_cached_at": time.time()}
    return key


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def report_cache_key(document_ids: list[str]) -> str:
    return _make_key("report", sorted(document_ids), _RETRIEVAL_CONFIG_FINGERPRINT)


def get_cached_report(document_ids: list[str]) -> dict | None:
    return _report_cache.get(report_cache_key(document_ids))


def set_cached_report(document_ids: list[str], payload: dict) -> str:
    key = report_cache_key(document_ids)
    _report_cache[key] = {**payload, "_cached_at": time.time()}
    return key


def get_report_by_id(report_id: str) -> dict | None:
    """Report download endpoints look reports up by their cache key directly."""
    return _report_cache.get(report_id)


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------

def clear_all() -> None:
    """Call on any document upload or removal — see module docstring."""
    _answer_cache.clear()
    _report_cache.clear()


def stats() -> dict:
    """Diagnostic only — never exposed to the normal UI."""
    return {"answer_entries": len(_answer_cache), "report_entries": len(_report_cache)}
