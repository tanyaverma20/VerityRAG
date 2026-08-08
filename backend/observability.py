"""
observability.py — Phase 4 LLMOps Structured Observability

Provides a lightweight, file-backed structured event logger.
Each query event is appended as a JSON line to logs/verityrag_events.jsonl.

NEVER logs:
  - API keys
  - secrets
  - credentials
  - full document bodies unnecessarily

Usage:
    from observability import log_query_event
    log_query_event(
        query="attention mechanism",
        question_type="factual",
        retrieval_strategy="hybrid",
        ...
    )
"""

from __future__ import annotations
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Log directory — placed next to backend/
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "verityrag_events.jsonl"


def _ensure_log_dir() -> None:
    _LOG_DIR.mkdir(exist_ok=True)


def log_query_event(
    *,
    query: str,
    question_type: str = "unknown",
    retrieval_strategy: str = "hybrid",
    documents_searched: int = 0,
    documents_contributing: int = 0,
    retrieved_chunk_count: int = 0,
    selected_evidence_count: int = 0,
    estimated_context_tokens: int = 0,
    synthesis_attempted: bool = False,
    verification_attempted: bool = False,
    verification_statuses: list[str] | None = None,
    retry_count: int = 0,
    retrieval_latency_s: float | None = None,
    synthesis_latency_s: float | None = None,
    verification_latency_s: float | None = None,
    total_latency_s: float | None = None,
    fallback_triggered: bool = False,
    errors: list[str] | None = None,
    model_name: str = "",
    prompt_version: str = "v1",
    # --- Phase 6 Adaptive Fields ---
    research_type: str = "simple",
    research_iterations: int = 1,
    adaptive_queries: int = 0,
    evidence_gaps: int = 0,
    contradiction_count: int = 0,
    research_gap_count: int = 0,
    confidence: str = "UNAVAILABLE",
    # -------------------------------
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Append one structured JSON event to the JSONL log file.
    Returns the event dict for inspection/testing.

    The query string is stored but intentionally truncated at 500 chars to
    avoid accidental PII logging of very long prompts.
    """
    _ensure_log_dir()

    event: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query_preview": query[:500],
        "question_type": question_type,
        "retrieval_strategy": retrieval_strategy,
        "documents_searched": documents_searched,
        "documents_contributing": documents_contributing,
        "retrieved_chunk_count": retrieved_chunk_count,
        "selected_evidence_count": selected_evidence_count,
        "estimated_context_tokens": estimated_context_tokens,
        "synthesis_attempted": synthesis_attempted,
        "verification_attempted": verification_attempted,
        "verification_statuses": verification_statuses or [],
        "retry_count": retry_count,
        "retrieval_latency_s": retrieval_latency_s,
        "synthesis_latency_s": synthesis_latency_s,
        "verification_latency_s": verification_latency_s,
        "total_latency_s": total_latency_s,
        "fallback_triggered": fallback_triggered,
        "errors": errors or [],
        "model_name": model_name,
        "prompt_version": prompt_version,
        "research_type": research_type,
        "research_iterations": research_iterations,
        "adaptive_queries": adaptive_queries,
        "evidence_gaps": evidence_gaps,
        "contradiction_count": contradiction_count,
        "research_gap_count": research_gap_count,
        "confidence": confidence,
    }
    if extra:
        event["extra"] = extra

    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        # Never crash the main pipeline because observability failed
        pass

    return event


def read_recent_events(n: int = 50) -> list[dict[str, Any]]:
    """
    Read the N most recent logged events. Returns [] if log is empty.
    """
    _ensure_log_dir()
    if not _LOG_FILE.exists():
        return []
    try:
        lines = _LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
        recent = lines[-n:]
        return [json.loads(line) for line in recent if line.strip()]
    except Exception:
        return []


def get_log_path() -> Path:
    return _LOG_FILE
