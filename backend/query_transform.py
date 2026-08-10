"""
query_transform.py — Lightweight, LLM-optional query transformation.

Two operations are provided:
  1. rewrite_query()     — make a vague query retrieval-friendly
  2. decompose_query()   — split a complex multi-aspect question into sub-queries

Both functions are OPTIONAL helpers.  Retrieval never depends on successful
transformation; every function falls back to the original query on any error.

Design constraints (Phase 2):
  - No LangGraph, no agents, no planning loops.
  - LLM calls are only made when GROQ_API_KEY is present AND the query
    appears to need transformation.
  - All callers receive usable queries even if the LLM is unavailable.
"""

import json
import re
import contextvars
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_FALLBACK_MODEL, MAX_ANSWER_TOKENS

# ---------------------------------------------------------------------------
# LLM call observability (item 17) — every physical Groq HTTP request made
# through with_model_fallback() is recorded here, per-request. Uses a
# contextvar (not a plain module global) so it's safe under FastAPI's
# threadpool-per-request execution without threading a tracker object through
# every graph node's function signature.
#
# This is diagnostic-only: it never influences what the app does, only what
# gets reported afterward. Call start_call_tracking() once per request.
# ---------------------------------------------------------------------------

_call_log: contextvars.ContextVar[list | None] = contextvars.ContextVar("_call_log", default=None)


def start_call_tracking() -> None:
    """Call once at the top of a request to begin recording LLM calls made during it."""
    _call_log.set([])


def get_call_log() -> list[dict]:
    """Returns the calls recorded since the last start_call_tracking() in this request."""
    log = _call_log.get()
    return list(log) if log is not None else []


def get_token_totals() -> dict:
    """Aggregate input/output/total tokens across every physical call this
    request made so far — item 15's input_tokens/output_tokens/total_tokens."""
    log = get_call_log()
    prompt = sum(c.get("prompt_tokens", 0) for c in log)
    completion = sum(c.get("completion_tokens", 0) for c in log)
    return {"input_tokens": prompt, "output_tokens": completion, "total_tokens": prompt + completion}


def _record_call(model: str, role: str, success: bool, usage: dict | None = None) -> None:
    log = _call_log.get()
    if log is not None:
        usage = usage or {}
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        log.append({
            "model": model, "role": role, "success": success,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })

# ---------------------------------------------------------------------------
# Heuristics — avoid LLM calls for simple queries
# ---------------------------------------------------------------------------

_COMPARISON_SIGNALS = [
    "compare", "comparison", "versus", "vs", "difference between",
    "how do", "contrast", "both", "all papers", "across papers",
    "each paper", "different papers",
]

_VAGUE_SIGNALS = [
    "how did they", "what did they", "explain it", "tell me about",
    "what is it", "can you", "describe",
]

# ---------------------------------------------------------------------------
# Shared user-facing copy — one place so main.py / synthesizer.py / verifier.py
# never drift into showing raw exceptions, "VERIFICATION_UNAVAILABLE", or
# stale "Fallback Answer: ..." strings to end users.
# ---------------------------------------------------------------------------

NO_DOCUMENTS_MESSAGE = (
    "Upload at least one paper to get started — I can only answer questions "
    "about documents in your current workspace."
)
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find enough information in the uploaded papers to answer this reliably."
)
RATE_LIMIT_MESSAGE = (
    "The AI reasoning service is temporarily rate-limited. Your documents are "
    "safe — please try again in a bit."
)
GENERATION_FAILED_MESSAGE = (
    "I couldn't generate a reliable answer right now. Please try again shortly."
)


def _looks_vague(query: str) -> bool:
    q = query.lower().strip()
    return any(s in q for s in _VAGUE_SIGNALS) and len(q.split()) < 8


def _looks_complex(query: str) -> bool:
    q = query.lower().strip()
    return any(s in q for s in _COMPARISON_SIGNALS)


# ---------------------------------------------------------------------------
# Deterministic (zero-LLM-cost) query decomposition
#
# Used by graph/planner.py in NORMAL mode so a genuinely multi-aspect
# question (e.g. "Compare methodologies, datasets and results") retrieves
# evidence for each aspect independently, without ever calling the LLM to
# decide how to split it. This is intentionally NOT the same thing as
# decompose_query() above (which is LLM-backed and Deep-Research-only) —
# normal mode must stay at zero planning LLM calls.
# ---------------------------------------------------------------------------

_ASPECT_NOUNS = {
    "methodology", "methodologies", "method", "methods",
    "dataset", "datasets", "data",
    "result", "results", "finding", "findings",
    "metric", "metrics", "performance",
    "architecture", "architectures", "model", "models",
    "limitation", "limitations",
    "contribution", "contributions",
    "approach", "approaches", "technique", "techniques",
    "evaluation", "evaluations", "baseline", "baselines",
    "conclusion", "conclusions", "experiment", "experiments",
}

# Splits a comma/"and"-joined list into its items, e.g.
# "methodologies, datasets and results" -> ["methodologies", "datasets", "results"]
_LIST_ITEM_SPLIT_RE = re.compile(r',\s*(?:and\s+)?|\s+and\s+')

# Bounds sub-query count so decomposition can never blow up retrieval/rerank/
# token cost — a handful of aspects is the realistic ceiling for one question.
MAX_DECOMPOSED_SUB_QUERIES = 4


def decompose_query_deterministic(query: str) -> list[str]:
    """
    Rule-based, zero-LLM-cost decomposition for questions with multiple
    distinct information needs. Never splits arbitrarily by word count or
    fixed length — only triggers when BOTH are true:
      1. The question contains a comparison/aggregation signal (_looks_complex).
      2. The question contains an explicit comma/"and"-joined list of 2+
         recognised research-paper aspect nouns (methodology, datasets,
         results, architecture, ...).

    "Compare methodologies, datasets and results" ->
        ["methodologies: Compare methodologies, datasets and results",
         "datasets: Compare methodologies, datasets and results",
         "results: Compare methodologies, datasets and results"]

    Every other question — including ones that merely say "compare" without
    an aspect list (e.g. "Compare the two papers", "What's the difference
    between BERT and GPT?") — returns [query] unchanged, so retrieval stays
    a single search, exactly as required for simple questions.
    """
    q = query.strip()
    if not q:
        return [query]

    if not _looks_complex(q):
        return [q]

    lowered = q.lower().rstrip("?.! ")
    tokens = [t.strip() for t in _LIST_ITEM_SPLIT_RE.split(lowered) if t.strip()]

    aspects: list[str] = []
    for tok in tokens:
        for word in tok.split():
            if word in _ASPECT_NOUNS and word not in aspects:
                aspects.append(word)
                break  # one aspect label per list item is enough

    if len(aspects) < 2:
        return [q]

    aspects = aspects[:MAX_DECOMPOSED_SUB_QUERIES]
    # Keep the full original question as context in every sub-query (paper
    # names, "compare" framing, etc. all still matter for retrieval) while
    # anchoring each one on a distinct aspect so dense/BM25 actually surface
    # different top chunks per sub-query.
    return [f"{aspect}: {q}" for aspect in aspects]


# ---------------------------------------------------------------------------
# LLM-backed transformation (Groq, same client as llm.py uses)
# ---------------------------------------------------------------------------

def _call_groq_raw(prompt: str, max_tokens: int = MAX_ANSWER_TOKENS) -> str:
    """
    Direct Groq call that returns raw text; raises on any error.

    Tries GROQ_MODEL first. On a temporary failure only, retries once against
    GROQ_FALLBACK_MODEL (see `with_model_fallback`) — transparent to callers,
    who don't need to know which model actually answered.

    max_tokens defaults to the normal-answer budget; report generation
    passes a larger MAX_REPORT_ANSWER_TOKENS since one report call covers
    much more ground than one Q&A answer.
    """
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    def _attempt(model: str):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        usage_dict = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens} if usage else None
        return text, usage_dict

    return with_model_fallback(_attempt)


def _call_groq_vision_raw(prompt: str, image_base64: str, vision_model: str, max_tokens: int = MAX_ANSWER_TOKENS) -> str | None:
    """
    Multimodal Groq call: one text prompt + one base64 PNG image, sent to
    an explicitly-configured vision-capable model (figure_vision.py:
    VISION_MODEL — never guessed, never defaulted to a text model). Used
    ONLY by Explain Figure's optional visual path (analysis.py), and only
    when figure_vision.vision_model_available() is True.

    No fallback-model retry here (unlike _call_groq_raw): there is no
    second vision-capable model configured to fall back to, so a failure
    here — including "no vision model configured at all" — simply means
    the caller falls back to the existing text/caption-based explanation.
    Returns None on ANY failure rather than raising, so that fallback is
    always a clean, silent, honest path rather than an exception the
    caller has to specifically catch.
    """
    if not vision_model or not image_base64:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as e:
        print(f"[figure_vision] vision model call failed, falling back to text-based explanation: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

REWRITE_PROMPT = """You are a search-query optimizer for an academic research paper RAG system.

Given a vague user question, rewrite it as a concise, keyword-rich retrieval query.

Rules:
- Return ONLY the improved query string, nothing else.
- Keep it under 30 words.
- Do not add information not implied by the original question.
- If the original query is already good, return it unchanged.

Original question: {question}
Retrieval query:"""

DECOMPOSE_PROMPT = """You are a research question decomposer for a multi-paper RAG system.

Given a complex research question, decompose it into 2–6 simpler sub-questions
that can each be answered independently by searching individual research papers.

Return ONLY a JSON array of strings. Example:
["What architecture does Paper A use?", "What datasets are used?"]

Complex question: {question}
Sub-questions JSON:"""


def rewrite_query(query: str) -> str:
    """
    Return a retrieval-friendly version of *query*.

    Uses the LLM only when:
      - GROQ_API_KEY is set, AND
      - the query looks vague (heuristic).

    Always returns a non-empty string.  Falls back to original on any error.
    """
    query = query.strip()
    if not query:
        return query

    # Skip LLM for queries that are already specific enough
    if not _looks_vague(query) or not GROQ_API_KEY:
        return query

    try:
        rewritten = _call_groq_raw(REWRITE_PROMPT.format(question=query))
        # Sanity: reject empty or implausibly long responses
        if rewritten and len(rewritten) < 300:
            return rewritten
    except Exception:
        pass

    return query  # fallback


def is_rate_limit_error(exc: Exception) -> bool:
    """
    Best-effort classification of a Groq call failure.

    Distinguishes "temporarily rate-limited" from "genuinely no evidence" so the
    UI never mislabels a 429 as an empty result. String-matched rather than
    tied to a specific groq-sdk exception class so it survives SDK upgrades.
    """
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


def _is_temporary_failure(exc: Exception) -> bool:
    """
    True only for failures where retrying against a *different* model is
    likely to help: rate limits, 5xx/timeouts, or the primary model itself
    being unavailable/decommissioned.

    False for anything a fallback model can't fix — bad API key, malformed
    request, or other client-side (4xx) errors — so those fail fast instead
    of wasting a second call.

    Model-availability errors (e.g. Groq's "model_decommissioned") come back
    as HTTP 400 with type "invalid_request_error" — the same shape as a
    genuinely malformed request — so the message content must be checked
    BEFORE any blanket "4xx = don't retry" rule, not after.
    """
    if is_rate_limit_error(exc):
        return True

    msg = str(exc).lower()
    if any(s in msg for s in (
        "decommissioned", "model_not_found", "model not found",
        "does not exist", "has been deprecated", "no longer supported",
    )):
        return True  # different model needed — a fallback model can fix this

    if any(s in msg for s in ("invalid api key", "unauthorized", "authentication")):
        return False

    status = getattr(exc, "status_code", None)
    if status in (500, 502, 503, 504, 408):
        return True
    if status in (401, 403, 400, 422):
        return False

    if any(s in msg for s in (
        "timeout", "timed out", "connection", "temporarily unavailable",
        "service unavailable", "internal server error", "502", "503", "504",
    )):
        return True
    return False  # unknown error shape — safer to fail clean than to guess


def with_model_fallback(make_request):
    """
    Calls `make_request(model_name)` with GROQ_MODEL first.

    `make_request` must return `(result, usage)` where `usage` is either
    `None` or a dict with `prompt_tokens`/`completion_tokens` — this is how
    _record_call() gets real token counts for observability (item 15)
    without with_model_fallback needing to know anything about the Groq SDK
    response shape itself.

    On a *temporary* failure only (see `_is_temporary_failure`), retries
    exactly once against GROQ_FALLBACK_MODEL. Never retries beyond that, and
    never falls back for auth/validation errors a different model can't fix.

    If both attempts fail, re-raises the ORIGINAL primary-model exception so
    existing callers' rate-limit classification (`is_rate_limit_error`)
    keeps working unchanged.

    Records every physical attempt via _record_call() for observability —
    this is where the real, external call count is measured, not assumed.
    """
    try:
        result, usage = make_request(GROQ_MODEL)
        _record_call(GROQ_MODEL, "primary", True, usage)
        return result
    except Exception as primary_error:
        _record_call(GROQ_MODEL, "primary", False)
        if not GROQ_FALLBACK_MODEL or GROQ_FALLBACK_MODEL == GROQ_MODEL:
            raise
        if not _is_temporary_failure(primary_error):
            raise
        try:
            result, usage = make_request(GROQ_FALLBACK_MODEL)
            _record_call(GROQ_FALLBACK_MODEL, "fallback", True, usage)
            return result
        except Exception:
            _record_call(GROQ_FALLBACK_MODEL, "fallback", False)
            raise primary_error


def decompose_query(query: str) -> list[str]:
    """
    Decompose a complex question into a list of focused sub-queries.

    Uses the LLM only when:
      - GROQ_API_KEY is set, AND
      - the query looks like a comparison / multi-aspect question.

    Always returns a non-empty list.  Falls back to [original_query] on any error.
    """
    query = query.strip()
    if not query:
        return [query]

    if not _looks_complex(query) or not GROQ_API_KEY:
        return [query]

    try:
        raw = _call_groq_raw(DECOMPOSE_PROMPT.format(question=query))
        # Extract JSON array — tolerate markdown fences
        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if json_match:
            sub_qs = json.loads(json_match.group())
            if isinstance(sub_qs, list) and all(isinstance(q, str) for q in sub_qs):
                valid = [q.strip() for q in sub_qs if q.strip()]
                if valid:
                    return valid
    except Exception:
        pass

    return [query]  # fallback
