"""
json_extract.py — robust JSON-object extraction from raw LLM text output.

Root-cause fix (verified against a real vision-capable reasoning model,
`qwen/qwen3.6-27b` via GROQ_VISION_MODEL): every prompt in this codebase
that asks for "compact JSON" used to extract it with a naive
`re.search(r"\{.*\}", raw, re.DOTALL)` — a GREEDY match from the first `{`
in the entire response to the last `}`. That breaks in two real,
now-confirmed ways:

  1. A reasoning model's <think>...</think> preamble can itself contain
     balanced curly braces that have nothing to do with the answer (e.g.
     LaTeX math notation like `\\frac{QK^T}{\\sqrt{d_k}}`). The greedy regex
     then spans from one of THOSE braces all the way to the real answer's
     closing brace, producing a huge blob that isn't valid JSON at all —
     json.loads() fails, and the caller silently falls back to a
     degraded/text-only path, exactly as if the model had said nothing
     useful, even though it actually gave a complete, correct answer.
  2. The real JSON is wrapped in a markdown code fence (```json ... ```),
     which the old regex handled by accident (fences don't contain stray
     braces) but which is worth handling explicitly and first, since it's
     the most reliable signal when present.

Used by every JSON-expecting LLM call site in the codebase (normal Q&A's
one synthesis call — specifically its bounded fallback-model path, which
is exactly the "gpt-oss"-style reasoning model this was found against —
every backend/analysis.py mode, and graph/analyzer.py's Deep Research
nodes), so this one fix closes the gap everywhere at once rather than
patching each duplicated regex separately.
"""
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _find_balanced_json_objects(text: str) -> list[str]:
    """Scans for every top-level, brace-depth-balanced {...} span in text
    — real bracket matching, not a greedy regex, so an unrelated `{`/`}`
    pair earlier in the string (e.g. inside LaTeX notation in a reasoning
    preamble) can never pull a match across the actual JSON answer's own
    boundaries. Returns them in the order they appear."""
    spans: list[str] = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append(text[start:i + 1])
                    start = None
    return spans


def extract_json_object(raw: str) -> dict | None:
    """Best real attempt to extract the single JSON object a prompt asked
    for, trying the most reliable signal first. Never raises — returns
    None if nothing in the text parses as a JSON object, which every
    caller already treats as "generation failed, degrade honestly."
    """
    if not raw:
        return None

    # 1. A fenced ```json ... ``` (or bare ```...```) block, if present —
    #    the most explicit, least ambiguous signal a model can give.
    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except Exception:
            pass  # fall through — a malformed fence shouldn't block the other strategies

    # 2. Real brace-depth-balanced spans, tried LAST-to-first — a
    #    reasoning model's actual answer is reliably the last JSON-shaped
    #    thing it emits, after any <think> preamble; trying the last one
    #    first also means we don't have to know in advance whether a
    #    preamble is present at all.
    for span in reversed(_find_balanced_json_objects(raw)):
        try:
            obj = json.loads(span)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    # 3. Last resort: the original greedy first-to-last match, kept only
    #    for the simplest case (a bare JSON object with no preamble and no
    #    fence) as a final safety net — every call site's own downstream
    #    validation (e.g. Pydantic) still catches anything that parses but
    #    doesn't match the expected schema.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None
