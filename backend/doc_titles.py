"""
doc_titles.py — deterministic, zero-LLM-call resolution of a human-readable
DISPLAY title for a document.

This is purely a presentation-layer concern: `document_id` itself is never
touched, replaced, or reinterpreted anywhere else in the system — retrieval,
scoping, citations, caching keys, and persistence all keep using the real
document_id exactly as before. This module only decides what string a human
should see instead of that id (or a technical filename) in report/compare/
literature-matrix/evaluate-paper/research-gaps/knowledge-graph output.

Resolution order (never an extra LLM call):
  1. The document's own filename, humanized — used whenever it looks like a
     real title rather than a generated/technical name.
  2. A title the model ALREADY produced in the very same generation call
     (e.g. a report's per-paper "title" field) — free, since no new call is
     made to get it; only used when the filename itself wasn't usable.
  3. A safe generic label ("Document 1", "Untitled Document") — NEVER the
     raw document_id, hash, or any internal identifier.
"""
from __future__ import annotations

import re
from collections import Counter

# Filenames (after stripping .pdf) that are themselves technical/generated
# rather than a real title: hex hashes/ids, UUIDs, and generic placeholders
# like "scan", "untitled", "document3", or a bare number.
_MEANINGLESS_STEM_RE = re.compile(
    r"^("
    r"[0-9a-f]{6,}"
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"|(untitled|document|doc|scan|img|image|file|paper|copy|new|download|export|upload|attachment)[\s_-]*\d*"
    r"|\d+"
    r")$",
    re.IGNORECASE,
)


def _looks_meaningless(stem: str) -> bool:
    s = stem.strip()
    if not s:
        return True
    if _MEANINGLESS_STEM_RE.match(s):
        return True
    # A long unbroken run of hex characters with no spacing/words is almost
    # certainly a content hash or generated id, not a real title, even if it
    # didn't match one of the exact patterns above (e.g. a 40-char sha1).
    if len(s) >= 12 and re.fullmatch(r"[0-9a-f]+", s, re.IGNORECASE):
        return True
    return False


def humanize_filename(filename: str | None) -> str | None:
    """Strip .pdf, tidy separators/casing — returns None (not a guess) when
    the resulting stem looks technical/meaningless rather than a real
    title, so the caller knows to fall back to something else."""
    if not filename:
        return None
    stem = re.sub(r"\.pdf$", "", filename.strip(), flags=re.IGNORECASE)
    stem = re.sub(r"\(\d+\)$", "", stem).strip()  # trailing "(1)" from re-uploads/duplicates
    if _looks_meaningless(stem):
        return None
    spaced = re.sub(r"[_\-]+", " ", stem)
    spaced = re.sub(r"\s+", " ", spaced).strip()
    if not spaced or _looks_meaningless(spaced):
        return None
    # Only impose Title Case when the stem was ALL lower or ALL upper —
    # never touch a filename that already has deliberate mixed casing
    # (acronyms like "DBMS", "RRF", or an already-nice title).
    if spaced.isupper() or spaced.islower():
        spaced = spaced.title()
    return spaced


# ---------------------------------------------------------------------------
# Content-derived fallback — used ONLY when the filename itself is technical/
# meaningless and no usable fallback_title was already generated. Reads
# chunks already sitting in Chroma for this document_id via a plain
# metadata-filtered get() (the same call shape retrieval.py already uses for
# parent-context expansion) — no embeddings, no dense/BM25/RRF/reranking, no
# LLM call. Purely deterministic: the same stored chunks always produce the
# same extracted phrase.
# ---------------------------------------------------------------------------

# Common short connector words allowed INSIDE a multi-word phrase (never at
# its start or end) — e.g. "Theory of Computation".
_PHRASE_CONNECTORS = {"of", "and", "the", "for", "in", "on", "to", "a", "an", "with", "by", "vs"}

# Generic words that are technically capitalized-looking (start of a
# sentence, a lecture-note artifact, etc.) but never a meaningful document
# topic on their own — excluded from the single-word fallback.
_GENERIC_SINGLE_WORDS = {
    "this", "that", "these", "those", "chapter", "section", "page", "figure",
    "table", "note", "notes", "example", "introduction", "overview", "summary",
    "definition", "lecture", "lec", "unit", "module", "part", "topic", "topics",
    "content", "contents", "abstract", "conclusion", "references", "appendix",
}


def _normalize_word(w: str) -> str:
    # Keep short all-caps tokens as acronyms (DBMS, RRF, API, OS) rather than
    # mangling them into "Dbms"; otherwise Title-case the word.
    if w.isupper() and 2 <= len(w) <= 5:
        return w
    return w[:1].upper() + w[1:].lower()


def _extract_phrases(text: str) -> list[str]:
    """Sequences of 2-5 capitalized words, optionally joined by a single
    lowercase connector, e.g. "Database Management System",
    "Entity Relationship Model", "Theory of Computation"."""
    words = re.findall(r"[A-Za-z][A-Za-z\-]*", text)
    phrases: list[str] = []
    n = len(words)
    i = 0
    while i < n:
        w = words[i]
        if w[:1].isupper() and len(w) >= 3 and w.lower() not in _PHRASE_CONNECTORS:
            phrase = [w]
            cap_count = 1
            j = i + 1
            while j < n and cap_count < 5:
                nxt = words[j]
                if nxt[:1].isupper() and len(nxt) >= 3:
                    phrase.append(nxt)
                    cap_count += 1
                    j += 1
                elif nxt.lower() in _PHRASE_CONNECTORS and j + 1 < n and words[j + 1][:1].isupper() and len(words[j + 1]) >= 3:
                    phrase.append(nxt.lower())
                    j += 1
                else:
                    break
            if cap_count >= 2:
                phrases.append(" ".join(phrase))
                i = j
                continue
        i += 1
    return phrases


def _top_phrase(text: str, min_occurrences: int = 2) -> str | None:
    counts: Counter[str] = Counter()
    canonical: dict[str, str] = {}
    for phrase in _extract_phrases(text):
        words = phrase.split(" ")
        key = " ".join(w.lower().rstrip("s") for w in words)
        counts[key] += 1
        display = " ".join(_normalize_word(w) for w in words)
        if key not in canonical or len(display) >= len(canonical[key]):
            canonical[key] = display
    if not counts:
        return None
    best_key, best_count = counts.most_common(1)[0]
    if best_count < min_occurrences:
        return None
    return canonical[best_key]


def _top_single_word(text: str, min_occurrences: int = 3) -> str | None:
    counts: Counter[str] = Counter()
    canonical: dict[str, str] = {}
    for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text):
        if not w[:1].isupper() or w.lower() in _GENERIC_SINGLE_WORDS:
            continue
        key = w.lower().rstrip("s")
        if key in _GENERIC_SINGLE_WORDS:
            continue
        counts[key] += 1
        canonical.setdefault(key, _normalize_word(w))
    if not counts:
        return None
    best_key, best_count = counts.most_common(1)[0]
    if best_count < min_occurrences:
        return None
    return canonical[best_key]


def derive_topic_from_content(document_id: str, max_chunks: int = 40) -> str | None:
    """A short topic phrase pulled from THIS document's own already-stored
    chunk text — never from another document, never invented. Returns None
    (not a guess) if nothing sufficiently repeated/meaningful is found."""
    if not document_id:
        return None
    try:
        from ingest import get_collection
        col = get_collection()
        res = col.get(where={"document_id": document_id}, include=["documents", "metadatas"], limit=max_chunks)
    except Exception:
        return None

    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    if not docs:
        return None

    # Earliest pages first — intro/title material carries the most signal.
    paired = sorted(zip(docs, metas), key=lambda dm: (dm[1] or {}).get("page_number") or 0)
    text = " ".join(d for d, _ in paired if d)
    if not text.strip():
        return None

    return _top_phrase(text) or _top_single_word(text)


def resolve_display_title(
    document_id: str,
    filename: str | None,
    fallback_title: str | None = None,
    index: int | None = None,
) -> str:
    """The single source of truth used everywhere a document needs a
    human-readable label. `document_id` is accepted only so every call site
    is explicit about which document this title is for — it is never
    itself returned.

    Order: humanized filename -> a title the model already produced in this
    SAME generation call -> a topic phrase deterministically extracted from
    the document's own stored content (no LLM call) -> a safe generic label
    as an absolute last resort."""
    title = humanize_filename(filename)
    if title:
        return title
    if fallback_title and fallback_title.strip() and not _looks_meaningless(fallback_title.strip()):
        return fallback_title.strip()
    content_title = derive_topic_from_content(document_id)
    if content_title:
        return content_title
    return f"Document {index + 1}" if index is not None else "Untitled Document"
