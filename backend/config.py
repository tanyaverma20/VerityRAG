import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Used once, only on a temporary primary-model failure. Never hardcode a
# model name elsewhere — every call site reads GROQ_MODEL/GROQ_FALLBACK_MODEL
# from here.
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "verityrag_docs_v2")

import pathlib
REGISTRY_DB_PATH = os.getenv("REGISTRY_DB_PATH", str(pathlib.Path(__file__).parent.parent / "data" / "registry.db"))

# Retrieval tuning
DENSE_TOP_K = 15        # candidates pulled from vector search
BM25_TOP_K = 15         # candidates pulled from keyword search
RERANK_TOP_K = 5        # final chunks sent to the LLM after reranking
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 100

# Phase 2 — Advanced Retrieval
RRF_K = 60              # RRF constant: RRF(d) = Σ 1/(k + rank(d))
MAX_CONTEXT_TOKENS = 3000   # rough token budget for final LLM context (input)
MAX_CHUNKS_PER_DOC = 3     # max chunks per document in final selection (diversity)

# Output token budget for the one synthesis call. Some fallback models (e.g.
# "gpt-oss" reasoning models) spend part of this budget on internal
# reasoning before emitting visible content — too small a cap can truncate
# the response before any usable answer/JSON appears (finish_reason="length"
# with empty content). 1024 is generous enough for a full multi-paper
# comparison answer while staying a bounded, controlled output cap.
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "1024"))

# Explain Figure's vision call gets its OWN, larger budget. Confirmed via a
# real end-to-end test against a real configured vision-capable model
# (Phase 15 vision verification): a reasoning-style vision model can spend
# several hundred tokens on internal <think>...</think> reasoning about a
# genuinely complex multimodal prompt (describing an architecture diagram's
# components/layout, not just answering a short factual question) before
# ever emitting the actual JSON answer — MAX_ANSWER_TOKENS (1024) was
# repeatedly observed to truncate the response mid-reasoning, before any
# JSON appeared at all, silently degrading every vision call to the
# text-only fallback. Still a bounded, controlled cap, just sized for this
# specific call's real observed behavior rather than reused from the
# text-only synthesis path.
MAX_VISION_ANSWER_TOKENS = int(os.getenv("MAX_VISION_ANSWER_TOKENS", "4096"))

# Report generation (deterministic retrieval, still ONE LLM call). A report
# needs broader per-document coverage than a single Q&A answer, so these are
# intentionally larger than RERANK_TOP_K/MAX_CONTEXT_TOKENS/MAX_CHUNKS_PER_DOC
# above — but still bounded, never "send the whole PDF".
REPORT_CHUNKS_PER_DOC = int(os.getenv("REPORT_CHUNKS_PER_DOC", "6"))
MAX_REPORT_CONTEXT_TOKENS = int(os.getenv("MAX_REPORT_CONTEXT_TOKENS", "6000"))
MAX_REPORT_ANSWER_TOKENS = int(os.getenv("MAX_REPORT_ANSWER_TOKENS", "2048"))

# Deep Research explicit limits — NORMAL mode makes exactly one LLM call and
# never touches these. Deep Research's multi-step loop (planner, adaptive
# retrieval, cross-paper analysis, verification, retry) is bounded by these
# so it can never run away or loop indefinitely.
DEEP_RESEARCH_MAX_ITERATIONS = int(os.getenv("DEEP_RESEARCH_MAX_ITERATIONS", "3"))
DEEP_RESEARCH_MAX_LLM_CALLS = int(os.getenv("DEEP_RESEARCH_MAX_LLM_CALLS", "12"))
DEEP_RESEARCH_MAX_RETRIES = int(os.getenv("DEEP_RESEARCH_MAX_RETRIES", "1"))

# CORS (security audit finding: main.py previously hardcoded
# allow_origins=["*"], which — combined with real per-user authentication —
# would let ANY origin's JavaScript read another logged-in user's
# responses if a browser ever sent along their bearer token. Configurable
# via a comma-separated env var so a real deployment can lock this down to
# its actual frontend origin(s); the dev default covers Vite's default
# port (5173) and the vanilla-JS legacy frontend's typical static-server
# ports, never "*".
def parse_cors_origins(raw: str) -> list[str]:
    """Pure parsing logic, factored out so it's testable without reloading
    this module (reloading would re-run load_dotenv() and risks the exact
    env-var-restoration footgun documented in conftest.py)."""
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


CORS_ALLOWED_ORIGINS = parse_cors_origins(os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000",
))

# Upload size cap (Phase 13 security audit finding: /upload previously had
# no size limit at all — an unbounded upload could exhaust disk/memory).
# 50 MB is generous for a real academic paper (even ones with many figures)
# while bounding the worst case. Enforced in main.py:upload_document() by
# counting bytes while streaming to the temp file, not by trusting a
# client-supplied Content-Length header.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
