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
