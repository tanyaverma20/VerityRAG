import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
COLLECTION_NAME = "verityrag_docs_v2"

# Retrieval tuning
DENSE_TOP_K = 15        # candidates pulled from vector search
BM25_TOP_K = 15         # candidates pulled from keyword search
RERANK_TOP_K = 5        # final chunks sent to the LLM after reranking
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 100

# Phase 2 — Advanced Retrieval
RRF_K = 60              # RRF constant: RRF(d) = Σ 1/(k + rank(d))
MAX_CONTEXT_TOKENS = 3000   # rough token budget for final LLM context
MAX_CHUNKS_PER_DOC = 3     # max chunks per document in final selection (diversity)
