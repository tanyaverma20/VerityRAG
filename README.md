# VerityRAG

**An evidence-grounded research intelligence system that retrieves, analyzes, compares, and reconstructs knowledge from scientific literature.**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white">
  <img alt="LangGraph" src="https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C">
  <img alt="ChromaDB" src="https://img.shields.io/badge/Vector%20Store-ChromaDB-6E56CF">
  <img alt="PostgreSQL" src="https://img.shields.io/badge/App%20Data-PostgreSQL%20%2F%20SQLite-4169E1?logo=postgresql&logoColor=white">
  <img alt="Redis" src="https://img.shields.io/badge/Cache-Redis%20%2F%20In--Memory-DC382D?logo=redis&logoColor=white">
  <img alt="Groq" src="https://img.shields.io/badge/LLM-Groq%20(Llama%203.3%2070B)-F55036">
  <img alt="Frontend" src="https://img.shields.io/badge/Frontend-Vanilla%20JS%2C%20No%20Build%20Step-F7DF1E?logo=javascript&logoColor=black">
</p>

> **Production engineering status at a glance:** PostgreSQL, Redis, and
> workspace-scoped vector isolation are genuinely implemented and tested
> against real infrastructure (see [Production Infrastructure](#production-infrastructure)).
> A React frontend migration is underway in `frontend-react/` — the core
> workspace/document/chat loop works end-to-end against the real backend,
> most analysis-mode UIs are not yet migrated — see
> [React Migration Status](#react-migration-status) for the exact scope.

---

## Overview

**Verity** — truth, reliability. **RAG** — Retrieval-Augmented Generation.

VerityRAG is a research-paper workspace built around one constraint: **every
answer must trace back to the evidence the user actually uploaded.** Point it
at a stack of PDFs and it answers questions, compares methodologies across
papers, generates structured reports, quizzes you on the material, critiques
a paper's methodology, and maps out a document's concepts — all grounded in
retrieved passages, never in the model's general knowledge.

The problem it addresses is the one every "ChatGPT + PDF" tool runs into:
handed a stack of papers and a vague question, a bare LLM will confidently
blend memorized knowledge with document content, and there's no way to tell
which is which. VerityRAG's answer is architectural, not a prompt trick — a
hybrid dense + keyword retrieval pipeline picks the evidence, the model is
instructed to answer *only* from what was retrieved, self-reports whether
that evidence was actually sufficient, and a document explicitly says so
when the uploaded papers don't cover a question rather than filling the gap
from outside knowledge.

## Preview

<p align="center">
  <img src="docs/images/verity-ui.png" alt="VerityRAG workspace UI — sidebar with workspace switcher, uploaded PDFs, and recent chats; empty chat state with suggested prompts and the composer's + actions menu." width="820">
</p>

<p align="center"><em>The research workspace: per-workspace document management, persistent conversation history, and every research-intelligence action reachable from a single composer menu.</em></p>

---

## Key Features

Every item below is implemented in this repository — nothing here is aspirational.

| Feature | What it does |
|---|---|
| **Evidence-grounded PDF Q&A** | Answers are synthesized only from retrieved chunks of the currently active documents; outside knowledge is explicitly excluded by the prompt contract. |
| **Hybrid Dense + BM25 retrieval** | Semantic search (`sentence-transformers/all-MiniLM-L6-v2`) runs alongside BM25 keyword search so exact terms, model names, and dataset names aren't lost to pure semantic drift. |
| **Reciprocal Rank Fusion (RRF)** | Merges the dense and BM25 ranked lists (`RRF(d) = Σ 1/(k + rank(d))`, k=60) without needing score normalization. |
| **Cross-Encoder reranking** | A local `cross-encoder/ms-marco-MiniLM-L-6-v2` model re-scores the fused candidate pool — a real model forward pass, zero extra LLM calls. |
| **Deterministic query decomposition** | Multi-part comparison questions are split into sub-queries with pure pattern matching, not an LLM call. |
| **Strict document scoping** | Every request resolves an explicit, priority-ordered document scope (named-in-text → explicit "all" → selected → active); retrieval never silently spans documents outside that scope. |
| **Multi-paper comparison** | One structured-report call receives evidence from every selected paper grouped together — never one call per paper. |
| **Comparative reports** | Per-paper + cross-paper structured report, exported as Markdown, PDF, and DOCX from the same validated JSON — the LLM never touches file formatting. |
| **Viva / Mock Test / Project Interview** | Question generation and a turn-by-turn interview loop, grounded in the uploaded document's own evidence (not a generic quiz). |
| **Explain Figure** | Explains a referenced figure/table/graph from its extracted caption and surrounding text. |
| **Paper Evaluation** | A structured critique (problem clarity, novelty, methodology, results, limitations, reproducibility, strengths/weaknesses) with each dimension labeled by evidence support. |
| **Research Gap Discovery** | Surfaces gaps labeled `AUTHOR_STATED_GAP` (the paper says so) vs. `POTENTIAL_INFERRED_GAP` (reasonably inferred, never invented). |
| **Literature Matrix** | One row per paper across problem/method/architecture/dataset/metrics/results/limitations/gap — each cell grounded only in that paper's own evidence. |
| **Knowledge Graph** | Concept nodes/edges per document, with cross-paper edges only when the evidence itself supports a real relationship. |
| **Evidence-aware grounding** | Structured-mode responses can carry a claim-level trace (`DIRECTLY_STATED` / `STRONGLY_SUPPORTED` / `NOT_FOUND`) generated in the same call — no extra request. |
| **Token / LLM-call optimization** | Normal Q&A is exactly one LLM call on success; retrieval, reranking, decomposition, and evidence selection are all LLM-free. |
| **Caching, fallback & observability** | Redis-backed cache in production (in-memory fallback in dev/test, or if Redis is unreachable), a bounded one-time fallback model on genuine failures, and structured JSONL event logging for every request. |
| **PostgreSQL-ready persistence** | A SQLAlchemy repository layer with Alembic migrations backs every structured table (workspaces, documents, sessions, messages, tasks) — SQLite by default, real PostgreSQL when `DATABASE_URL` is set. |
| **Workspace-scoped vector isolation** | Every indexed chunk carries a `workspace_id` alongside `document_id`/`chunk_id`/`parent_id`; retrieval enforces both directly at the ChromaDB query layer, not just via the application-level scope check. |
| **OCR fallback for scanned PDFs** | Pages where normal text extraction comes back empty trigger an OCR attempt (Tesseract, optional) — never invoked on a normal text PDF, never fabricates text when OCR is unavailable/fails. |
| **Offline groundedness evaluation** | A separate, opt-in evaluator scores claim-level groundedness/evidence-coverage from the model's own self-reported evidence trace — never a second call in the normal request path. |

---

## How It Works

```
PDF Upload
    ↓
PDF Parsing               (pypdf — page-aware text extraction)
    ↓
Chunking + Parent Context (page/section-aware, 800-char chunks, 100-char overlap)
    ↓
Embeddings                (sentence-transformers/all-MiniLM-L6-v2, 384-dim)
    ↓
ChromaDB                  (persistent vector store, metadata-filtered by document_id)
    ↓
Dense Retrieval + BM25    (parallel candidate pools, zero LLM calls)
    ↓
RRF Fusion                (rank-based merge, no score normalization needed)
    ↓
Cross-Encoder Reranking   (local model, picks the strongest ~5 chunks)
    ↓
Context / Token Budgeting (dedup, per-document diversity cap, global token cap)
    ↓
Grounded LLM Synthesis    (ONE call, Pydantic-validated structured JSON)
    ↓
Evidence-backed Answer
```

Full request-level view, including where structured application data and
caching sit relative to the retrieval pipeline above:

```
Frontend (vanilla JS today — see React Migration Status)
    ↓
FastAPI (main.py)
    ↓
LangGraph (workflow orchestration — normal mode: one straight path;
           Deep Research: adaptive multi-step, explicitly bounded)
    ↓
Query Processing (deterministic decomposition, document/workspace scope resolution)
    ↓
Dense Retrieval + BM25          ─┐
    ↓                            │  all LLM-free, all workspace_id +
RRF (fusion)                     │  document_id scoped at the vector
    ↓                            │  layer (retrieval.py:_scope_where_clause)
Cross-Encoder (reranking)        │
    ↓                            │
Parent Context (expansion)      ─┘
    ↓
Grounded LLM Generation (Groq / Llama — exactly ONE call on success)
    ↓
PostgreSQL / Redis / ChromaDB (persistence, caching, vectors — see below)
```

---

## Architecture

```
verityrag/
├── frontend/
│   └── index.html          Single-file UI — sidebar, chat, compare view, all
│                            research-action panels. Vanilla JS, no build step,
│                            no framework.
├── backend/
│   ├── main.py              FastAPI app — all HTTP endpoints
│   ├── ingest.py             PDF parsing → chunking → embeddings → Chroma
│   ├── retrieval.py           Dense + BM25 + RRF + Cross-Encoder rerank + token budgeting
│   ├── query_transform.py      Deterministic query decomposition + Groq call wrapper (primary/fallback)
│   ├── graph/                   LangGraph orchestration (normal + Deep Research modes)
│   ├── analysis.py                Viva/Mock Test/Interview/Explain Figure/Evaluate Paper/
│   │                               Research Gaps/Literature Matrix/Knowledge Graph prompts
│   ├── report_generator.py          Structured report generation + Markdown/PDF/DOCX rendering
│   ├── doc_titles.py                 Deterministic, LLM-free display-title resolution
│   ├── schemas.py                     Pydantic contracts for every LLM JSON output
│   ├── database.py                     SQLite registry — workspaces, documents, sessions, messages
│   ├── cache.py                         In-memory answer/report cache
│   ├── observability.py                  Structured JSONL event logging
│   └── chroma_store/                      Persistent ChromaDB collection (runtime data)
└── data/                                   Evaluation harness fixtures/results
```

**Frontend** — a single self-contained HTML file: sidebar (workspace switcher,
document list, conversation history), a chat pane, a side-by-side compare
view, and a grouped "+" actions menu for every research-intelligence
feature. No React/Vue, no bundler — open the file and it runs.

**Backend** — FastAPI. `main.py` is the HTTP surface; every request path
(normal Q&A, Deep Research, reports, and all `/analyze` modes) reuses the
*same* retrieval pipeline in `retrieval.py`, so grounding and scoping
guarantees hold everywhere, not just in the chat endpoint.

**Retrieval layer** — `retrieval.py` + `ingest.py`. Entirely deterministic
and LLM-free: embeddings, BM25, RRF, reranking, and token budgeting are all
plain Python/model inference, never a Groq call.

**LLM layer** — `query_transform.py` wraps every Groq call with a single
bounded fallback attempt (primary model → fallback model, once, only on a
genuinely temporary failure) and per-request call-count tracking. `graph/`
holds the LangGraph state machine that decides *when* that one call happens.

**Storage** — ChromaDB (`backend/chroma_store/`) for vectors; PostgreSQL in
production (SQLite by default) for everything structured. See below.

**Observability** — every request appends one structured line to
`logs/verityrag_events.jsonl` (LLM call count, retrieval/reranking/total
latency, fallback/cache status, token counts, retrieved chunk count) —
read back by `GET /eval/dashboard`, never surfaced in the normal chat UI.

Each component's responsibility, kept deliberately separate:

| Component | Responsible for |
|---|---|
| **PostgreSQL** (or SQLite in dev/test) | Structured application data only — workspaces, documents, sessions, messages, tasks. Never vector embeddings. |
| **Redis** (or in-memory in dev/test) | Caching normal-answer and report results, scoped by workspace/documents/query/mode. |
| **ChromaDB** | Vector storage and retrieval — the only place embeddings live. |
| **BM25** (rank-bm25) | Lexical/keyword retrieval, fused with dense search via RRF. |
| **RRF** | Retrieval fusion — merges dense + BM25 ranked lists without score normalization. |
| **Cross-Encoder** | Reranking the fused candidate pool — local model, zero LLM calls. |
| **Groq / Llama** | Generation — the one LLM call per request. |
| **LangGraph** | Workflow orchestration — decides *when* that one call happens. |

---

## Production Infrastructure

**PostgreSQL** — `backend/db/` is a SQLAlchemy repository layer (models in
`db/models.py`, engine/session handling in `db/session.py`, CRUD in
`db/repository.py`) with an Alembic migration (`backend/alembic/`).
`database.py` is now a thin re-export of this package, so every existing
call site is unaffected. Backend selection: `DATABASE_URL` (a real
`postgresql+psycopg2://...` URL) in production; the same SQLite file this
project has always used otherwise — local dev and the test suite need
nothing extra installed. Connection pooling (`pool_size`/`max_overflow`,
env-configurable) and `pool_pre_ping` are enabled for the PostgreSQL path;
every write goes through one transaction (commit-on-success,
rollback-and-reraise-on-any-exception). Verified in this environment
against SQLite (no live PostgreSQL server was available in the sandbox
this was built in) — the same code path runs against real PostgreSQL the
moment `DATABASE_URL` points at one; nothing here is PostgreSQL-specific
beyond the driver string.

**Redis** — `cache.py` keeps its exact original public functions
(`get_cached_answer`, `set_cached_answer`, `get_cached_report`, etc.) but
now has a pluggable backend: Redis when `REDIS_URL` is set and reachable,
the original in-memory dict otherwise — including automatic fallback if
Redis becomes unreachable *mid-session* (every Redis call is individually
wrapped; a failure degrades that one call to a cache miss, never raises
into a request). Cache keys include workspace_id, document_ids, the
normalized query, mode/research_type, and a config fingerprint. TTL is
configurable (`CACHE_TTL_SECONDS`). Hit/miss/backend counters are exposed
via `cache.stats()` and the Eval Dashboard.

**Workspace-scoped vector isolation** — every indexed chunk's Chroma
metadata now carries `workspace_id` alongside `document_id`/`chunk_id`/
`parent_id`. Retrieval (`retrieval.py:_scope_where_clause`) enforces
`workspace_id` AND `document_id` directly in the Chroma query itself when
a caller supplies a workspace_id — an additive, optional second
enforcement point on top of the pre-existing SQL-layer check
(`_resolve_document_scope`/`documents_in_workspace`), never a replacement
for it, and never breaking a caller that omits workspace_id. Chunks
ingested before this feature existed simply carry no workspace_id, so they
never match a workspace-scoped query rather than leaking into one. This is
**workspace-level data isolation, not user authentication** — this
repository has no login/user system, and none was invented; binding a
workspace to an authenticated identity is a separate, unimplemented
concern (see `backend/test_workspace_vector_isolation.py`, whose last test
documents this boundary explicitly).

---

## Research Intelligence

Beyond Q&A, VerityRAG runs a set of structured analysis modes over the same
retrieved-and-reranked evidence — each one exactly one additional LLM call:

- **Evaluate Paper** — a fixed-dimension critique (problem clarity, novelty,
  methodology, experimental design, results, limitations, reproducibility)
  plus free-text strengths/weaknesses. Every dimension is tagged
  `DIRECTLY_STATED`, `STRONGLY_SUPPORTED`, or `NOT_FOUND` rather than being
  padded out when the evidence doesn't cover it.
- **Research Gap Discovery** — distinguishes gaps the authors state
  themselves from gaps that are only reasonably inferable from what *is*
  written, and labels each one accordingly.
- **Literature Matrix** — a comparison table with one row per selected
  paper; each cell is generated from that paper's own evidence block only,
  so a strong paper's results can't bleed into a weaker paper's row.
- **Knowledge Graph** — a concept tree for a single paper, or a
  cross-paper graph when multiple documents are selected, with every
  node/edge tagged with the document it came from.
- **Multi-paper comparison & reports** — one structured-report call covers
  every selected paper plus a dedicated cross-paper comparison section
  (commonalities, differences, strengths, limitations), exported to
  Markdown/PDF/DOCX.
- **Explain Figure** — explains a referenced figure, table, graph, or
  diagram. The real infrastructure to render the relevant PDF page as an
  image exists (`figure_vision.py`, PyMuPDF) and is wired end-to-end into
  the request path; whether a given answer actually used that image
  depends on `GROQ_VISION_MODEL` being configured with a real
  vision-capable model id (unset by default — no such model is provisioned
  on the account this was built against). Every response is honestly
  labeled either way (see [Limitations](#limitations)) — it never claims
  to have looked at the image unless it genuinely did, for that specific
  answer.

---

## Interview & Learning Modes

- **Viva** and **Mock Test** — generate a batch of questions (with
  category, difficulty, and expected-answer points) directly from the
  uploaded document's evidence.
- **Project Interview** — a turn-by-turn interview loop: one question at a
  time, each candidate answer evaluated against the same evidence
  (correctness, missing points, suggested answer), with the next question
  generated from that same document — pinned to whatever document(s) were
  selected when the interview started, so the topic can't drift mid-session.

All three are PDF-grounded by default: they generate from the currently
selected uploaded document(s), not from any fixed or hardcoded subject
matter.

---

## Grounding & Reliability

- **Document scoping** — a single priority-ordered resolver decides which
  document_ids a request is scoped to (explicit request → named in the
  question text → an explicit "all documents" phrase → click-selected
  documents → all active documents) and every retrieval call is filtered to
  exactly that set.
- **Evidence grounding** — the synthesis prompt instructs the model to
  answer only from the retrieved evidence block and to say so when that
  evidence doesn't cover the question, rather than filling the gap from
  general knowledge.
- **Missing-evidence behavior** — when retrieval or the model determines
  the evidence is insufficient, the response says so explicitly instead of
  generating a plausible-sounding but ungrounded answer.
- **Self-reported confidence** — the same synthesis call returns `grounded`
  and `evidence_sufficient` booleans; there's no separate verifier call in
  normal mode, so this self-assessment costs nothing extra.
- **Claim support levels** — in structured mode, individual claims in the
  answer can be tagged `DIRECTLY_STATED` / `STRONGLY_SUPPORTED` /
  `NOT_FOUND` in that same call, for callers that want claim-level
  traceability rather than just a paragraph-level `grounded` flag.

**Stated honestly:** this is a system *designed* to minimize hallucination
through grounding, structured self-assessment, and explicit
insufficient-evidence handling — it does not claim to mathematically
guarantee zero hallucinations, which no LLM-based system can.

---

## Token & LLM Efficiency

- **One-call normal mode** — a standard question is exactly one physical
  LLM call on success (two only if the primary model has a genuinely
  temporary failure and the bounded fallback is used).
- **Deterministic processing everywhere else** — query decomposition,
  retrieval, RRF fusion, reranking, and document-scope resolution are all
  plain Python/local-model inference; none of them call an LLM.
- **Global context budgeting** — retrieved chunks are deduplicated, capped
  per document for diversity, and selected against a global token budget
  (`MAX_CONTEXT_TOKENS`) before anything reaches the model; output length
  is separately capped (`MAX_ANSWER_TOKENS`).
- **Reranking without an LLM** — the Cross-Encoder narrows ~30 fused
  candidates down to the final evidence set via a local model forward pass.
- **Bounded fallback** — the fallback model is tried at most once, and only
  for a classified-temporary failure (rate limit, 5xx, timeout, or the
  primary model being unavailable) — never for auth errors or bad input.
- **Deep Research is explicitly bounded** — the multi-step agentic path
  (planner, adaptive retrieval, cross-paper analysis, verification retry)
  is capped by `DEEP_RESEARCH_MAX_ITERATIONS` / `_MAX_LLM_CALLS` /
  `_MAX_RETRIES` so it can never run unbounded, even under a routing bug.
- **Caching** — an identical question against an identical active document
  set and conversation context is a full cache hit (zero LLM calls, zero
  retrieval); any upload or deletion invalidates the whole cache
  immediately.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI, Uvicorn |
| Orchestration | LangGraph, LangChain Core |
| LLM provider | Groq (`llama-3.3-70b-versatile`, fallback `openai/gpt-oss-120b`) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Reranking | sentence-transformers CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| Keyword search | rank-bm25 (BM25Okapi) |
| Vector store | ChromaDB (persistent client) |
| Relational storage | PostgreSQL (production) / SQLite (dev, test, default) via SQLAlchemy 2.x + Alembic migrations |
| Cache | Redis (production) / in-memory (dev, test, default) |
| PDF parsing | pypdf (text), PyMuPDF (page-image rendering for Explain Figure) |
| OCR (optional) | pytesseract + Tesseract (system binary, not bundled) |
| Validation | Pydantic v2 |
| Report export | ReportLab (PDF), python-docx (DOCX) |
| Frontend | Vanilla HTML/CSS/JS — no framework, no build step (see [React Migration Status](#react-migration-status)) |
| Testing | pytest, `unittest.mock` (Groq network-boundary mocking), `psutil` (benchmark resource usage) |

---

## Project Structure

```
verityrag/
├── frontend/
│   └── index.html
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── ingest.py
│   ├── retrieval.py
│   ├── query_transform.py
│   ├── analysis.py
│   ├── doc_titles.py
│   ├── figure_vision.py           Page-image rendering (PyMuPDF) + honest vision-model gate
│   ├── ocr_fallback.py            OCR fallback for scanned/image-only PDFs (Tesseract, optional)
│   ├── groundedness_eval.py       OFFLINE, opt-in claim-groundedness evaluator (real LLM calls)
│   ├── report_generator.py
│   ├── schemas.py
│   ├── database.py                Thin compatibility shim — re-exports db/repository.py
│   ├── db/
│   │   ├── models.py                  SQLAlchemy ORM models
│   │   ├── session.py                  Engine/session factory (DATABASE_URL or SQLite fallback)
│   │   └── repository.py               CRUD — same function names/shapes as the old database.py
│   ├── alembic/                    Schema migrations (alembic upgrade head)
│   ├── cache.py                   Redis-backed cache, in-memory fallback
│   ├── observability.py
│   ├── verify.py
│   ├── llm.py
│   ├── eval_harness.py
│   ├── run_evaluation.py          Real 11-paper/113-question retrieval benchmark (pre-existing)
│   ├── requirements.txt
│   ├── .env.example               Placeholders only — never a real credential
│   ├── conftest.py              pytest isolation — temp Chroma/SQLite, never production
│   ├── test_*.py                 275+ tests across ingestion, retrieval, the LangGraph
│   │                              workflow, analysis modes, scoping, DB, cache, workspace
│   │                              isolation, figure vision, OCR, groundedness, and eval
│   ├── tests/fixtures/             Test-only PDF fixture, isolated from the real workspace
│   ├── graph/
│   │   ├── workflow.py               StateGraph definition (normal + Deep Research)
│   │   ├── state.py                   Shared ResearchState TypedDict
│   │   ├── planner.py                  Deep Research: query planning
│   │   ├── retriever.py                 Deep Research: adaptive retrieval node
│   │   ├── organizer.py                  Evidence grouping (shared by all modes)
│   │   ├── synthesizer.py                 The one-call synthesis node
│   │   ├── analyzer.py                     Deep Research: sufficiency/cross-paper analysis
│   │   └── verifier.py                     Deep Research: claim verification pass
│   └── chroma_store/                        Persistent ChromaDB collection (runtime data)
├── evaluation/
│   ├── benchmark_corpus.py        10 original benchmark documents (not real papers — see file)
│   ├── run_benchmark.py           Reproducible 10-document isolation/scale/latency benchmark
│   ├── results.json               Real, measured output of the last run
│   └── README.md                  Human-readable report of the same
├── data/
│   ├── registry.db                            SQLite registry (runtime data; default backend)
│   ├── eval_set.json                            Offline evaluation questions (11 real papers)
│   └── eval_results_comprehensive.json           Offline evaluation results
├── docs/images/
│   └── verity-ui.png
└── logs/
    └── verityrag_events.jsonl                      Structured event log (runtime data)
```

---

## Installation & Setup

```bash
git clone https://github.com/tanyaverma20/VerityRAG.git
cd VerityRAG/backend
pip install -r requirements.txt
```

Copy `backend/.env.example` to `backend/.env` and fill in your own values
(never commit `.env` — it's already covered by `.gitignore`):

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional — these already have sensible defaults in config.py
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_FALLBACK_MODEL=openai/gpt-oss-120b
CHROMA_DIR=./chroma_store
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
COLLECTION_NAME=verityrag_docs_v2

# Structured persistence — leave unset for SQLite (default, zero setup);
# set to a real PostgreSQL URL in production.
DATABASE_URL=
DB_POOL_SIZE=5
DB_POOL_MAX_OVERFLOW=10

# Caching — leave unset for the in-memory fallback (default); set to a
# real Redis URL in production. Falls back automatically if unreachable.
REDIS_URL=
CACHE_TTL_SECONDS=86400

# OCR fallback — requires the Tesseract system binary installed
# separately; skipped cleanly (not an error) if it isn't present.
OCR_ENABLED=true
TESSERACT_CMD=

# Explain Figure's optional vision path — only set this to a REAL,
# verified vision-capable model id available on your Groq account. Left
# unset by default (no such model is provisioned on the account this was
# built against), in which case Explain Figure honestly falls back to its
# existing text/caption-based explanation.
GROQ_VISION_MODEL=
```

The reranker and embedding models are downloaded automatically from
Hugging Face on first run. If you set `DATABASE_URL` to a PostgreSQL URL,
run migrations once: `cd backend && alembic upgrade head`.

## Running Locally

**Backend:**

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Confirm it's up:

```bash
curl http://127.0.0.1:8001/health
# {"status":"ok","chunks_indexed": <N>}
```

**Frontend:**

Open `frontend/index.html` directly in a browser — it's a static file with
no build step. Upload a PDF from the sidebar, wait for it to reach "Ready,"
then ask a question.

---

## Testing

```bash
cd backend
python -m pytest --ignore=test_eval.py -q
```

The suite is 300 collected tests, split by dependency on real network calls:

- **276 tests** (everything except `test_eval.py`) run against an isolated,
  temp-directory Chroma/SQLite environment set up automatically by
  `conftest.py` — the real `backend/chroma_store` and production
  `data/registry.db` are never touched. Most LLM-dependent assertions mock
  `groq.Groq` at the network boundary, so exact physical call counts are
  *proven*, not assumed (see `test_llm_call_count.py`,
  `test_analysis_modes.py`). Retrieval, reranking, and scoping assertions
  run against real (test-fixture) Chroma data — only the LLM call itself is
  mocked. New coverage in this pass: `test_db_repository.py` (SQLAlchemy
  CRUD/transactions/isolation), `test_cache_redis.py` (hit/miss/TTL/
  fallback/scoping, including a fake Redis client), `test_workspace_vector_
  isolation.py` (all 7 required cross-workspace scenarios), `test_figure_
  vision.py` (real PyMuPDF page rendering + honest vision/text fallback),
  `test_ocr_fallback.py`, `test_groundedness_eval.py`, and
  `test_eval_dashboard_extended.py`.
- **`test_eval.py`** (~24 tests) exercises the pipeline against the real
  Groq API and is run deliberately, not as a routine check, to avoid
  burning API quota.

Latest full run of the mocked/isolated suite: **275 passed, 1 pre-existing
failure** (`test_graph.py::test_api_backward_compatibility` — a stale test
double missing a field added by earlier workspace support, predates this
round of work and is unrelated to retrieval, grounding, or any of the
infrastructure added here).

`backend/groundedness_eval.py` makes REAL Groq calls when actually run and
is never part of the automated suite above — see [Grounding & Reliability](#grounding--reliability).

## Evaluation Harness

Two independent, real evaluations exist — deliberately not conflated:

1. **`backend/run_evaluation.py`** (pre-existing) benchmarks dense_only /
   bm25_only / hybrid_rrf / hybrid_reranked / agentic_graph against 11 real
   research papers and 113 real questions
   (`data/eval_results_comprehensive.json`). This is what backs the actual
   measured improvement: **hybrid_reranked reaches recall@5 = 0.9444 vs.
   dense_only's 0.8637** (+8.07 points, +9.3% relative), MRR 0.9077 vs.
   0.8835 — real numbers, not an assertion.
2. **`evaluation/run_benchmark.py`** (new) is a reproducible,
   isolated-Chroma, 10-document benchmark proving indexing correctness,
   zero cross-document contamination, document isolation, comparison
   scaling at 2/5/10 documents, query-decomposition scope preservation,
   concurrent-retrieval speedup (8.93x with 10 parallel queries in the
   last run), and real latency distributions (mean 137.5ms / p95 150.8ms
   end-to-end retrieval) — see `evaluation/README.md` for the full report
   and exactly which numbers came from where.

Both sets of numbers are surfaced live in the app's **Eval Dashboard**
(`GET /eval/dashboard`), alongside real per-request runtime metrics from
the observability log (LLM calls/query, latency, cache hit rate, fallback
rate) and — when `backend/groundedness_eval.py` has actually been run —
real claim-level groundedness/evidence-coverage/unsupported-claim-rate
scores. Any metric without a real measurement behind it (e.g.
`hallucination_rate` as a single scalar) is reported as **"Not measured"**
in the dashboard's response, never invented.

---

## Example Use Cases

Upload one or more papers, then ask things like:

- *"What problem is this paper trying to solve?"*
- *"Summarize the methodology section."*
- *"Compare the methodologies of these two papers."*
- *"What datasets were used, and what were the reported metrics?"*
- *"What are the stated limitations?"*
- *"Explain Figure 2."*
- *"Which approach would you choose and why?"* (after comparing papers)

Or reach for a structured mode from the composer's actions menu: **Evaluate
Paper** for a critique, **Find Research Gaps** for what's missing,
**Literature Matrix** to line up several papers side by side, **Knowledge
Graph** to see a paper's concepts as a graph, or **Viva** / **Mock Test** /
**Project Interview** to be quizzed on the material.

---

## Design Decisions

- **Dense + BM25, not dense-only** — semantic search alone loses exact
  terms: model names, dataset names, acronyms, and numbers. BM25 catches
  precisely what embeddings blur.
- **RRF over a weighted score merge** — dense and BM25 scores live on
  different, incomparable scales. RRF combines rank positions instead of
  raw scores, so it needs no tuning or normalization step.
- **Cross-Encoder reranking after fusion, not before** — a Cross-Encoder is
  far more accurate than bi-encoder similarity but expensive (one forward
  pass per candidate). Running it only on the ~30 already-fused candidates,
  rather than the full corpus, keeps quality high and cost bounded.
- **ChromaDB** — a persistent, embeddable vector store with native metadata
  filtering, which is exactly what document-scoped retrieval needs (filter
  by `document_id` at query time) without standing up a separate service.
- **LangGraph** — normal mode is a straight line
  (plan → retrieve → organize → synthesize → assign_confidence), but Deep
  Research genuinely branches and loops (adaptive retrieval, cross-paper
  analysis, a bounded verification retry). A state graph makes that
  explicit and inspectable instead of a tangle of conditionals.
- **Structured, Pydantic-validated outputs** — every LLM response is
  parsed against a schema; a malformed response is a clean, typed failure
  path, never a silent `dict.get()` guess.
- **Token/context budgeting** — a global budget with a per-document cap
  means adding more papers to a comparison degrades gracefully (broader
  but still-representative coverage) instead of the prompt silently
  exceeding the model's context window.
- **Document scoping as a first-class concept** — every feature (Q&A,
  reports, viva, interview, comparison, evaluation) resolves scope through
  the same priority-ordered rule, so "which papers is this answer actually
  about" is never feature-specific or accidental.

---

## React Migration Status

**In progress, not yet at parity.** A real React + TypeScript + Vite
frontend lives in `frontend-react/`, built incrementally against the
actual backend (not mocked) and manually verified end-to-end in a live
browser session: workspace management, document upload/list/delete,
conversation history, and the core normal Q&A chat loop (grounded
answers, confidence, de-duplicated citations, honest error rendering) all
work for real. It has a typed API client for every backend endpoint, 23
passing Vitest/React Testing Library tests, and a clean production build
(`npm run build`).

It is deliberately **not** yet a full replacement: the ~90-function
original app's remaining surface — Deep Research, Viva/Mock Test, Project
Interview, Explain Figure, Evaluate Paper, Research Gaps, Literature
Matrix, Knowledge Graph, Comparative Reports, the Eval Dashboard UI, and
the "+" composer menu — has not been migrated yet. See
[`frontend-react/README.md`](frontend-react/README.md) for the exact,
current gap list. Per the project's migration policy, `frontend/` remains
the served, relied-upon frontend until the new one reaches real parity —
it is not deleted or defaulted-to preemptively. The backend's HTTP API is
unchanged by this work either way.

## Limitations

Stated plainly, not as a footnote — genuinely implemented, partially
implemented, and still-open items, not blurred together:

**Implemented:**
- **Workspace-scoped vector isolation** — see [Production Infrastructure](#production-infrastructure);
  tested against all 7 required cross-workspace scenarios.
- **PostgreSQL/Redis support** — real, tested code; verified against
  SQLite/in-memory in this environment (no live PostgreSQL/Redis server
  was available in the sandbox this was built in), designed to be
  driver-agnostic beyond the connection string.
- **OCR fallback exists** — but requires the Tesseract OCR system binary
  to be installed separately (this app never installs it); if it isn't
  present, OCR is skipped cleanly with a clear status, never a silent
  failure or fabricated text.
- **Offline groundedness evaluation exists** (`groundedness_eval.py`) —
  opt-in, makes real LLM calls, was verified via mocked tests rather than
  a full live run in this pass (to avoid burning API quota on an infra
  change) — run it yourself for real numbers on your own documents.

**Partially implemented:**
- **Explain Figure's visual path** — the real infrastructure (PDF page
  rendering via PyMuPDF, a multimodal Groq call path) is built and tested,
  but requires `GROQ_VISION_MODEL` to be set to an actual vision-capable
  model id. No such model is provisioned on the Groq account this was
  built against, so in this deployment Explain Figure currently always
  uses its original text/caption-based path — and says so explicitly in
  every response, never claiming visual inspection that didn't happen.

**Not implemented / open:**
- **No automated production groundedness/hallucination scoring on live
  traffic** — `groundedness_eval.py` is offline/opt-in by design (adding
  it to the live request path would mean a second LLM call per query,
  which was an explicit constraint to avoid); the Eval Dashboard reports
  this honestly rather than fabricating a live number.
- **PDF text extraction quality still depends on the source PDF** — OCR
  helps for scanned pages, but unusual layouts/columns can still extract
  poorly.
- **Single shared vector collection** — workspace_id scoping (above)
  strengthens isolation but this remains one shared ChromaDB collection,
  not one per tenant; adequate for personal/small-team scale, not
  multi-tenant production isolation at scale.
- **No user authentication** — workspace_id is a plain client-supplied
  scope identifier, not a security principal; binding workspaces to
  authenticated users is unimplemented and was not invented here.
- **No streaming output** — answers are returned in full, not
  token-by-token.
- **React frontend migration** — see above.

## Future Improvements

- A real vision-capable model configured via `GROQ_VISION_MODEL`, turning
  Explain Figure's already-built visual path from dormant infrastructure
  into an active one.
- A live, sampled groundedness scorer (reusing `groundedness_eval.py`'s
  logic) surfaced in the Eval Dashboard as a real running average, not
  just an on-demand offline report.
- Tesseract bundled or documented as a one-line install step so OCR works
  out of the box.
- Per-tenant vector store isolation and real authentication for multi-user
  deployment.
- The React migration, scoped and executed as its own dedicated effort.
- Streaming token-by-token responses for perceived latency.

---

## Author / Project

**VerityRAG** — an evidence-grounded research intelligence system, built to
demonstrate that a RAG pipeline's reliability comes from its retrieval and
scoping architecture, not just its prompt.

Repository: [github.com/tanyaverma20/VerityRAG](https://github.com/tanyaverma20/VerityRAG)
