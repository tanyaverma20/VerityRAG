# VerityRAG

VerityRAG is a research-focused AI assistant designed to stay grounded in
the user's uploaded papers while minimizing unnecessary LLM usage.

It is not "ChatGPT with a PDF attached." It is a hybrid-retrieval RAG
pipeline — dense + BM25 + RRF fusion + cross-encoder reranking — with
document-scoped grounding, a one-LLM-call normal mode, a controlled
fallback model, structured/validated JSON output, multi-paper comparison,
research report generation, answer caching, and an isolated test
environment that never touches production data.

**Design goal, stated honestly:** VerityRAG is *designed to minimize
hallucination* through evidence grounding, structured self-assessment
(`grounded` / `evidence_sufficient`), and explicit insufficient-evidence
handling. It does not claim to mathematically guarantee zero
hallucinations — no LLM-based system can.

## The 15 differentiators

1. **Evidence-first answers** — every answer is generated from chunks
   retrieved from the user's currently active uploaded documents, never
   from the whole PDF and never from outside knowledge.
2. **Hallucination-resistant design** — the synthesis call self-reports
   `grounded`/`evidence_sufficient` as part of its one structured JSON
   response; no second "verifier" call in normal mode.
3. **Token-efficient architecture** — a normal query is exactly one LLM
   call on success (two only if the primary genuinely fails and the
   fallback is used). No LLM calls for planning, query rewriting,
   retrieval, reranking, or evidence selection.
4. **Hybrid retrieval** — dense (semantic) + BM25 (exact terms, model/
   dataset names, abbreviations), fused with Reciprocal Rank Fusion.
5. **Cross-encoder reranking** — a local model, zero LLM calls.
6. **Context/token budgeting** — deduplication, per-document diversity
   caps, and a global token budget before anything reaches the LLM;
   output length is separately capped and configurable.
7. **Multi-paper comparison** — evidence from every selected paper is
   grouped and handed to the *same single call*, never one call per paper.
8. **Strict document workspace isolation** — the user's uploaded PDFs are
   the only source of truth for normal queries; the canonical/legacy
   Chroma corpus and `data/*.pdf` test fixtures never participate unless
   explicitly scoped in.
9. **Controlled fallback model** — primary `llama-3.3-70b-versatile`,
   fallback `openai/gpt-oss-120b`, used at most once, only for genuinely
   temporary failures (never for bad input, auth errors, or insufficient
   evidence).
10. **Structured JSON pipeline** — every LLM output is Pydantic-validated;
    a malformed response is a clean failure, never a reason to retry
    blindly, and raw JSON is never shown to the normal user.
11. **Research report generation** — one call produces a structured report
    (per-paper sections + cross-paper comparison), rendered deterministically
    into Markdown, PDF, and DOCX — the LLM never formats the file itself.
12. **Answer caching** — identical question + identical active document set
    + identical conversation context ⇒ zero LLM calls. Any upload/removal
    invalidates the cache immediately.
13. **Research-focused UX** — a ChatGPT/Claude-style workspace: sidebar with
    conversation history, per-document processing pipeline, clean answers
    with no exposed chunk IDs, UUIDs, or raw scores.
14. **Deep Research mode** — a separate LangGraph-orchestrated path with a
    planner, adaptive retrieval, and cross-paper analysis, explicitly
    bounded by `DEEP_RESEARCH_MAX_ITERATIONS` / `_MAX_LLM_CALLS` / `_MAX_RETRIES`
    so it can never loop unbounded.
15. **Observable pipeline** — every request logs LLM call count, which
    model answered, fallback/cache status, token counts, and latency
    internally (`logs/verityrag_events.jsonl`) — never surfaced in the
    normal UI.

## Project structure

```
verityrag/
  backend/
    main.py               FastAPI app — /query, /report, /upload, /health, sessions
    ingest.py              PDF parsing, chunking, embedding, Chroma storage
    retrieval.py            Hybrid retrieval: dense + BM25 + RRF + cross-encoder rerank
    cache.py                In-memory answer/report cache
    query_transform.py      Primary/fallback model call + LLM observability
    schemas.py               Pydantic contracts for all LLM JSON output
    report_generator.py      Structured report generation + Markdown/PDF/DOCX rendering
    observability.py         Structured event logging (logs/verityrag_events.jsonl)
    database.py               SQLite registry: documents, collections, sessions, messages
    graph/                    LangGraph workflow (normal mode + Deep Research mode)
    conftest.py                Phase 8 test isolation — temp Chroma/SQLite, never production
    tests/fixtures/             Test-only PDFs, never part of the user workspace
    chroma_store/                Canonical production ChromaDB — never rebuilt/re-ingested by tooling
    requirements.txt
  frontend/
    index.html               Sidebar + chat UI: upload, ask, compare, generate reports
  data/
    *.pdf                     Source PDFs behind the canonical corpus + eval harness fixtures
    eval_set.json, eval_results_comprehensive.json   Evaluation harness inputs/outputs
  .env.example
```

## Setup

```bash
cd backend
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `GROQ_API_KEY`. Defaults for
`GROQ_MODEL` / `GROQ_FALLBACK_MODEL` / retrieval tuning live in
`backend/config.py` and can all be overridden via environment variables.

## Running it

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

`GET http://127.0.0.1:8001/health` should return `{"status":"ok","chunks_indexed": <N>}`.

Open `frontend/index.html` directly in a browser (no build step). Upload a
PDF, wait for it to reach "Ready," then ask a question.

## Testing

```bash
cd backend
python -m pytest test_llm_call_count.py test_reports_and_caching.py test_multi_document_scoping.py test_observability_and_budget.py test_ingestion.py -v
```

These run against an isolated temp Chroma/SQLite environment (see
`conftest.py`) and never touch `backend/chroma_store`. Most LLM-dependent
assertions are mocked at the network boundary (`groq.Groq`) so the call
count itself is proven, not assumed — see `test_llm_call_count.py`.

The full historical regression suite (103+ tests across ingestion,
retrieval, the LangGraph workflow, evaluation, and conversational memory)
also exists but makes many real Groq calls; run it deliberately, not as a
routine check, to avoid burning API quota.

## Evaluation harness

`backend/eval_harness.py` compares naive / dense-only / hybrid+verify
pipelines on `data/eval_set.json` and reports accuracy proxy, latency, and
groundedness per pipeline — useful for backing up "grounded" and
"low-hallucination" with actual numbers rather than a claim.

## Data safety

The canonical Chroma collection (`verityrag_docs_v2`, in
`backend/chroma_store/`) is production data and is never rebuilt, migrated,
or re-ingested by any script or test in this repository. `data/*.pdf` are
historical/eval fixtures, not part of any user's active workspace — the
frontend never lists them, and the backend never searches them unless a
request explicitly scopes them in by `document_id`.
