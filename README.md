# VerityRAG

A research-paper QA system with hybrid retrieval (BM25 + dense + reranking)
and claim-level verification — plus an evaluation harness that measures it
against baselines instead of just claiming it works.

## Project structure

```
verityrag/
  backend/
    main.py            FastAPI app (upload, query, health endpoints)
    ingest.py           PDF parsing, chunking, embedding, Chroma storage
    retrieval.py         Hybrid retrieval: BM25 + dense + cross-encoder rerank
    verify.py            Claim-level fact verification against sources
    eval_harness.py       Compares naive / dense-only / hybrid+verify pipelines
    llm.py               Groq API wrapper
    config.py            All tunable settings in one place
    requirements.txt
  frontend/
    index.html           No-build-step UI: upload, ask, see verification
  data/
    eval_set.json         Your held-out test questions (replace the sample)
    eval_results.csv       Generated after running eval_harness.py
  .env.example
```

## Setup (in VS Code)

1. Open the `verityrag` folder in VS Code.
2. Open a terminal (`` Ctrl+` ``) and create a virtual environment:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Get a free Groq API key at https://console.groq.com/keys
4. In the project root, copy `.env.example` to `.env` and paste in your key:
   ```bash
   cd ..
   cp .env.example .env
   ```
   Edit `.env` and set `GROQ_API_KEY=...`

## Running it

**Start the backend** (from the `backend/` folder, with venv active):
```bash
uvicorn main:app --reload --port 8001
```
Visit `http://localhost:8001/health` — you should see `{"status":"ok","chunks_indexed":0}`.

**Open the frontend**: just open `frontend/index.html` directly in your browser
(no build step, no dev server needed). Upload a PDF, then ask a question.

## Running the evaluation harness

This is the part that turns this from "a RAG demo" into something with real
numbers behind it.

1. Ingest a few real papers first (via the frontend upload, or `POST /upload`).
2. Replace the placeholder questions in `data/eval_set.json` with real
   questions you know the answer to from those papers.
3. Run:
   ```bash
   cd backend
   python eval_harness.py
   ```
4. Read the printed summary table and check `data/eval_results.csv` for the
   full breakdown. This gives you real, defensible numbers like:
   - accuracy proxy (naive vs dense-only vs hybrid+verify)
   - average latency per pipeline
   - average groundedness score from the verifier

Put these numbers in your README and resume line instead of vague claims.

## What to build next, in order

1. Get the base pipeline running end-to-end with 2-3 real papers.
2. Build out `data/eval_set.json` to at least 20-30 real questions.
3. Run `eval_harness.py`, save the comparison table — this is your strongest
   talking point in interviews.
4. Only after that works: consider adding the LangGraph planner/writer
   agent split described in the architecture diagram. Don't add it before
   the core pipeline is measured and solid — an agent wrapper around a
   pipeline you haven't evaluated just adds surface area to defend.

## Notes on scope

This intentionally does NOT include Kubernetes, RBAC, multi-tenancy, or a
built React app. Those are easy to add once the core pipeline is solid, but
adding them first means you end up with a lot of surface area you can't
speak to under questioning. Get the retrieval + verification + evaluation
loop right first — that's the part that's actually rare.
