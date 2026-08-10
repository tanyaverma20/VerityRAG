# VerityRAG — React Frontend (in progress)

A real, incrementally-built React + TypeScript + Vite replacement for
`frontend/index.html` (the original ~3,100-line single-file vanilla-JS
app). This is **not yet feature-complete** — see "What's covered" below —
and the original frontend remains the one actually served/relied on until
parity is demonstrated and this directory is explicitly promoted.

## What's real and working here

Everything below has been built against the actual FastAPI backend (not
mocked) and manually verified end-to-end in a live browser session against
a real Postgres-backed, Redis-cached backend instance:

- **Typed API client** (`src/api/client.ts`, `src/api/types.ts`) covering
  every real endpoint the old frontend calls: workspaces, documents
  (list/get/delete/upload), sessions (list/create/rename/delete/messages),
  `/query`, `/analyze`, `/report` (+ download URLs), `/eval/dashboard`.
- **Workspace management** — list, switch (persisted via `localStorage`,
  same pattern as the old frontend), create.
- **Document upload/list/delete**, scoped per workspace exactly like the
  original (`workspace_id` threaded through every call).
- **Conversation list** per workspace, new/select/delete, with real
  message-history restore from `GET /sessions/{id}/messages`.
- **Normal Q&A chat** (`/query`) — the core product loop: ask a question
  scoped to the workspace's INDEXED documents, render the grounded answer,
  confidence pill, and de-duplicated citation pills; correctly renders a
  real backend error state (e.g. rate-limiting) without crashing.
- **23 passing unit/component tests** (Vitest + React Testing Library):
  every API client function, empty-state logic, document-readiness
  filtering, submit/disable behavior, citation dedup — see `npm test`.
- **Clean production build** (`npm run build` → `tsc -b && vite build`,
  zero errors) and a working dev server (`npm run dev`).

## What is NOT yet migrated (honest gap list)

The original frontend has ~90 functions covering a much larger surface.
These remain **only in `frontend/index.html`** for now:

- Deep Research mode (async `/research` + task polling)
- Viva / Mock Test question generation
- Project Interview (start/evaluate loop)
- Why This Design? / System Design (non-PDF-grounded modes)
- Explain Figure (incl. the vision-model image-rendering path)
- Evaluate Paper, Research Gaps, Literature Matrix, Knowledge Graph cards
- Comparative Report generation + rendering + PDF/DOCX/Markdown download
- Compare / Recommend views
- Eval Dashboard UI (the backend endpoint is in the typed client already;
  no screen renders it yet)
- The "+" composer menu that groups all of the above
- Drag-and-drop upload (click-to-upload works; drag zone does not yet)
- Document-scope pill picker for a specific question (chat currently
  always scopes to all INDEXED documents in the active workspace)

## Architecture

```
src/
  api/         typed fetch client + response/request interfaces
  hooks/       useWorkspace (workspace+documents+sessions), useChat (messages+send)
  components/  Sidebar, ChatWindow, MessageBubble
  App.tsx      wires hooks + components into the app shell
```

State management is plain React hooks (`useState`/`useCallback`) — no
external state library. This was a deliberate choice for the current
scope; if/when the remaining analysis-mode UIs are added, revisit whether
a shared context or a library like Zustand pulls its weight.

## Running it

```bash
cd frontend-react
npm install
npm run dev      # http://localhost:5173, expects the backend on :8001
npm run build    # production build → dist/
npm test         # Vitest, 23 tests
```

Set `VITE_API_BASE` in `.env` (defaults to `http://127.0.0.1:8001`) if the
backend runs elsewhere.

## Migration plan

Per the project's explicit migration policy: **the old `frontend/`
remains the served, relied-upon frontend until this one reaches real
feature parity and is explicitly verified against the real backend for
every major flow** — not deleted or defaulted-to preemptively. Each
remaining item above is a self-contained, addable slice (its own
component + hook + tests) rather than a rewrite of what already exists
here.
