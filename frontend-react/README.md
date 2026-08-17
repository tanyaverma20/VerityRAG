# VerityRAG — React Frontend

The canonical, production-ready frontend for the VerityRAG research platform, built with **React 19**, **Vanilla JavaScript** (`.jsx`/`.js`, no TypeScript), and **Vite**.

It replaces the legacy single-file static interface with a modular component architecture, full authentication flow, workspace management, document scoping, and comprehensive research analysis interfaces.

---

## Features

- **Authentication & Security**: Complete user registration, login, token management (`Bearer` authentication header), and server-side session lifecycle integration (`src/hooks/useAuth.js`).
- **Workspace & Document Management**: Workspace switching, creation, and document upload/deletion scoped strictly to the authenticated owner (`src/hooks/useWorkspace.js`).
- **Document-Scoped Retrieval**: Automatic, explicit, and click-to-select document scoping (`src/utils/scope.js`) for targeted paper analysis.
- **Normal Q&A Chat & Deep Research**: Grounded Q&A with citation metadata, confidence badges, and synchronous Deep Research mode toggle.
- **Structured Research Analysis**:
  - **Research Gaps**: Inferred vs. author-stated research gap extraction.
  - **Literature Matrix**: Side-by-side comparative HTML table for multi-document review.
  - **Knowledge Graph**: Entity/concept relationship mapping.
  - **Evaluate Paper**: 7-dimension paper critique with support evidence badges.
  - **Explain Figure**: Multimodal figure/table vision analysis with graceful text fallback.
  - **Comparative Reports**: Unified multi-paper comparative synthesis (`/report`).
- **Learning & Interview Suite**:
  - **Viva & Mock Test**: Quiz question generation with custom difficulty and question counts.
  - **Project Interview**: Interactive multi-turn interview loop with 11-topic setup steering.
  - **Why This Design? & System Design**: Architecture question lists grounded in platform design decisions.
- **Eval Dashboard**: Live telemetry metrics (LLM latency, cache hit rate, token counts) and offline benchmark scores.

---

## Architecture

```
frontend-react/src/
├── api/
│   └── client.js              Fetch API client for FastAPI backend endpoints
├── components/
│   ├── AnalysisResultCard.jsx Card dispatcher for structured analysis modes
│   ├── AuthScreen.jsx         User login and registration interface
│   ├── ChatWindow.jsx         Main chat stream, composer, and Deep Research banner
│   ├── ComposerMenu.jsx       Actions menu (+) grouping analysis tools
│   ├── EvalDashboard.jsx      Live telemetry and evaluation metric dashboard modal
│   ├── MessageBubble.jsx      Chat message bubble (Q&A, citations, status)
│   ├── SetupModal.jsx         Configuration modal for interview/quiz/figure/design modes
│   └── Sidebar.jsx            Workspace switcher, document list, and chat history
├── hooks/
│   ├── useAuth.js             Authentication & token state management
│   ├── useChat.js             Chat session, query execution, and analysis state
│   └── useWorkspace.js        Workspaces, document uploads, and session persistence
├── utils/
│   ├── interviewConstants.js Fixed topic and design question constants
│   └── scope.js               Document scope resolution logic
└── test/
    └── setup.js               Vitest & Testing Library environment setup
```

---

## API Integration

The frontend connects to the FastAPI backend via `src/api/client.js`. All authenticated requests automatically include the user's opaque session token in the `Authorization: Bearer <token>` header.

Key endpoint mappings:
- **Auth**: `POST /auth/register`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`
- **Workspaces**: `GET /workspaces`, `POST /workspaces`, `GET /workspaces/{id}`
- **Documents**: `GET /documents`, `POST /upload`, `DELETE /documents/{id}`
- **Q&A & Research**: `POST /query`, `POST /analyze`, `POST /report`
- **Sessions**: `GET /sessions`, `POST /sessions`, `GET /sessions/{id}/messages`
- **Dashboard**: `GET /eval/dashboard`

---

## Development & Testing

### Running Locally
```bash
# Install dependencies
npm install

# Start Vite development server (http://localhost:5173)
npm run dev
```

### Running Test Suite
```bash
# Execute Vitest test suite (113 passing tests)
npm test
```

### Production Build
```bash
# Compile client bundle for production (output to dist/)
npm run build
```

---

## Canonical Status

`frontend-react` is configured as the default served frontend application across dev and production environments. Legacy static files (`frontend/index.html`) remain solely as historical reference artifacts.
