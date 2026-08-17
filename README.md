# VerityRAG

Production-hardened multi-tenant AI research platform for grounded, citation-aware analysis of scientific literature.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19.0-61DAFB?style=flat&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-8.2-646CFF?style=flat&logo=vite&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-4169E1?style=flat&logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7.0-DC382D?style=flat&logo=redis&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F61?style=flat)

---

## Overview

**VerityRAG** is an evidence-grounded research intelligence system designed for researchers, analysts, and domain experts who require rigorous, verifiable insights from scientific literature.

### The Problem
Traditional document Q&A tools and naive RAG implementations suffer from severe structural limitations:
- **Hallucinations & Unsupported Claims**: Synthesizing plausible-sounding answers without explicit source grounding or verifiable evidence traces.
- **Lost Context in Retrieval**: Truncated chunking losing section-level context around matched text fragments.
- **Single-Document Bottlenecks**: Inability to synthesize cross-paper literature matrices or identify underlying research gaps.
- **Security & Multi-Tenant Data Leakage**: Inadequate workspace and vector isolation exposing sensitive documents across user boundaries.

### The VerityRAG Solution
VerityRAG addresses these challenges by combining a hybrid multi-stage retrieval architecture (dense vector search, BM25 keyword matching, Reciprocal Rank Fusion, Cross-Encoder reranking, and parent-context expansion) with strict server-side tenant isolation, structured analytical workflows, multimodal figure inspection, and continuous observability.

---

## Key Features

### Research & RAG
- **Grounded Document Q&A**: Citation-backed answer generation using direct source excerpts with de-duplicated evidence metadata.
- **Hybrid Retrieval**: Merges dense vector semantic search with sparse BM25 keyword matching via Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking**: Re-scores candidate passages using deep relevance cross-encoders (`ms-marco-MiniLM-L-6-v2`).
- **Parent-Context Expansion**: Dynamic retrieval expansion preserving section-level context around matched text chunks.
- **Multi-Document Synthesis**: Synthesizes comparative insights across multiple papers concurrently.

### Research Workflows
- **Deep Research**: Multi-pass research mode (`research_type: "deep"`) for complex technical queries.
- **Research Gaps**: Identifies explicit author-acknowledged limitations alongside inferred methodology and dataset gaps.
- **Literature Matrix**: Constructs a side-by-side comparative table summarizing objectives, methods, datasets, and findings.
- **Knowledge Graph**: Maps key concepts, entities, and relationship statements into interactive graph tags.
- **Comparative Reports**: Synthesizes multi-paper comparative analysis reports in Markdown, PDF, and DOCX formats.
- **Paper Evaluation**: 7-dimension paper critique assessing methodology, claims, and ground truth alignment.

### Learning / Interview
- **Viva & Mock Test**: Quiz question generation with custom difficulty and question counts.
- **Project Interview**: Interactive interview simulator covering 11 domain topics with real-time technical depth feedback.

### Document Intelligence
- **PDF Extraction**: Structure-preserving layout parsing using PyMuPDF (`fitz`).
- **Figure & Page Rendering**: High-resolution page image generation for visual analysis.
- **Vision-Based Figure Analysis (Explain Figure)**: Multimodal page inspection via Groq vision models (`qwen/qwen3.6-27b`), with graceful text fallback.
- **OCR Fallback**: Scanned PDF page detection with non-intrusive fallback handling when system Tesseract binary is available.

### Platform Infrastructure
- **Authentication**: Server-side user registration, password hashing (`bcrypt`), SHA-256 session token storage, 24h expiration, and revocation.
- **Authorization & Isolation**: Server-side workspace and resource ownership enforcement; mandatory vector metadata filtering.
- **Persistence & Caching**: PostgreSQL 18 relational storage with Alembic migrations; Redis response caching with automatic in-memory fallback during outages.
- **Observability**: Live telemetry logging (`logs/verityrag_events.jsonl`) and real-time metrics dashboard (`/eval/dashboard`).

---

## Architecture

```mermaid
flowchart TD
    User([User / Browser]) <--> ReactApp[React 19 + Vite Frontend]
    ReactApp <-->|REST API + Bearer Auth| FastAPI[FastAPI Server]
    
    subgraph Security & Auth
        FastAPI --> Authz[_require_workspace_owner / _require_resource_owner]
        Authz --> SessionStore[PostgreSQL Sessions & Passwords]
    end

    subgraph Application Services
        FastAPI --> RAGPipe[RAG Pipeline]
        FastAPI --> Workflows[Research Workflows]
        FastAPI --> DocIntel[Document Intelligence]
    end

    subgraph RAG Pipeline Details
        RAGPipe --> QueryProc[Query Processing & Expansion]
        QueryProc --> DenseSearch[Dense Vector Search - MiniLM-L6-v2]
        QueryProc --> SparseSearch[Sparse BM25 Keyword Search]
        DenseSearch & SparseSearch --> RRF[Reciprocal Rank Fusion - RRF]
        RRF --> Reranker[Cross-Encoder Reranker - MS MARCO]
        Reranker --> ParentExp[Parent-Context Expansion]
        ParentExp --> Evidence[Evidence Trace & Prompt Assembly]
    end

    subgraph Research Workflows
        Workflows --> DeepResearch[Deep Research]
        Workflows --> ResearchGaps[Research Gap Discovery]
        Workflows --> LitMatrix[Literature Matrix]
        Workflows --> KnowGraph[Knowledge Graph]
        Workflows --> CompReport[Comparative Reports]
        Workflows --> PaperEval[Paper Evaluation]
    end

    subgraph Document Intelligence
        DocIntel --> PDFExtract[PDF Parsing & Layout Extraction]
        DocIntel --> VisionInspect[Groq Vision Model - Qwen 3.6 27B]
        DocIntel --> OCRFallback[Deterministic OCR Fallback Engine]
    end

    subgraph Storage & External Services
        SessionStore & Workflows --> Postgres[(PostgreSQL 18)]
        FastAPI <-->|Report & Query Caching| Redis[(Redis Cache)]
        DenseSearch <-->|Workspace-Scoped Metadata| Chroma[(ChromaDB Vector Store)]
        Evidence & VisionInspect <-->|LLM Inference| Groq[Groq API - Llama 3.3 70B / Qwen 27B Vision]
    end
```

---

## RAG Pipeline

The VerityRAG retrieval and generation pipeline follows a 15-stage architecture designed for strict evidence grounding:

### Ingestion Stages
1. **Document Ingestion**: PDF documents uploaded via `/upload` are received and verified.
2. **Validation**: Enforces `%PDF-` magic-byte checking and streaming 50 MB file size limits.
3. **Parsing**: Extracts text, page numbers, and document structure using PyMuPDF (`fitz`).
4. **Chunking**: Splits document text into 500-token chunks with 50-token overlap.
5. **Metadata Assignment**: Tags each chunk with `workspace_id`, `document_id`, `page_number`, and `chunk_id`.
6. **Embedding Generation**: Vectorizes text chunks using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
7. **ChromaDB Indexing**: Stores vector embeddings and metadata in local persistent ChromaDB collections.

### Query & Generation Stages
1. **Query Processing**: Input queries are sanitized, normalized, and expanded.
2. **Dense Retrieval**: Retrieves top candidate chunks from ChromaDB filtered strictly by `workspace_id` and `document_id`.
3. **BM25 Sparse Search**: Evaluates lexical keyword relevance over candidate chunks using an in-memory BM25 index.
4. **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse candidate rankings using standard RRF scoring ($k=60$).
5. **Cross-Encoder Reranking**: Re-scores fused candidate passages using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
6. **Parent-Context Expansion**: Dynamically expands matched chunks to include adjacent surrounding page context.
7. **Evidence Construction**: Formats top reranked context blocks into structured, citation-indexed context payloads.
8. **LLM Generation**: Submits evidence payload to `llama-3.3-70b-versatile` via Groq API.
9. **Citation & Response Assembly**: Assembles answer with inline bracketed citations (`[1]`, `[2]`), confidence metadata, and evidence traces.

> **LLM Efficiency**: Standard Q&A queries execute in exactly **1 physical LLM call** under normal operations, minimizing latency and token overhead.

---

## Security & Multi-Tenancy

Security and multi-tenant isolation are enforced at every layer of the backend application:

- **Authentication & Passwords**: User registration (`/auth/register`) and login (`/auth/login`) store passwords hashed via `bcrypt` with unique per-user salts.
- **Session Security**: Generates 256-bit cryptographically secure session tokens (`secrets.token_urlsafe`), stored as SHA-256 hashes with 24-hour expiration and server-side revocation (`/auth/logout`).
- **Authorization Enforcement**: Every protected route executes explicit `_require_workspace_owner()` or `_require_resource_owner()` checks against the authenticated user token.
- **Resource Scoping Guarantee**: **`workspace_id` is treated as a resource scope, not an authentication principal.** Client-supplied identifiers are never trusted without server-side verification against the authenticated user.
- **Vector Metadata Filtering**: ChromaDB queries enforce mandatory, non-bypassable `workspace_id` and `document_id` metadata filters preventing cross-tenant vector contamination.
- **Cache Key Isolation**: Redis keys are scoped with isolated prefixes (`verityrag:report:...`).
- **Upload Hardening**: Enforces `%PDF-` magic-byte verification, strict filename sanitization (`secure_filename`), 50 MB streaming caps, and automatic temporary file cleanup.
- **Path Traversal Protection**: Uses strict `Path.resolve()` boundary checks preventing directory traversal attacks.
- **CORS Hardening**: Explicit origin configuration (`CORS_ALLOWED_ORIGINS`) with total removal of wildcard `*` origins.
- **Adversarial Verification**: Validated by 25 dedicated cross-user isolation and authorization security tests.

---

## Tech Stack

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Frontend** | React 19 | Modular component UI architecture |
| **Language** | Vanilla JavaScript (`.jsx`/`.js`) | Frontend client application logic (no TypeScript build step) |
| **Build Tool** | Vite 8 | Development server & production bundler |
| **Backend** | Python 3.13 | Backend service implementation |
| **API Layer** | FastAPI | Async REST API framework |
| **ORM** | SQLAlchemy 2.0 | PostgreSQL database access & connection pooling |
| **Migrations** | Alembic | Relational schema versioning |
| **Database** | PostgreSQL 18 | Persistent relational multi-tenant storage |
| **Cache** | Redis | Report caching & outage-resilient fallback |
| **Vector DB** | ChromaDB | Local vector storage & metadata retrieval |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 384-dimensional dense text embeddings |
| **Sparse Search** | BM25 (`rank_bm25`) | Lexical keyword retrieval engine |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Dense + sparse rank merging |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Deep relevance reranking |
| **LLM Inference** | Groq (`llama-3.3-70b-versatile`) | Grounded answer generation |
| **Vision Inference** | Groq (`qwen/qwen3.6-27b`) | Multimodal figure & page inspection |
| **PDF Parser** | PyMuPDF (`fitz`) / `pypdf` | PDF text extraction & page image rendering |
| **OCR (Optional)** | Tesseract OCR | Scanned PDF page fallback engine |
| **Backend Testing** | `pytest` / `httpx` | Backend unit, integration, & security tests |
| **Frontend Testing** | Vitest / React Testing Library | Component, hook, & API client tests |

---

## Project Structure

```
verityrag/
├── backend/
│   ├── alembic/                Database schema migration scripts
│   ├── db/                     SQLAlchemy models, repository pattern, session setup
│   ├── graph/                  LangGraph / workflow analysis engines
│   ├── tests/                  Fixture files and test resources
│   ├── analysis.py             Analysis mode execution handlers
│   ├── auth.py                 Authentication & session token management
│   ├── cache.py                Redis cache wrapper with in-memory fallback
│   ├── config.py               Environment variables and application settings
│   ├── database.py             SQLite/PostgreSQL database initialization helpers
│   ├── figure_vision.py        PDF page rendering and Groq vision pipeline
│   ├── groundedness_eval.py    Offline groundedness evaluation harness
│   ├── ingest.py               Document parsing, chunking, and ChromaDB vector ingestion
│   ├── main.py                 FastAPI application, routes, and ownership middleware
│   ├── observability.py        Structured telemetry logger (`verityrag_events.jsonl`)
│   ├── ocr_fallback.py         Text insufficiency detection and OCR fallback
│   ├── query_transform.py      Query decomposition and transformation handlers
│   ├── report_generator.py     Comparative report generation (MD, PDF, DOCX)
│   ├── retrieval.py            Dense search, BM25, RRF, reranking, parent-context expansion
│   ├── schemas.py              Pydantic request and response models
│   └── test_*.py               422 passing backend test modules
├── frontend-react/
│   ├── public/                 Static public assets (`icons.svg`)
│   ├── src/
│   │   ├── api/               Fetch API client (`client.js`)
│   │   ├── components/        React UI components (Sidebar, ChatWindow, etc.)
│   │   ├── hooks/             Custom React hooks (`useAuth`, `useChat`, `useWorkspace`)
│   │   ├── utils/             Scoping and constant utilities
│   │   ├── App.jsx            Main application container
│   │   └── main.jsx           React application entry point
│   ├── package.json           Frontend dependencies and scripts
│   └── vite.config.js         Vite build and server configuration
├── evaluation/
│   ├── run_benchmark.py       10-document isolated retrieval benchmark runner
│   └── results.json           Verified retrieval benchmark outputs
├── data/                       Local storage for uploaded document files
├── logs/                       Structured JSONL telemetry event logs
├── .env                        Environment configuration file
└── README.md                   Root project documentation
```

---

## Research Workflows

- **Deep Research**: Multi-pass research mode (`research_type: "deep"`) synthesizing broader document context across query variations.
- **Research Gaps**: Identifies explicit author-acknowledged limitations alongside inferred methodology and dataset gaps.
- **Literature Matrix**: Constructs a side-by-side comparative table summarizing objectives, methods, datasets, and findings across papers.
- **Knowledge Graph**: Extracts key concepts, entity definitions, and relationship triplets rendered as visual tags and relationship sentences.
- **Comparative Reports**: Synthesizes comprehensive multi-paper analysis reports downloadable in Markdown, PDF, and DOCX formats.
- **Evaluate Paper**: Generates a 7-dimension paper critique assessing research question clarity, methodology rigor, evidence support, and limitations.
- **Explain Figure**: Renders the targeted PDF page as a high-resolution PNG image, submitting it to Groq vision models (`qwen/qwen3.6-27b`) for visual analysis, with automatic fallback to text captions.
- **Viva / Mock Test**: Generates oral examination questions or multiple-choice mock tests with customizable question counts and difficulty tiers.
- **Project Interview**: Interactive interview simulator featuring 11 domain topics (e.g., System Architecture, Data Pipelines, RAG Tradeoffs) with real-time feedback on technical depth and candidate answers.

---

## Data & Infrastructure

### PostgreSQL 18
Serves as the primary relational persistence store managed via SQLAlchemy ORM and Alembic migrations. Stores:
- User accounts and password hashes (`users`)
- Active session tokens (`sessions`)
- Workspaces and document metadata (`workspaces`, `documents`)
- Conversation histories and analysis records (`chat_sessions`, `chat_messages`)

### Redis
Provides high-performance caching for generated reports and query results. Features:
- Automatic key expiration (TTL)
- Scoped cache key prefixes
- **Outage Resilience**: Automatic transparent fallback to in-memory dictionary cache if Redis connection fails or drops.

### ChromaDB
Local persistent vector store storing 384-dimensional chunk embeddings along with document and workspace scoping metadata.

### Alembic
Handles database schema versioning and zero-downtime database migrations (`backend/alembic/`).

---

## Evaluation & Benchmarks

Retrieval performance and system accuracy are verified through automated benchmark runs:

| Evaluation Metric | Benchmark Result | Measurement Source |
| :--- | :---: | :--- |
| **Benchmark Dataset** | **10 Documents** | `evaluation/run_benchmark.py` |
| **Recall@3** | **1.0** | Hybrid RAG benchmark evaluation |
| **Mean Reciprocal Rank (MRR)** | **1.0** | Hybrid RAG benchmark evaluation |
| **Mean Retrieval Latency** | **111.65 ms** | Benchmark timing log (`evaluation/results.json`) |
| **Cross-Document Isolation Violations** | **0** | Verified cross-tenant vector isolation suite |
| **Backend Test Pass Rate** | **422 passed / 10 skipped** | `pytest --ignore=temp_chroma` (0 failures) |
| **Frontend Test Pass Rate** | **113 passed** | `npm test` in `frontend-react` (0 failures) |
| **Security Isolation Test Pass Rate** | **25 passed** | `pytest test_adversarial_isolation.py` |
| **Live Postgres / Redis Integration** | **6 passed** | `pytest test_postgres_live.py test_redis_live.py` |
| **Frontend Production Build** | **PASS** | `npm run build` (0 compilation errors) |

### Methodology & Integrity Guarantee
- Benchmark evaluations run in isolated vector collections to ensure zero cross-document contamination.
- Groundedness evaluation operates offline via `groundedness_eval.py` to prevent injecting secondary LLM overhead into live user requests.
- Claims of universal perfection or instant latency are explicitly avoided; Recall@3 = 1.0 reflects performance on the benchmark evaluation dataset and does not imply universally zero errors across arbitrary out-of-domain text.

---

## Testing

Execute full automated verification across all project layers using these exact commands:

### 1. Backend Test Suite
```bash
cd backend
pytest --ignore=temp_chroma
```

### 2. Security & Multi-Tenancy Isolation Suite
```bash
cd backend
pytest test_adversarial_isolation.py -v
```

### 3. Live PostgreSQL & Redis Integration Tests
```bash
cd backend
pytest test_postgres_live.py test_redis_live.py -v
```

### 4. Frontend Component & Hook Tests
```bash
cd frontend-react
npm test -- --run
```

### 5. Frontend Production Build Check
```bash
cd frontend-react
npm run build
```

### 6. Retrieval Benchmark Suite
```bash
cd evaluation
python run_benchmark.py
```

---

## Setup

### Prerequisites
- **Python 3.11+**
- **Node.js 18+** and **npm**
- **PostgreSQL 18** (Optional; falls back to SQLite for local development)
- **Redis** (Optional; falls back to in-memory dict for local development)

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tanyaverma20/VerityRAG.git
   cd VerityRAG
   ```

2. **Create Python virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install backend dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory (or update `backend/.env`):
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_VISION_MODEL=qwen/qwen3.6-27b
   DATABASE_URL=postgresql://postgres:password@localhost:5432/verityrag
   REDIS_URL=redis://localhost:6379/0
   CORS_ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
   ```

5. **Run database migrations (PostgreSQL)**:
   ```bash
   cd backend
   alembic upgrade head
   cd ..
   ```

6. **Install frontend dependencies**:
   ```bash
   cd frontend-react
   npm install
   cd ..
   ```

---

## Environment Variables

| Variable Name | Required / Optional | Description |
| :--- | :---: | :--- |
| `GROQ_API_KEY` | **Required** | API key for Groq LLM and Vision inference calls |
| `GROQ_VISION_MODEL` | Optional | Vision model ID for Explain Figure mode (default: `qwen/qwen3.6-27b`) |
| `DATABASE_URL` | Optional | PostgreSQL connection string (falls back to SQLite `data/registry.db`) |
| `REDIS_URL` | Optional | Redis server URL (falls back to in-memory dictionary cache) |
| `CORS_ALLOWED_ORIGINS` | Optional | Comma-separated CORS allowed origin URLs |
| `CHROMA_DIR` | Optional | Directory path for local ChromaDB storage |
| `OCR_ENABLED` | Optional | Enable or disable OCR processing fallback (`true`/`false`) |

---

## Running the Application

### Start Backend API Server
```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```
The FastAPI backend will run at: `http://127.0.0.1:8001` (Swagger API docs at `http://127.0.0.1:8001/docs`).

### Start Frontend Application
```bash
cd frontend-react
npm run dev
```
The canonical React frontend will launch at: **`http://localhost:5173`**

---

## Observability

VerityRAG records detailed telemetry for system health, latency, and resource utilization:

- **Telemetry Logger (`observability.py`)**: Appends structured JSON logs to `logs/verityrag_events.jsonl` containing request durations, LLM token counts, retrieval candidates, reranking latency, and cache outcomes.
- **Metrics Dashboard (`/eval/dashboard`)**: Aggregates live runtime metrics:
  - Total logged requests
  - Average LLM calls per query (nominal baseline: 1.01)
  - Average total request latency (retrieval + reranking + LLM synthesis)
  - Cache hit and fallback rates
  - Average prompt and response token counts
  - Reranking Recall@5 delta improvement

---

## Limitations

- **OCR System Binary Dependency**: Real Tesseract OCR text recovery was not verified in the development environment because the system Tesseract binary was intentionally not installed. The application detects insufficient extracted text, supports an OCR fallback when Tesseract is available, and degrades gracefully when it is unavailable.

---

## Future Improvements

- Production deployment manifests for cloud deployment (Kubernetes / Cloud Run).
- Managed multi-node vector database migration (e.g., Qdrant / Milvus) for horizontal scale.
- Streaming token-by-token response generation over Server-Sent Events (SSE) or WebSockets.
- Optional pre-packaged Tesseract OCR container integration for zero-setup scanned document handling.

---

## Resume-Ready Highlights

- **Built a Production-Grade Multi-Tenant RAG Platform**: Architected a hybrid retrieval pipeline using dense vector embeddings, BM25 keyword matching, Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking, achieving **1.0 Recall@3** and **1.0 MRR** on benchmark evaluation datasets with **111.65 ms** mean retrieval latency.
- **Hardened SaaS Multi-Tenancy & Security**: Engineered server-side user authentication, SHA-256 session token hashing, and strict workspace authorization, verified by **25 automated adversarial security tests** preventing cross-tenant vector data leakage.
- **Designed Resilient Micro-Infrastructure**: Built PostgreSQL persistence with SQLAlchemy ORM and Alembic migrations, alongside a Redis response cache featuring transparent automatic fallback to in-memory storage during Redis outages.
- **Delivered Multimodal Document Intelligence**: Created visual figure inspection using PyMuPDF rendering and Groq vision models (`qwen/qwen3.6-27b`), backed by a comprehensive suite of **422 backend pytest cases** and **113 frontend Vitest cases**.
