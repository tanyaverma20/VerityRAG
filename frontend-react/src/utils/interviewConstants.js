// Fixed option lists ported verbatim from the original vanilla frontend
// (frontend/index.html: PROJECT_INTERVIEW_TOPICS / WHY_DESIGN_QUESTIONS /
// SYSTEM_DESIGN_QUESTIONS) so Project Interview's topic picker and the Why
// This Design?/System Design pickers show the exact same real options.

export const PROJECT_INTERVIEW_TOPICS = [
  "Architecture", "Methodology / Approach", "ML Models / Algorithms", "Dataset",
  "Features", "Pipeline", "Backend", "AI Components", "Results",
  "Design Decisions", "Limitations"
];

export const WHY_DESIGN_QUESTIONS = [
  "Why ChromaDB?", "Why Dense + BM25?", "Why RRF?", "Why Cross-Encoder reranking?",
  "Why one LLM call?", "Why Groq/Llama?", "Why LangGraph?", "Why Pydantic?",
  "Why document_id scoping?", "Why token budgeting?", "Why a fallback model?", "Why caching?"
];

export const SYSTEM_DESIGN_QUESTIONS = [
  "How would you scale to millions of PDFs?", "How would you support many concurrent users?",
  "How would you implement multi-tenancy?", "How would you handle vector DB failure?",
  "How would you reduce latency?", "How would you handle huge PDFs?",
  "How would you improve retrieval quality?", "How would you evaluate the RAG system?"
];
