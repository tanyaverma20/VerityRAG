"""
analysis.py — Interview / design-defense / figure-explanation modes.

Every function here makes exactly ONE physical LLM call (via
query_transform._call_groq_raw, which already carries the one-fallback-on-
genuine-failure contract) — no separate verifier call, no chain of calls.
PDF-grounded modes (viva, mock test, explain-figure, recommend) are handed
already-retrieved, already-reranked evidence text by the caller (main.py) —
this module does not perform retrieval itself, so the existing hybrid
retrieval/rerank/token-budget pipeline is reused unmodified.

Project-Interview / Why-This-Design / System-Design modes are NOT grounded
in retrieved PDF evidence — there is nothing to retrieve; they're grounded
in REAL_ARCHITECTURE_CONTEXT below, a factual description of VerityRAG's
actual implementation (kept honest and current — see the file references in
each line). This is the mechanism that satisfies "do NOT invent architecture
that does not exist": the model is only ever asked to reason about what's
written here, never to freelance about a hypothetical RAG system.
"""
import json
import re

from query_transform import _call_groq_raw, INSUFFICIENT_EVIDENCE_MESSAGE
from schemas import (
    AnswerJSON, QuestionSetJSON, AnswerEvaluationJSON,
    PaperEvaluationJSON, ResearchGapsJSON, LiteratureMatrixJSON, KnowledgeGraphJSON,
)
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Ground truth for non-PDF modes — a factual snapshot of the real, current
# implementation. Update this alongside any real architecture change; the
# model is instructed to reason ONLY from these facts.
# ---------------------------------------------------------------------------
REAL_ARCHITECTURE_CONTEXT = """
VERITYRAG — ACTUAL SYSTEM ARCHITECTURE (ground truth; do not invent beyond this)

Backend: FastAPI (main.py). PDF parsing via pypdf (extract_pages_from_pdf),
page/section-aware paragraph chunking (ingest.py: chunk_page), CHUNK_SIZE=800
chars, CHUNK_OVERLAP=100 chars, deterministic chunk_id/parent_id per page.
document_id = sha256(file bytes)[:16] — content-addressed, so re-uploading
identical bytes reuses the same ID instead of duplicating.

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim), wrapped as a
ChromaDB EmbeddingFunction (ingest.py).

Vector store: a single persistent ChromaDB collection ("verityrag_docs_v2"
by default) at backend/chroma_store. There is NOT a separate Chroma
collection per workspace/user — isolation is enforced by document_id
metadata filtering at query time (Chroma `where` clauses) plus a SQLite
registry (database.py) that tracks which workspace_id each document_id
belongs to; /query cross-checks requested document_ids against that
registry before retrieval ever runs (main.py: _resolve_document_scope /
documents_in_workspace).

Retrieval (retrieval.py), zero LLM calls:
  1. dense_search() — Chroma nearest-neighbor query, document_id-filtered.
  2. bm25_search() — rank_bm25 (BM25Okapi) over an in-memory index rebuilt
     after every ingest/delete (build_bm25_index), also document_id-filtered.
  3. reciprocal_rank_fusion() — RRF(d) = sum(1/(k+rank)), k=60 (RRF_K),
     merges the dense and BM25 ranked lists without needing score
     normalization.
  4. rerank() — a local CrossEncoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
     scores the fused candidate pool; a real model forward pass, but never
     an LLM call.
  5. expand_with_parent_context() — attaches each chunk's full parent-page
     text alongside the precise child chunk.
  6. deduplicate_chunks(), then select_within_token_budget() — greedy
     selection under MAX_CONTEXT_TOKENS (3000 by default) with a per-document
     cap (MAX_CHUNKS_PER_DOC=3) so one document can't crowd out the others in
     a multi-paper question.

Deterministic query decomposition (query_transform.py:
decompose_query_deterministic) — zero LLM calls. Only splits a question into
multiple sub-queries when it BOTH matches a comparison/aggregation signal
AND contains a comma/"and"-joined list of recognised research-paper aspect
nouns (methodology, datasets, results, ...); otherwise the question is used
verbatim as a single search query. Each sub-query's dense+BM25 results are
fused with ONE global RRF pass and reranked ONCE against the original
question (retrieval.py: retrieve_multi) — never once per sub-query.

Orchestration: LangGraph (graph/workflow.py). NORMAL mode runs
plan -> retrieve -> organize -> synthesize -> assign_confidence, where plan/
retrieve/organize/assign_confidence are all pure Python (zero LLM calls) and
synthesize is the ONE LLM call. DEEP RESEARCH mode (explicitly opted into)
adds an LLM-backed planner, an adaptive retrieval loop, cross-paper
analysis, and a separate verification pass with one bounded retry — all
capped by DEEP_RESEARCH_MAX_ITERATIONS / _MAX_LLM_CALLS / _MAX_RETRIES so it
can never run unbounded.

Generation: Groq API. Primary model llama-3.3-70b-versatile, fallback
openai/gpt-oss-120b. with_model_fallback() (query_transform.py) tries the
primary once; ONLY on a classified-temporary failure (rate limit, 5xx/
timeout, or the primary model itself being decommissioned) does it retry
ONCE against the fallback model — never for auth errors or malformed
requests. A normal successful query is exactly ONE physical LLM call, two
only when the primary genuinely failed.

Output contract: the model is required to return one compact JSON object
(schemas.py: AnswerJSON — answer/grounded/evidence_sufficient/document_ids),
Pydantic-validated. A malformed/unparseable response is treated as a clean
synthesis failure — never a reason to retry the SAME model, never a reason
to invent data to satisfy the schema. The model self-reports
grounded/evidence_sufficient in that SAME call; there is no separate
verifier call in normal mode.

Grounding: the synthesis prompt instructs the model to answer ONLY from the
evidence block built from retrieved chunks, to say so explicitly when
evidence is insufficient, and never to fill gaps from outside knowledge.

Token efficiency: MAX_CONTEXT_TOKENS=3000 (input budget), MAX_ANSWER_TOKENS
=1024 (output budget, higher for the fallback reasoning model which spends
part of its budget on internal reasoning tokens), MAX_CHUNKS_PER_DOC=3,
RERANK_TOP_K=5 final chunks. The full PDF text is never sent to the model —
only the reranked, budgeted evidence.

Caching (cache.py): an in-memory cache keyed by normalized question +
sorted document_ids + research_type + a fingerprint of the last 4 chat
messages + the current retrieval-config values. A cache hit costs zero LLM
calls and zero retrieval. Any document upload or removal clears the cache
entirely so a stale answer can never survive a workspace change.

Observability (observability.py): every request appends one structured JSON
line to logs/verityrag_events.jsonl — llm_calls, fallback_triggered, model
name, input/output/total tokens, cache_hit, latency, confidence — read back
via read_recent_events()/filter_events(). Never exposed in the normal answer
UI.

Testing: FakeGroqClient-style tests that patch groq.Groq at the network
boundary (unittest.mock.patch) so exact physical call counts are proven, not
assumed (test_llm_call_count.py, test_query_decomposition.py,
test_workspace_isolation.py, test_natural_qa.py). Real-Chroma tests run only
inside an isolated Phase-8 fixture (conftest.py) that redirects CHROMA_DIR/
COLLECTION_NAME/REGISTRY_DB_PATH to temp locations — the production
chroma_store is never touched by tests.

Multi-document workspaces: a workspaces table (database.py) plus
workspace_id columns on documents/sessions; /documents and /sessions accept
a workspace_id filter, and /query cross-checks any requested document_ids
against that workspace before retrieval — not just a frontend display
convention.

Known limitations (accurate, not a marketing gap-fill): no automated live
groundedness/hallucination-rate scoring in production (only an offline eval
harness, eval_harness.py, comparing retrieval-quality metrics across
pipeline variants); no per-tenant Chroma isolation (single shared
collection, filtered by metadata); no horizontal scaling / sharding of the
vector store; no streaming token-by-token output.
""".strip()


# ---------------------------------------------------------------------------
# Shared JSON-extraction helper
# ---------------------------------------------------------------------------
def _extract_json_object(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PDF-grounded modes: viva / mock test question generation
# ---------------------------------------------------------------------------
QUESTION_GEN_PROMPT = """You are generating {mode_label} questions for a research paper or project, using ONLY the evidence below.

Evidence was retrieved from the user's selected uploaded document(s). Base every question strictly on this evidence — do not invent details, numbers, datasets, or claims that are not present. Do NOT ask about anything other than what this evidence actually describes — never substitute a different project or system you may know about.

DIFFICULTY: {difficulty}
NUMBER OF QUESTIONS: {num_questions}
FOCUS (optional — prefer these categories/topics if given and the evidence supports them; otherwise cover the material broadly): {topics}

EVIDENCE:
{evidence_text}

Generate {num_questions} {difficulty}-difficulty questions that test understanding of THIS uploaded document's actual content — its problem/motivation, methodology, architecture, algorithms/models, datasets, experiments, metrics, results, limitations, contributions, technical decisions — whatever the evidence actually covers. For each question, include a short category label (e.g. "Architecture", "Dataset", "Results", "Limitations" — whatever fits this specific document) and 1-3 expected_answer_points drawn from the evidence.

If the evidence is too thin to responsibly generate {num_questions} distinct questions, generate as many as are genuinely supported and set evidence_sufficient to false.

Output ONLY compact JSON:
{{
  "questions": [
    {{"question": "...", "category": "...", "difficulty": "{difficulty}", "expected_answer_points": ["..."]}}
  ],
  "grounded": true,
  "evidence_sufficient": true
}}
JSON:"""


def generate_pdf_questions(evidence_text: str, mode_label: str, difficulty: str, num_questions: int, topics: list[str] | None = None) -> QuestionSetJSON | None:
    """Viva / Mock Test / (PDF-grounded) Project Interview question
    generation. ONE LLM call. Returns None on any failure (caller decides
    the user-facing message)."""
    prompt = QUESTION_GEN_PROMPT.format(
        mode_label=mode_label, difficulty=difficulty, num_questions=num_questions, evidence_text=evidence_text,
        topics=", ".join(topics) if topics else "any",
    )
    try:
        raw = _call_groq_raw(prompt)
        obj = _extract_json_object(raw)
        if obj is None:
            return None
        return QuestionSetJSON.model_validate(obj)
    except (ValidationError, Exception):
        return None


# ---------------------------------------------------------------------------
# PDF-grounded: explain a figure/table/graph/diagram
# ---------------------------------------------------------------------------
EXPLAIN_FIGURE_PROMPT = """You are explaining a specific figure/table/graph/diagram from a research paper, using ONLY the evidence below (retrieved from the user's selected uploaded document(s)).

The user is asking about: {figure_reference}

EVIDENCE (surrounding text, captions, and any table/figure text extracted from the PDF):
{evidence_text}

Explain what was asked about, adapting the explanation to what it actually is:
- FIGURE: what it represents, main components, relationships, important observations, conclusions the paper draws from it.
- ARCHITECTURE DIAGRAM: components, component-to-component flow, inputs/outputs, processing stages, overall architecture.
- GRAPH: axes, curves/series, trends, comparisons, important values, conclusions the paper draws from it.
- TABLE: important columns/rows, key comparisons, relevant values, best/worst results where directly stated, conclusions the paper draws from it.

Rules:
- Use ONLY what the evidence actually contains. Do not invent axis labels, values, or components that aren't in the text.
- If the evidence doesn't contain enough about this specific figure/table to explain it reliably (e.g. the caption/reference wasn't retrieved), say so explicitly rather than guessing.
- Do not include chunk IDs or technical citations in the answer text.
- HONESTY: the "evidence" above is the figure/table's caption and surrounding extracted TEXT — you have not seen the actual image/rendering. Never write as if you visually inspected the figure (e.g. never claim to see specific colors, exact pixel positions, or line styles that aren't described in the text). If the caption/text doesn't state a detail, say it isn't available from the extracted text rather than inferring it from what such a figure would typically look like.

Output ONLY compact JSON:
{{
  "answer": "...",
  "grounded": true or false,
  "evidence_sufficient": true or false,
  "document_ids": ["..."]
}}
JSON:"""


# Used ONLY when an actual page image was rendered (figure_vision.py) AND a
# vision-capable model is configured (figure_vision.vision_model_available())
# — asks the model to look at the image directly, but still grounds it in
# the same retrieved evidence text so a genuinely-visual answer never
# contradicts what the paper's own text says.
EXPLAIN_FIGURE_VISION_PROMPT = """You are looking at an image of one page from a research paper's PDF, and explaining a specific figure/table/graph/diagram that appears on it.

The user is asking about: {figure_reference}

SUPPORTING TEXT EVIDENCE from this same document (captions, surrounding text — use this to confirm labels/values you can also see in the image, and to fill in anything not visible in the image itself):
{evidence_text}

Explain what was asked about, using what you can actually SEE in the image (components, layout, curves, axis labels, table values, colors/shapes only if visually distinguishable) combined with the supporting text evidence:
- FIGURE: what it represents, main components, relationships, important observations.
- ARCHITECTURE DIAGRAM: components, component-to-component flow, inputs/outputs, processing stages.
- GRAPH: axes, curves/series, visible trends, comparisons, values you can actually read off the image.
- TABLE: columns/rows, key comparisons, values you can actually read.

Rules:
- Only describe what is genuinely visible in the image or stated in the text evidence — never invent a value, label, or component you cannot actually see or read in the evidence.
- Do not include chunk IDs or technical citations in the answer text.

Output ONLY compact JSON:
{{
  "answer": "...",
  "grounded": true or false,
  "evidence_sufficient": true or false,
  "document_ids": ["..."]
}}
JSON:"""


def explain_figure(evidence_text: str, figure_reference: str, image_base64: str | None = None, vision_model: str | None = None) -> tuple[AnswerJSON | None, bool]:
    """
    Returns (result, visual_inspection_used). visual_inspection_used is
    True ONLY when an image was actually provided to a real vision-capable
    model AND that call succeeded — the caller (main.py) uses this exact
    flag to decide which honesty caveat to append; it is never inferred or
    assumed from "an image was available" alone.
    """
    if image_base64 and vision_model:
        from query_transform import _call_groq_vision_raw
        vision_prompt = EXPLAIN_FIGURE_VISION_PROMPT.format(
            figure_reference=figure_reference or "the referenced figure/table", evidence_text=evidence_text,
        )
        raw = _call_groq_vision_raw(vision_prompt, image_base64, vision_model)
        if raw is not None:
            obj = _extract_json_object(raw)
            if obj is not None:
                try:
                    return AnswerJSON.model_validate(obj), True
                except (ValidationError, Exception):
                    pass
        # Vision call unavailable/failed — fall through to the text-only
        # path below rather than returning an error; the honesty caveat
        # main.py appends will correctly say text-based, not lie either way.

    prompt = EXPLAIN_FIGURE_PROMPT.format(figure_reference=figure_reference or "the referenced figure/table", evidence_text=evidence_text)
    try:
        raw = _call_groq_raw(prompt)
        obj = _extract_json_object(raw)
        if obj is None:
            return None, False
        return AnswerJSON.model_validate(obj), False
    except (ValidationError, Exception):
        return None, False


# ---------------------------------------------------------------------------
# PDF-grounded: recommendation after comparison
# ---------------------------------------------------------------------------
RECOMMEND_PROMPT = """You are making a recommendation between research papers being compared, using ONLY the evidence below (retrieved from the user's selected uploaded documents).

EVIDENCE (grouped by document):
{evidence_text}

The user asked: "Which approach would you choose and why?"

Rules:
- First state what each paper actually reports (methodology, results, trade-offs) — strictly from the evidence.
- Then give a recommendation that follows FROM those reported differences only. Do not introduce outside facts, outside benchmarks, or general AI knowledge not present in the evidence.
- If the evidence doesn't support a confident recommendation (e.g. the papers address different problems, or evidence is too thin), say so explicitly instead of forcing a pick.
- Clearly separate "what the papers report" from "what I recommend and why" in the answer text.

Output ONLY compact JSON:
{{
  "answer": "...",
  "grounded": true or false,
  "evidence_sufficient": true or false,
  "document_ids": ["..."]
}}
JSON:"""


def recommend_from_comparison(evidence_text: str) -> AnswerJSON | None:
    prompt = RECOMMEND_PROMPT.format(evidence_text=evidence_text)
    try:
        raw = _call_groq_raw(prompt)
        obj = _extract_json_object(raw)
        if obj is None:
            return None
        return AnswerJSON.model_validate(obj)
    except (ValidationError, Exception):
        return None


# ---------------------------------------------------------------------------
# Non-PDF: Why This Design / System Design — grounded in REAL_ARCHITECTURE_CONTEXT
# ---------------------------------------------------------------------------
DESIGN_QUESTION_PROMPT = """You are answering a design-defense / system-design question about VerityRAG, a real, already-implemented RAG system. Answer using ONLY the ground-truth architecture description below — never invent components, models, or techniques that aren't described in it.

{architecture_context}

MODE: {mode_label}
{mode_instructions}

QUESTION: {question}

Rules:
- Ground every factual claim about "what VerityRAG does" in the architecture description above.
- Explain trade-offs and realistic alternatives (why this choice over the alternatives), but be explicit that the alternatives are alternatives, not what's implemented.
- If asked about scaling/future improvements, clearly separate CURRENT IMPLEMENTATION from PROPOSED FUTURE DESIGN — never blur the two.
- If the ground truth doesn't cover something the question assumes, say so plainly rather than guessing.

Output ONLY compact JSON:
{{
  "answer": "...",
  "grounded": true,
  "evidence_sufficient": true,
  "document_ids": []
}}
JSON:"""

_MODE_INSTRUCTIONS = {
    "why_design": "Explain WHY this specific design choice was made — the trade-offs against realistic alternatives (e.g. why ChromaDB over another vector store, why dense+BM25 over dense-only, why RRF, why a CrossEncoder, why one LLM call, why Groq/Llama, why LangGraph, why Pydantic validation, why document_id scoping, why token budgeting, why a fallback model, why caching).",
    "system_design": "This is a system-design / scaling question. Clearly separate what VerityRAG ACTUALLY does today from a PROPOSED FUTURE design for the scenario asked about (e.g. millions of PDFs, many concurrent users, multi-tenancy, vector DB failure, latency reduction, huge PDFs, retrieval-quality improvement, evaluation). Be concrete and specific to this system's real components, not generic RAG platitudes.",
}


def answer_design_question(question: str, mode: str) -> AnswerJSON | None:
    mode_label = "Why This Design" if mode == "why_design" else "System Design Interview"
    prompt = DESIGN_QUESTION_PROMPT.format(
        architecture_context=REAL_ARCHITECTURE_CONTEXT, mode_label=mode_label,
        mode_instructions=_MODE_INSTRUCTIONS.get(mode, ""), question=question,
    )
    try:
        raw = _call_groq_raw(prompt)
        obj = _extract_json_object(raw)
        if obj is None:
            return None
        return AnswerJSON.model_validate(obj)
    except (ValidationError, Exception):
        return None


# ---------------------------------------------------------------------------
# PDF-grounded: Project Interview — one question at a time, grounded in the
# CURRENTLY SELECTED/REFERENCED UPLOADED DOCUMENT(S), not in VerityRAG's own
# architecture. This is the default interview source; if the uploaded
# document happens to itself describe VerityRAG, the questions will
# naturally be about VerityRAG because that's what the evidence contains —
# but nothing here special-cases that, it's the same evidence-grounded path
# as Viva/Mock Test. ("Why This Design?" / "System Design" stay separately
# grounded in REAL_ARCHITECTURE_CONTEXT below — those modes are explicitly
# about VerityRAG's own design choices, not about the uploaded paper.)
#
# Opening question reuses generate_pdf_questions() for exactly ONE question
# (no separate prompt/schema needed). Evaluation + next-question is ONE
# combined call per turn, mirroring the same shape as the old
# architecture-grounded version, just evidence-grounded instead.
# ---------------------------------------------------------------------------
PDF_INTERVIEW_EVALUATE_PROMPT = """You are conducting a technical interview about the uploaded document/project below, using ONLY the evidence retrieved from it. Evaluate the candidate's answer to the last question, then ask ONE new question — grounded in this SAME evidence, about this SAME document. Never switch to a different project or system.

DIFFICULTY: {difficulty}
FOCUS: {topics}

EVIDENCE:
{evidence_text}

QUESTION ASKED: {question}
CANDIDATE'S ANSWER: {user_answer}

Evaluate against the evidence above:
- correctness: "correct", "partially_correct", "incorrect", or "unclear"
- missing_points: specific facts from the evidence the answer missed (empty list if none)
- technical_depth: one short sentence on precision/depth shown
- suggested_answer: a better, evidence-accurate answer (concise)
- next_question: the next interview question, grounded in the SAME evidence (a different aspect/category than the last question where possible; stay within the focus if given)
- next_question_category: its category label

If the evidence doesn't cover enough to evaluate the answer reliably, say so honestly in technical_depth rather than guessing, and ask a next_question that IS well-supported by the evidence.

Output ONLY compact JSON:
{{
  "correctness": "...",
  "missing_points": ["..."],
  "technical_depth": "...",
  "suggested_answer": "...",
  "next_question": "...",
  "next_question_category": "..."
}}
JSON:"""


def start_project_interview(evidence_text: str, difficulty: str, topics: list[str] | None) -> AnswerJSON | None:
    """The interview's opening question, grounded in the uploaded document's
    evidence — reuses generate_pdf_questions() requesting a single question."""
    result = generate_pdf_questions(evidence_text, "Project Interview", difficulty, num_questions=1, topics=topics)
    if result is None or not result.questions:
        return None
    q = result.questions[0]
    return AnswerJSON(answer=q.question, grounded=result.grounded, evidence_sufficient=result.evidence_sufficient, document_ids=[])


def evaluate_interview_answer(evidence_text: str, question: str, user_answer: str, difficulty: str, topics: list[str] | None) -> AnswerEvaluationJSON | None:
    prompt = PDF_INTERVIEW_EVALUATE_PROMPT.format(
        evidence_text=evidence_text, difficulty=difficulty,
        topics=", ".join(topics) if topics else "any",
        question=question, user_answer=user_answer,
    )
    try:
        raw = _call_groq_raw(prompt)
        obj = _extract_json_object(raw)
        if obj is None:
            return None
        return AnswerEvaluationJSON.model_validate(obj)
    except (ValidationError, Exception):
        return None


# ---------------------------------------------------------------------------
# PDF-grounded: Evaluate Paper — a structured critique across fixed
# dimensions, each one distinguishing what the paper directly states, what's
# strongly implied, and what simply isn't in the evidence. ONE LLM call.
# ---------------------------------------------------------------------------
EVALUATE_PAPER_PROMPT = """You are critically evaluating a single research paper/project, using ONLY the evidence below (retrieved from the user's selected uploaded document).

EVIDENCE:
{evidence_text}

For EACH dimension below, give a concise assessment and mark its support level:
- "DIRECTLY_STATED": the paper explicitly says this (e.g. states its own limitation, states its own novelty claim).
- "STRONGLY_SUPPORTED": not stated outright, but clearly and reasonably inferable from what IS stated (e.g. inferring "the method is not evaluated on production traffic" because only offline datasets are mentioned).
- "NOT_FOUND": the evidence doesn't cover this dimension at all — do not guess; leave assessment empty and set support to "NOT_FOUND".

Dimensions: problem_clarity, novelty, methodology, experimental_design, results, limitations, reproducibility.

Also list concrete strengths and weaknesses (each grounded in the evidence, not generic), and a 2-4 sentence overall_assessment that is honest about both merit and gaps.

Rules:
- Never invent a claim, number, or comparison that isn't in the evidence.
- A paper with thin evidence should have MORE "NOT_FOUND" dimensions, not a padded-out assessment.

Output ONLY compact JSON:
{{
  "problem_clarity": {{"assessment": "...", "support": "DIRECTLY_STATED"}},
  "novelty": {{"assessment": "...", "support": "STRONGLY_SUPPORTED"}},
  "methodology": {{"assessment": "...", "support": "DIRECTLY_STATED"}},
  "experimental_design": {{"assessment": "...", "support": "DIRECTLY_STATED"}},
  "results": {{"assessment": "...", "support": "DIRECTLY_STATED"}},
  "limitations": {{"assessment": "...", "support": "DIRECTLY_STATED"}},
  "reproducibility": {{"assessment": "...", "support": "NOT_FOUND"}},
  "strengths": ["..."],
  "weaknesses": ["..."],
  "overall_assessment": "...",
  "evidence_sufficient": true
}}
JSON:"""


def evaluate_paper(evidence_text: str, document_id: str) -> PaperEvaluationJSON | None:
    prompt = EVALUATE_PAPER_PROMPT.format(evidence_text=evidence_text)
    try:
        # 7 dimensions + strengths/weaknesses/overall routinely exceed the
        # default MAX_ANSWER_TOKENS (1024) and get cut off mid-JSON — the
        # SAME root cause as the token-budget bug fixed in retrieval.py:
        # something silently overrunning its configured budget. A wider
        # output budget here, not a second call.
        raw = _call_groq_raw(prompt, max_tokens=2048)
        obj = _extract_json_object(raw)
        if obj is None:
            return None
        obj["document_id"] = document_id
        return PaperEvaluationJSON.model_validate(obj)
    except (ValidationError, Exception):
        return None


# ---------------------------------------------------------------------------
# PDF-grounded: Research Gap Discovery — labels each gap by provenance.
# ---------------------------------------------------------------------------
RESEARCH_GAPS_PROMPT = """You are identifying research gaps in the document(s) below, using ONLY this evidence (retrieved from the user's selected uploaded document(s), grouped by "--- Document ID: X ---").

EVIDENCE:
{evidence_text}

Find gaps and label each one by where it came from:
- "AUTHOR_STATED_GAP": the authors explicitly name this as a limitation, open problem, or future work item.
- "POTENTIAL_INFERRED_GAP": not stated by the authors, but a reasonable gap you can infer from what the evidence DOES say (e.g. no ablation study is described, so its absence is an inferred gap) — be conservative, only infer what the evidence actually supports the absence of.

For each gap, include the specific evidence text/reasoning it's based on, and which document_id it applies to (use the exact Document ID string from the evidence headers).

Rules:
- Do not invent gaps unrelated to what's in the evidence.
- If the evidence is too thin to respinsibly infer anything beyond what's stated, only return AUTHOR_STATED_GAP entries and set evidence_sufficient to false if there are none of those either.

Output ONLY compact JSON:
{{
  "gaps": [
    {{"gap": "...", "label": "AUTHOR_STATED_GAP", "evidence": "...", "document_id": "..."}}
  ],
  "evidence_sufficient": true
}}
JSON:"""


def find_research_gaps(evidence_text: str) -> ResearchGapsJSON | None:
    prompt = RESEARCH_GAPS_PROMPT.format(evidence_text=evidence_text)
    try:
        raw = _call_groq_raw(prompt, max_tokens=1536)  # a multi-paper gap list can exceed the 1024 default
        obj = _extract_json_object(raw)
        if obj is None:
            return None
        return ResearchGapsJSON.model_validate(obj)
    except (ValidationError, Exception):
        return None


# ---------------------------------------------------------------------------
# PDF-grounded: Literature Matrix — one row per paper, each cell grounded
# ONLY in that paper's own evidence block (never borrowed from another
# paper's section, even when the model can see both in the same prompt).
# ---------------------------------------------------------------------------
LITERATURE_MATRIX_PROMPT = """You are building a literature comparison matrix from the evidence below (retrieved from the user's selected uploaded documents, grouped by "--- Document ID: X ---").

EVIDENCE:
{evidence_text}

Produce exactly ONE row per Document ID that appears in the evidence above (use the exact Document ID string as document_id). For EACH row, fill problem / method / architecture / dataset / metrics / results / limitations / research_gap using ONLY that paper's own evidence block — never copy or infer a value from a different paper's section, even if it seems similar. If a paper's evidence doesn't cover a field, use exactly "Not available in uploaded evidence" for that field — never leave it blank or guess.

Output ONLY compact JSON:
{{
  "rows": [
    {{"document_id": "...", "title": "...", "problem": "...", "method": "...", "architecture": "...", "dataset": "...", "metrics": "...", "results": "...", "limitations": "...", "research_gap": "..."}}
  ],
  "evidence_sufficient": true
}}
JSON:"""


def build_literature_matrix(evidence_text: str) -> LiteratureMatrixJSON | None:
    prompt = LITERATURE_MATRIX_PROMPT.format(evidence_text=evidence_text)
    try:
        # 8 text fields per paper, one row per paper — scales with N papers,
        # easily exceeding the 1024-token default for 3+ papers.
        raw = _call_groq_raw(prompt, max_tokens=2560)
        obj = _extract_json_object(raw)
        if obj is None:
            return None
        return LiteratureMatrixJSON.model_validate(obj)
    except (ValidationError, Exception):
        return None


# ---------------------------------------------------------------------------
# PDF-grounded: Knowledge Graph — single-paper concept tree, or multi-paper
# evidence-supported relationships. Every node/edge carries the document_id
# it came from so a multi-paper graph never blurs provenance.
# ---------------------------------------------------------------------------
KNOWLEDGE_GRAPH_PROMPT = """You are extracting a knowledge graph from the evidence below (retrieved from the user's selected uploaded document(s), grouped by "--- Document ID: X ---").

EVIDENCE:
{evidence_text}

Build nodes (id, label, type, document_id) and edges (source id, target id, relation, document_id). Node "type" is one of: paper, method, dataset, result, concept, limitation.

If there is exactly one Document ID in the evidence: build a tree rooted at a "paper" node for that document, branching into its method/dataset/result/limitation concepts, each edge relation describing the relationship (e.g. "uses", "evaluated_on", "reports", "limited_by").

If there are multiple Document IDs: additionally add edges directly BETWEEN nodes belonging to different documents ONLY when the evidence itself supports a real relationship (e.g. both use the same dataset, one extends or contrasts with the other's method) — never invent a cross-paper relationship the evidence doesn't support. Every node and edge must carry the document_id it came from (a genuinely cross-paper edge may reference either contributing document_id in "document_id", but only add such an edge when both endpoints' evidence supports it).

Rules:
- Every node/edge must trace to the evidence — do not invent concepts, datasets, or relationships not present.
- Keep the graph compact: prefer the most important 8-14 nodes over an exhaustive list — a smaller, complete, correctly-closed JSON graph is far more useful than a larger one that gets cut off.

Output ONLY compact JSON, with SHORT labels/relations (a few words each, not full sentences):
{{
  "nodes": [{{"id": "...", "label": "...", "type": "method", "document_id": "..."}}],
  "edges": [{{"source": "...", "target": "...", "relation": "...", "document_id": "..."}}],
  "evidence_sufficient": true
}}
JSON:"""


def build_knowledge_graph(evidence_text: str) -> KnowledgeGraphJSON | None:
    prompt = KNOWLEDGE_GRAPH_PROMPT.format(evidence_text=evidence_text)
    try:
        # A node/edge graph is one of the more verbose outputs here — the
        # 1024-token default was observed to truncate mid-JSON on real
        # multi-node graphs, which _extract_json_object then correctly
        # fails to parse (returns None) rather than accepting broken JSON.
        # Same fix pattern as evaluate_paper/literature_matrix: a wider
        # output budget, still exactly ONE call.
        raw = _call_groq_raw(prompt, max_tokens=2048)
        obj = _extract_json_object(raw)
        if obj is None:
            return None
        return KnowledgeGraphJSON.model_validate(obj)
    except (ValidationError, Exception):
        return None
