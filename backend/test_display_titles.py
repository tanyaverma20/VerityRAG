"""
test_display_titles.py — proves document_id/UUIDs/hashes never leak into
displayed titles across Comparative Report, Literature Matrix, Evaluate
Paper, Research Gaps, and Knowledge Graph, while document_id itself stays
completely unchanged for retrieval/scoping/citations. See doc_titles.py.

Runs against the isolated Phase 8 fixture; only the Groq network boundary
is mocked.
"""
import json
import re
import uuid

import pytest
from fastapi.testclient import TestClient

import config
from ingest import get_collection
from retrieval import build_bm25_index
from main import app

client = TestClient(app)

_call_count = {"n": 0}


def _push_chunk(document_id: str, suffix: str, text: str, filename: str) -> None:
    col = get_collection()
    chunk_id = f"doc_{document_id}_{suffix}"
    col.add(
        documents=[text],
        ids=[chunk_id],
        metadatas=[{
            "document_id": document_id, "filename": filename, "source": filename,
            "page_number": 1, "section": "Body", "chunk_id": chunk_id,
            "parent_id": f"{document_id}_{suffix}_parent", "chunk_type": "child",
        }],
    )


class _FakeMessage:
    def __init__(self, content): self.content = content
class _FakeChoice:
    def __init__(self, content): self.message = _FakeMessage(content)
class _FakeUsage:
    prompt_tokens = 80
    completion_tokens = 40
class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeCompletions:
    def create(self, model, messages, temperature=0.0, max_tokens=None):
        _call_count["n"] += 1
        prompt = messages[0]["content"]
        evidence_section = prompt.split("EVIDENCE:", 1)[-1]
        doc_ids = list(dict.fromkeys(re.findall(r"Document ID:\s*(\S+)", evidence_section)))

        if '"problem_clarity"' in prompt:
            return _FakeResponse(json.dumps({
                "problem_clarity": {"assessment": "Clear.", "support": "DIRECTLY_STATED"},
                "novelty": {"assessment": "", "support": "NOT_FOUND"},
                "methodology": {"assessment": "", "support": "NOT_FOUND"},
                "experimental_design": {"assessment": "", "support": "NOT_FOUND"},
                "results": {"assessment": "", "support": "NOT_FOUND"},
                "limitations": {"assessment": "", "support": "NOT_FOUND"},
                "reproducibility": {"assessment": "", "support": "NOT_FOUND"},
                "strengths": [], "weaknesses": [], "overall_assessment": "Fine.",
                "evidence_sufficient": True,
            }))
        if '"gaps"' in prompt:
            return _FakeResponse(json.dumps({
                "gaps": [{"gap": "No ablation.", "label": "POTENTIAL_INFERRED_GAP", "evidence": "e", "document_id": d}
                         for d in doc_ids],
                "evidence_sufficient": True,
            }))
        if '"rows"' in prompt:
            # Deliberately echo the raw document_id back as "title" — a
            # real model could do this; must not leak through either.
            rows = [{
                "document_id": d, "title": d, "problem": "p", "method": "m", "architecture": "a",
                "dataset": "d", "metrics": "m", "results": "r", "limitations": "l", "research_gap": "g",
            } for d in doc_ids]
            return _FakeResponse(json.dumps({"rows": rows, "evidence_sufficient": True}))
        if '"nodes"' in prompt:
            nodes = [{"id": f"{d}_n", "label": "Concept", "type": "concept", "document_id": d} for d in doc_ids]
            return _FakeResponse(json.dumps({"nodes": nodes, "edges": [], "evidence_sufficient": True}))
        if '"comparison":' in prompt or "COMPARATIVE report" in prompt:
            papers = [{
                "document_id": d, "title": None, "overview": "o", "main_contribution": "c", "methodology": "m",
                "architecture": "a", "datasets": [], "evaluation_metrics": [], "key_results": [],
                "important_calculations": [], "limitations": [], "final_summary": "s",
            } for d in doc_ids]
            return _FakeResponse(json.dumps({
                "title": "Report", "overview": "o", "papers": papers, "comparison": None,
                "conclusion": "c", "evidence_sufficient": True,
            }))
        return _FakeResponse(json.dumps({"answer": "x", "grounded": True, "evidence_sufficient": True, "document_ids": []}))


class _FakeChat:
    def __init__(self): self.completions = _FakeCompletions()
class _FakeGroqClient:
    def __init__(self, api_key=None): self.chat = _FakeChat()


from unittest.mock import patch
_groq_patch = patch("groq.Groq", _FakeGroqClient)


@pytest.fixture(autouse=True)
def _reset():
    _call_count["n"] = 0
    _groq_patch.start()
    yield
    _groq_patch.stop()


def _looks_like_leaked_id(text: str, doc_id: str) -> bool:
    return doc_id in (text or "")


@pytest.fixture
def real_and_hash_named_docs():
    """One document with a genuinely meaningful filename, one whose filename
    IS itself the hash/id (mirrors the real d926a10c1260ae70.pdf case)."""
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"
    doc_a = uuid.uuid4().hex[:16]  # meaningful-filename document
    doc_b = uuid.uuid4().hex[:16]  # hash-named-file document — filename == its own id
    _push_chunk(doc_a, "c1", "Operating systems manage processes, threads, and memory allocation.", "OS_Full_Notes.pdf")
    _push_chunk(doc_b, "c1", "Database systems use indexing to speed up query performance.", f"{doc_b}.pdf")
    build_bm25_index()
    return doc_a, doc_b


# ---------------------------------------------------------------------------
def test_evaluate_paper_title_never_leaks_document_id(real_and_hash_named_docs):
    doc_a, doc_b = real_and_hash_named_docs
    resp = client.post("/analyze", json={"mode": "evaluate_paper", "document_ids": [doc_a]})
    data = resp.json()
    assert data["ok"] is True
    assert data["evaluation"]["title"] == "OS Full Notes"
    assert data["evaluation"]["document_id"] == doc_a, "document_id itself must stay exactly as-is"

    resp2 = client.post("/analyze", json={"mode": "evaluate_paper", "document_ids": [doc_b]})
    data2 = resp2.json()
    assert data2["ok"] is True
    assert not _looks_like_leaked_id(data2["evaluation"]["title"], doc_b), f"Title must never contain the raw document_id, got {data2['evaluation']['title']!r}"
    assert data2["evaluation"]["document_id"] == doc_b


def test_research_gaps_document_titles_map_never_leaks_id(real_and_hash_named_docs):
    doc_a, doc_b = real_and_hash_named_docs
    resp = client.post("/analyze", json={"mode": "research_gaps", "document_ids": [doc_a, doc_b]})
    data = resp.json()
    assert data["ok"] is True
    titles = data["document_titles"]
    assert titles[doc_a] == "OS Full Notes"
    assert not _looks_like_leaked_id(titles[doc_b], doc_b)
    # The raw document_id must still be present on each gap (internal use —
    # scoping/isolation is unaffected), only the DISPLAY title is resolved.
    assert {g["document_id"] for g in data["gaps"]} == {doc_a, doc_b}


def test_literature_matrix_rejects_model_echoing_id_as_title(real_and_hash_named_docs):
    doc_a, doc_b = real_and_hash_named_docs
    resp = client.post("/analyze", json={"mode": "literature_matrix", "document_ids": [doc_a, doc_b]})
    data = resp.json()
    assert data["ok"] is True
    row_a = next(r for r in data["rows"] if r["document_id"] == doc_a)
    row_b = next(r for r in data["rows"] if r["document_id"] == doc_b)
    assert row_a["title"] == "OS Full Notes"
    # The fake model echoed doc_b's own id back as "title" — must be caught
    # and replaced, never displayed.
    assert row_b["title"] != doc_b
    assert not _looks_like_leaked_id(row_b["title"], doc_b)
    assert re.match(r"^Document \d+$", row_b["title"]), f"Expected a safe generic fallback label, got {row_b['title']!r}"


def test_knowledge_graph_document_titles_map_never_leaks_id(real_and_hash_named_docs):
    doc_a, doc_b = real_and_hash_named_docs
    resp = client.post("/analyze", json={"mode": "knowledge_graph", "document_ids": [doc_a, doc_b]})
    data = resp.json()
    assert data["ok"] is True
    titles = data["document_titles"]
    assert titles[doc_a] == "OS Full Notes"
    assert not _looks_like_leaked_id(titles[doc_b], doc_b)
    assert {n["document_id"] for n in data["nodes"]} == {doc_a, doc_b}, "document_id on nodes stays exactly as-is for internal use"


def test_comparative_report_title_never_leaks_document_id(real_and_hash_named_docs):
    doc_a, doc_b = real_and_hash_named_docs
    resp = client.post("/report", json={"document_ids": [doc_a, doc_b]})
    data = resp.json()
    assert data["ok"] is True
    paper_a = next(p for p in data["papers"] if p["document_id"] == doc_a)
    paper_b = next(p for p in data["papers"] if p["document_id"] == doc_b)
    assert paper_a["title"] == "OS Full Notes"
    assert not _looks_like_leaked_id(paper_b["title"], doc_b)
    assert paper_b["title"], "Every paper must always get a non-empty display title"
    # document_id remains fully correct/unchanged for both papers.
    assert paper_a["document_id"] == doc_a and paper_b["document_id"] == doc_b


def test_comparative_report_markdown_export_never_leaks_document_id(real_and_hash_named_docs):
    doc_a, doc_b = real_and_hash_named_docs
    resp = client.post("/report", json={"document_ids": [doc_a, doc_b]})
    data = resp.json()
    md_resp = client.get(f"/report/{data['report_id']}/markdown")
    assert md_resp.status_code == 200
    md_text = md_resp.text
    assert doc_b not in md_text, "Exported Markdown must never contain the raw document_id"
    assert "OS Full Notes" in md_text
