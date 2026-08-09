"""
test_evidence_trace.py — claim-level evidence tracing (DIRECTLY_STATED /
STRONGLY_SUPPORTED / NOT_FOUND) rides on the SAME one-call structured-mode
synthesis path; normal mode (structured_mode=False, the default for every
ordinary chat question) must be completely unaffected — same prompt, same
schema, same ONE call, no claims field requested or returned.
"""
import json
from unittest.mock import patch

from graph.synthesizer import synthesize_answer

RETRIEVAL_RESULTS = [{
    "text": "The router uses a learned load-balancing loss.",
    "parent_context": "The router uses a learned load-balancing loss to distribute tokens across experts.",
    "metadata": {"document_id": "doc_A", "parent_id": "p1", "chunk_id": "c1", "source": "paperA.pdf", "page_number": 3},
}]


def _state(structured_mode: bool) -> dict:
    return {
        "original_query": "How does the router balance load?",
        "evidence_by_document": {
            "doc_A": [{
                "parent_context": RETRIEVAL_RESULTS[0]["parent_context"],
                "children": [{
                    "chunk_id": "c1", "text": RETRIEVAL_RESULTS[0]["text"],
                    "source": "paperA.pdf", "page_number": 3, "section": "Body",
                }],
            }],
        },
        "structured_mode": structured_mode,
    }


def test_structured_mode_returns_claim_evidence_trace():
    fake_json = json.dumps({
        "answer": "The router balances load using a learned load-balancing loss.",
        "grounded": True, "evidence_sufficient": True, "document_ids": ["doc_A"],
        "structured": {
            "architecture": "MoE router", "datasets": [], "metrics": [], "contributions": [],
            "methodology": None, "key_calculations": [], "limitations": [], "final_summary": None,
        },
        "claims": [
            {"claim": "The router uses a learned load-balancing loss.", "support": "DIRECTLY_STATED", "document_ids": ["doc_A"]},
            {"claim": "This improves throughput by 20%.", "support": "NOT_FOUND", "document_ids": []},
        ],
    })
    with patch("graph.synthesizer._call_groq_raw", return_value=fake_json), \
         patch("graph.synthesizer.GROQ_API_KEY", "fake-key-for-test"):
        res = synthesize_answer(_state(structured_mode=True))

    assert res["status"] == "synthesized"
    trace = res["claim_evidence_trace"]
    assert len(trace) == 2
    assert trace[0]["support"] == "DIRECTLY_STATED"
    assert trace[0]["document_ids"] == ["doc_A"]
    assert trace[1]["support"] == "NOT_FOUND", "A claim the evidence doesn't back must be labeled NOT_FOUND, not silently dropped"
    assert res["structured_data"]["architecture"] == "MoE router"


def test_normal_mode_never_requests_or_returns_claims():
    fake_json = json.dumps({
        "answer": "The router balances load using a learned load-balancing loss.",
        "grounded": True, "evidence_sufficient": True, "document_ids": ["doc_A"],
    })
    captured_prompt = {}

    def _capture(prompt):
        captured_prompt["text"] = prompt
        return fake_json

    with patch("graph.synthesizer._call_groq_raw", side_effect=_capture), \
         patch("graph.synthesizer.GROQ_API_KEY", "fake-key-for-test"):
        res = synthesize_answer(_state(structured_mode=False))

    assert res["status"] == "synthesized"
    assert res["claim_evidence_trace"] == [], "Normal mode must never populate the claim trace"
    assert res["structured_data"] is None
    assert '"claims"' not in captured_prompt["text"], "Normal mode's prompt must stay exactly the existing one-call shape — no extra requested field"
    assert "DIRECTLY_STATED" not in captured_prompt["text"]


def test_claim_trace_empty_when_model_omits_it():
    # A structured-mode response that (validly) doesn't include claims must
    # not fail synthesis — the field is optional, not a hard requirement.
    fake_json = json.dumps({
        "answer": "Answer without claims.", "grounded": True, "evidence_sufficient": True, "document_ids": ["doc_A"],
        "structured": {"architecture": None, "datasets": [], "metrics": [], "contributions": [], "methodology": None, "key_calculations": [], "limitations": [], "final_summary": None},
    })
    with patch("graph.synthesizer._call_groq_raw", return_value=fake_json), \
         patch("graph.synthesizer.GROQ_API_KEY", "fake-key-for-test"):
        res = synthesize_answer(_state(structured_mode=True))
    assert res["status"] == "synthesized"
    assert res["claim_evidence_trace"] == []
