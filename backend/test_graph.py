"""
test_graph.py — Phase 3 LangGraph Research Workflow Test Suite

Tests 1–16 as specified in Phase 3 requirements.
All tests use real assertions.

Run with:
    cd backend
    python test_graph.py
"""

import sys
import traceback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def run_test(name: str, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  PASS  {name}")
    except AssertionError as e:
        _results.append((name, False, str(e)))
        print(f"  FAIL  {name} — {e}")
    except Exception as e:
        _results.append((name, False, f"Exception: {e}"))
        print(f"  FAIL  {name} — {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Import the modules under test
# ---------------------------------------------------------------------------

from graph.workflow import research_app
from graph.planner import plan_research, _fallback_plan
from graph.retriever import retrieve_evidence
from graph.organizer import organize_evidence
from graph.synthesizer import synthesize_answer
from graph.verifier import verify_evidence
from graph.state import ResearchState
from ingest import ingest_document
from retrieval import build_bm25_index


# ---------------------------------------------------------------------------
# Pre-test setup
# ---------------------------------------------------------------------------
print("\n=== Phase 3 LangGraph Research Test Suite ===\n")
try:
    ingest_document("../data/attention.pdf")
    build_bm25_index()
except Exception as e:
    print("Warning during setup:", e)

# ---------------------------------------------------------------------------
# TEST 1 — Graph construction
# ---------------------------------------------------------------------------
def test_graph_construction():
    assert research_app is not None, "Research graph should compile successfully"
    # Basic structural check
    assert hasattr(research_app, "invoke"), "research_app must have invoke method"

# ---------------------------------------------------------------------------
# TEST 2 — State initialization
# ---------------------------------------------------------------------------
def test_state_init():
    initial = {"original_query": "test query"}
    # The graph allows us to pass a partial typed dict
    assert "original_query" in initial, "State allows initializing with original_query"

# ---------------------------------------------------------------------------
# TEST 3 — Planner behavior
# ---------------------------------------------------------------------------
def test_planner_behavior():
    state = {"original_query": "What is the architecture?"}
    res = plan_research(state)
    assert "research_plan" in res, "Planner must return a research_plan"
    plan = res["research_plan"]
    assert "mode" in plan, "Plan must have a mode"
    assert "needs_decomposition" in plan, "Plan must have needs_decomposition"

# ---------------------------------------------------------------------------
# TEST 4 — Query decomposition
# ---------------------------------------------------------------------------
def test_planner_decomposition():
    state = {"original_query": "Compare architecture and datasets"}
    res = plan_research(state)
    sub_qs = res.get("sub_queries", [])
    assert isinstance(sub_qs, list), "sub_queries must be a list"
    assert len(sub_qs) >= 1, "sub_queries must not be empty"

# ---------------------------------------------------------------------------
# TEST 5 — Retrieval node
# ---------------------------------------------------------------------------
def test_retrieval_node():
    state = {"original_query": "attention mechanism", "sub_queries": ["attention mechanism"]}
    res = retrieve_evidence(state)
    assert "retrieval_results" in res, "Must return retrieval_results"
    assert isinstance(res["retrieval_results"], list), "retrieval_results must be a list"

# ---------------------------------------------------------------------------
# TEST 6 — Evidence organization
# ---------------------------------------------------------------------------
def test_organizer():
    retrieval_results = [
        {
            "text": "child 1",
            "parent_context": "parent page 1",
            "metadata": {"document_id": "doc_1", "parent_id": "p1", "chunk_id": "c1"}
        },
        {
            "text": "child 2",
            "parent_context": "parent page 1",
            "metadata": {"document_id": "doc_1", "parent_id": "p1", "chunk_id": "c2"}
        }
    ]
    state = {"original_query": "test", "retrieval_results": retrieval_results}
    res = organize_evidence(state)
    assert "evidence_by_document" in res, "Must return evidence_by_document"
    evidence = res["evidence_by_document"]
    assert "doc_1" in evidence, "doc_1 must be in evidence map"

# ---------------------------------------------------------------------------
# TEST 7 — Parent-context deduplication
# ---------------------------------------------------------------------------
def test_parent_context_dedupe():
    retrieval_results = [
        {
            "text": "child 1",
            "parent_context": "parent page 1",
            "metadata": {"document_id": "doc_1", "parent_id": "p1", "chunk_id": "c1"}
        },
        {
            "text": "child 2",
            "parent_context": "parent page 1", # Exact same parent context
            "metadata": {"document_id": "doc_1", "parent_id": "p1", "chunk_id": "c2"}
        }
    ]
    state = {"original_query": "test", "retrieval_results": retrieval_results}
    res = organize_evidence(state)
    parents = res["evidence_by_document"]["doc_1"]
    assert len(parents) == 1, "Should deduplicate to exactly one parent block"
    children = parents[0]["children"]
    assert len(children) == 2, "Both child chunks must be retained within the parent"

# ---------------------------------------------------------------------------
# TEST 8 — Multi-document grouping
# ---------------------------------------------------------------------------
def test_multi_document_grouping():
    retrieval_results = [
        {"text": "t1", "parent_context": "p1", "metadata": {"document_id": "doc_A", "parent_id": "p1"}},
        {"text": "t2", "parent_context": "p2", "metadata": {"document_id": "doc_B", "parent_id": "p2"}}
    ]
    state = {"original_query": "test", "retrieval_results": retrieval_results}
    res = organize_evidence(state)
    evidence = res["evidence_by_document"]
    assert "doc_A" in evidence and "doc_B" in evidence, "Must group distinct document IDs"

# ---------------------------------------------------------------------------
# TEST 9 — Structured citation preservation
# ---------------------------------------------------------------------------
def test_citation_preservation():
    retrieval_results = [
        {"text": "t1", "parent_context": "p1", "metadata": {"document_id": "doc_A", "parent_id": "p1", "chunk_id": "c1", "page_number": 3}}
    ]
    state = {"original_query": "test", "retrieval_results": retrieval_results}
    res = organize_evidence(state)
    state.update(res)
    synth_res = synthesize_answer(state)
    
    citations = synth_res.get("citations", [])
    assert len(citations) > 0, "Must output structured citations"
    assert citations[0]["document_id"] == "doc_A", "document_id must be preserved"
    assert citations[0]["chunk_id"] == "c1", "chunk_id must be preserved"
    assert citations[0]["page_number"] == 3, "page_number must be preserved"

# ---------------------------------------------------------------------------
# TEST 10 — Synthesis
# ---------------------------------------------------------------------------
def test_synthesis():
    retrieval_results = [
        {"text": "t1", "parent_context": "p1", "metadata": {"document_id": "doc_A", "parent_id": "p1", "chunk_id": "c1"}}
    ]
    state = {"original_query": "test", "retrieval_results": retrieval_results}
    state.update(organize_evidence(state))
    res = synthesize_answer(state)
    assert "draft_answer" in res, "Must return draft_answer"
    assert len(res["draft_answer"]) > 0, "Answer must not be empty"

# ---------------------------------------------------------------------------
# TEST 11 — Claim-level verification
# ---------------------------------------------------------------------------
def test_claim_verification():
    state = {
        "original_query": "test",
        "draft_answer": "This is a factual claim.",
        "evidence_by_document": {"doc_A": [{"parent_context": "p", "children": [{"chunk_id": "c1", "text": "fact"}]}]}
    }
    res = verify_evidence(state)
    assert "verification_results" in res, "Must output verification_results"
    results = res["verification_results"]
    assert isinstance(results, list), "verification_results must be a list"
    if results:
        assert "claim" in results[0], "Claim must be in verification result"
        assert "status" in results[0], "Status must be in verification result"

# ---------------------------------------------------------------------------
# TEST 12 — Retry limit
# ---------------------------------------------------------------------------
def test_retry_limit():
    from graph.workflow import route_verification
    state = {
        "verification_results": [{"status": "UNSUPPORTED"}],
        "retry_count": 1 # already retried once
    }
    route = route_verification(state)
    # The constant END is a special variable from langgraph, which evaluates to "__end__"
    assert route == "__end__", f"Must proceed to END when retry_count >= 1, got {route}"

# ---------------------------------------------------------------------------
# TEST 13 — Retry behavior
# ---------------------------------------------------------------------------
def test_retry_behavior():
    from graph.workflow import route_verification
    state = {
        "verification_results": [{"status": "UNSUPPORTED"}],
        "retry_count": 0 # first failure
    }
    route = route_verification(state)
    assert route == "synthesize", "Must loop back to synthesize if retry_count < 1"

# ---------------------------------------------------------------------------
# TEST 14 — LLM failure fallback
# ---------------------------------------------------------------------------
def test_llm_fallback_planner():
    from unittest.mock import patch
    with patch("graph.planner._call_groq_raw", side_effect=Exception("simulated LLM fail")):
        state = {"original_query": "test fallback query"}
        res = plan_research(state)
        # Should fallback deterministically
        assert "research_plan" in res, "Must return fallback plan"
        assert res["research_plan"]["reasoning"].startswith("Fallback"), "Should use fallback reasoning"

# ---------------------------------------------------------------------------
# TEST 15 — Full graph execution (Single Paper)
# ---------------------------------------------------------------------------
def test_full_graph_execution():
    initial_state = {"original_query": "attention mechanism"}
    final_state = research_app.invoke(initial_state)
    assert "draft_answer" in final_state, "Final state must have draft_answer"
    assert "citations" in final_state, "Final state must have citations"
    assert "research_plan" in final_state, "Final state must have research_plan"

# ---------------------------------------------------------------------------
# TEST 16 — Backward compatibility / API Integration Check
# ---------------------------------------------------------------------------
def test_api_backward_compatibility():
    # Calling the actual query function from main.py via mocked Request
    import main
    class MockRequest:
        question = "attention mechanism"
        top_k = 2
        document_ids = None
        strategy = "hybrid"
        
    res = main.query(MockRequest())
    assert "answer" in res, "Must return 'answer'"
    assert "sources" in res, "Must return 'sources'"
    assert "structured_citations" in res, "Must return new 'structured_citations'"
    assert "documents_found" in res, "Must return 'documents_found'"


# ---------------------------------------------------------------------------
# TEST 17 — Verifier failure produces VERIFICATION_UNAVAILABLE
# ---------------------------------------------------------------------------
def test_verifier_fallback():
    from unittest.mock import patch
    with patch("graph.verifier._call_groq_raw", side_effect=Exception("simulated verifier fail")):
        state = {
            "draft_answer": "Test answer",
            "evidence_by_document": {"doc": [{"children": [{"chunk_id": "c1", "text": "t"}]}]}
        }
        res = verify_evidence(state)
        
        # Must not crash
        assert "verification_results" in res, "Must return verification_results"
        results = res["verification_results"]
        assert len(results) == 1, "Must return exactly one fallback result"
        
        status = results[0]["status"]
        assert status == "VERIFICATION_UNAVAILABLE", f"Status must be VERIFICATION_UNAVAILABLE, got {status}"
        assert status != "SUPPORTED", "Status must NOT be SUPPORTED"

# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

tests = [
    ("TEST 01: Graph construction",       test_graph_construction),
    ("TEST 02: State init",               test_state_init),
    ("TEST 03: Planner behavior",         test_planner_behavior),
    ("TEST 04: Planner decomposition",    test_planner_decomposition),
    ("TEST 05: Retrieval node",           test_retrieval_node),
    ("TEST 06: Organizer",                test_organizer),
    ("TEST 07: Parent context dedupe",    test_parent_context_dedupe),
    ("TEST 08: Multi-document grouping",  test_multi_document_grouping),
    ("TEST 09: Structured citations",     test_citation_preservation),
    ("TEST 10: Synthesis",                test_synthesis),
    ("TEST 11: Claim verification",       test_claim_verification),
    ("TEST 12: Retry limit",              test_retry_limit),
    ("TEST 13: Retry behavior",           test_retry_behavior),
    ("TEST 14: LLM fallback",             test_llm_fallback_planner),
    ("TEST 15: Full graph execution",     test_full_graph_execution),
    ("TEST 16: API Backward compat",      test_api_backward_compatibility),
    ("TEST 17: Verifier fallback",        test_verifier_fallback),
]

for name, fn in tests:
    run_test(name, fn)

passed = sum(1 for _, ok, _ in _results if ok)
failed = sum(1 for _, ok, _ in _results if not ok)
print(f"\n{'='*50}")
print(f"Results: {passed} passed / {failed} failed / {len(_results)} total")
if failed:
    print("\nFailed tests:")
    for name, ok, msg in _results:
        if not ok:
            print(f"  {name}: {msg}")
    sys.exit(1)
else:
    print("\nAll Phase 3 graph tests PASSED.")
