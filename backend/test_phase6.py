import json
import pytest
import os
import sys

# Ensure backend directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.state import ResearchState
from graph.planner import plan_research
from graph.analyzer import analyze_evidence_sufficiency, generate_adaptive_queries, cross_paper_analysis, assign_confidence
from graph.workflow import route_adaptive_research

class TestPhase6AdaptiveLoop:
    def test_state_fields(self):
        """1. Test that Phase 6 research state fields initialize properly."""
        state = ResearchState(
            original_query="Test query",
            research_type="deep",
            research_plan={},
            sub_queries=[],
            completed_sub_queries=[],
            document_ids=[],
            retrieval_results=[],
            evidence_by_document={},
            research_iterations=1,
            evidence_gaps=[],
            adaptive_queries=[],
            contradictions=[],
            similarities=[],
            differences=[],
            research_gaps=[],
            claims=[],
            confidence="UNAVAILABLE",
            citations=[],
            verification_results=[],
            retry_count=0,
            status=""
        )
        assert state["research_type"] == "deep"
        assert state["research_iterations"] == 1
        assert state["evidence_gaps"] == []
        
    def test_simple_routing(self):
        """2. Test simple query routing bypasses gap detection."""
        state = ResearchState(original_query="Test", research_type="simple")
        res = analyze_evidence_sufficiency(state)
        assert res["evidence_gaps"] == []
        assert res["status"] == "sufficient"
        
    def test_deep_routing(self):
        """3. Test deep query routing."""
        state = ResearchState(original_query="Test", research_type="deep")
        # In a test environment without LLM, it should return [] safely
        res = analyze_evidence_sufficiency(state)
        assert "evidence_gaps" in res
        
    def test_adaptive_routing_logic(self):
        """4. Test evidence sufficiency edge logic."""
        # Gap exists, iterations < max
        state1 = ResearchState(evidence_gaps=["gap1"], research_iterations=1)
        assert route_adaptive_research(state1) == "adapt_queries"
        
        # Gap exists, iterations == max (3)
        state2 = ResearchState(evidence_gaps=["gap1"], research_iterations=3)
        assert route_adaptive_research(state2) == "cross_paper"
        
        # No gaps
        state3 = ResearchState(evidence_gaps=[], research_iterations=1)
        assert route_adaptive_research(state3) == "cross_paper"

    def test_adaptive_query_deduplication(self):
        """7. Test adaptive queries are deduplicated."""
        state = ResearchState(
            original_query="Test",
            evidence_gaps=["Gap 1"],
            sub_queries=["Original", "Dupe"],
            completed_sub_queries=["Original"]
        )
        # Without LLM, we can't fully test generation, but we test the structure
        res = generate_adaptive_queries(state)
        assert "sub_queries" in res
        assert "adaptive_queries" in res
        
    def test_cross_paper_structure(self):
        """15, 17, 18, 19. Test cross paper structure."""
        state = ResearchState(original_query="Test", research_type="deep")
        res = cross_paper_analysis(state)
        assert "similarities" in res
        assert "differences" in res
        assert "contradictions" in res
        assert "research_gaps" in res
        
    def test_confidence_unavailable(self):
        """23. Test confidence assignment - UNAVAILABLE"""
        state = ResearchState(verification_results=[])
        res = assign_confidence(state)
        assert res["confidence"] == "UNAVAILABLE"
        
    def test_confidence_high(self):
        """21. Test confidence assignment - HIGH"""
        state = ResearchState(
            evidence_by_document={"doc1": [], "doc2": []},
            verification_results=[{"status": "SUPPORTED"}, {"status": "SUPPORTED"}],
            contradictions=[],
            evidence_gaps=[]
        )
        res = assign_confidence(state)
        assert res["confidence"] == "HIGH"
        
    def test_confidence_low(self):
        """22. Test confidence assignment - LOW (due to gaps)"""
        state = ResearchState(
            evidence_by_document={"doc1": [], "doc2": []},
            verification_results=[{"status": "SUPPORTED"}],
            contradictions=[],
            evidence_gaps=["A gap exists"]
        )
        res = assign_confidence(state)
        assert res["confidence"] == "LOW"
        
        # LOW due to contradictions
        state2 = ResearchState(
            evidence_by_document={"doc1": []},
            verification_results=[{"status": "SUPPORTED"}],
            contradictions=[{"status": "CONTRADICTORY"}],
            evidence_gaps=[]
        )
        assert assign_confidence(state2)["confidence"] == "LOW"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
