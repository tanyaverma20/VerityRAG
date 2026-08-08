"""
organizer.py — Graph node for Evidence Organization
"""
from typing import Any
from .state import ResearchState

def organize_evidence(state: ResearchState) -> dict[str, Any]:
    """
    Groups retrieved chunks by document_id and deduplicates redundant parent contexts.
    
    Returns:
      evidence_by_document: dict mapping document_id to a list of deduplicated context blocks.
                            Each block retains the exact child texts and citation mappings.
    """
    retrieval_results = state.get("retrieval_results", [])
    
    # 1. Group by document_id
    grouped_chunks = {}
    for chunk in retrieval_results:
        doc_id = chunk.get("metadata", {}).get("document_id", "unknown_doc")
        grouped_chunks.setdefault(doc_id, []).append(chunk)
        
    evidence_by_document = {}
    
    # 2. Deduplicate parent context per document
    for doc_id, chunks in grouped_chunks.items():
        # Map: parent_id -> dict with 'context', 'children'
        parents_map = {}
        
        for c in chunks:
            parent_id = c.get("metadata", {}).get("parent_id", "no_parent")
            
            # Use parent_context if available, fallback to chunk text
            context_text = c.get("parent_context", c["text"])
            
            if parent_id not in parents_map:
                parents_map[parent_id] = {
                    "parent_context": context_text,
                    "children": []
                }
            
            # Retain child exact text and metadata for citation mapping
            parents_map[parent_id]["children"].append({
                "chunk_id": c.get("metadata", {}).get("chunk_id", ""),
                "text": c["text"],
                "source": c.get("metadata", {}).get("source", ""),
                "page_number": c.get("metadata", {}).get("page_number", ""),
                "section": c.get("metadata", {}).get("section", ""),
            })
            
        evidence_by_document[doc_id] = list(parents_map.values())
        
    return {
        "evidence_by_document": evidence_by_document,
        "status": f"organized into {len(evidence_by_document)} documents"
    }
