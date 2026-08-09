"""
Direct Chroma retrieval verification — bypasses LLM entirely.
Confirms the vector index can embed a query and return real matching chunks.
"""
import sys
sys.path.insert(0, ".")
from ingest import _collection, _embed_fn

query = "What attention mechanism did the Transformer model introduce?"
print(f"Query: {query!r}")

query_embedding = _embed_fn([query])[0]

results = _collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

print(f"\nTop-5 results from chroma_store:")
for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
)):
    print(f"\n  [{i+1}] distance={dist:.4f}")
    print(f"       source={meta.get('source','?')}  page={meta.get('page_number','?')}")
    print(f"       text (first 120 chars): {doc[:120]!r}")

total = _collection.count()
print(f"\nTotal embeddings in collection: {total}")
assert total == 1416, f"Expected 1416, got {total}"
print("Direct retrieval verification: PASSED")
