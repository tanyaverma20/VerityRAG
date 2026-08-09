"""
Verify the canonical chroma_store collection and chunk count.
Read-only. No modifications.
"""
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "verityrag_docs_v2"

print(f"ChromaDB version: {chromadb.__version__}")
print(f"Opening PersistentClient at: {CHROMA_DIR}")

client = chromadb.PersistentClient(path=CHROMA_DIR)

collections = client.list_collections()
print(f"\nCollections found: {len(collections)}")
for c in collections:
    print(f"  - {c.name}")

col = client.get_collection(name=COLLECTION_NAME)
count = col.count()
print(f"\nCollection '{COLLECTION_NAME}':")
print(f"  Embedding count: {count}")

# Sample a couple of records to confirm metadata is intact
sample = col.peek(limit=3)
print(f"\nSample IDs:    {sample['ids']}")
print(f"Sample docs (first 80 chars each):")
for d in (sample.get('documents') or []):
    print(f"  {repr(d[:80])}")
print(f"Sample metadata keys: {[list(m.keys()) for m in (sample.get('metadatas') or [])]}")

print("\n=== DONE (no changes made) ===")
