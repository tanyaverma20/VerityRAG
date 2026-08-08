"""
ingest_all.py — Temporary ingestion script for Phase 4 setup.
Run once to ingest all available PDFs and print their document_ids.
"""
import sys, json
sys.path.insert(0, '.')
from ingest import ingest_document
from retrieval import build_bm25_index
from pathlib import Path

data_dir = Path('../data')
pdfs = sorted(data_dir.glob('*.pdf'))
results = []
for p in pdfs:
    try:
        r = ingest_document(str(p))
        entry = {
            'file': p.name,
            'status': r.get('status'),
            'document_id': r.get('document_id', ''),
            'chunks_added': r.get('chunks_added', 0)
        }
        results.append(entry)
        print(f"OK   {p.name}  doc_id={entry['document_id']}  chunks={entry['chunks_added']}")
    except Exception as e:
        entry = {'file': p.name, 'status': 'error', 'document_id': '', 'chunks_added': 0, 'error': str(e)}
        results.append(entry)
        print(f"ERR  {p.name}: {e}")

build_bm25_index()
print("\n--- Ingestion Summary ---")
print(json.dumps(results, indent=2))

# Save to a temp file so we can read it back
with open('../data/ingestion_manifest.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nManifest saved to data/ingestion_manifest.json")
