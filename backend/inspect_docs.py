import sys, json, os
sys.path.insert(0, '.')
from ingest import get_collection
coll = get_collection()
manifest = json.load(open('../data/ingestion_manifest.json'))

for m in manifest:
    doc_id = m['document_id']
    if not doc_id: continue
    res = coll.get(where={'document_id': doc_id}, limit=1)
    if res['documents'] and res['documents'][0]:
        print(f"{m['file']} -> {res['documents'][0][:150]}")
    else:
        print(f"{m['file']} -> NO CHUNKS FOUND")
