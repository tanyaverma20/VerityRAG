import urllib.request
import json
import sys

BASE = 'http://127.0.0.1:8001'

try:
    # /health
    r = urllib.request.urlopen(BASE + '/health')
    health = json.loads(r.read())
    print('GET /health:', health)
    assert health['status'] == 'ok', 'health check failed'
    assert health['chunks_indexed'] == 1367, f"Expected 1367 chunks, got {health['chunks_indexed']}"

    # /docs (just check it returns 200)
    r2 = urllib.request.urlopen(BASE + '/docs')
    print('GET /docs: HTTP', r2.status)

    # /documents
    r3 = urllib.request.urlopen(BASE + '/documents')
    docs = json.loads(r3.read())
    print('GET /documents: returned', len(docs), 'documents')

    # /collections
    r4 = urllib.request.urlopen(BASE + '/collections')
    cols = json.loads(r4.read())
    print('GET /collections: returned', len(cols), 'collections')

    print('\nAll endpoint checks PASSED')
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)
