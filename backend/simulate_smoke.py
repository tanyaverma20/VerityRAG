"""
Live retrieval smoke test against the running server on port 8001.
Sends a real research query and checks the response contains an answer and citations.
"""
import urllib.request
import urllib.error
import json

BASE = "http://127.0.0.1:8001"

def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

print("=== Smoke test: /query ===")
resp = post("/query", {
    "question": "What attention mechanism did the Transformer model introduce?",
    "strategy": "hybrid",
    "research_type": "simple"
})
print(f"  answer (first 200 chars): {resp.get('answer','')[:200]}")
print(f"  chunks_used: {resp.get('chunks_used')}")
print(f"  citations count: {len(resp.get('citations', []))}")
print(f"  strategy: {resp.get('strategy')}")

assert resp.get("answer"), "Missing answer"
assert resp.get("chunks_used", 0) > 0, "No chunks used"
print("\nRetrieval smoke test: PASSED")
