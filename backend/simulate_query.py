import urllib.request
import urllib.error
import json
import sys

BASE = 'http://127.0.0.1:8001'

payload = {
    "question": "What is the main contribution of the paper?",
    "research_type": "simple"
}
data = json.dumps(payload).encode()
req = urllib.request.Request(
    BASE + '/query',
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read())
        print(f"Success: {resp.get('answer', '')[:100]}...")
except urllib.error.HTTPError as e:
    print(f"HTTPError: {e.code} {e.reason}")
    print(e.read().decode())
except Exception as e:
    print(f"Error: {e}")
