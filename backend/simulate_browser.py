import urllib.request
import urllib.error
import json

BASE = 'http://127.0.0.1:8001'

payload = {
    "question": "What is the main contribution of the paper?",
    "research_type": "simple"
}
data = json.dumps(payload).encode()
req = urllib.request.Request(
    BASE + '/query',
    data=data,
    headers={
        "Content-Type": "application/json",
        "Origin": "http://127.0.0.1:5500",
        "Accept": "*/*"
    },
    method="POST"
)

try:
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
        with open('output.json', 'w', encoding='utf-8') as f:
            json.dump(resp, f, indent=2, ensure_ascii=False)
        print("Success! Response written to output.json")
except urllib.error.HTTPError as e:
    print(f"HTTPError: {e.code} {e.reason}")
    print(e.read().decode())
except Exception as e:
    print(f"Error: {e}")
