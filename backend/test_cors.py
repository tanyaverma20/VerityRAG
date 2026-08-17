"""
test_cors.py — CORS is configurable and restricted, never wide-open
(security audit finding: main.py previously hardcoded
allow_origins=["*"]). Real requests through Starlette's CORSMiddleware via
TestClient, not a mock of the middleware.
"""
from fastapi.testclient import TestClient

import config
from main import app

client = TestClient(app)


def test_default_dev_cors_origins_are_specific_not_wildcard():
    assert "*" not in config.CORS_ALLOWED_ORIGINS, "CORS must never be wide-open, even by default"
    assert len(config.CORS_ALLOWED_ORIGINS) > 0
    assert all(o.startswith("http://") or o.startswith("https://") for o in config.CORS_ALLOWED_ORIGINS)


def test_allowed_origin_receives_cors_headers():
    allowed = config.CORS_ALLOWED_ORIGINS[0]
    resp = client.get("/health", headers={"Origin": allowed})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == allowed


def test_disallowed_origin_receives_no_cors_header():
    resp = client.get("/health", headers={"Origin": "https://evil-attacker.example.com"})
    assert resp.status_code == 200  # the request itself isn't blocked server-side...
    # ...but the browser-enforced CORS header must NOT authorize that origin.
    assert resp.headers.get("access-control-allow-origin") != "https://evil-attacker.example.com"
    assert resp.headers.get("access-control-allow-origin") is None


def test_cors_allowed_origins_is_configurable_via_env():
    # Tests the real parsing function config.py's module-level
    # CORS_ALLOWED_ORIGINS is built from, without reloading the module
    # itself (a reload would re-run load_dotenv(), risking the exact
    # env-var-restoration footgun conftest.py documents at length).
    assert config.parse_cors_origins("https://custom-frontend.example.com") == ["https://custom-frontend.example.com"]
    assert config.parse_cors_origins("https://a.example.com, https://b.example.com ,https://c.example.com") == [
        "https://a.example.com", "https://b.example.com", "https://c.example.com",
    ]
    assert config.parse_cors_origins("") == []
