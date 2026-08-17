"""
test_auth_endpoints.py — real HTTP integration tests for /auth/register,
/auth/login, /auth/logout, /auth/me (main.py). Uses FastAPI's TestClient
against the real `app` object (same pattern as the rest of this suite,
e.g. test_adversarial_isolation.py) — real request/response cycle through
FastAPI's routing + dependency injection, not a mock of auth.py. Runs
against the isolated SQLite fixture (conftest.py), never production
PostgreSQL.
"""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _unique_email(tag: str) -> str:
    import uuid
    return f"{tag}-{uuid.uuid4().hex[:8]}@example.com"


# ---------------------------------------------------------------------------
# /auth/register
# ---------------------------------------------------------------------------
def test_register_returns_201_with_user_and_token():
    email = _unique_email("reg")
    resp = client.post("/auth/register", json={"email": email, "password": "correctpassword1"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["user"]["email"] == email
    assert "password_hash" not in body["user"]
    assert len(body["token"]) > 20


def test_register_duplicate_email_returns_409():
    email = _unique_email("dup")
    client.post("/auth/register", json={"email": email, "password": "correctpassword1"})
    resp = client.post("/auth/register", json={"email": email, "password": "differentpassword1"})
    assert resp.status_code == 409


def test_register_invalid_email_returns_400():
    resp = client.post("/auth/register", json={"email": "not-an-email", "password": "correctpassword1"})
    assert resp.status_code == 400


def test_register_short_password_returns_400():
    resp = client.post("/auth/register", json={"email": _unique_email("short"), "password": "short"})
    assert resp.status_code == 400


def test_register_missing_fields_returns_422():
    resp = client.post("/auth/register", json={"email": _unique_email("missing")})
    assert resp.status_code == 422  # FastAPI/pydantic validation, not a 500


def test_register_error_response_never_leaks_a_stack_trace():
    resp = client.post("/auth/register", json={"email": "bad", "password": "x"})
    assert "Traceback" not in resp.text
    assert "raise" not in resp.text.lower()


# ---------------------------------------------------------------------------
# /auth/login
# ---------------------------------------------------------------------------
def test_login_with_correct_credentials_returns_200_with_token():
    email = _unique_email("login")
    client.post("/auth/register", json={"email": email, "password": "correctpassword1"})
    resp = client.post("/auth/login", json={"email": email, "password": "correctpassword1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["user"]["email"] == email
    assert len(body["token"]) > 20


def test_login_with_wrong_password_returns_401():
    email = _unique_email("wrongpw")
    client.post("/auth/register", json={"email": email, "password": "correctpassword1"})
    resp = client.post("/auth/login", json={"email": email, "password": "totallywrongpw1"})
    assert resp.status_code == 401


def test_login_with_unknown_email_returns_401_not_404():
    """Must not leak whether the account exists via status code either."""
    resp = client.post("/auth/login", json={"email": _unique_email("never"), "password": "anypassword1"})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /auth/me
# ---------------------------------------------------------------------------
def test_me_with_no_authorization_header_returns_401():
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_me_with_invalid_token_returns_401():
    resp = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


def test_me_with_valid_token_returns_the_real_user():
    email = _unique_email("me")
    reg = client.post("/auth/register", json={"email": email, "password": "correctpassword1"})
    token = reg.json()["token"]
    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["email"] == email
    assert "password_hash" not in resp.json()


# ---------------------------------------------------------------------------
# /auth/logout
# ---------------------------------------------------------------------------
def test_logout_invalidates_the_token_a_subsequent_me_call_then_401s():
    email = _unique_email("logout")
    reg = client.post("/auth/register", json={"email": email, "password": "correctpassword1"})
    token = reg.json()["token"]
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200

    logout_resp = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout_resp.status_code == 200

    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_logout_with_no_token_is_a_safe_noop_not_an_error():
    resp = client.post("/auth/logout")
    assert resp.status_code == 200


def test_logout_with_an_already_invalid_token_is_a_safe_noop():
    resp = client.post("/auth/logout", headers={"Authorization": "Bearer never-issued-token"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# End-to-end: register -> me -> logout -> me(401) -> login again -> me
# ---------------------------------------------------------------------------
def test_full_register_me_logout_login_me_roundtrip():
    email = _unique_email("roundtrip")
    reg = client.post("/auth/register", json={"email": email, "password": "correctpassword1"})
    token_a = reg.json()["token"]
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token_a}"}).status_code == 200

    client.post("/auth/logout", headers={"Authorization": f"Bearer {token_a}"})
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token_a}"}).status_code == 401

    login = client.post("/auth/login", json={"email": email, "password": "correctpassword1"})
    token_b = login.json()["token"]
    assert token_b != token_a
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token_b}"})
    assert me.status_code == 200
    assert me.json()["email"] == email
