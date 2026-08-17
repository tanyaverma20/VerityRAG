"""
test_auth_helpers.py — shared helpers for tests that exercise endpoints
now behind real authentication/ownership (see auth.py, main.py's
_require_workspace_owner / _require_resource_owner). Not itself a test
file (no test_ functions) — imported by other test_*.py files.
"""
import uuid

from fastapi.testclient import TestClient


def unique_email(tag: str = "user") -> str:
    return f"{tag}-{uuid.uuid4().hex[:10]}@example.com"


def register(client: TestClient, tag: str = "user") -> tuple[dict, dict]:
    """Registers a brand-new user and returns (user_dict, auth_headers)."""
    email = unique_email(tag)
    resp = client.post("/auth/register", json={"email": email, "password": "correctpassword1"})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    return body["user"], {"Authorization": f"Bearer {body['token']}"}


def owned_workspace(client: TestClient, headers: dict, name: str = "Test Workspace") -> str:
    """Creates a workspace owned by whoever `headers` authenticates as,
    returns its workspace_id."""
    resp = client.post("/workspaces", json={"name": name}, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["workspace_id"]


def registered_user_with_workspace(client: TestClient, tag: str = "user") -> tuple[dict, str]:
    """Convenience: register a new user + create one workspace they own.
    Returns (auth_headers, workspace_id)."""
    _, headers = register(client, tag)
    ws_id = owned_workspace(client, headers, name=f"{tag} workspace")
    return headers, ws_id
