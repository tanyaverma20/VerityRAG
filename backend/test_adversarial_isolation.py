"""
test_adversarial_isolation.py — Phase 4 multi-tenant/vector isolation
security audit: ADVERSARIAL tests (a malicious or buggy client deliberately
claiming/guessing another user's real workspace/document/task/report
identifiers), not just happy-path scoping checks. Complements:
  - test_workspace_isolation.py       (/query, /documents, /sessions — same owner)
  - test_workspace_vector_isolation.py (retrieval.py internals)
  - test_authorization.py             (cross-user adversarial coverage, general)

This file's distinct coverage: /analyze rejection across every PDF-grounded
mode (not just /query), background task results, and /report — all against
TWO REAL, SEPARATELY REGISTERED USERS.

UPDATED (real authentication landed): every test in this file originally
exercised the pre-auth model, where a bare client-supplied workspace_id was
itself treated as the isolation boundary and several tests explicitly
documented "omitting workspace_id keeps the original unscoped behavior" as
a real, intentional escape hatch. That escape hatch is exactly the
vulnerability real authentication closes ("Do NOT use workspace_id
supplied by the client as the security principal" — see main.py's
_require_workspace_owner()/_require_resource_owner()). Every test below
now uses two real, separately registered users; the handful of tests whose
entire premise was "no workspace_id => unscoped access with no auth at
all" have been rewritten (not deleted) to assert the new, correct
behavior instead — real authentication is required, and workspace_id is
now mandatory (and ownership-checked) for /report same as /query/analyze.
The one genuinely-unchanged backward-compatibility guarantee — a
workspace-LESS resource (no workspace_id at all, e.g. a legacy task) stays
reachable to any authenticated caller — is still covered and still true.

Runs entirely against the ISOLATED test Chroma/SQLite fixture (see
conftest.py) — never backend/chroma_store or the real data/registry.db.
"""
from pathlib import Path
from unittest.mock import patch
import json
import time

import pytest
from fastapi.testclient import TestClient

import config
from main import app
from database import get_task, task_belongs_to_workspace
from test_auth_helpers import registered_user_with_workspace

client = TestClient(app)

FIXTURE_PDF = Path(__file__).parent / "tests" / "fixtures" / "attention.pdf"


def _upload(workspace_id: str, headers: dict) -> dict:
    with open(FIXTURE_PDF, "rb") as f:
        resp = client.post(
            "/upload",
            files={"file": ("attention.pdf", f, "application/pdf")},
            data={"workspace_id": workspace_id},
            headers=headers,
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _two_real_users_with_docs():
    """Two REAL, separately registered users, each owning their own real
    workspace and document — the actual adversarial setup now that
    workspace_id alone is never trusted as a security principal."""
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated test fixture"
    headers_a, ws_a = registered_user_with_workspace(client, "adv_a")
    headers_b, ws_b = registered_user_with_workspace(client, "adv_b")
    doc_a = _upload(ws_a, headers_a)["document_id"]
    return headers_a, ws_a, headers_b, ws_b, doc_a


# ---------------------------------------------------------------------------
# /analyze — every PDF-grounded mode must reject a cross-user document,
# not just /query. User B explicitly names User A's real document_id AND
# claims User A's real workspace_id — both real, neither actually theirs.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode,extra", [
    ("viva", {}),
    ("mock_test", {}),
    ("evaluate_paper", {}),
    ("recommend", {}),
    ("explain_figure", {"figure_reference": "Figure 1"}),
])
def test_analyze_rejects_cross_user_document(mode, extra):
    headers_a, ws_a, headers_b, ws_b, doc_a = _two_real_users_with_docs()

    resp = client.post("/analyze", json={
        "mode": mode,
        "document_ids": [doc_a],   # a REAL document_id, but owned by user A
        "workspace_id": ws_a,      # user A's REAL workspace_id — but B doesn't own it
        **extra,
    }, headers=headers_b)
    assert resp.status_code == 404, (
        f"mode={mode}: a workspace/document belonging to a different user must never be "
        f"usable, even naming real IDs — got {resp.status_code}: {resp.text}"
    )


def test_analyze_same_owner_still_works_normally():
    """Sanity check that the rejection above is really about ownership, not
    that /analyze is broken for evaluate_paper generally."""
    headers_a, ws_a, headers_b, ws_b, doc_a = _two_real_users_with_docs()
    resp = client.post("/analyze", json={
        "mode": "evaluate_paper",
        "document_ids": [doc_a],
        "workspace_id": ws_a,  # A acting on A's own workspace
    }, headers=headers_a)
    assert resp.status_code == 200
    data = resp.json()
    # Groq isn't mocked here, so this only asserts we got PAST the
    # data-layer scoping check (evidence was found) — not that generation
    # itself succeeded (which needs a real/mocked LLM call, covered by
    # test_new_analysis_modes.py).
    assert data.get("error") != "You haven't uploaded any documents yet. Please upload a PDF to get started."


# ---------------------------------------------------------------------------
# Background tasks — GET /task/{task_id} must not leak another user's
# research result just because the caller learned/guessed the task_id.
# ---------------------------------------------------------------------------
def test_task_result_not_readable_by_a_different_user():
    headers_a, ws_a, headers_b, ws_b, doc_a = _two_real_users_with_docs()

    class _FakeMessage:
        def __init__(self, content): self.content = content
    class _FakeChoice:
        def __init__(self, content): self.message = _FakeMessage(content)
    class _FakeUsage:
        prompt_tokens = 40
        completion_tokens = 12
    class _FakeResponse:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]
            self.usage = _FakeUsage()
    class _FakeCompletions:
        def create(self, model, messages, temperature=0.0, max_tokens=None):
            return _FakeResponse(json.dumps({
                "answer": "The Transformer relies on self-attention.",
                "grounded": True, "evidence_sufficient": True, "document_ids": [doc_a],
            }))
    class _FakeChat:
        def __init__(self): self.completions = _FakeCompletions()
    class _FakeGroqClient:
        def __init__(self, api_key=None): self.chat = _FakeChat()

    with patch("groq.Groq", _FakeGroqClient):
        resp = client.post("/research", json={
            "question": "What is the main contribution?",
            "document_ids": [doc_a],
            "workspace_id": ws_a,
            "research_type": "simple",
        }, headers=headers_a)
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

        # BackgroundTasks run synchronously within TestClient's request
        # lifecycle (no separate worker process in tests), so the task
        # should already be COMPLETED by the time this returns.
        for _ in range(20):
            task = get_task(task_id)
            if task and task["status"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.1)

    assert task is not None and task["status"] == "COMPLETED", f"Task did not complete: {task}"
    assert task["workspace_id"] == ws_a

    # The task truly belongs to ws_a, not ws_b.
    assert task_belongs_to_workspace(task_id, ws_a) is True
    assert task_belongs_to_workspace(task_id, ws_b) is False

    # Owner (A) can read it via the API.
    own_resp = client.get(f"/task/{task_id}", headers=headers_a)
    assert own_resp.status_code == 200
    assert own_resp.json()["result_payload"] is not None

    # A different, real, authenticated user (B) must be rejected — not
    # shown the answer, citations, or evidence from A's task — even though
    # B is a real, valid, logged-in user.
    foreign_resp = client.get(f"/task/{task_id}", headers=headers_b)
    assert foreign_resp.status_code == 404, (
        f"User B must never be able to read User A's task result, got {foreign_resp.status_code}: {foreign_resp.text}"
    )


def test_task_without_workspace_id_remains_reachable_to_any_authenticated_user():
    """A task created with no workspace_id at all (legacy/no-workspace
    usage) has no ownership to enforce — it stays reachable by any
    AUTHENTICATED caller (unchanged backward-compat guarantee), but an
    unauthenticated request is still rejected — auth itself is never
    optional, only the workspace-ownership check is skipped for a
    workspace-less resource."""
    from database import create_task
    headers_a, _ = registered_user_with_workspace(client, "adv_legacy_task")

    task_id = "legacy_task_" + Path(__file__).stem
    create_task(task_id, session_id=None)  # no workspace_id at all

    assert client.get(f"/task/{task_id}").status_code == 401, "Auth itself is never optional"

    resp = client.get(f"/task/{task_id}", headers=headers_a)
    assert resp.status_code == 200

    # A second, unrelated authenticated user can reach it too — there's no
    # owner to contradict.
    headers_b, _ = registered_user_with_workspace(client, "adv_legacy_task_b")
    resp2 = client.get(f"/task/{task_id}", headers=headers_b)
    assert resp2.status_code == 200


def test_nonexistent_task_id_is_404():
    headers_a, _ = registered_user_with_workspace(client, "adv_missing_task")
    resp = client.get("/task/definitely_does_not_exist", headers=headers_a)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /report — report_id is itself a deterministic, guessable-format cache
# key. A different user must not be able to generate OR read back a
# comparative report over another user's documents just by naming their
# real content-hash document_ids / real workspace_id / real report_id.
# ---------------------------------------------------------------------------
def test_report_rejects_cross_user_documents():
    headers_a, ws_a, headers_b, ws_b, doc_a = _two_real_users_with_docs()

    resp = client.post("/report", json={
        "document_ids": [doc_a],  # real document, owned by user A
        "workspace_id": ws_a,     # user A's real workspace — B doesn't own it
    }, headers=headers_b)
    assert resp.status_code == 404, f"Expected rejection, got {resp.status_code}: {resp.text}"


def test_delete_document_rejects_a_different_users_document():
    """A malicious/buggy client (User B) must not be able to DELETE a
    document that belongs to User A, even though document_id is a
    deterministic hash it could compute on its own (e.g. by owning an
    identical copy of the same public PDF)."""
    headers_a, ws_a, headers_b, ws_b, doc_a = _two_real_users_with_docs()

    resp = client.delete(f"/documents/{doc_a}", headers=headers_b)
    assert resp.status_code == 404, f"Expected rejection, got {resp.status_code}: {resp.text}"

    # The document must still exist, fully intact, for its real owner.
    still_there = client.get(f"/documents/{doc_a}", headers=headers_a)
    assert still_there.status_code == 200
    assert still_there.json()["document_id"] == doc_a


def test_get_document_rejects_a_different_users_document():
    headers_a, ws_a, headers_b, ws_b, doc_a = _two_real_users_with_docs()
    resp = client.get(f"/documents/{doc_a}", headers=headers_b)
    assert resp.status_code == 404


def test_delete_document_still_works_for_its_real_owner():
    headers_a, ws_a, headers_b, ws_b, doc_a = _two_real_users_with_docs()
    resp = client.delete(f"/documents/{doc_a}", headers=headers_a)
    assert resp.status_code == 200


def test_document_and_report_endpoints_require_real_authentication():
    """UPDATED (was: 'stays unscoped when workspace_id omitted'). Omitting
    workspace_id no longer means 'unscoped access with no auth' — that
    escape hatch is exactly what real authentication closes. Every one of
    these now requires a real, valid session; workspace_id itself became
    mandatory (not merely optional-but-unscoped) for /report specifically,
    matching /query and /analyze."""
    headers_a, ws_a, headers_b, ws_b, doc_a = _two_real_users_with_docs()

    assert client.get(f"/documents/{doc_a}").status_code == 401  # no auth at all
    assert client.get(f"/documents/{doc_a}", headers=headers_a).status_code == 200  # real owner, real auth

    resp = client.post("/report", json={"document_ids": [doc_a]}, headers=headers_a)  # workspace_id omitted
    assert resp.status_code == 400, "workspace_id is now a required field for /report, not an optional unscoped mode"
