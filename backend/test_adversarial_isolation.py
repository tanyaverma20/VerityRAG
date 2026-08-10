"""
test_adversarial_isolation.py — Phase 4 multi-tenant/vector isolation
security audit: ADVERSARIAL tests (a malicious or buggy client deliberately
claiming/guessing another workspace's identifiers), not just happy-path
scoping checks. Complements:
  - test_workspace_isolation.py       (/query, /documents, /sessions)
  - test_workspace_vector_isolation.py (retrieval.py internals)

This file covers the gaps those didn't: /analyze (every PDF-grounded mode,
not just /query), and background task results (/research + GET
/task/{task_id}).

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

client = TestClient(app)

FIXTURE_PDF = Path(__file__).parent / "tests" / "fixtures" / "attention.pdf"


def _upload(workspace_id: str) -> dict:
    with open(FIXTURE_PDF, "rb") as f:
        resp = client.post(
            "/upload",
            files={"file": ("attention.pdf", f, "application/pdf")},
            data={"workspace_id": workspace_id},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _two_workspaces_with_docs():
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated test fixture"
    ws_a = client.post("/workspaces", json={"name": "Adversarial A"}).json()["workspace_id"]
    ws_b = client.post("/workspaces", json={"name": "Adversarial B"}).json()["workspace_id"]
    doc_a = _upload(ws_a)["document_id"]
    return ws_a, ws_b, doc_a


# ---------------------------------------------------------------------------
# /analyze — every PDF-grounded mode must reject a cross-workspace document,
# not just /query. A buggy or malicious client from workspace B explicitly
# names workspace A's real document_id.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode,extra", [
    ("viva", {}),
    ("mock_test", {}),
    ("evaluate_paper", {}),
    ("recommend", {}),
    ("explain_figure", {"figure_reference": "Figure 1"}),
])
def test_analyze_rejects_cross_workspace_document(mode, extra):
    ws_a, ws_b, doc_a = _two_workspaces_with_docs()

    resp = client.post("/analyze", json={
        "mode": mode,
        "document_ids": [doc_a],   # a REAL document_id, but owned by ws_a
        "workspace_id": ws_b,      # claiming a DIFFERENT workspace
        **extra,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False, (
        f"mode={mode}: a document belonging to a different workspace must never be usable, "
        f"got ok=True with payload keys {list(data.keys())}"
    )
    # Must fail via the "no documents scoped" path (data-layer rejection),
    # not some unrelated downstream error that happens to also return ok=False.
    assert "document" in data.get("error", "").lower() or "upload" in data.get("error", "").lower(), (
        f"mode={mode}: expected a no-documents-scoped rejection, got: {data.get('error')!r}"
    )


def test_analyze_same_workspace_still_works_normally():
    """Sanity check that the rejection above is really about the workspace
    mismatch, not that /analyze is broken for evaluate_paper generally."""
    ws_a, ws_b, doc_a = _two_workspaces_with_docs()
    resp = client.post("/analyze", json={
        "mode": "evaluate_paper",
        "document_ids": [doc_a],
        "workspace_id": ws_a,  # CORRECT workspace this time
    })
    assert resp.status_code == 200
    data = resp.json()
    # Groq isn't mocked here, so this only asserts we got PAST the
    # data-layer scoping check (evidence was found) — not that generation
    # itself succeeded (which needs a real/mocked LLM call, covered by
    # test_new_analysis_modes.py).
    assert data.get("error") != "You haven't uploaded any documents yet. Please upload a PDF to get started."


# ---------------------------------------------------------------------------
# Background tasks — GET /task/{task_id} must not leak another workspace's
# research result just because the caller learned/guessed the task_id.
# ---------------------------------------------------------------------------
def test_task_result_not_readable_from_a_different_workspace():
    ws_a, ws_b, doc_a = _two_workspaces_with_docs()

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
        })
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

    # Owner can read it via the API.
    own_resp = client.get(f"/task/{task_id}", params={"workspace_id": ws_a})
    assert own_resp.status_code == 200
    assert own_resp.json()["result_payload"] is not None

    # A different workspace claiming the SAME task_id must be rejected —
    # not shown the answer, citations, or evidence from ws_a's task.
    foreign_resp = client.get(f"/task/{task_id}", params={"workspace_id": ws_b})
    assert foreign_resp.status_code == 404, (
        f"workspace_id={ws_b} must never be able to read ws_a's task result, got {foreign_resp.status_code}: {foreign_resp.text}"
    )


def test_task_without_workspace_id_remains_backward_compatible():
    """A task created by a caller that never supplied workspace_id (legacy /
    no-workspace usage) has nothing to enforce — it must stay reachable by
    ANY caller exactly as before this fix, including one that omits
    workspace_id on the GET too."""
    from database import create_task
    task_id = "legacy_task_" + Path(__file__).stem
    create_task(task_id, session_id=None)  # no workspace_id at all

    resp = client.get(f"/task/{task_id}")
    assert resp.status_code == 200

    # Even a caller that DOES pass a workspace_id must still be able to read
    # a workspace-less legacy task (there's no owner to contradict).
    resp2 = client.get(f"/task/{task_id}", params={"workspace_id": "some_workspace"})
    assert resp2.status_code == 200


def test_nonexistent_task_id_is_404_regardless_of_workspace():
    resp = client.get("/task/definitely_does_not_exist", params={"workspace_id": "anything"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /report — previously had NO workspace scoping at all (ReportRequest had no
# workspace_id field, and req.document_ids was used directly with no
# ownership check). A client from workspace B must not be able to generate
# a comparative report over workspace A's documents just by naming their
# content-hash document_ids.
# ---------------------------------------------------------------------------
def test_report_rejects_cross_workspace_documents():
    ws_a, ws_b, doc_a = _two_workspaces_with_docs()

    resp = client.post("/report", json={
        "document_ids": [doc_a],  # real document, owned by ws_a
        "workspace_id": ws_b,     # claiming a DIFFERENT workspace
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False, f"Expected rejection, got: {data}"
    assert data["report_id"] is None
    assert "document" in data["error"].lower() or "upload" in data["error"].lower()


def test_delete_document_rejects_a_different_workspaces_document():
    """A malicious/buggy client from workspace B must not be able to
    DELETE a document that belongs to workspace A, even though document_id
    is a deterministic hash it could compute on its own (e.g. by owning an
    identical copy of the same public PDF)."""
    ws_a, ws_b, doc_a = _two_workspaces_with_docs()

    resp = client.delete(f"/documents/{doc_a}", params={"workspace_id": ws_b})
    assert resp.status_code == 404, f"Expected rejection, got {resp.status_code}: {resp.text}"

    # The document must still exist, fully intact, in its real workspace.
    still_there = client.get(f"/documents/{doc_a}", params={"workspace_id": ws_a})
    assert still_there.status_code == 200
    assert still_there.json()["document_id"] == doc_a


def test_get_document_rejects_a_different_workspaces_document():
    ws_a, ws_b, doc_a = _two_workspaces_with_docs()
    resp = client.get(f"/documents/{doc_a}", params={"workspace_id": ws_b})
    assert resp.status_code == 404


def test_delete_document_still_works_for_its_own_workspace():
    ws_a, ws_b, doc_a = _two_workspaces_with_docs()
    resp = client.delete(f"/documents/{doc_a}", params={"workspace_id": ws_a})
    assert resp.status_code == 200

    gone = client.get(f"/documents/{doc_a}")  # unscoped GET still finds the row (status FAILED/removed downstream)
    # get_document() returns the row until fully purged by delete_document_from_index;
    # the important assertion is that the SAME-workspace delete was accepted (200), not rejected.


def test_document_endpoints_stay_unscoped_when_workspace_id_omitted():
    """Backward compatibility: a caller that never supplies workspace_id at
    all must see the exact original (pre-this-fix) behavior."""
    ws_a, ws_b, doc_a = _two_workspaces_with_docs()
    resp = client.get(f"/documents/{doc_a}")  # no workspace_id param at all
    assert resp.status_code == 200
    assert resp.json()["document_id"] == doc_a


def test_report_without_workspace_id_keeps_legacy_unscoped_behavior():
    """Omitting workspace_id entirely must reproduce the exact pre-existing
    behavior (document_ids alone govern scope) — this is the same
    backward-compatibility contract every other workspace_id parameter in
    this API already honors."""
    ws_a, ws_b, doc_a = _two_workspaces_with_docs()

    with patch("main.generate_report") as mock_gen:
        mock_gen.return_value = {"ok": False, "error": "no evidence (mocked, avoids a real LLM call)"}
        resp = client.post("/report", json={"document_ids": [doc_a]})  # no workspace_id at all
    assert resp.status_code == 200
    # Must have reached generate_report() at all (i.e. NOT rejected at the
    # scoping layer) — the mock controls what happens after that.
    mock_gen.assert_called_once()
    called_ids = mock_gen.call_args[0][0]
    assert doc_a in called_ids
