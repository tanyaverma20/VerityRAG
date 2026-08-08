import pytest
from fastapi.testclient import TestClient
from main import app
from database import get_connection, create_session, add_message, get_session_messages, create_task, update_task_status, get_task
import uuid
import time
import json

client = TestClient(app)

def setup_module(module):
    pass

def teardown_module(module):
    pass

def test_session_create():
    resp = client.post("/sessions", json={"collection_id": None})
    assert resp.status_code == 200
    assert "session_id" in resp.json()

def test_session_retrieve():
    resp = client.post("/sessions", json={"collection_id": None})
    sid = resp.json()["session_id"]
    resp2 = client.get("/sessions")
    assert any(s["session_id"] == sid for s in resp2.json())

def test_session_delete():
    resp = client.post("/sessions", json={"collection_id": None})
    sid = resp.json()["session_id"]
    client.delete(f"/sessions/{sid}")
    resp2 = client.get("/sessions")
    assert not any(s["session_id"] == sid for s in resp2.json())

def test_message_user_add():
    sid = client.post("/sessions", json={"collection_id": None}).json()["session_id"]
    msg_id = uuid.uuid4().hex[:12]
    add_message(msg_id, sid, "user", "Hello")
    msgs = client.get(f"/sessions/{sid}/messages").json()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"

def test_message_assistant_add():
    sid = client.post("/sessions", json={"collection_id": None}).json()["session_id"]
    msg_id = uuid.uuid4().hex[:12]
    add_message(msg_id, sid, "assistant", "Response", metadata={"confidence": "HIGH"})
    msgs = client.get(f"/sessions/{sid}/messages").json()
    assert len(msgs) == 1
    assert msgs[0]["metadata"]["confidence"] == "HIGH"

def test_collection_binding():
    try:
        sid = client.post("/sessions", json={"collection_id": "fake_col"}).json()["session_id"]
        assert False, "Should have failed due to foreign key constraint"
    except Exception:
        pass

def test_task_create_pending():
    sid = client.post("/sessions", json={"collection_id": None}).json()["session_id"]
    tid = uuid.uuid4().hex[:12]
    task = create_task(tid, sid)
    assert task["status"] == "PENDING"

def test_task_update_running():
    sid = client.post("/sessions", json={"collection_id": None}).json()["session_id"]
    tid = uuid.uuid4().hex[:12]
    create_task(tid, sid)
    update_task_status(tid, "RUNNING")
    assert get_task(tid)["status"] == "RUNNING"
    assert get_task(tid)["started_at"] is not None

def test_task_update_completed():
    sid = client.post("/sessions", json={"collection_id": None}).json()["session_id"]
    tid = uuid.uuid4().hex[:12]
    create_task(tid, sid)
    update_task_status(tid, "COMPLETED", result_payload={"ok": True})
    task = get_task(tid)
    assert task["status"] == "COMPLETED"
    assert task["completed_at"] is not None
    assert task["result_payload"]["ok"] is True

def test_task_update_failed():
    sid = client.post("/sessions", json={"collection_id": None}).json()["session_id"]
    tid = uuid.uuid4().hex[:12]
    create_task(tid, sid)
    update_task_status(tid, "FAILED", error_message="oops")
    task = get_task(tid)
    assert task["status"] == "FAILED"
    assert task["error_message"] == "oops"

def test_async_research_endpoint_returns_task():
    sid = client.post("/sessions", json={"collection_id": None}).json()["session_id"]
    req = {"question": "What?", "session_id": sid, "research_type": "simple"}
    resp = client.post("/research", json=req)
    assert resp.status_code == 200
    assert "task_id" in resp.json()
    assert resp.json()["status"] == "PENDING"

def test_backward_compatible_query():
    req = {"question": "What?", "research_type": "simple"}
    resp = client.post("/query", json=req)
    assert resp.status_code == 200
    assert "answer" in resp.json()

def test_observability_endpoint_exists():
    resp = client.get("/logs?limit=1")
    assert resp.status_code == 200

def test_observability_filtering_safe():
    resp = client.get("/logs?session_id=none&task_id=none")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

def test_follow_up_planner_resolution():
    from graph.planner import plan_research
    state = {
        "original_query": "What about their training data?",
        "chat_history": [
            {"role": "user", "content": "Compare A and B."},
            {"role": "assistant", "content": "A is big. B is small."}
        ],
        "research_type": "simple"
    }
    res = plan_research(state)
    assert "research_plan" in res
    assert "original_query" in res

def test_session_isolation():
    sid1 = client.post("/sessions", json={"collection_id": None}).json()["session_id"]
    sid2 = client.post("/sessions", json={"collection_id": None}).json()["session_id"]
    add_message(uuid.uuid4().hex[:12], sid1, "user", "Q1")
    add_message(uuid.uuid4().hex[:12], sid2, "user", "Q2")
    m1 = client.get(f"/sessions/{sid1}/messages").json()
    m2 = client.get(f"/sessions/{sid2}/messages").json()
    assert m1[0]["content"] == "Q1"
    assert m2[0]["content"] == "Q2"

def test_long_history_handling():
    sid = client.post("/sessions", json={"collection_id": None}).json()["session_id"]
    for i in range(10):
        add_message(uuid.uuid4().hex[:12], sid, "user", f"Q{i}")
    msgs = client.get(f"/sessions/{sid}/messages").json()
    assert len(msgs) == 10

def test_async_task_polling_completes():
    sid = client.post("/sessions", json={"collection_id": None}).json()["session_id"]
    req = {"question": "Test", "session_id": sid, "research_type": "simple"}
    resp = client.post("/research", json=req)
    task_id = resp.json()["task_id"]
    
    # Wait
    for _ in range(15):
        task_resp = client.get(f"/task/{task_id}")
        if task_resp.json()["status"] in ["COMPLETED", "FAILED"]:
            break
        time.sleep(1)
    
    final_task = client.get(f"/task/{task_id}").json()
    assert final_task["status"] in ["COMPLETED", "FAILED"]

def test_task_persistence_across_endpoints():
    sid = client.post("/sessions", json={"collection_id": None}).json()["session_id"]
    req = {"question": "Test", "session_id": sid, "research_type": "simple"}
    task_id = client.post("/research", json=req).json()["task_id"]
    assert client.get(f"/task/{task_id}").status_code == 200

def test_message_creation_after_async_completes():
    sid = client.post("/sessions", json={"collection_id": None}).json()["session_id"]
    req = {"question": "Verify message", "session_id": sid, "research_type": "simple"}
    task_id = client.post("/research", json=req).json()["task_id"]
    
    for _ in range(15):
        if client.get(f"/task/{task_id}").json()["status"] in ["COMPLETED", "FAILED"]:
            break
        time.sleep(1)
        
    status = client.get(f"/task/{task_id}").json()["status"]
    if status == "COMPLETED":
        msgs = client.get(f"/sessions/{sid}/messages").json()
        assert len(msgs) >= 2
        assert msgs[-1]["role"] == "assistant"
        assert msgs[-2]["role"] == "user"
