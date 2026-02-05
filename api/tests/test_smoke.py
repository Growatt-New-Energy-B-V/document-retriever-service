"""Smoke tests for the document-retriever API."""
import io
import json

import pytest

pytestmark = pytest.mark.asyncio


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_post_task(client):
    schema = json.dumps({"product": {"name": "string"}})
    files = {"file": ("test.pdf", b"%PDF-1.4 fake content", "application/pdf")}
    data = {"schema": schema}
    resp = await client.post("/tasks", files=files, data=data)
    assert resp.status_code == 202
    body = resp.json()
    assert "task_id" in body
    assert len(body["task_id"]) > 0


async def test_post_task_invalid_schema(client):
    files = {"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")}
    data = {"schema": "not-json{{{"}
    resp = await client.post("/tasks", files=files, data=data)
    assert resp.status_code == 400


async def test_post_task_unsupported_format(client):
    schema = json.dumps({"key": "value"})
    files = {"file": ("test.pptx", b"fake pptx", "application/vnd.ms-powerpoint")}
    data = {"schema": schema}
    resp = await client.post("/tasks", files=files, data=data)
    assert resp.status_code == 400
    assert "Unsupported" in resp.json()["detail"]


async def test_get_task_status(client):
    # Create a task first
    schema = json.dumps({"key": "value"})
    files = {"file": ("test.txt", b"hello world", "text/plain")}
    data = {"schema": schema}
    create_resp = await client.post("/tasks", files=files, data=data)
    task_id = create_resp.json()["task_id"]

    # Get status
    resp = await client.get(f"/tasks/{task_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["status"] in ("queued", "running", "succeeded", "failed")


async def test_get_task_not_found(client):
    resp = await client.get("/tasks/nonexistent-id")
    assert resp.status_code == 404


async def test_result_not_ready(client):
    schema = json.dumps({"key": "value"})
    files = {"file": ("test.txt", b"hello", "text/plain")}
    data = {"schema": schema}
    create_resp = await client.post("/tasks", files=files, data=data)
    task_id = create_resp.json()["task_id"]

    resp = await client.get(f"/tasks/{task_id}/result")
    assert resp.status_code == 400
    assert "not complete" in resp.json()["detail"]
