"""Tests for SQLite task store."""
import asyncio
import json
import os
import tempfile

import pytest

# Override DB path before import
_tmpdir = tempfile.mkdtemp()
os.environ["DB_PATH"] = os.path.join(_tmpdir, "test_tasks.db")
os.environ["DATA_DIR"] = _tmpdir

from app.store import (
    init_db,
    close_db,
    create_task,
    get_task,
    update_task_status,
    update_task_result,
    update_task_error,
)


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize DB for each test."""
    asyncio.get_event_loop().run_until_complete(init_db())
    yield
    asyncio.get_event_loop().run_until_complete(close_db())


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_create_and_get_task():
    task_id = run(create_task("test.pdf", '{"key": "value"}'))
    assert task_id is not None
    task = run(get_task(task_id))
    assert task is not None
    assert task["task_id"] == task_id
    assert task["status"] == "queued"
    assert task["original_filename"] == "test.pdf"


def test_get_nonexistent_task():
    task = run(get_task("nonexistent"))
    assert task is None


def test_update_status():
    task_id = run(create_task("test.pdf", "{}"))
    run(update_task_status(task_id, "running"))
    task = run(get_task(task_id))
    assert task["status"] == "running"


def test_update_result():
    task_id = run(create_task("test.pdf", "{}"))
    run(update_task_status(task_id, "running"))
    result = {"data": {"key": "val"}, "evidence": {}, "warnings": []}
    timing = {"total_seconds": 10.5}
    run(update_task_result(task_id, result, timing))
    task = run(get_task(task_id))
    assert task["status"] == "succeeded"
    assert json.loads(task["result_json"])["data"]["key"] == "val"


def test_update_error():
    task_id = run(create_task("test.pdf", "{}"))
    run(update_task_status(task_id, "running"))
    run(update_task_error(task_id, "Something broke", ["detail1"]))
    task = run(get_task(task_id))
    assert task["status"] == "failed"
    error = json.loads(task["error_json"])
    assert error["message"] == "Something broke"
