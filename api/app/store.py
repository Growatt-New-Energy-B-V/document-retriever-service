"""SQLite task persistence."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import aiosqlite
import ulid

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("DB_PATH", "/data/tasks.db")

_db: Optional[aiosqlite.Connection] = None

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'queued',
    original_filename TEXT NOT NULL,
    schema_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    result_json TEXT,
    error_json TEXT
)
"""


async def init_db() -> None:
    """Initialize database connection and create tables."""
    global _db
    if _db is not None:
        return
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    _db = await aiosqlite.connect(DB_PATH)
    _db.row_factory = aiosqlite.Row
    await _db.execute(CREATE_TABLE)
    await _db.commit()
    logger.info(f"Database initialized at {DB_PATH}")


async def close_db() -> None:
    """Close database connection."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None


async def _get_db() -> aiosqlite.Connection:
    if _db is None:
        await init_db()
    return _db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_task(original_filename: str, schema_json: str) -> str:
    """Create a new task. Returns task_id."""
    db = await _get_db()
    task_id = str(ulid.new())
    now = _now()
    await db.execute(
        "INSERT INTO tasks (task_id, status, original_filename, schema_json, created_at, updated_at) "
        "VALUES (?, 'queued', ?, ?, ?, ?)",
        (task_id, original_filename, schema_json, now, now),
    )
    await db.commit()
    logger.info(f"Task {task_id} created")
    return task_id


async def get_task(task_id: str) -> Optional[dict]:
    """Get task by ID. Returns dict or None."""
    db = await _get_db()
    async with db.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)) as cursor:
        row = await cursor.fetchone()
        return dict(row) if row else None


async def update_task_status(task_id: str, status: str) -> None:
    """Update task status."""
    db = await _get_db()
    now = _now()
    updates = {"status": status, "updated_at": now}
    if status == "running":
        updates["started_at"] = now
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [task_id]
    await db.execute(f"UPDATE tasks SET {set_clause} WHERE task_id = ?", values)
    await db.commit()


async def update_task_result(task_id: str, result: dict, timing: dict) -> None:
    """Mark task as succeeded with result."""
    db = await _get_db()
    now = _now()
    result_with_timing = {**result, "timing": timing}
    await db.execute(
        "UPDATE tasks SET status = 'succeeded', result_json = ?, finished_at = ?, updated_at = ? WHERE task_id = ?",
        (json.dumps(result_with_timing), now, now, task_id),
    )
    await db.commit()


async def update_task_error(task_id: str, message: str, details: list[str] | None = None) -> None:
    """Mark task as failed with error."""
    db = await _get_db()
    now = _now()
    error = {"message": message, "details": details or []}
    await db.execute(
        "UPDATE tasks SET status = 'failed', error_json = ?, finished_at = ?, updated_at = ? WHERE task_id = ?",
        (json.dumps(error), now, now, task_id),
    )
    await db.commit()
