"""Background task queue and worker for document extraction."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event

from .config import settings
from .store import get_task, update_task_status, update_task_result, update_task_error

logger = logging.getLogger(__name__)

_task_queue: Queue[str] = Queue()
_workers: list[Thread] = []
_shutdown = Event()
_loop: asyncio.AbstractEventLoop | None = None


def enqueue_task(task_id: str) -> None:
    """Add a task to the processing queue."""
    _task_queue.put(task_id)
    logger.info(f"Task {task_id} enqueued")


def _worker_thread(worker_id: int, loop: asyncio.AbstractEventLoop) -> None:
    """Worker thread that processes tasks."""
    logger.info(f"Worker {worker_id} started")
    while not _shutdown.is_set():
        try:
            task_id = _task_queue.get(timeout=1.0)
        except Empty:
            continue

        logger.info(f"Worker {worker_id} processing task {task_id}")
        try:
            future = asyncio.run_coroutine_threadsafe(_process_task(task_id), loop)
            future.result(timeout=settings.AGENT_TIMEOUT + 60)
        except Exception as e:
            logger.exception(f"Worker {worker_id} error on task {task_id}: {e}")
            asyncio.run_coroutine_threadsafe(
                update_task_error(task_id, f"Worker error: {str(e)[:500]}"),
                loop,
            )
        finally:
            _task_queue.task_done()

    logger.info(f"Worker {worker_id} stopped")


async def _process_task(task_id: str) -> None:
    """Process a single extraction task."""
    task = await get_task(task_id)
    if not task:
        logger.error(f"Task {task_id} not found")
        return

    await update_task_status(task_id, "running")
    start_time = time.time()

    try:
        # Find the uploaded file
        upload_dir = Path(settings.DATA_DIR) / "uploads" / task_id
        files = list(upload_dir.iterdir()) if upload_dir.exists() else []
        if not files:
            raise FileNotFoundError(f"No uploaded file found for task {task_id}")
        file_path = str(files[0])
        file_size = Path(file_path).stat().st_size
        logger.info(f"Task {task_id}: file={file_path} ({file_size} bytes)")

        schema = json.loads(task["schema_json"])
        logger.info(f"Task {task_id}: schema has {len(schema)} top-level keys: {list(schema.keys())}")
        logger.info(
            f"Task {task_id}: config timeout={settings.AGENT_TIMEOUT}s, "
            f"retries={settings.AGENT_MAX_RETRIES}, mcp_url={settings.MCP_TOOLS_URL}"
        )

        # Run extraction agent with retry
        from .agent import run_extraction

        last_error = None
        for attempt in range(settings.AGENT_MAX_RETRIES + 1):
            attempt_start = time.time()
            logger.info(f"Task {task_id}: attempt {attempt+1}/{settings.AGENT_MAX_RETRIES+1}")
            try:
                result = await asyncio.wait_for(
                    run_extraction(file_path, schema),
                    timeout=settings.AGENT_TIMEOUT,
                )
                total_time = time.time() - start_time
                attempt_time = time.time() - attempt_start

                sdk_usage = result.pop("usage", {})
                timing = {
                    "total_seconds": round(total_time, 1),
                    **sdk_usage,
                }

                logger.info(
                    f"Task {task_id}: attempt {attempt+1} succeeded in {attempt_time:.1f}s "
                    f"(total {total_time:.1f}s)"
                )
                await update_task_result(task_id, result, timing)
                logger.info(f"Task {task_id} result saved")
                return

            except asyncio.TimeoutError:
                attempt_time = time.time() - attempt_start
                last_error = TimeoutError(
                    f"Agent timed out after {attempt_time:.0f}s (limit {settings.AGENT_TIMEOUT}s)"
                )
                logger.error(f"Task {task_id}: attempt {attempt+1} TIMEOUT after {attempt_time:.0f}s")
                if attempt < settings.AGENT_MAX_RETRIES:
                    logger.info(f"Task {task_id}: retrying in 2s...")
                    await asyncio.sleep(2)

            except Exception as e:
                attempt_time = time.time() - attempt_start
                last_error = e
                logger.warning(
                    f"Task {task_id}: attempt {attempt+1} failed after {attempt_time:.1f}s: "
                    f"{type(e).__name__}: {e}"
                )
                if attempt < settings.AGENT_MAX_RETRIES:
                    logger.info(f"Task {task_id}: retrying in 2s...")
                    await asyncio.sleep(2)

        raise last_error

    except Exception as e:
        total_time = time.time() - start_time
        logger.exception(f"Task {task_id} failed after {total_time:.1f}s: {type(e).__name__}: {e}")
        await update_task_error(task_id, str(e)[:500])


def start_workers() -> None:
    """Start worker threads."""
    global _loop
    _loop = asyncio.get_event_loop()
    _shutdown.clear()
    for i in range(settings.MAX_CONCURRENT_TASKS):
        t = Thread(target=_worker_thread, args=(i, _loop), daemon=True)
        t.start()
        _workers.append(t)
    logger.info(f"Started {settings.MAX_CONCURRENT_TASKS} workers")


def stop_workers() -> None:
    """Stop worker threads."""
    _shutdown.set()
    for t in _workers:
        t.join(timeout=5.0)
    _workers.clear()
    logger.info("Workers stopped")
