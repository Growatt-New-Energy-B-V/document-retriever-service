"""FastAPI application for document-retriever."""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .config import settings
from .models import (
    TaskCreateResponse,
    TaskError,
    TaskFailedResponse,
    TaskResultResponse,
    TaskStatusResponse,
    TaskTiming,
)
from .store import init_db, close_db, create_task, get_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(settings.DATA_DIR) / "uploads"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup and shutdown."""
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    await init_db()
    logger.info("Database initialized")

    # Start background worker
    from .worker import start_workers, stop_workers
    start_workers()
    logger.info("Workers started")

    yield

    stop_workers()
    await close_db()
    logger.info("Shutdown complete")


app = FastAPI(
    title="document-retriever-service",
    description="Extract structured data from documents using Claude + MCP tools",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/tasks", response_model=TaskCreateResponse, status_code=202)
async def create_extraction_task(
    file: UploadFile = File(...),
    schema: str = Form(...),
):
    """Upload a document and schema, start async extraction."""
    # Validate schema is valid JSON
    try:
        json.loads(schema)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in schema field")

    # Validate file extension
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Create task
    task_id = await create_task(filename, schema)

    # Save uploaded file
    task_upload_dir = UPLOAD_DIR / task_id
    task_upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = task_upload_dir / filename

    size = 0
    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.MAX_UPLOAD_SIZE:
                f.close()
                file_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large")
            f.write(chunk)

    logger.info(f"Task {task_id} created for {filename} ({size} bytes)")

    # Enqueue for processing
    from .worker import enqueue_task
    enqueue_task(task_id)

    return TaskCreateResponse(task_id=task_id)


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get task status."""
    task = await get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    error = None
    if task["error_json"]:
        err_data = json.loads(task["error_json"])
        error = TaskError(message=err_data["message"], details=err_data.get("details", []))

    return TaskStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        created_at=task["created_at"],
        updated_at=task["updated_at"],
        error=error,
    )


@app.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    """Get task result (only if succeeded or failed)."""
    task = await get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] == "failed":
        error_data = json.loads(task["error_json"]) if task["error_json"] else {"message": "Unknown error", "details": []}
        return JSONResponse(
            status_code=200,
            content={
                "task_id": task_id,
                "status": "failed",
                "error": error_data,
            },
        )

    if task["status"] != "succeeded":
        raise HTTPException(
            status_code=400,
            detail=f"Task not complete (status: {task['status']})",
        )

    result = json.loads(task["result_json"])
    return JSONResponse(
        status_code=200,
        content={
            "task_id": task_id,
            "data": result.get("data", {}),
            "evidence": result.get("evidence", {}),
            "warnings": result.get("warnings", []),
            "timing": result.get("timing", {}),
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
