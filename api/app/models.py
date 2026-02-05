"""Pydantic request/response models for the document-retriever API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TaskCreateResponse(BaseModel):
    task_id: str


class TaskError(BaseModel):
    message: str
    details: list[str] = Field(default_factory=list)


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: str
    updated_at: str
    error: Optional[TaskError] = None


class TaskTiming(BaseModel):
    total_seconds: float
    index_seconds: Optional[float] = None
    extraction_seconds: Optional[float] = None


class TaskResultResponse(BaseModel):
    task_id: str
    data: dict[str, Any]
    evidence: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    timing: TaskTiming


class TaskFailedResponse(BaseModel):
    task_id: str
    status: str = "failed"
    error: TaskError
