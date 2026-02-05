"""HTTP client for the markitdown-vision-service."""
from __future__ import annotations

import logging
import asyncio
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class MarkitdownClient:
    """Async client for markitdown-vision-service."""

    def __init__(self, base_url: str = "http://markitdown:8000", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def convert_to_markdown(self, file_path: str) -> str:
        """Upload a file to markitdown, poll until done, return markdown text.

        Args:
            file_path: Path to the file to convert.

        Returns:
            The markdown text output.

        Raises:
            RuntimeError: If conversion fails or times out.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # POST file
            with open(file_path, "rb") as f:
                files = {"file": (path.name, f, "application/octet-stream")}
                resp = await client.post(f"{self.base_url}/tasks", files=files)
                resp.raise_for_status()
                task_data = resp.json()

            task_id = task_data["task_id"]
            logger.info(f"Markitdown task created: {task_id}")

            # Poll until completed
            elapsed = 0.0
            poll_interval = 2.0
            while elapsed < self.timeout:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                status_resp = await client.get(f"{self.base_url}/tasks/{task_id}")
                status_resp.raise_for_status()
                status_data = status_resp.json()
                status = status_data["status"]

                if status == "completed":
                    # Download the markdown file
                    outputs = status_data.get("outputs", [])
                    md_files = [o for o in outputs if o.endswith(".md")]
                    if not md_files:
                        raise RuntimeError(f"No markdown output for task {task_id}")

                    md_resp = await client.get(
                        f"{self.base_url}/tasks/{task_id}/files/{md_files[0]}"
                    )
                    md_resp.raise_for_status()
                    logger.info(f"Markitdown conversion complete: {len(md_resp.text)} chars")
                    return md_resp.text

                elif status == "failed":
                    error_msg = status_data.get("error_message", "Unknown error")
                    raise RuntimeError(f"Markitdown conversion failed: {error_msg}")

                logger.debug(f"Markitdown task {task_id}: {status} ({elapsed:.0f}s)")

            raise RuntimeError(f"Markitdown conversion timed out after {self.timeout}s")

    async def health_check(self) -> bool:
        """Check if the markitdown service is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False
