"""Claude Agent SDK integration for document extraction."""
from __future__ import annotations

import json
import logging
import re

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)

from .config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a document data extraction agent. You extract structured data from
documents using the provided tools. You MUST follow this protocol exactly:

STEP 1 — INDEX
Call doc_index_build with the provided file_path. Note the index_id,
chunk_count, and toc.

STEP 2 — EXTRACT FIELD BY FIELD
For each field in the target schema:
  a) Call doc_search with a query describing the field. Use top_k=5.
  b) Review the returned snippets and scores.
  c) For promising hits, call doc_read with the locator to get full content.
  d) Extract the value. Record the locator and exact snippet as evidence.

RULES:
- NEVER call doc_read without first calling doc_search for that field.
- NEVER guess or hallucinate values. If not found, set the value to null.
- For every extracted value, record evidence: { locator, snippet }.
- For per-model values (tables with multiple models), return a dict keyed
  by model name.
- For lists, return arrays.
- Preserve original units. Standardize formatting only (e.g. "1000V" not
  "1000 V" or "1kV").

STEP 3 — VALIDATE
Call json_validate_and_normalize with your extracted data and the target
schema. Fix any errors reported.

STEP 4 — RESPOND
Return exactly two JSON objects, nothing else:
{"data": { ... }, "evidence": { ... }}

The "evidence" object mirrors the "data" structure but each leaf value is:
{"locator": "h3-c2", "snippet": "exact text from document"}

If a field is null, its evidence should be:
{"locator": null, "snippet": null, "warning": "Not found in document"}

No prose. No explanation. Only the two JSON objects.
"""


def _build_options() -> ClaudeAgentOptions:
    """Build agent options for extraction."""
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={
            "doc-tools": {
                "type": "sse",
                "url": settings.MCP_TOOLS_URL,
            }
        },
        allowed_tools=[
            "mcp__doc-tools__doc_index_build",
            "mcp__doc-tools__doc_search",
            "mcp__doc-tools__doc_read",
            "mcp__doc-tools__json_validate_and_normalize",
        ],
        max_turns=50,
        permission_mode="acceptEdits",
    )


async def run_extraction(file_path: str, schema: dict) -> dict:
    """Run the extraction agent and return structured result.

    Args:
        file_path: Path to the uploaded document.
        schema: The target schema skeleton.

    Returns:
        Dict with "data", "evidence", and "warnings" keys.
    """
    prompt = (
        f"Extract data from the document at: {file_path}\n\n"
        f"Target schema:\n{json.dumps(schema, indent=2)}"
    )

    options = _build_options()
    final_text = ""

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            if isinstance(msg, (AssistantMessage, ResultMessage)):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        final_text += block.text

    return _parse_agent_output(final_text)


def _parse_agent_output(text: str) -> dict:
    """Parse the agent's output into data + evidence + warnings.

    The agent should return two JSON objects: {"data": ...} and {"evidence": ...}
    """
    # Try to find JSON objects in the text
    json_objects = []
    # Find all top-level JSON objects
    brace_depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start : i + 1])
                    json_objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    data = {}
    evidence = {}
    warnings = []

    if len(json_objects) >= 2:
        # Agent returned two objects: data and evidence
        data = json_objects[0].get("data", json_objects[0])
        evidence = json_objects[1].get("evidence", json_objects[1])
    elif len(json_objects) == 1:
        obj = json_objects[0]
        data = obj.get("data", {})
        evidence = obj.get("evidence", {})
        warnings = obj.get("warnings", [])

    # Extract warnings from evidence (null fields with warnings)
    if not warnings:
        warnings = _extract_warnings(evidence)

    return {"data": data, "evidence": evidence, "warnings": warnings}


def _extract_warnings(evidence: dict, path: str = "") -> list[str]:
    """Extract warnings from evidence tree (null locators indicate missing data)."""
    warnings = []
    if isinstance(evidence, dict):
        if "warning" in evidence and evidence.get("locator") is None:
            field_name = path or "unknown"
            warnings.append(evidence["warning"])
        else:
            for key, val in evidence.items():
                child_path = f"{path}.{key}" if path else key
                if key in ("locator", "snippet", "warning"):
                    continue
                warnings.extend(_extract_warnings(val, child_path))
    return warnings
