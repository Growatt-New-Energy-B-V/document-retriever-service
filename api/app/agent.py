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
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
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
    msg_count = 0
    turn_count = 0
    sdk_usage = {}

    logger.info(f"Starting extraction: file={file_path}, schema_keys={list(schema.keys())}")
    logger.info(f"MCP URL: {settings.MCP_TOOLS_URL}, max_turns: 50")

    async with ClaudeSDKClient(options=options) as client:
        logger.info("SDK client connected, sending query...")
        await client.query(prompt)
        logger.info("Query sent, waiting for response stream...")

        async for msg in client.receive_response():
            msg_count += 1
            msg_type = type(msg).__name__

            if isinstance(msg, AssistantMessage):
                turn_count += 1
                block_summary = []
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        final_text += block.text
                        preview = block.text[:200].replace("\n", "\\n")
                        block_summary.append(f"Text({len(block.text)} chars): {preview}")
                    elif isinstance(block, ToolUseBlock):
                        input_preview = json.dumps(block.input)[:150]
                        block_summary.append(f"ToolUse({block.name}, {input_preview})")
                    elif isinstance(block, ToolResultBlock):
                        content_preview = str(block.content)[:150]
                        block_summary.append(f"ToolResult(err={block.is_error}, {content_preview})")
                    elif isinstance(block, ThinkingBlock):
                        block_summary.append(f"Thinking({len(block.thinking)} chars)")
                    else:
                        block_summary.append(f"Unknown({type(block).__name__})")
                logger.info(
                    f"[msg {msg_count}] turn={turn_count} AssistantMessage "
                    f"blocks={len(msg.content)}: {'; '.join(block_summary)}"
                )
            elif isinstance(msg, ResultMessage):
                result_preview = (msg.result or "")[:300].replace("\n", "\\n")
                logger.info(
                    f"[msg {msg_count}] ResultMessage: is_error={msg.is_error}, "
                    f"turns={msg.num_turns}, cost=${msg.total_cost_usd}, "
                    f"duration={msg.duration_ms}ms, usage={msg.usage}, "
                    f"result({len(msg.result or '')} chars): {result_preview}"
                )
                sdk_usage = {
                    "num_turns": msg.num_turns,
                    "duration_ms": msg.duration_ms,
                    "duration_api_ms": msg.duration_api_ms,
                    "total_cost_usd": msg.total_cost_usd,
                    "is_error": msg.is_error,
                    **(msg.usage or {}),
                }
                if msg.result:
                    final_text += msg.result
            elif isinstance(msg, SystemMessage):
                logger.info(f"[msg {msg_count}] SystemMessage: subtype={msg.subtype}")
            else:
                logger.info(f"[msg {msg_count}] {msg_type}: {str(msg)[:200]}")

    logger.info(
        f"Extraction stream complete: {msg_count} messages, {turn_count} turns, "
        f"final_text={len(final_text)} chars"
    )

    result = _parse_agent_output(final_text)
    result["usage"] = sdk_usage
    logger.info(
        f"Parsed output: data_keys={list(result.get('data', {}).keys())}, "
        f"evidence_keys={list(result.get('evidence', {}).keys())}, "
        f"warnings={len(result.get('warnings', []))}, "
        f"usage={sdk_usage}"
    )
    return result


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
