"""Pure parsers for Claude Code's stream-json and session-JSONL output.

Kept independent of any container/CLI machinery so they can be unit-tested
against fixture strings.

Stream-json shape (one JSON object per line) is documented at
https://docs.anthropic.com/en/docs/agents/claude-code. The terminating
event is ``{"type": "result", ...}`` and is the source of truth for cost
and turn count. Session JSONL lives under
``$CLAUDE_CONFIG_DIR/projects/<slug>/<session-id>/*.jsonl`` and contains
the raw assistant/user/tool turns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamSummary:
    """Aggregate stats extracted from the final ``result`` event."""

    status: str = "Error"
    cost: float = 0.0
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    raw_result: dict[str, Any] = field(default_factory=dict)


def _iter_json_lines(text: str):
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def parse_stream_json(text: str) -> StreamSummary:
    """Extract cost/tokens/status from ``claude --output-format=stream-json`` output.

    The final ``{"type": "result", ...}`` event is authoritative.  If it's
    missing (CLI crashed, killed by timeout), status is ``"Error"`` and
    counters are zero.
    """
    for event in _iter_json_lines(text):
        if event.get("type") != "result":
            continue
        usage = event.get("usage") or {}
        is_error = bool(event.get("is_error")) or event.get("subtype", "success") != "success"
        status = "Submitted"
        if is_error:
            subtype = event.get("subtype", "")
            if "max_turns" in subtype or "limit" in subtype or "budget" in subtype:
                status = "LimitsExceeded"
            else:
                status = "Error"
        return StreamSummary(
            status=status,
            cost=float(event.get("total_cost_usd") or 0.0),
            steps=int(event.get("num_turns") or 0),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
            cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            raw_result=event,
        )
    return StreamSummary()


def _content_blocks_to_text(content: Any) -> str:
    """Flatten a Claude message ``content`` field to a plain string.

    Anthropic messages use a list of typed blocks (text/tool_use/
    tool_result/thinking).  Downstream CooperBench code does
    ``"send_message" in msg["content"]`` so we must always return a string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
        elif btype == "thinking":
            thought = block.get("thinking") or block.get("text") or ""
            if thought:
                parts.append(f"[thinking] {thought}")
        elif btype == "tool_use":
            name = block.get("name", "")
            args = block.get("input")
            try:
                args_str = json.dumps(args, ensure_ascii=False)
            except TypeError:
                args_str = str(args)
            parts.append(f"[tool_use {name}] {args_str}")
        elif btype == "tool_result":
            inner = block.get("content")
            if isinstance(inner, list):
                inner_text = "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in inner)
            elif inner is None:
                inner_text = ""
            else:
                inner_text = str(inner)
            parts.append(f"[tool_result] {inner_text}")
        else:
            try:
                parts.append(json.dumps(block, ensure_ascii=False))
            except TypeError:
                parts.append(str(block))
    return "\n".join(p for p in parts if p)


def parse_session_jsonl(text: str) -> list[dict[str, str]]:
    """Convert Claude Code session JSONL to OpenAI-style chat messages.

    Returns a list of ``{"role": ..., "content": ...}`` dicts, sorted by
    ``timestamp``.  ``content`` is always a string.
    """
    events: list[dict[str, Any]] = list(_iter_json_lines(text))
    events.sort(key=lambda e: e.get("timestamp") or "")

    messages: list[dict[str, str]] = []
    for event in events:
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"user", "assistant", "system"}:
            continue
        content_text = _content_blocks_to_text(message.get("content"))
        messages.append({"role": role, "content": content_text})
    return messages
