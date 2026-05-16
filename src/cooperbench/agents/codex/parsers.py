"""Pure parsers for the Codex CLI's ``--json`` event stream.

The schema is documented at
https://github.com/openai/codex/blob/main/codex-rs/exec/src/exec_events.rs.

Codex emits one ``ThreadEvent`` JSON object per line.  We summarize:

  - ``status``       — "Submitted" / "Error" based on whether the stream
                       ended cleanly (no ``error`` or ``turn.failed``)
  - ``steps``        — count of ``turn.completed`` events
  - token totals     — summed across all ``turn.completed`` usage objects
  - ``cost``         — always 0.0 (Codex does not report cost in events)
  - ``is_model_error`` — true when the terminal error looks like a
                         model-not-found, so the adapter can fall back

Messages come from ``item.completed`` events whose ``item.details.type``
is one of ``agent_message`` / ``reasoning`` / ``command_execution``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamSummary:
    """Aggregate stats extracted from a Codex JSONL stream."""

    status: str = "Error"
    cost: float = 0.0
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    is_model_error: bool = False
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


def _looks_like_model_error(message: str) -> bool:
    """Heuristic: does an error message indicate the model id was rejected?

    We're matching the OpenAI API's typical phrasing:
        "model 'gpt-5.5' does not exist"
        "invalid_request_error: model not found"
        "the model `xyz` does not exist"
    """
    msg = (message or "").lower()
    if "model" not in msg:
        return False
    return any(
        needle in msg
        for needle in (
            "does not exist",
            "not found",
            "is not supported",
            "invalid model",
            "unknown model",
        )
    )


def parse_stream_jsonl(text: str) -> StreamSummary:
    """Aggregate Codex's JSONL event stream into a single summary.

    The terminal status is "Error" if we see any ``error`` or
    ``turn.failed`` event, otherwise "Submitted" (provided we saw at
    least one ``turn.completed``).  Empty input is "Error".
    """
    saw_completed_turn = False
    saw_terminal_error = False
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    is_model_error = False
    last_error: dict[str, Any] = {}

    for event in _iter_json_lines(text):
        etype = event.get("type")
        if etype == "turn.completed":
            saw_completed_turn = True
            usage = event.get("usage") or {}
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
            cache_read += int(usage.get("cached_input_tokens") or 0)
        elif etype == "turn.failed":
            saw_terminal_error = True
            err = event.get("error") or {}
            last_error = {"message": err.get("message", "")}
            if _looks_like_model_error(last_error["message"]):
                is_model_error = True
        elif etype == "error":
            saw_terminal_error = True
            last_error = {"message": event.get("message", "")}
            if _looks_like_model_error(last_error["message"]):
                is_model_error = True

    if not saw_completed_turn and not saw_terminal_error and not input_tokens:
        return StreamSummary(status="Error")

    return StreamSummary(
        status="Error" if saw_terminal_error else "Submitted",
        cost=0.0,
        steps=_count_turns(text),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=0,  # Codex doesn't separate cache creation vs read
        is_model_error=is_model_error,
        raw_result=last_error,
    )


def _count_turns(text: str) -> int:
    return sum(1 for ev in _iter_json_lines(text) if ev.get("type") == "turn.completed")


def _item_to_message(item: dict[str, Any]) -> dict[str, str] | None:
    """Convert one ``ThreadItemDetails`` payload to a chat message.

    Returns None for item kinds that aren't worth surfacing
    individually (file_change, mcp_tool_call, etc) — those are visible
    through the tool-result text where they matter.
    """
    item_type = item.get("type")
    if item_type == "agent_message":
        text = item.get("text") or ""
        return {"role": "assistant", "content": text}
    if item_type == "reasoning":
        text = item.get("text") or ""
        return {"role": "assistant", "content": f"[reasoning] {text}"}
    if item_type == "command_execution":
        cmd = item.get("command") or ""
        out = item.get("aggregated_output") or ""
        exit_code = item.get("exit_code")
        parts = [f"[command] {cmd}"]
        if out:
            parts.append(out.rstrip())
        if exit_code not in (None, 0):
            parts.append(f"[exit {exit_code}]")
        return {"role": "assistant", "content": "\n".join(parts)}
    return None


def parse_messages(text: str) -> list[dict[str, str]]:
    """Walk ``item.completed`` events into OpenAI-style chat messages."""
    out: list[dict[str, str]] = []
    for event in _iter_json_lines(text):
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        msg = _item_to_message(item)
        if msg is not None:
            # content is always a string (downstream code does ``"x" in content``).
            if msg["content"] is None:
                msg["content"] = ""
            out.append(msg)
    return out
