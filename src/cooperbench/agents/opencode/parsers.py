"""Lightweight parsers for ``opencode run`` stdout/stderr output."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches ANSI CSI escape sequences (colors, cursor moves) that opencode's
# TUI-style output embeds.  These break naive regex scraping of the numbers
# and labels, so strip them before parsing.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from ``text``."""
    return _ANSI_RE.sub("", text)


@dataclass
class RunSummary:
    status: str = "Submitted"
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


def parse_run_output(text: str) -> RunSummary:
    """Extract steps / token counts / cost from opencode run stdout.

    opencode's run output is primarily human-readable streaming text wrapped
    in ANSI color codes.  Strip those first, then best-effort scrape the
    summary figures; all fields default to 0 when a pattern is absent so
    callers always get a valid dataclass back.
    """
    summary = RunSummary()
    text = strip_ansi(text)

    # Cost: "$0.0023", "cost: $0.0023", "total cost: $0.0023".
    cost_match = re.search(r"(?:cost|total)\s*[:=]?\s*\$?([\d]+\.[\d]+)", text, re.IGNORECASE)
    if not cost_match:
        cost_match = re.search(r"\$([\d]+\.[\d]+)", text)
    if cost_match:
        try:
            summary.cost = float(cost_match.group(1))
        except ValueError:
            pass

    # Tokens: "1,234 input tokens", "input: 1234", "input tokens 1234".
    input_match = re.search(r"(\d[\d,]*)\s+input\s+tokens?", text, re.IGNORECASE)
    if not input_match:
        input_match = re.search(r"input[:\s]+(\d[\d,]*)", text, re.IGNORECASE)
    if input_match:
        summary.input_tokens = _to_int(input_match.group(1))

    output_match = re.search(r"(\d[\d,]*)\s+output\s+tokens?", text, re.IGNORECASE)
    if not output_match:
        output_match = re.search(r"output[:\s]+(\d[\d,]*)", text, re.IGNORECASE)
    if output_match:
        summary.output_tokens = _to_int(output_match.group(1))

    # Fallback: a single total "tokens: 1234" when input/output are absent.
    if not summary.input_tokens and not summary.output_tokens:
        tok_match = re.search(r"tokens?[:\s]+(\d[\d,]*)", text, re.IGNORECASE)
        if tok_match:
            summary.input_tokens = _to_int(tok_match.group(1))

    # Steps: "12 steps", "steps: 12".  If no explicit count is printed, fall
    # back to counting tool invocations opencode logs (one line per tool call,
    # rendered with a leading bullet or the tool name in brackets).
    steps_match = re.search(r"(\d+)\s+steps?", text, re.IGNORECASE)
    if not steps_match:
        steps_match = re.search(r"steps?[:\s]+(\d+)", text, re.IGNORECASE)
    if steps_match:
        summary.steps = int(steps_match.group(1))
    else:
        tool_calls = len(re.findall(r"^\s*[|•]\s*\w+\s", text, re.MULTILINE))
        if tool_calls:
            summary.steps = tool_calls

    return summary


def _to_int(raw: str) -> int:
    """Parse an integer that may contain thousands separators."""
    try:
        return int(raw.replace(",", ""))
    except ValueError:
        return 0
