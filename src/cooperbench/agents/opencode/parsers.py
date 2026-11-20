"""Lightweight parsers for ``opencode run`` stdout/stderr output."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RunSummary:
    status: str = "Submitted"
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


def parse_run_output(text: str) -> RunSummary:
    """Extract steps / token counts / cost from opencode run stdout.

    opencode's run output is primarily human-readable streaming text.
    We use best-effort regex scraping; all fields default to 0 when the
    pattern is absent so callers always get a valid dataclass back.
    """
    summary = RunSummary()

    # Cost: "$0.0023" or "cost: $0.0023" or "total cost: $0.0023"
    cost_match = re.search(r"(?:cost|total)[:\s]+\$?([\d.]+)", text, re.IGNORECASE)
    if cost_match:
        try:
            summary.cost = float(cost_match.group(1))
        except ValueError:
            pass

    # Tokens: "1234 tokens", "tokens: 1234", "1234 input tokens"
    input_match = re.search(r"(\d[\d,]+)\s+input\s+tokens?", text, re.IGNORECASE)
    if input_match:
        summary.input_tokens = int(input_match.group(1).replace(",", ""))

    output_match = re.search(r"(\d[\d,]+)\s+output\s+tokens?", text, re.IGNORECASE)
    if output_match:
        summary.output_tokens = int(output_match.group(1).replace(",", ""))

    # Fallback: "tokens: 1234" (total)
    if not summary.input_tokens and not summary.output_tokens:
        tok_match = re.search(r"tokens?[:\s]+(\d[\d,]+)", text, re.IGNORECASE)
        if tok_match:
            summary.input_tokens = int(tok_match.group(1).replace(",", ""))

    # Steps: "12 steps", "steps: 12", "turn 12"
    steps_match = re.search(r"(\d+)\s+steps?", text, re.IGNORECASE)
    if not steps_match:
        steps_match = re.search(r"steps?[:\s]+(\d+)", text, re.IGNORECASE)
    if steps_match:
        summary.steps = int(steps_match.group(1))

    return summary
