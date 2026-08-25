"""Convert Codex ``codex_stream_log`` trajectories to ``mini_swe_agent_v2`` format.

The teacher trajectories in the team-coop dataset (gpt-5.5-hao via Codex) embed
shell commands and their outputs inline in ``assistant`` message content:

    [command] /bin/bash -lc 'some-command'
    <stdout>
    [exit N]          ← non-zero exits only; success has no marker

This script converts those to the structured format Qwen uses at inference time
(role: assistant with tool_calls + role: tool messages), so the teacher data can
be used for SFT without a format mismatch.

Usage:
    uv run python scripts/distill/convert_codex_to_miniswe.py
    uv run python scripts/distill/convert_codex_to_miniswe.py --src data/team-coop/cmp-full-team
    uv run python scripts/distill/convert_codex_to_miniswe.py --out data/converted_teacher.jsonl
    uv run python scripts/distill/convert_codex_to_miniswe.py --stats   # dry-run, stats only

Output JSONL schema: same as successful.jsonl (extract_successful.py) but with
``trajectory_format: "mini_swe_agent_v2"`` and ``messages`` replaced with converted messages.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SRC = Path(__file__).resolve().parent.parent.parent / "data" / "team-coop"
DEFAULT_OUT = Path(__file__).resolve().parent.parent.parent / "data" / "converted_teacher.jsonl"

# The system prompt used by mini_swe_agent_v2 runs
SYSTEM_PROMPT = (
    "You are a software engineer working alongside a colleague on a shared codebase. "
    "You each have your own workspace and are implementing different features in parallel. "
    "You communicate naturally — like engineers on the same team — to make sure "
    "your combined work integrates cleanly."
)

# Regexes
_CMD_PREFIX = re.compile(r"^\[command\] /bin/bash -l?c\s+")
_TIMING_LINE = re.compile(r"^[ \t]*(?:succeeded|exited\s+\S+)\s+in\s+\d+ms:\s*$")
_EXIT_LINE = re.compile(r"^\[exit\s+(-?\d+)\]\s*$")
_DIFF_START = re.compile(r"^diff --git ")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParsedTurn:
    reasoning: str = ""        # text before [command], if any
    has_command: bool = False
    command: str = ""          # the shell command string (after bash -lc)
    output: str = ""           # stdout/stderr (diff and timing stripped)
    returncode: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ConversionStats:
    total: int = 0
    skipped_empty: int = 0
    converted: int = 0
    warnings: int = 0
    turns_reasoning: int = 0
    turns_command: int = 0
    coop_commands: Counter = field(default_factory=Counter)

    def report(self) -> str:
        lines = [
            f"Total trajectories:   {self.total}",
            f"Skipped (≤1 msg):     {self.skipped_empty}",
            f"Converted:            {self.converted}",
            f"Parse warnings:       {self.warnings}",
            f"Reasoning turns:      {self.turns_reasoning}",
            f"Command turns:        {self.turns_command}",
        ]
        if self.coop_commands:
            lines.append("Top coop-task commands:")
            for cmd, n in self.coop_commands.most_common(10):
                lines.append(f"  {cmd}: {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Codex message parser
# ---------------------------------------------------------------------------

def _extract_shell_command(after_prefix: str) -> tuple[str, str]:
    """Extract the shell command and remaining content from text after '/bin/bash -lc '.

    Returns (command_string, rest_of_content) where rest_of_content starts on
    the line after the command line.
    """
    text = after_prefix.lstrip()
    # The command is quoted with either ' or " or unquoted (rare)
    if text.startswith("'"):
        # Single-quoted — find the closing ' that isn't escaped
        # Codex uses -lc '...' where the command may contain escaped quotes
        end = 1
        while end < len(text):
            if text[end] == "'" and text[end - 1] != "\\":
                break
            end += 1
        cmd = text[1:end]
        rest = text[end + 1:]
    elif text.startswith('"'):
        # Double-quoted
        end = 1
        while end < len(text):
            if text[end] == '"' and text[end - 1] != "\\":
                break
            end += 1
        cmd = text[1:end]
        # Unescape inner \" → "
        cmd = cmd.replace('\\"', '"')
        rest = text[end + 1:]
    else:
        # Unquoted — take until end of line
        nl = text.find("\n")
        if nl == -1:
            cmd, rest = text, ""
        else:
            cmd, rest = text[:nl], text[nl:]

    # rest should start with a newline; strip it
    if rest.startswith("\n"):
        rest = rest[1:]
    return cmd, rest


def _strip_output(raw_output: str) -> tuple[str, int, list[str]]:
    """Strip timing lines, trailing diff blocks, and [exit N] from raw output.

    Returns (cleaned_output, returncode, warnings).
    """
    warnings: list[str] = []
    lines = raw_output.split("\n")
    returncode = 0

    # Strip leading timing line if present
    start = 0
    if lines and _TIMING_LINE.match(lines[0]):
        # Extract exit code from timing line if exited N
        m = re.match(r"^ *exited\s+(-?\d+)\s+in", lines[0])
        if m:
            returncode = int(m.group(1))
        start = 1

    # Walk backwards: strip [exit N] and trailing diff block
    end = len(lines)

    # Check for [exit N] at the end (possibly after a diff block)
    # Strategy: scan from the end for [exit N], then find where diff starts
    exit_idx = None
    for i in range(end - 1, start - 1, -1):
        line = lines[i]
        m = _EXIT_LINE.match(line)
        if m:
            if returncode == 0:  # timing line takes precedence if both present
                returncode = int(m.group(1))
            exit_idx = i
            end = i  # don't include the [exit N] line
            break

    # Strip trailing diff block (workspace context injected by Codex)
    # Find the last `diff --git` line before end and cut there
    # But only strip it if it's after real output (not if the entire output is a diff)
    diff_start_idx = None
    for i in range(end - 1, start - 1, -1):
        if _DIFF_START.match(lines[i]):
            diff_start_idx = i
            break

    if diff_start_idx is not None:
        # Heuristic: if output before diff_start is non-empty, strip the diff
        pre_diff = "\n".join(lines[start:diff_start_idx]).strip()
        if pre_diff:
            end = diff_start_idx
        else:
            # The entire output IS the diff — keep it (rare, but happens)
            pass

    output = "\n".join(lines[start:end])
    # Normalise: strip trailing blank lines
    output = output.rstrip("\n")
    return output, returncode, warnings


def parse_codex_message(content: str) -> ParsedTurn:
    """Parse a single Codex assistant message content string into a ParsedTurn."""
    turn = ParsedTurn()

    # Find the [command] prefix
    m = _CMD_PREFIX.search(content)
    if m is None:
        # Pure reasoning — no command
        turn.reasoning = content
        return turn

    turn.has_command = True

    # Text before [command] = reasoning
    reasoning_end = content.rfind("\n", 0, m.start())
    if reasoning_end == -1:
        turn.reasoning = content[: m.start()].strip()
    else:
        turn.reasoning = content[:reasoning_end].strip()

    # Extract command string and rest (output)
    after_prefix = content[m.end():]
    cmd, raw_output = _extract_shell_command(after_prefix)
    turn.command = cmd

    # Parse output
    output, returncode, warnings = _strip_output(raw_output)
    turn.output = output
    turn.returncode = returncode
    turn.warnings = warnings

    return turn


# ---------------------------------------------------------------------------
# mini_swe_agent_v2 message builders
# ---------------------------------------------------------------------------

def _new_tool_call_id() -> str:
    return f"chatcmpl-tool-{uuid.uuid4().hex[:16]}"


def _assistant_reasoning(text: str) -> dict:
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": None,
        "function_call": None,
        "provider_specific_fields": {"refusal": None, "reasoning": None},
        "extra": None,
    }


def _assistant_command(reasoning: str, command: str, tc_id: str) -> dict:
    return {
        "role": "assistant",
        "content": reasoning,
        "tool_calls": [
            {
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
        "function_call": None,
        "provider_specific_fields": {"refusal": None, "reasoning": None},
        "extra": {
            "actions": [{"tool_name": "bash", "tool_call_id": tc_id, "command": command}],
            "response": None,
            "cost": 0.0,
            "timestamp": None,
        },
    }


def _tool_result(tc_id: str, output: str, returncode: int) -> dict:
    content = json.dumps({"returncode": returncode, "output": output})
    return {
        "role": "tool",
        "tool_call_id": tc_id,
        "content": content,
        "extra": {
            "raw_output": output,
            "returncode": returncode,
            "timestamp": None,
            "exception_info": "",
        },
    }


def _exit_message(status: str = "Submitted") -> dict:
    return {
        "role": "exit",
        "content": "",
        "extra": {"exit_status": status, "submission": ""},
    }


# ---------------------------------------------------------------------------
# Full trajectory conversion
# ---------------------------------------------------------------------------

def convert_trajectory(
    codex_messages: list[dict],
    agent_status: str,
    stats: ConversionStats,
) -> list[dict]:
    """Convert a list of Codex messages to mini_swe_agent_v2 format."""
    out: list[dict] = []

    # System message
    out.append({"role": "system", "content": SYSTEM_PROMPT})

    for msg in codex_messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "user":
            # Pass through verbatim (task description)
            out.append({"role": "user", "content": content})
            continue

        if role != "assistant":
            continue

        turn = parse_codex_message(content)

        if turn.warnings:
            stats.warnings += len(turn.warnings)

        if not turn.has_command:
            # Pure reasoning turn
            if turn.reasoning:
                stats.turns_reasoning += 1
                out.append(_assistant_reasoning(turn.reasoning))
        else:
            stats.turns_command += 1
            tc_id = _new_tool_call_id()
            out.append(_assistant_command(turn.reasoning, turn.command, tc_id))
            out.append(_tool_result(tc_id, turn.output, turn.returncode))

            # Track coop-task command usage
            cmd_stripped = turn.command.strip()
            m = re.match(r"(coop-task[a-z-]*)", cmd_stripped)
            if m:
                stats.coop_commands[m.group(1)] += 1

    # Exit message
    out.append(_exit_message(agent_status))
    return out


# ---------------------------------------------------------------------------
# Dataset traversal
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _iter_successful_pairs(src: Path, run_filter: str | None) -> list[tuple[str, Path]]:
    """Yield (run_name, pair_dir) for successful pairs under src."""
    pairs: list[tuple[str, Path]] = []

    def _scan_run(run_name: str, run_root: Path) -> None:
        coop_root = run_root / "coop"
        if not coop_root.is_dir():
            return
        for pair_dir in coop_root.rglob("f*_f*"):
            if not pair_dir.is_dir():
                continue
            eval_data = _load_json(pair_dir / "eval.json")
            if not eval_data or not isinstance(eval_data, dict):
                continue
            if eval_data.get("correct") and eval_data.get("verified"):
                pairs.append((run_name, pair_dir))

    if run_filter and src.name == run_filter and (src / "summary.json").exists() and (src / "coop").is_dir():
        _scan_run(src.name, src)
    else:
        for child in sorted(src.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if not (child / "summary.json").exists() or not (child / "coop").is_dir():
                continue
            if run_filter and child.name != run_filter:
                continue
            _scan_run(child.name, child)

    return pairs


def _meta_from_pair(run_name: str, pair_dir: Path) -> dict:
    """Extract metadata fields for the output record."""
    meta = _load_json(pair_dir / "metadata.json") or {}
    eval_data = _load_json(pair_dir / "eval.json") or {}
    return {
        "run": run_name,
        "repo": meta.get("repo", pair_dir.parts[-3]),
        "task_id": meta.get("task_id"),
        "features": meta.get("features", meta.get("source_features")),
        "model": meta.get("model"),
        "agent_framework": meta.get("agent_framework"),
        "team_features": meta.get("team_features", {}),
        "tasks": meta.get("tasks", []),
        "task_log": meta.get("task_log", []),
        "metrics": meta.get("metrics", {}),
        "lead_agent": meta.get("lead_agent", "agent1"),
        "duration_seconds": meta.get("duration_seconds"),
        "score": eval_data.get("score", 1.0),
        "trajectory_format": "mini_swe_agent_v2",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC,
                   help=f"Root of downloaded dataset (default: {DEFAULT_SRC})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output JSONL (default: {DEFAULT_OUT})")
    p.add_argument("--run", default=None, metavar="RUN_NAME",
                   help="Restrict to a single run, e.g. 'cmp-full-team'")
    p.add_argument("--stats", action="store_true",
                   help="Dry-run: print conversion stats without writing output")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.src.exists():
        print(f"error: {args.src} not found", file=sys.stderr)
        print("Run scripts/distill/download_team_coop.py first.", file=sys.stderr)
        return 1

    # Only convert Codex (teacher) runs — skip Qwen baseline runs
    codex_runs = {"cmp-full-team", "cmp-full-team-noproto"}

    pairs = _iter_successful_pairs(args.src, args.run)
    # Filter to Codex runs only
    if args.run is None:
        pairs = [(r, p) for r, p in pairs if r in codex_runs]
    print(f"Found {len(pairs)} successful pairs to convert")

    stats = ConversionStats()

    if not args.stats:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_fh = args.out.open("w")
    else:
        out_fh = None

    try:
        for run_name, pair_dir in pairs:
            meta = _meta_from_pair(run_name, pair_dir)
            lead_agent = meta["lead_agent"]

            for agent_id in ("agent1", "agent2"):
                traj_path = pair_dir / f"{agent_id}_traj.json"
                traj_data = _load_json(traj_path)
                if not traj_data or not isinstance(traj_data, dict):
                    continue

                codex_msgs = traj_data.get("messages", [])
                stats.total += 1

                # Skip empty or single-message (summary-only) trajectories
                if len(codex_msgs) <= 1:
                    stats.skipped_empty += 1
                    continue

                agent_status = traj_data.get("status", "Submitted")
                converted = convert_trajectory(codex_msgs, agent_status, stats)
                stats.converted += 1

                role = "lead" if agent_id == lead_agent else "member"
                record = {
                    **meta,
                    "agent_id": agent_id,
                    "role": role,
                    "messages": converted,
                }

                if out_fh is not None:
                    out_fh.write(json.dumps(record) + "\n")

    finally:
        if out_fh is not None:
            out_fh.close()

    sep = "─" * 60
    print(sep)
    print(stats.report())
    print(sep)
    if not args.stats and out_fh is not None:
        print(f"Written to: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
