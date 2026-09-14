"""Analyze scenario coverage in the extracted successful trajectories.

Reads the JSONL produced by extract_successful.py and classifies each
trajectory against the 8 canonical coordination scenarios we need for
distillation training.  Prints a detailed report showing:

  - Per-scenario counts and example trajectory IDs
  - Gap analysis: which scenarios are underrepresented or absent
  - Per-run and per-role breakdowns

Canonical scenarios
-------------------
1. solo_task_lifecycle      - agent claims, works, and marks done with no
                               coordination (single task, single agent active)
2. parallel_independent     - both agents claim separate pre-assigned tasks
                               and work independently with no messaging
3. lead_creates_subtask     - lead agent creates an additional task mid-run
                               (task_log has a "create" event by an agent, not bench-runner)
4. request_respond          - at least one request/respond pair in task_log
                               (kind == "request" or "response")
5. blocked_task             - any task reaches status "blocked"
6. claim_after_list         - agent calls list then claims (inferred from
                               task_log ordering: list events precede claim)
7. cross_agent_dependency   - lead waits for member: lead's lead_task stays
                               in_progress while member's task completes first
8. wait_for_message         - trajectory contains a wait_for_message MCP tool call

Scenarios are not mutually exclusive — one trajectory can cover several.

Usage:
    uv run python scripts/distill/analyze_coverage.py
    uv run python scripts/distill/analyze_coverage.py --src data/successful.jsonl
    uv run python scripts/distill/analyze_coverage.py --src data/successful.jsonl --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_SRC = Path(__file__).resolve().parent.parent.parent / "data" / "successful.jsonl"

SCENARIOS = [
    "solo_task_lifecycle",
    "parallel_independent",
    "lead_creates_subtask",
    "request_respond",
    "blocked_task",
    "claim_after_list",
    "cross_agent_dependency",
    "wait_for_message",
]

# Minimum examples we want per scenario before we consider it "covered"
COVERAGE_THRESHOLD = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help=f"Extracted JSONL from extract_successful.py (default: {DEFAULT_SRC})",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print up to 3 example trajectory IDs per scenario.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the report as JSON (for downstream use).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scenario detectors — each takes one record dict and returns bool
# ---------------------------------------------------------------------------

def _task_log(rec: dict) -> list[dict]:
    return rec.get("task_log") or []


def _trajectory(rec: dict) -> list:
    return rec.get("trajectory") or []


def _is_solo_task_lifecycle(rec: dict) -> bool:
    """Agent has at least one full open→in_progress→done arc in the task_log."""
    log = _task_log(rec)
    me = rec["agent_id"]
    claimed = set()
    updated_done = set()
    for ev in log:
        if ev.get("kind") == "claim" and ev.get("by") == me:
            claimed.add(ev["task_id"])
        if ev.get("kind") == "update" and ev.get("by") == me and ev.get("status") == "done":
            updated_done.add(ev["task_id"])
    return bool(claimed & updated_done)


def _is_parallel_independent(rec: dict) -> bool:
    """Both agents claim different tasks; no request/respond events at all."""
    log = _task_log(rec)
    has_messaging = any(ev.get("kind") in ("request", "response") for ev in log)
    if has_messaging:
        return False
    agents_claiming: set[str] = set()
    tasks_claimed: set[str] = set()
    for ev in log:
        if ev.get("kind") == "claim":
            agents_claiming.add(ev.get("by", ""))
            tasks_claimed.add(ev.get("task_id", ""))
    # two different agents each claimed a different task
    return len(agents_claiming) >= 2 and len(tasks_claimed) >= 2


def _is_lead_creates_subtask(rec: dict) -> bool:
    """An agent (not bench-runner) creates a task during the run."""
    log = _task_log(rec)
    return any(ev.get("kind") == "create" and ev.get("by") not in ("bench-runner", None) for ev in log)


def _is_request_respond(rec: dict) -> bool:
    """task_log contains at least one request or response event."""
    log = _task_log(rec)
    return any(ev.get("kind") in ("request", "response") for ev in log)


def _is_blocked_task(rec: dict) -> bool:
    """Any task reaches status 'blocked'."""
    log = _task_log(rec)
    return any(ev.get("kind") == "update" and ev.get("status") == "blocked" for ev in log)


def _is_claim_after_list(rec: dict) -> bool:
    """task_log has a list event before a claim by the same agent.

    The task_log itself doesn't record list calls (those are CLI-only), so
    we infer from the trajectory: look for a shell command containing
    'coop-task list' followed by a claim in the task_log.
    """
    traj = _trajectory(rec)
    log = _task_log(rec)
    me = rec["agent_id"]

    claim_ts = min(
        (ev["ts"] for ev in log if ev.get("kind") == "claim" and ev.get("by") == me),
        default=None,
    )
    if claim_ts is None:
        return False

    # Search trajectory steps for a coop-task list call that occurred
    # before the first claim timestamp (heuristic: step index as proxy)
    claim_step = None
    for i, step in enumerate(traj):
        if ev_ts := _step_ts(step):
            if ev_ts >= claim_ts:
                claim_step = i
                break

    for i, step in enumerate(traj):
        if claim_step is not None and i >= claim_step:
            break
        if _step_contains(step, "coop-task list") or _step_contains(step, "coop-task pending"):
            return True
    return False


def _is_cross_agent_dependency(rec: dict) -> bool:
    """Lead's lead_task stays in_progress while member's task completes first."""
    if rec.get("role") != "lead":
        return False
    log = _task_log(rec)
    # Find lead task id (created by bench-runner with lead_task metadata — but
    # we don't have per-event metadata here, so proxy: title contains "Lead-only")
    # Also works: lead claims their task and then member's done event appears
    # before lead's done event.
    lead_in_progress_ts = None
    member_done_ts = None
    lead_done_ts = None

    me = rec["agent_id"]
    # identify other agent
    all_agents = {ev.get("by") for ev in log if ev.get("by") not in (None, "bench-runner")}
    other_agents = all_agents - {me}

    for ev in log:
        if ev.get("kind") == "update" and ev.get("by") == me and ev.get("status") == "in_progress":
            lead_in_progress_ts = ev.get("ts")
        if ev.get("kind") == "update" and ev.get("by") in other_agents and ev.get("status") == "done":
            member_done_ts = ev.get("ts")
        if ev.get("kind") == "update" and ev.get("by") == me and ev.get("status") == "done":
            lead_done_ts = ev.get("ts")

    if lead_in_progress_ts and member_done_ts and lead_done_ts:
        # Lead was in_progress, member finished, then lead finished — classic dependency pattern
        return lead_in_progress_ts < member_done_ts < lead_done_ts
    return False


def _is_wait_for_message(rec: dict) -> bool:
    """Trajectory contains a wait_for_message MCP tool call."""
    traj = _trajectory(rec)
    for step in traj:
        if _step_contains(step, "wait_for_message"):
            return True
    return False


# ---------------------------------------------------------------------------
# Trajectory step helpers
# ---------------------------------------------------------------------------

def _step_contains(step, text: str) -> bool:
    """Check whether any string field in a trajectory step contains text."""
    if isinstance(step, str):
        return text in step
    if isinstance(step, dict):
        return any(
            text in v
            for v in step.values()
            if isinstance(v, str)
        ) or any(_step_contains(v, text) for v in step.values() if isinstance(v, (dict, list)))
    if isinstance(step, list):
        return any(_step_contains(item, text) for item in step)
    return False


def _step_ts(step) -> float | None:
    if isinstance(step, dict):
        for key in ("ts", "timestamp", "time"):
            if key in step:
                try:
                    return float(step[key])
                except (TypeError, ValueError):
                    pass
    return None


DETECTORS: dict[str, object] = {
    "solo_task_lifecycle": _is_solo_task_lifecycle,
    "parallel_independent": _is_parallel_independent,
    "lead_creates_subtask": _is_lead_creates_subtask,
    "request_respond": _is_request_respond,
    "blocked_task": _is_blocked_task,
    "claim_after_list": _is_claim_after_list,
    "cross_agent_dependency": _is_cross_agent_dependency,
    "wait_for_message": _is_wait_for_message,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if not args.src.exists():
        print(f"error: {args.src} not found.", file=sys.stderr)
        print("Run scripts/distill/extract_successful.py first.", file=sys.stderr)
        return 1

    # Counters
    scenario_hits: dict[str, list[str]] = defaultdict(list)   # scenario → [traj_key, ...]
    scenario_by_run: dict[str, Counter] = defaultdict(Counter)
    scenario_by_role: dict[str, Counter] = defaultdict(Counter)
    total = 0
    multi_scenario_counts: Counter = Counter()

    with args.src.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            key = f"{rec.get('run')}/{rec.get('repo')}/{rec.get('task_id')}/{rec.get('features')}/{rec.get('agent_id')}"
            run = rec.get("run", "unknown")
            role = rec.get("role", "unknown")

            matched = []
            for scenario, detector in DETECTORS.items():
                if detector(rec):  # type: ignore[operator]
                    scenario_hits[scenario].append(key)
                    scenario_by_run[scenario][run] += 1
                    scenario_by_role[scenario][role] += 1
                    matched.append(scenario)
            multi_scenario_counts[len(matched)] += 1

    # --- Report -----------------------------------------------------------
    sep = "─" * 72

    print(sep)
    print(f"  Team-Coop Scenario Coverage Report")
    print(f"  Source:  {args.src}")
    print(f"  Records: {total} (one per agent per successful pair)")
    print(sep)
    print()

    covered = []
    gaps = []

    for scenario in SCENARIOS:
        hits = scenario_hits[scenario]
        n = len(hits)
        status = "OK " if n >= COVERAGE_THRESHOLD else "GAP"
        bar_len = min(40, n // max(1, total // 400))
        bar = "█" * bar_len
        print(f"  [{status}]  {scenario:<28}  {n:>5} examples  {bar}")
        if args.verbose and hits:
            for ex in hits[:3]:
                print(f"            ↳ {ex}")
        if n >= COVERAGE_THRESHOLD:
            covered.append(scenario)
        else:
            gaps.append((scenario, n))

    print()
    print(sep)
    print(f"  Covered (≥{COVERAGE_THRESHOLD}): {len(covered)}/{len(SCENARIOS)}")
    print()

    if gaps:
        print("  GAPS — need synthetic scenario generation:")
        for scenario, n in gaps:
            needed = COVERAGE_THRESHOLD - n
            print(f"    {scenario:<28}  {n} found  →  need ~{needed} more synthetic examples")
    else:
        print("  All scenarios covered — no synthetic generation needed.")

    print()
    print("  Scenario overlap (how many scenarios one trajectory covers):")
    for n_scenarios in sorted(multi_scenario_counts):
        print(f"    {n_scenarios} scenarios: {multi_scenario_counts[n_scenarios]} trajectories")

    print()
    print("  Per-run breakdown:")
    for scenario in SCENARIOS:
        if scenario_by_run[scenario]:
            breakdown = "  ".join(f"{r}:{c}" for r, c in sorted(scenario_by_run[scenario].items()))
            print(f"    {scenario:<28}  {breakdown}")

    print()
    print("  Per-role breakdown:")
    for scenario in SCENARIOS:
        if scenario_by_role[scenario]:
            breakdown = "  ".join(f"{r}:{c}" for r, c in sorted(scenario_by_role[scenario].items()))
            print(f"    {scenario:<28}  {breakdown}")

    print(sep)

    if args.out:
        report = {
            "total_records": total,
            "coverage_threshold": COVERAGE_THRESHOLD,
            "scenarios": {
                s: {
                    "count": len(scenario_hits[s]),
                    "covered": len(scenario_hits[s]) >= COVERAGE_THRESHOLD,
                    "by_run": dict(scenario_by_run[s]),
                    "by_role": dict(scenario_by_role[s]),
                    "examples": scenario_hits[s][:10],
                }
                for s in SCENARIOS
            },
            "gaps": [
                {"scenario": s, "count": n, "needed": COVERAGE_THRESHOLD - n}
                for s, n in gaps
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
