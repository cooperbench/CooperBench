"""Extract successful trajectories from a downloaded team-coop dataset.

A trajectory pair (agent1 + agent2) is "successful" when eval.json has
``correct == true`` AND ``verified == true``.  We write one JSON record per
agent per successful pair to a JSONL output file, keeping:

    - the full task_log from metadata.json  (tool call sequence)
    - the conversation from conversation.json
    - the agent's own trajectory from agentN_traj.json
    - key scalars from metadata.json / eval.json / result.json

Usage:
    uv run python scripts/distill/extract_successful.py
    uv run python scripts/distill/extract_successful.py --src data/team-coop/cmp-full-team
    uv run python scripts/distill/extract_successful.py --src data/team-coop --out data/successful.jsonl

Output JSONL schema (one object per line, one line per agent per pair):
    {
      "run":         str,   # top-level run dir name, e.g. "cmp-full-team"
      "repo":        str,   # e.g. "dspy_task"
      "task_id":     int,
      "features":    [int, int],
      "agent_id":    str,   # "agent1" or "agent2"
      "role":        str,   # "lead" or "member"
      "model":       str,
      "agent_framework": str,
      "team_features": dict,
      "tasks":       list,  # final task objects from metadata.json
      "task_log":    list,  # full coop-task event log
      "conversation": list, # inter-agent messages
      "trajectory":  list,  # raw agent trajectory steps
      "metrics":     dict,
      "duration_seconds": float,
      "score":       float
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_SRC = Path(__file__).resolve().parent.parent.parent / "data" / "team-coop"
DEFAULT_OUT = Path(__file__).resolve().parent.parent.parent / "data" / "successful.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help=f"Root of the downloaded dataset (default: {DEFAULT_SRC}). "
        "Can point to the top-level dir (scans all runs) or a single run dir.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output JSONL file (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--run",
        default=None,
        metavar="RUN_NAME",
        help="Restrict to a single run, e.g. 'cmp-full-team'.",
    )
    p.add_argument(
        "--require-verified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require eval.json verified==true (default: on). Use --no-require-verified to keep unverified passes.",
    )
    return p.parse_args()


def _load_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _find_pair_dirs(src: Path, run_filter: str | None) -> list[tuple[str, Path]]:
    """Return (run_name, pair_dir) for every f<x>_f<y> leaf directory."""
    pairs: list[tuple[str, Path]] = []

    def _scan_run(run_name: str, run_root: Path) -> None:
        coop_root = run_root / "coop"
        if not coop_root.is_dir():
            return
        for pair_dir in coop_root.rglob("f*_f*"):
            if pair_dir.is_dir() and (pair_dir / "eval.json").exists():
                pairs.append((run_name, pair_dir))

    # src might be the top-level (contains multiple run dirs) or a single run dir.
    # Scan children that look like run dirs (have both coop/ and summary.json).
    # If --run is given, also accept src itself as the run dir.
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


def _extract_pair(run_name: str, pair_dir: Path, require_verified: bool) -> list[dict] | None:
    eval_data = _load_json(pair_dir / "eval.json")
    if not eval_data or not isinstance(eval_data, dict):
        return None
    if not eval_data.get("correct"):
        return None
    if require_verified and not eval_data.get("verified"):
        return None

    meta = _load_json(pair_dir / "metadata.json") or {}
    result = _load_json(pair_dir / "result.json") or {}
    conversation = _load_json(pair_dir / "conversation.json") or []

    lead_agent = meta.get("lead_agent", "agent1")

    records = []
    for agent_id in ("agent1", "agent2"):
        traj_file = pair_dir / f"{agent_id}_traj.json"
        trajectory = _load_json(traj_file) or []

        records.append({
            "run": run_name,
            "repo": meta.get("repo", pair_dir.parts[-3]),
            "task_id": meta.get("task_id"),
            "features": meta.get("features", meta.get("source_features")),
            "agent_id": agent_id,
            "role": "lead" if agent_id == lead_agent else "member",
            "model": meta.get("model") or result.get("model"),
            "agent_framework": meta.get("agent_framework") or result.get("agent_framework"),
            "team_features": meta.get("team_features", {}),
            "tasks": meta.get("tasks", []),
            "task_log": meta.get("task_log", []),
            "conversation": conversation,
            "trajectory": trajectory,
            "metrics": meta.get("metrics", {}),
            "duration_seconds": meta.get("duration_seconds"),
            "score": eval_data.get("score", 1.0),
        })
    return records


def main() -> int:
    args = parse_args()

    if not args.src.exists():
        print(f"error: source path does not exist: {args.src}", file=sys.stderr)
        print("Run scripts/distill/download_team_coop.py first.", file=sys.stderr)
        return 1

    pairs = _find_pair_dirs(args.src, args.run)
    print(f"Found {len(pairs)} trajectory pairs under {args.src}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    total_records = 0
    skipped = 0

    with args.out.open("w") as fh:
        for run_name, pair_dir in pairs:
            records = _extract_pair(pair_dir=pair_dir, run_name=run_name, require_verified=args.require_verified)
            if records is None:
                skipped += 1
                continue
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
            total_pairs += 1
            total_records += len(records)

    print(f"Successful pairs:  {total_pairs}")
    print(f"Skipped (failed):  {skipped}")
    print(f"Records written:   {total_records}  →  {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
