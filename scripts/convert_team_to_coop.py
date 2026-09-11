"""Convert cooperbench team-run logs into the coop layout from CooperData PR #98.

Source: ``logs/<run>/team/<repo>/<task_id>/<f1_fN>/`` (2-agent lead/member runs).
Target: ``<out>/<run>/coop/<repo>/<task_id>/f1_f2/`` matching the schema in
``cooperbench/CooperData`` PR #98 (``convert_swechat.write_run``).

Mapping:
- agent1 = team lead (asymmetric: holds integration responsibility).
- agent2 = team member (single-feature implementer).
- Inter-agent ``conversation.json`` is rebuilt from ``task_log.json`` events
  (create/claim/update task events with timestamps) plus ``coop-send`` /
  ``coop-broadcast`` invocations extracted from each agent's bash tool calls.
- ``result.json`` keeps the cooperbench shape and folds team-specific extras
  (``team_role``, ``metrics``, ``team_features``) into a ``team`` provenance
  block — analogous to PR #98's ``swechat`` block.
- ``eval.json`` is marked ``verified: true`` because cooperbench runs held-out
  tests (unlike SWE-chat).
- ``metadata.json`` ships the full ``tasks.json`` and ``task_log.json`` so the
  coordination state is fully reconstructible.

Only 2-agent pairs are converted; >2-agent dirs are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("convert_team_to_coop")

SOURCE = "cooperbench-team"
TASK_NAME = "cooperbench_team"
FEATURES = [1, 2]
FEATURE_STR = "f1_f2"

# Extract Redis-backed CLI invocations from bash commands inside tool calls.
# Conservative: matches ``coop-send`` / ``coop-broadcast`` with a -m/--message
# value, which is how mini_swe agents send free-form messages in team mode.
_COOP_SEND_RE = re.compile(
    r"coop-(send|broadcast)\s+(?:--to\s+(\S+)\s+)?(?:-m|--message)\s+"
    r"""(?P<q>['"])(?P<msg>.*?)(?P=q)""",
    re.DOTALL,
)


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("could not parse %s", path)
        return None


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _to_iso(ts: float | str | None) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, str):
        return ts
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _ts_float(ts: Any) -> float | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _agent_index_by_role(team_result: dict[str, Any]) -> dict[str, int]:
    """Return {source_agent_id: 1|2} mapping lead→1, member→2.

    Falls back to natural-sort order of agent ids if roles are absent.
    """
    agents = team_result.get("agents") or {}
    lead = team_result.get("lead_agent")
    if lead and lead in agents:
        members = [a for a in agents if a != lead]
        if len(members) == 1:
            return {lead: 1, members[0]: 2}
    if len(agents) == 2:
        ordered = sorted(agents.keys())
        return {ordered[0]: 1, ordered[1]: 2}
    return {}


def _feature_id_for(agent_summary: dict[str, Any]) -> int | None:
    fid = agent_summary.get("feature_id")
    return int(fid) if fid is not None else None


def _flatten_messages(full_traj: dict[str, Any]) -> list[dict[str, Any]]:
    """All solver-segment messages concatenated (litellm shape preserved).

    mini_swe alternates ``solver`` and ``summarizer`` segments; summarizer
    output is a condensed view of the prior solver, so skipping them and
    concatenating solver messages yields a faithful turn-by-turn log.
    The system message from the first segment is kept once; subsequent
    segments' system messages are dropped.

    Falls back to the top-level ``messages`` field for trajectories that
    don't expose a segments list (codex / claude-code / other one-shot
    adapters write the cooperbench traj envelope directly).
    """
    segments = full_traj.get("segments") or []
    solver_segments = [s for s in segments if s.get("kind") == "solver"]
    if not solver_segments:
        return full_traj.get("messages") or []

    out: list[dict[str, Any]] = []
    for i, seg in enumerate(solver_segments):
        for j, m in enumerate(seg.get("messages", [])):
            if j == 0 and m.get("role") == "system" and i > 0:
                continue
            out.append(m)
    return out


def _load_agent_trajectory(pair_dir: Path, src_id: str) -> dict[str, Any]:
    """Prefer mini_swe's segmented ``_full_traj.json``; fall back to
    the plain cooperbench traj envelope written by codex / claude-code."""
    full = pair_dir / f"{src_id}_full_traj.json"
    if full.exists():
        return _read_json(full) or {}
    return _read_json(pair_dir / f"{src_id}_traj.json") or {}


def _load_sent_messages(pair_dir: Path, src_id: str) -> list[dict[str, Any]]:
    """Parse cooperbench's structured ``<agent>_sent.jsonl`` send log."""
    path = pair_dir / f"{src_id}_sent.jsonl"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("malformed sent.jsonl line in %s", path)
    return out


def _extract_coop_sends(
    traj: dict[str, Any],
    *,
    from_agent: str,
) -> list[dict[str, Any]]:
    """Pull (from, to, message, ts) tuples from ``coop-send/-broadcast``
    bash invocations inside the agent's tool calls.

    Team mode has no structured conversation log; agents message peers by
    running the Redis-backed CLI. We surface that traffic so the coop-style
    ``conversation.json`` isn't empty.
    """
    out: list[dict[str, Any]] = []
    msgs = _flatten_messages(traj)
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        ts = ((m.get("extra") or {}).get("timestamp")) or 0.0
        for tc in m.get("tool_calls") or []:
            args_raw = ((tc.get("function") or {}).get("arguments")) or ""
            try:
                cmd = json.loads(args_raw).get("command", "") if args_raw else ""
            except (json.JSONDecodeError, AttributeError):
                cmd = args_raw if isinstance(args_raw, str) else ""
            for match in _COOP_SEND_RE.finditer(cmd or ""):
                kind = match.group(1)
                to = match.group(2) or "*"  # broadcast → "*"
                text = match.group("msg")
                out.append(
                    {
                        "from": from_agent,
                        "to": to,
                        "message": text,
                        "timestamp": _to_iso(ts),
                        "channel": "coop-cli",
                        "kind": kind,
                    }
                )
    return out


def _conversation_from_task_log(
    task_log: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    role_by_id: dict[str, int],
) -> list[dict[str, Any]]:
    """Each task event becomes a typed broadcast on the task-log channel.

    Schema notes (intentionally drops `to` from the legacy P2P shape because
    these events aren't directed messages — they're broadcasts visible to
    every peer reading the shared task list):

      - ``sender`` — who emitted the event (an agent id, or "bench-runner"
        for system-emitted ``create`` events).
      - ``sender_role`` — ``"system"`` for bench-runner, ``"agent"`` for any
        peer-emitted claim/update.  Lets a downstream consumer condition on
        sender type without string-matching the sender id.
      - ``owner`` — only present on ``create`` events; the pre-assigned task
        owner.  This is the field the old ``to`` was conflating into a
        recipient slot, which it never was.
      - ``channel`` — always ``"task-log"``; reinforces that this is a
        broadcast on a shared log, not P2P traffic.
      - No ``to``.  An ``update`` from the owner to themselves had to be
        rendered as a self-loop under the old shape; here it's just a log
        event with no recipient, which is semantically honest.
    """
    title_by_task = {t.get("id"): t.get("title") for t in (tasks or [])}
    assigned_by_task = {
        t.get("id"): (t.get("metadata") or {}).get("assigned_to") or t.get("owner") for t in (tasks or [])
    }

    out: list[dict[str, Any]] = []
    for ev in task_log or []:
        actor = ev.get("by") or ""
        tid = ev.get("task_id")
        kind = ev.get("kind")
        ts = _to_iso(ev.get("ts"))
        title = ev.get("title") or title_by_task.get(tid, "")
        assignee = assigned_by_task.get(tid)

        if kind == "create":
            sender = actor or "bench-runner"
            sender_role = "system" if sender == "bench-runner" or not actor else "agent"
            msg = f"[task-create #{tid}] {title}"
            if assignee:
                msg += f" → assigned to {assignee}"
            entry = {
                "sender": sender,
                "sender_role": sender_role,
                "owner": assignee or None,
                "message": msg,
                "timestamp": ts,
                "channel": "task-log",
                "kind": kind,
                "task_id": tid,
            }
        elif kind == "claim":
            sender = actor or "bench-runner"
            msg = f"[task-claim #{tid}] claimed: {title}"
            entry = {
                "sender": sender,
                "sender_role": "agent" if actor else "system",
                "message": msg,
                "timestamp": ts,
                "channel": "task-log",
                "kind": kind,
                "task_id": tid,
                "feature_id": role_by_id.get(actor) if actor in role_by_id else None,
            }
        elif kind == "update":
            sender = actor or "bench-runner"
            status = ev.get("status") or ""
            note = ev.get("note") or ""
            msg = f"[task-update #{tid}] status={status}" + (f" — {note}" if note else "")
            entry = {
                "sender": sender,
                "sender_role": "agent" if actor else "system",
                "message": msg,
                "timestamp": ts,
                "channel": "task-log",
                "kind": kind,
                "task_id": tid,
                "status": status or None,
                "feature_id": role_by_id.get(actor) if actor in role_by_id else None,
            }
        else:
            continue

        out.append({k: v for k, v in entry.items() if v is not None})
    return out


def _agent_traj_doc(
    repo: str,
    task_id: int,
    feature_id: int,
    coop_agent_id: str,
    model: str,
    status: str,
    steps: int,
    cost: float,
    messages: list[dict[str, Any]],
    team_block: dict[str, Any],
) -> dict[str, Any]:
    """PR #98 agent{N}_traj.json shape + a ``team`` provenance block."""
    return {
        "repo": repo,
        "task_id": task_id,
        "feature_id": feature_id,
        "agent_id": coop_agent_id,
        "model": model,
        "status": status,
        "cost": float(cost or 0.0),
        "steps": int(steps or 0),
        "messages": messages,
        "team": team_block,
    }


def convert_pair(
    pair_dir: Path,
    out_root: Path,
    run_name: str,
    *,
    source_run_name: str,
) -> dict[str, Any] | None:
    """Convert one team pair directory. Returns a summary row, or None on skip."""
    result = _read_json(pair_dir / "result.json")
    if not result:
        logger.warning("skip %s: no result.json", pair_dir)
        return None

    agents = result.get("agents") or {}
    if len(agents) != 2:
        logger.warning("skip %s: only 2-agent pairs supported (got %d)", pair_dir, len(agents))
        return None

    role_by_id = _agent_index_by_role(result)
    if set(role_by_id.values()) != {1, 2}:
        logger.warning("skip %s: could not assign agent1/agent2 mapping", pair_dir)
        return None

    inv_role = {idx: src for src, idx in role_by_id.items()}  # {1: 'agent1', 2: 'agent2'}
    src_agent1 = inv_role[1]
    src_agent2 = inv_role[2]

    repo = result.get("repo") or pair_dir.parents[1].name
    task_id = int(result.get("task_id") or 0)
    # Coop SLOTS are always (1, 2), but each task in cooperbench has
    # many feature pairs (f1_f3, f1_f4, f2_f3, …). Use the source pair's
    # feature ids in the dir name so they don't collide.
    src_f1 = _feature_id_for(agents.get(src_agent1) or {})
    src_f2 = _feature_id_for(agents.get(src_agent2) or {})
    feature_dir = f"f{src_f1}_f{src_f2}" if src_f1 and src_f2 else FEATURE_STR
    out_dir = out_root / run_name / "coop" / repo / str(task_id) / feature_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_doc = _read_json(pair_dir / "eval.json") or {}
    tasks_doc = _read_json(pair_dir / "tasks.json") or []
    task_log_doc = _read_json(pair_dir / "task_log.json") or []

    team_block_common = {
        "source_run": source_run_name,
        "source_pair_dir": str(pair_dir),
        "lead_agent": result.get("lead_agent"),
        "team_features": result.get("team_features") or {},
        "metrics": result.get("metrics") or {},
        "setting": result.get("setting") or "team",
        "agent_framework": result.get("agent_framework"),
        "model": result.get("model"),
        "duration_seconds": result.get("duration_seconds"),
        "run_id": result.get("run_id"),
    }

    # Per-agent trajectories + patches.
    convo: list[dict[str, Any]] = []
    agent_docs: dict[int, dict[str, Any]] = {}
    for idx, src_id in inv_role.items():
        summary = agents.get(src_id) or {}
        fid = _feature_id_for(summary)
        # mini_swe writes per-feature_id patch files; codex/team also follow that.
        patch_path = pair_dir / f"agent{fid}.patch" if fid else None
        patch_text = patch_path.read_text() if patch_path and patch_path.exists() else ""
        (out_dir / f"agent{idx}.patch").write_text(patch_text)

        full_traj = _load_agent_trajectory(pair_dir, src_id)
        messages = _flatten_messages(full_traj)

        agent_doc = _agent_traj_doc(
            repo=repo,
            task_id=task_id,
            feature_id=fid or idx,
            coop_agent_id=f"agent{idx}",
            model=result.get("model") or "",
            status=summary.get("status") or "Unknown",
            steps=summary.get("steps") or 0,
            cost=summary.get("cost") or 0.0,
            messages=messages,
            team_block={
                **team_block_common,
                "source_agent_id": src_id,
                "team_role": summary.get("team_role"),
                "source_feature_id": fid,
                "input_tokens": summary.get("input_tokens", 0),
                "output_tokens": summary.get("output_tokens", 0),
                "cache_read_tokens": summary.get("cache_read_tokens", 0),
                "cache_write_tokens": summary.get("cache_write_tokens", 0),
                "patch_lines": summary.get("patch_lines", 0),
            },
        )
        # Preserve the full mini_swe segments / info losslessly.
        agent_doc["mini_swe_segments"] = full_traj.get("segments")
        agent_doc["mini_swe_info"] = full_traj.get("info")
        agent_doc["mini_swe_trajectory_format"] = full_traj.get("trajectory_format")
        _dump(out_dir / f"agent{idx}_traj.json", agent_doc)
        agent_docs[idx] = agent_doc

        convo.extend(_extract_coop_sends(full_traj, from_agent=f"agent{idx}"))
        # Codex / claude-code adapters log structured sends to
        # ``<agent>_sent.jsonl``; map ``to`` to the coop slot id.
        for sent in _load_sent_messages(pair_dir, src_id):
            to_src = sent.get("to") or ""
            to_idx = role_by_id.get(to_src)
            convo.append(
                {
                    "from": f"agent{idx}",
                    "to": f"agent{to_idx}" if to_idx else (to_src or "*"),
                    "message": sent.get("content") or "",
                    "timestamp": sent.get("timestamp_iso") or _to_iso(sent.get("timestamp")),
                    "channel": "sent-log",
                    "kind": "send",
                }
            )

    # Conversation = task-log events + structured sent log + coop-send/broadcast.
    convo.extend(_conversation_from_task_log(task_log_doc, tasks_doc, role_by_id))
    convo.sort(key=lambda m: _ts_float(m.get("timestamp")) or 0.0)
    _dump(out_dir / "conversation.json", convo)

    # result.json — coop shape + team provenance.
    correct = bool(eval_doc.get("both_passed"))
    apply_status = eval_doc.get("apply_status") or {}
    merge = eval_doc.get("merge") or {}

    def _coop_agent_summary(idx: int) -> dict[str, Any]:
        src = inv_role[idx]
        summary = agents.get(src) or {}
        return {
            "feature_id": _feature_id_for(summary) or idx,
            "status": summary.get("status") or "Unknown",
            "cost": float(summary.get("cost") or 0.0),
            "steps": int(summary.get("steps") or 0),
            "input_tokens": int(summary.get("input_tokens") or 0),
            "output_tokens": int(summary.get("output_tokens") or 0),
            "cache_read_tokens": int(summary.get("cache_read_tokens") or 0),
            "cache_write_tokens": int(summary.get("cache_write_tokens") or 0),
            "patch_lines": int(summary.get("patch_lines") or 0),
            "error": summary.get("error"),
        }

    log_dir_rel = str(Path(run_name) / "coop" / repo / str(task_id) / feature_dir)
    coop_result = {
        "repo": repo,
        "task_id": task_id,
        "features": FEATURES,
        "setting": "coop",
        "run_id": result.get("run_id"),
        "run_name": run_name,
        "agent_framework": result.get("agent_framework"),
        "model": result.get("model"),
        "started_at": result.get("started_at"),
        "ended_at": result.get("ended_at"),
        "duration_seconds": result.get("duration_seconds") or 0.0,
        "agents": {f"agent{idx}": _coop_agent_summary(idx) for idx in (1, 2)},
        "total_cost": float(result.get("total_cost") or 0.0),
        "total_steps": int(result.get("total_steps") or 0),
        "messages_sent": len(convo),
        "log_dir": log_dir_rel,
        "team": {
            **team_block_common,
            "source_features": [
                _feature_id_for(agents.get(src_agent1) or {}),
                _feature_id_for(agents.get(src_agent2) or {}),
            ],
            "apply_status": apply_status,
            "merge_status": merge.get("status"),
            "merge_strategy": merge.get("strategy"),
        },
    }
    _dump(out_dir / "result.json", coop_result)

    # eval.json — preserve the team eval verbatim, recast to coop schema.
    # ``verified`` flags a POSITIVE outcome confirmed by held-out tests —
    # set True only when ``correct`` is True. Failures get
    # ``verified: false`` even though cooperbench did run tests, so
    # downstream filters can treat verified-positive trajectories as the
    # headline-success subset.
    coop_eval = {
        "correct": correct,
        "score": 1.0 if correct else 0.0,
        "eval": "pass" if correct else "fail",
        "verified": correct,
        "eval_source": "cooperbench_held_out_tests",
        "both_passed": correct,
        "feature1": eval_doc.get("feature1"),
        "feature2": eval_doc.get("feature2"),
        "apply_status": apply_status,
        "merge": merge,
        "error": eval_doc.get("error"),
        "evaluated_at": eval_doc.get("evaluated_at"),
    }
    _dump(out_dir / "eval.json", coop_eval)

    # metadata.json — full coordination state + provenance.
    metadata = {
        "source": SOURCE,
        "source_run": source_run_name,
        "source_pair_dir": str(pair_dir),
        "task_name": TASK_NAME,
        "repo": repo,
        "task_id": task_id,
        "features": FEATURES,
        "source_features": [
            _feature_id_for(agents.get(src_agent1) or {}),
            _feature_id_for(agents.get(src_agent2) or {}),
        ],
        "team_features": result.get("team_features") or {},
        "lead_agent": result.get("lead_agent"),
        "metrics": result.get("metrics") or {},
        "agent_framework": result.get("agent_framework"),
        "model": result.get("model"),
        "started_at": result.get("started_at"),
        "ended_at": result.get("ended_at"),
        "duration_seconds": result.get("duration_seconds"),
        "agent_id_mapping": {f"agent{idx}": inv_role[idx] for idx in (1, 2)},
        "tasks": tasks_doc,
        "task_log": task_log_doc,
        "converted_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    _dump(out_dir / "metadata.json", metadata)

    return {
        "task": f"{repo}/{task_id}/{src_f1},{src_f2}",
        "source_pair_dir": str(pair_dir),
        "status": "completed",
        "cost": float(result.get("total_cost") or 0.0),
        "eval": "pass" if correct else "fail",
        "score": 1.0 if correct else 0.0,
    }


def _iter_pair_dirs(team_root: Path):
    # logs/<run>/team/<repo>/<task_id>/<f1_fN>/
    for repo_dir in sorted(p for p in team_root.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in repo_dir.iterdir() if p.is_dir()):
            yield from sorted(p for p in task_dir.iterdir() if p.is_dir())


def convert_run(
    source_run: Path,
    out_root: Path,
    *,
    run_name: str | None = None,
) -> dict[str, Any]:
    team_root = source_run / "team"
    if not team_root.is_dir():
        raise SystemExit(f"no team dir: {team_root}")

    run_name = run_name or source_run.name
    source_run_name = source_run.name
    started_at = datetime.now(tz=timezone.utc).isoformat()

    rows: list[dict[str, Any]] = []
    for pair_dir in _iter_pair_dirs(team_root):
        row = convert_pair(pair_dir, out_root, run_name, source_run_name=source_run_name)
        if row:
            rows.append(row)
            logger.info("converted %s → %s (%s)", pair_dir, row["task"], row["eval"])

    # Run-level config.json + summary.json (PR #98 shape).
    source_config = _read_json(source_run / "config.json") or {}
    source_summary = _read_json(source_run / "summary.json") or {}
    run_root = out_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    _dump(
        run_root / "config.json",
        {
            "run_name": run_name,
            "agent_framework": source_config.get("agent_framework"),
            "model": source_config.get("model"),
            "setting": "coop",
            "source": SOURCE,
            "source_run": source_run_name,
            "source_config": source_config,
            "total_tasks": len(rows),
            "started_at": started_at,
        },
    )
    graded = [r for r in rows if r["eval"] in ("pass", "fail")]
    passed = sum(1 for r in rows if r["eval"] == "pass")
    _dump(
        run_root / "summary.json",
        {
            "run_name": run_name,
            "completed_at": datetime.now(tz=timezone.utc).isoformat(),
            "source": SOURCE,
            "source_run": source_run_name,
            "total_tasks": len(rows),
            "completed": len(rows),
            "pass_rate": (passed / len(graded)) if graded else None,
            "total_cost": sum(r["cost"] for r in rows),
            "results": rows,
            "source_summary": source_summary,
        },
    )
    return {"run_name": run_name, "rows": rows, "pass_rate": (passed / len(graded)) if graded else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path, help="path to logs/<run>/")
    parser.add_argument("--out", type=Path, default=Path("data"), help="output root (default: ./data)")
    parser.add_argument(
        "--run-name", type=str, default=None, help="override run name in output (default: source dir name)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    res = convert_run(args.source_run, args.out, run_name=args.run_name)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
