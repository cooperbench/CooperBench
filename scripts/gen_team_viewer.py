"""Generate the team-trajectory viewer: an interactive, self-contained HTML that
indexes every team run under ``logs/`` and lets you drill into any pair to read
its *coordination story* — the lead/member split, the task-list timeline
(create/claim/update/done), inter-agent messages, the final task board, the
eval result, and (collapsible, secondary) each agent's step trajectory + patch.

Like the other ``gen_*`` scripts this reads ``logs/`` at generation time and
emits one self-contained file (inline CSS/JS, all data embedded — no server, no
external assets) so it works opened locally *and* deployed to Pages.

Usage:
    uv run python scripts/gen_team_viewer.py                # all team runs
    uv run python scripts/gen_team_viewer.py --runs msa_    # substring filter
    uv run python scripts/gen_team_viewer.py -o out.html

Message bodies, notes, test output and patches are truncated to keep the file
bounded (the focus is coordination, not full reasoning replay).
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs"
DEFAULT_OUT = REPO / "docs" / f"{date.today().isoformat()}-team-viewer.html"

# Truncation budgets (chars / lines) — keep the embedded JSON bounded.
# Coordination content (notes/messages/test output) is the focus; agent
# trajectories and patches are secondary, so they get tighter budgets.
NOTE_MAX = 300
MSG_MAX = 800
TRAJ_MSG_MAX = 300
TRAJ_MSGS_MAX = 16
TESTOUT_MAX = 600
PATCH_LINES_MAX = 22


def trunc(s: str | None, n: int) -> str:
    if not s:
        return ""
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + f"\n… (+{len(s) - n} more chars)"


def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def epoch(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp()
    except Exception:
        return None


def short(tid: str | None) -> str:
    return (tid or "")[:8]


def pair_status(eval_: dict | None, agents: dict) -> str:
    agent_err = any((a.get("error") or a.get("status") == "Error") for a in agents.values())
    if eval_:
        if eval_.get("both_passed"):
            return "pass"
        f1 = (eval_.get("feature1") or {}).get("passed")
        f2 = (eval_.get("feature2") or {}).get("passed")
        if f1 or f2:
            return "partial"
        return "fail"
    return "error" if agent_err else "unknown"


def build_timeline(task_log: list | None, conversation: list | None, t0: float | None) -> list:
    events: list[dict] = []
    for e in task_log or []:
        if not isinstance(e, dict):
            continue
        events.append(
            {
                "kind": e.get("kind", "?"),
                "by": e.get("by", "?"),
                "task": short(e.get("task_id")),
                "title": trunc(e.get("title"), NOTE_MAX),
                "status": e.get("status"),
                "note": trunc(e.get("note"), NOTE_MAX),
                "ts": e.get("ts"),
            }
        )
    for m in conversation or []:
        if not isinstance(m, dict):
            continue
        events.append(
            {
                "kind": "message",
                "by": m.get("from", "?"),
                "to": m.get("to"),
                "note": trunc(m.get("message"), MSG_MAX),
                "ts": m.get("ts") or m.get("timestamp"),
                "feature_id": m.get("feature_id"),
            }
        )
    events.sort(key=lambda e: (e.get("ts") is None, e.get("ts") or 0))
    if t0:
        for e in events:
            ts = e.get("ts")
            e["rel"] = round(ts - t0, 1) if ts else None
    return events


def extract_traj(log_dir: Path, agent_id: str, feature_id: int) -> dict:
    """Per-agent trajectory: prefer the rich ``*_full_traj.json`` (mini-swe-agent,
    keyed by agent index) and fall back to the ``agent{feature}_traj.json``
    summary. Messages truncated; secondary content for the viewer."""
    out: dict = {"messages": [], "source": None, "cost": None, "api_calls": None}
    full = load_json(log_dir / f"{agent_id}_full_traj.json")
    if isinstance(full, dict) and isinstance(full.get("messages"), list):
        out["source"] = "full_traj"
        stats = (full.get("info") or {}).get("model_stats") or {}
        out["cost"] = stats.get("instance_cost")
        out["api_calls"] = stats.get("api_calls")
        msgs = full["messages"]
        for m in msgs[:TRAJ_MSGS_MAX]:
            if not isinstance(m, dict):
                continue
            out["messages"].append(
                {
                    "role": m.get("role"),
                    "content": trunc(m.get("content"), TRAJ_MSG_MAX),
                    "cost": (m.get("extra") or {}).get("cost"),
                }
            )
        if len(msgs) > TRAJ_MSGS_MAX:
            out["omitted"] = len(msgs) - TRAJ_MSGS_MAX
        return out
    summ = load_json(log_dir / f"agent{feature_id}_traj.json")
    if isinstance(summ, dict) and isinstance(summ.get("messages"), list):
        out["source"] = "traj"
        for m in summ["messages"][:TRAJ_MSGS_MAX]:
            if isinstance(m, dict):
                out["messages"].append({"role": m.get("role"), "content": trunc(m.get("content"), TRAJ_MSG_MAX)})
    return out


MCP_RE = re.compile(r"coop-task-[a-z-]+")
SHARED_RE = re.compile(r"/workspace/shared/[^\s`\"'<>|)]*")
FEAT_EVENTS_MAX = 240


def _msg_times(msgs: list, t0: float | None, duration: float | None) -> tuple[list, bool]:
    """Per-message relative time. Exact from assistant ``extra.timestamp`` when
    present (mini-swe-agent), else interpolated across the run by step index
    (codex summary trajs carry no timestamps)."""
    n = len(msgs)
    have_real = t0 is not None and any(
        isinstance(m, dict) and m.get("role") == "assistant" and (m.get("extra") or {}).get("timestamp") for m in msgs
    )
    times: list[float | None] = [None] * n
    if have_real:
        last: float | None = None
        for i, m in enumerate(msgs):
            if isinstance(m, dict) and m.get("role") == "assistant":
                ts = (m.get("extra") or {}).get("timestamp")
                if ts:
                    last = round(ts - t0, 1)  # type: ignore[operator]
            times[i] = last  # forward-fill onto tool/refresh messages
        return times, True
    if duration and n > 1:
        times = [round(duration * i / (n - 1), 1) for i in range(n)]
    return times, False


STREAM_END_RE = re.compile(r"^\s*(succeeded|exited|failed|error|aborted)\b", re.I)


def _scan_cmd(text: str, include_protocol: bool) -> list[tuple[str, str]]:
    """Feature signals in one command / assistant turn → (feat, label) pairs."""
    out: list[tuple[str, str]] = []
    cmds = list(dict.fromkeys(MCP_RE.findall(text)))
    if cmds:
        out.append(("mcp", ", ".join(cmds[:3])))
    if include_protocol and "coop-send" in text:
        out.append(("protocol", "coop-send"))
    sh = SHARED_RE.search(text)
    if sh or "/workspace/shared" in text:
        out.append(("scratchpad", trunc(sh.group(0) if sh else "/workspace/shared", 80)))
    return out


def _scan_stream(path: Path, duration: float | None, aid: str) -> list[dict]:
    """Parse a raw codex ``*_stream.log`` for feature usage. Codex marks real
    executions with an ``exec`` block whose command ends ``in /workspace/...`` —
    we scan those blocks (not the prompt's command documentation). No timestamps
    in the stream, so events are placed by line position (approximate)."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception:
        return []
    n = len(lines)
    if n < 2:
        return []
    out: list[dict] = []
    i = 0
    while i < n:
        if lines[i].strip() == "exec":
            j, cmd = i + 1, []
            while j < n and lines[j].strip() != "exec" and not STREAM_END_RE.match(lines[j]) and len(cmd) < 8:
                cmd.append(lines[j])
                j += 1
            t = round(duration * i / (n - 1), 1) if duration else None
            for feat, label in _scan_cmd(" ".join(cmd), include_protocol=True):
                out.append({"feat": feat, "t": t, "by": aid, "label": label})
            i = max(j, i + 1)
        else:
            i += 1
    return out


def extract_feature_usage(
    log_dir: Path, agents: dict, t0: float | None, duration: float | None
) -> tuple[list[dict], bool]:
    """Per-feature usage events (mcp / scratchpad / auto_refresh / protocol-sends)
    scanned from each agent's trajectory: the parsed messages when present
    (``*_full_traj.json`` exact timestamps, multi-message ``*_traj.json``
    interpolated), else the raw codex ``*_stream.log``. task_list + the
    conversation side of protocol are derived client-side from the embedded
    timeline, so they are not repeated here."""
    events: list[dict] = []
    approx = False
    for aid, a in agents.items():
        fid = a.get("feature_id")
        a_events: list[dict] = []
        a_approx = False
        src = load_json(log_dir / f"{aid}_full_traj.json")
        if not (isinstance(src, dict) and isinstance(src.get("messages"), list)):
            src = load_json(log_dir / f"agent{fid}_traj.json")
        msgs = src.get("messages") if isinstance(src, dict) else None
        if isinstance(msgs, list) and len(msgs) > 1:
            times, exact = _msg_times(msgs, t0, duration)
            a_approx = not exact
            for i, m in enumerate(msgs):
                if not isinstance(m, dict):
                    continue
                content = str(m.get("content", ""))
                t = times[i]
                if m.get("role") == "assistant":
                    for feat, label in _scan_cmd(content, include_protocol=False):
                        a_events.append({"feat": feat, "t": t, "by": aid, "label": label})
                elif "[Team task list]" in content:
                    a_events.append({"feat": "auto_refresh", "t": t, "by": aid, "label": "task-list state injected"})
        if not a_events:  # codex full-dataset runs only keep the raw stream
            stream = log_dir / f"{aid}_stream.log"
            if stream.exists():
                a_events = _scan_stream(stream, duration, aid)
                a_approx = True
        if a_events:
            events.extend(a_events)
            approx = approx or a_approx
    events = [e for e in events if e["t"] is not None]
    events.sort(key=lambda e: e["t"])
    # collapse back-to-back identical events (e.g. repeated coop-task-list polls)
    deduped: list[dict] = []
    prev: tuple | None = None
    for e in events:
        key = (e["feat"], e.get("by"), e["label"])
        if key != prev:
            deduped.append(e)
            prev = key
    if len(deduped) > FEAT_EVENTS_MAX:
        deduped = deduped[:FEAT_EVENTS_MAX]
    return deduped, approx


def read_patch(p: Path) -> str | None:
    if not p.exists():
        return None
    try:
        lines = p.read_text().splitlines()
    except Exception:
        return None
    if not lines:
        return None
    if len(lines) > PATCH_LINES_MAX:
        extra = len(lines) - PATCH_LINES_MAX
        lines = lines[:PATCH_LINES_MAX] + [f"… (+{extra} more lines)"]
    return "\n".join(lines)


PLAN_MARKER = "> /workspace/shared/PLAN.md"
PROTO_MSGS_MAX = 40
PROTO_MSG_MAX = 400


def _extract_plan(lines: list[str]) -> str | None:
    """Best-effort PLAN.md content from a single-line ``'<content>' > .../PLAN.md``
    write (heredoc / multi-line writes are skipped). Cheap and bounded."""
    for line in lines:
        if PLAN_MARKER not in line:
            continue
        seg = line[: line.index(PLAN_MARKER)]
        q = max(seg.rfind("'"), seg.rfind('"'))
        if q <= 0:
            continue
        start = seg.rfind(seg[q], 0, q)
        if start >= 0 and q - start > 20:
            return trunc(seg[start + 1 : q], 1200)
    return None


def extract_scratchpad_plan(log_dir: Path, agents: dict) -> str | None:
    """Recover the lead's PLAN.md from whichever agent wrote it (trajectory
    message or raw stream)."""
    for aid, a in agents.items():
        fid = a.get("feature_id")
        src = load_json(log_dir / f"{aid}_full_traj.json") or load_json(log_dir / f"agent{fid}_traj.json")
        msgs = src.get("messages") if isinstance(src, dict) else None
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and PLAN_MARKER in str(m.get("content", "")):
                    plan = _extract_plan(str(m["content"]).splitlines())
                    if plan:
                        return plan
        stream = log_dir / f"{aid}_stream.log"
        if stream.exists():
            try:
                plan = _extract_plan(stream.read_text(errors="replace").splitlines())
            except Exception:
                plan = None
            if plan:
                return plan
    return None


def extract_protocol(log_dir: Path, conversation: list | None, task_log: list | None, t0: float | None) -> dict:
    """Inter-agent communication: free-text messages (per-agent ``*_sent.jsonl``,
    or conversation.json as fallback) + typed request/respond protocol events."""

    def rel(ts: float | None) -> float | None:
        return round(ts - t0, 1) if (ts and t0) else None

    messages: list[dict] = []
    for sf in sorted(log_dir.glob("*_sent.jsonl")):
        try:
            text = sf.read_text(errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except Exception:
                continue
            messages.append(
                {
                    "from": m.get("from"),
                    "to": m.get("to"),
                    "content": trunc(m.get("content") or m.get("message"), PROTO_MSG_MAX),
                    "t": rel(m.get("timestamp") or m.get("ts")),
                }
            )
    if not messages:
        for m in conversation or []:
            if isinstance(m, dict):
                messages.append(
                    {
                        "from": m.get("from"),
                        "to": m.get("to"),
                        "content": trunc(m.get("message") or m.get("content"), PROTO_MSG_MAX),
                        "t": rel(m.get("timestamp") or m.get("ts")),
                    }
                )
    messages.sort(key=lambda x: (x["t"] is None, x["t"] or 0))

    requests: list[dict] = []
    for e in task_log or []:
        if isinstance(e, dict) and e.get("kind") in ("request", "respond"):
            requests.append(
                {
                    "by": e.get("by"),
                    "to": e.get("to"),
                    "kind": e.get("request_kind") or e.get("kind"),
                    "verb": e.get("kind"),
                    "t": rel(e.get("ts")),
                }
            )
    return {"messages": messages[:PROTO_MSGS_MAX], "requests": requests[:PROTO_MSGS_MAX]}


def build_pair(rj: Path) -> dict | None:
    d = load_json(rj)
    if not isinstance(d, dict):
        return None
    log_dir = rj.parent
    agents = d.get("agents") or {}
    eval_ = load_json(log_dir / "eval.json")
    task_log = load_json(log_dir / "task_log.json")
    tasks = load_json(log_dir / "tasks.json")
    conversation = load_json(log_dir / "conversation.json")
    t0 = epoch(d.get("started_at"))
    if not t0 and isinstance(task_log, list) and task_log:
        ts_vals = [e["ts"] for e in task_log if isinstance(e, dict) and e.get("ts")]
        t0 = min(ts_vals) if ts_vals else None

    # eval: keep summary + truncated per-feature test output
    eval_out = None
    if isinstance(eval_, dict):
        eval_out = {
            "both_passed": eval_.get("both_passed"),
            "merge": (eval_.get("merge") or {}).get("status"),
            "apply_status": eval_.get("apply_status"),
            "error": eval_.get("error"),
            "features": [],
        }
        for fk in ("feature1", "feature2"):
            f = eval_.get(fk)
            if isinstance(f, dict):
                eval_out["features"].append(
                    {
                        "feature_id": f.get("feature_id"),
                        "passed": f.get("passed"),
                        "tests_passed": f.get("tests_passed"),
                        "tests_failed": f.get("tests_failed"),
                        "test_output": trunc(f.get("test_output"), TESTOUT_MAX),
                    }
                )

    trajectories = {}
    patches = {}
    for aid, a in agents.items():
        fid = a.get("feature_id")
        trajectories[aid] = extract_traj(log_dir, aid, fid)
        patch = read_patch(log_dir / f"agent{fid}.patch")
        if patch:
            patches[aid] = patch

    duration = d.get("duration_seconds")
    feat_events, ts_approx = extract_feature_usage(log_dir, agents, t0, duration)
    plan = extract_scratchpad_plan(log_dir, agents)
    protocol = extract_protocol(log_dir, conversation, task_log, t0)

    return {
        "run": d.get("run_name") or rj.parents[2].name,
        "repo": d.get("repo"),
        "task_id": d.get("task_id"),
        "features": d.get("features"),
        "framework": d.get("agent_framework"),
        "model": d.get("model"),
        "duration": d.get("duration_seconds"),
        "lead": d.get("lead_agent"),
        "status": pair_status(eval_, agents),
        "agents": {
            aid: {
                "feature_id": a.get("feature_id"),
                "role": a.get("team_role"),
                "status": a.get("status"),
                "steps": a.get("steps"),
                "cost": a.get("cost"),
                "input_tokens": a.get("input_tokens"),
                "output_tokens": a.get("output_tokens"),
                "patch_lines": a.get("patch_lines"),
                "error": trunc(a.get("error"), NOTE_MAX),
            }
            for aid, a in agents.items()
        },
        "metrics": d.get("metrics") or {},
        "team_features": d.get("team_features") or {},
        "eval": eval_out,
        "feat_events": feat_events,
        "ts_approx": ts_approx,
        "plan": plan,
        "protocol": protocol,
        "timeline": build_timeline(task_log, conversation, t0),
        "tasks": [
            {
                "id": short(t.get("id")),
                "title": trunc(t.get("title"), NOTE_MAX),
                "owner": t.get("owner"),
                "status": t.get("status"),
                "last_note": trunc(t.get("last_note"), NOTE_MAX),
                "created_by": t.get("created_by"),
            }
            for t in (tasks or [])
            if isinstance(t, dict)
        ],
        "trajectories": trajectories,
        "patches": patches,
    }


# The curated "study" set: the runs that back the published coordination report
# (full-dataset comparison, feature ablations, msa core) — excludes per-framework
# dev/smoke iterations (cc_/cx_/oh_ cores, *_v1.., *smoke*, probe-*).
STUDY_PREFIXES = ("cmp-full-", "ablate-", "msa_team_core")


def collect(patterns: list[str] | None) -> list[dict]:
    """Include a run if its name contains any of ``patterns`` (None = all)."""
    pairs: list[dict] = []
    for team_dir in sorted(LOGS.glob("*/team")):
        run = team_dir.parent.name
        if patterns and not any(pat in run for pat in patterns):
            continue
        for rj in sorted(team_dir.rglob("result.json")):
            p = build_pair(rj)
            if p:
                pairs.append(p)
    return pairs


# ----------------------------------------------------------------------------- HTML

STYLE = """
:root{--fg:#1f2937;--muted:#6b7280;--line:#e5e7eb;--bg:#fff;--panel:#f9fafb;
 --pass:#16a34a;--fail:#dc2626;--partial:#d97706;--err:#7c3aed;--unknown:#9ca3af;
 --a1:#2563eb;--a2:#16a34a;--a3:#d97706;--a4:#db2777;}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
 color:var(--fg);background:var(--bg);font-size:14px;line-height:1.5}
#app{display:grid;grid-template-columns:340px 1fr;height:100vh}
#side{border-right:1px solid var(--line);overflow-y:auto;background:var(--panel)}
#main{overflow-y:auto;padding:1.2rem 1.6rem}
.shead{position:sticky;top:0;background:var(--panel);padding:.7rem .8rem;border-bottom:1px solid var(--line);z-index:2}
.shead h1{font-size:1rem;margin:0 0 .4rem}
.shead .sub{color:var(--muted);font-size:.78rem;margin-bottom:.5rem}
#q{width:100%;padding:.35rem .5rem;border:1px solid var(--line);border-radius:6px;font-size:.85rem}
.filters{display:flex;gap:.3rem;margin-top:.4rem;flex-wrap:wrap}
.filters button{font-size:.72rem;padding:.2rem .45rem;border:1px solid var(--line);background:#fff;border-radius:99px;cursor:pointer;color:var(--muted)}
.filters button.on{background:var(--fg);color:#fff;border-color:var(--fg)}
.run{border-bottom:1px solid var(--line)}
.run>summary{cursor:pointer;padding:.5rem .8rem;font-weight:600;font-size:.85rem;list-style:none;display:flex;justify-content:space-between;align-items:center;gap:.4rem}
.run>summary::-webkit-details-marker{display:none}
.run>summary:hover{background:#f0f1f3}
.run .rate{font-size:.72rem;color:var(--muted);font-weight:500;font-variant-numeric:tabular-nums}
.pair{padding:.35rem .8rem .35rem 1.2rem;cursor:pointer;font-size:.8rem;display:flex;align-items:center;gap:.45rem;border-top:1px solid #f0f1f3}
.pair:hover{background:#eef2ff}
.pair.sel{background:#e0e7ff}
.dot{width:8px;height:8px;border-radius:99px;flex:none}
.dot.pass{background:var(--pass)}.dot.fail{background:var(--fail)}.dot.partial{background:var(--partial)}
.dot.error{background:var(--err)}.dot.unknown{background:var(--unknown)}
.pair .meta{color:var(--muted);font-size:.72rem}
.empty{padding:3rem 1rem;text-align:center;color:var(--muted)}
/* main */
.hdr{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin-bottom:.2rem}
.hdr h2{font-size:1.2rem;margin:0}
.badge{font-size:.72rem;font-weight:700;padding:.15rem .5rem;border-radius:99px;color:#fff;text-transform:uppercase;letter-spacing:.03em}
.badge.pass{background:var(--pass)}.badge.fail{background:var(--fail)}.badge.partial{background:var(--partial)}
.badge.error{background:var(--err)}.badge.unknown{background:var(--unknown)}
.crumbs{color:var(--muted);font-size:.85rem;margin-bottom:1rem}
.crumbs code{background:var(--panel);padding:.1rem .35rem;border-radius:3px}
section{margin:1.4rem 0}
section>h3{font-size:.95rem;margin:0 0 .6rem;padding-bottom:.25rem;border-bottom:2px solid var(--line)}
.cards{display:flex;gap:.7rem;flex-wrap:wrap}
.card{border:1px solid var(--line);border-radius:8px;padding:.6rem .8rem;min-width:210px;flex:1}
.card .top{display:flex;align-items:center;gap:.4rem;margin-bottom:.3rem}
.card .aid{font-weight:700}
.role{font-size:.68rem;font-weight:700;padding:.1rem .4rem;border-radius:99px;color:#fff}
.role.lead{background:#111827}.role.member{background:#6b7280}
.kv{display:grid;grid-template-columns:auto 1fr;gap:.05rem .6rem;font-size:.8rem}
.kv .k{color:var(--muted)}.kv .v{text-align:right;font-variant-numeric:tabular-nums}
.err{color:var(--fail);font-size:.78rem;margin-top:.3rem;white-space:pre-wrap;word-break:break-word}
.chips{display:flex;gap:.35rem;flex-wrap:wrap}
.chip{font-size:.72rem;padding:.15rem .5rem;border-radius:99px;border:1px solid var(--line)}
.chip.on{background:#f0fdf4;color:var(--pass);border-color:#bbf7d0}
.chip.off{background:#fef2f2;color:var(--fail);border-color:#fecaca;text-decoration:line-through}
.mline{font-size:.85rem;color:var(--muted);margin-bottom:.6rem}
.mline b{color:var(--fg)}
/* timeline */
.tl{border-left:2px solid var(--line);margin-left:.5rem;padding-left:0}
.ev{position:relative;padding:.35rem 0 .35rem 1.1rem;font-size:.84rem}
.ev::before{content:"";position:absolute;left:-5px;top:.7rem;width:8px;height:8px;border-radius:99px;background:var(--m,var(--muted))}
.ev .when{color:var(--muted);font-size:.72rem;font-variant-numeric:tabular-nums;margin-right:.4rem}
.ev .who{font-weight:700}
.ev .kind{font-size:.68rem;font-weight:700;padding:.05rem .4rem;border-radius:99px;color:#fff;margin:0 .35rem}
.kind-create{background:#0ea5e9}.kind-claim{background:#8b5cf6}.kind-update{background:#64748b}
.kind-message{background:#db2777}.kind-done{background:var(--pass)}
.ev .tid{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.72rem;color:var(--muted)}
.ev .note{display:block;margin-top:.15rem;color:#374151;white-space:pre-wrap;word-break:break-word}
.ev .stat{font-size:.72rem;font-weight:600}
/* task board */
.board{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.7rem}
.col h4{font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 .4rem}
.tcard{border:1px solid var(--line);border-radius:6px;padding:.45rem .55rem;margin-bottom:.45rem;font-size:.8rem;background:#fff}
.tcard .tt{display:block;margin-bottom:.2rem}
.tcard .ow{font-size:.72rem;color:var(--muted)}
/* collapsible */
details.box{border:1px solid var(--line);border-radius:8px;margin:.5rem 0}
details.box>summary{cursor:pointer;padding:.5rem .8rem;font-weight:600;font-size:.85rem}
details.box[open]>summary{border-bottom:1px solid var(--line)}
.boxbody{padding:.6rem .8rem}
.msg{border-left:3px solid var(--line);padding:.2rem .6rem;margin:.5rem 0}
.msg.system{border-color:#9ca3af}.msg.user{border-color:#0ea5e9}.msg.assistant{border-color:#16a34a}.msg.tool{border-color:#d97706}
.msg .role{display:inline-block;color:#fff;background:#6b7280;margin-bottom:.2rem}
.msg.assistant .role{background:#16a34a}.msg.user .role{background:#0ea5e9}.msg.tool .role{background:#d97706}.msg.system .role{background:#9ca3af}
.msg pre{margin:.2rem 0 0;white-space:pre-wrap;word-break:break-word;font-size:.8rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre.diff{background:#0d1117;color:#c9d1d9;border-radius:6px;padding:.7rem;overflow-x:auto;font-size:.76rem;line-height:1.4;white-space:pre}
pre.diff .a{color:#3fb950}pre.diff .d{color:#f85149}pre.diff .h{color:#58a6ff}pre.diff .m{color:#8b949e}
pre.testout{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.6rem;overflow-x:auto;font-size:.76rem;white-space:pre-wrap}
.muted{color:var(--muted)}
/* feature usage replay */
.replay{border:1px solid var(--line);border-radius:8px;padding:.7rem .9rem;background:var(--panel)}
.rctl{display:flex;align-items:center;gap:.6rem;margin-bottom:.7rem;flex-wrap:wrap}
.rctl button.play{font-size:.82rem;font-weight:600;padding:.3rem .7rem;border:1px solid var(--fg);background:var(--fg);color:#fff;border-radius:6px;cursor:pointer;min-width:84px}
.rctl input[type=range]{flex:1;min-width:160px;accent-color:#111827}
.rtime{font-variant-numeric:tabular-nums;font-size:.8rem;color:var(--muted)}
.spd{display:flex;gap:.2rem}
.spd button{font-size:.72rem;padding:.15rem .4rem;border:1px solid var(--line);background:#fff;border-radius:4px;cursor:pointer;color:var(--muted)}
.spd button.on{background:var(--fg);color:#fff;border-color:var(--fg)}
.lane{display:grid;grid-template-columns:118px 1fr;align-items:center;gap:.6rem;margin:.28rem 0}
.lane .lbl{font-size:.78rem;font-weight:600;display:flex;align-items:center;gap:.35rem;font-variant-numeric:tabular-nums}
.lane .lbl .sw{width:9px;height:9px;border-radius:2px;flex:none}
.lane .lbl.dis{opacity:.42}
.track{position:relative;height:22px;background:#eceef1;border-radius:4px;overflow:visible}
.track .dot{position:absolute;top:50%;transform:translate(-50%,-50%);width:9px;height:9px;border-radius:99px;opacity:.22;transition:opacity .12s ease}
.track .dot.past{opacity:1}
.playhead{position:absolute;top:-3px;bottom:-3px;width:2px;background:#111827;left:0;z-index:3;pointer-events:none}
.rnote{font-size:.74rem;color:#b45309;margin-top:.45rem}
.rfeed{margin-top:.6rem;font-size:.82rem;max-height:150px;overflow:auto;border-top:1px solid var(--line);padding-top:.4rem}
.rfeed .row{padding:.12rem 0;display:flex;gap:.45rem;align-items:baseline}
.rfeed .row .tt{min-width:48px;color:var(--muted);font-variant-numeric:tabular-nums;font-size:.74rem}
.rfeed .ft{font-size:.64rem;font-weight:700;color:#fff;padding:.04rem .35rem;border-radius:99px;flex:none}
/* task-list widget */
.tasklist{display:flex;flex-direction:column;gap:.5rem}
.tk{border:1px solid var(--line);border-radius:8px;padding:.5rem .7rem}
.tk .head{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}
.tk .st{font-size:.66rem;font-weight:700;color:#fff;padding:.06rem .45rem;border-radius:99px;text-transform:uppercase}
.tk .st.open{background:#9ca3af}.tk .st.in_progress{background:#0ea5e9}.tk .st.done{background:var(--pass)}
.tk .tt{font-weight:600;font-size:.86rem}
.tk .meta{font-size:.74rem;color:var(--muted)}
.tk .life{margin:.4rem 0 0;border-left:2px solid var(--line);padding-left:.7rem;display:flex;flex-direction:column;gap:.2rem}
.tk .life .l{font-size:.78rem;display:flex;gap:.4rem;align-items:baseline}
.tk .life .l .w{min-width:46px;color:var(--muted);font-variant-numeric:tabular-nums;font-size:.72rem}
.tk .life .l .vb{font-size:.62rem;font-weight:700;color:#fff;padding:.02rem .35rem;border-radius:99px;flex:none}
.tk .life .nt{color:#374151}
/* scratchpad widget */
.sp .path{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.78rem;color:var(--muted);margin-bottom:.4rem}
.spfile>summary{display:flex;align-items:center;gap:.4rem}
.spfile .fn{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:600}
.spfile .tag{font-size:.62rem;font-weight:700;color:#fff;background:#6b7280;padding:.04rem .35rem;border-radius:99px}
.spfile .tag.plan{background:#7c3aed}.spfile .tag.patch{background:#0d9488}
pre.plan{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.6rem;white-space:pre-wrap;word-break:break-word;font-size:.8rem}
/* protocol widget */
.proto .msgs{display:flex;flex-direction:column;gap:.45rem}
.pmsg{max-width:78%;border:1px solid var(--line);border-radius:10px;padding:.4rem .6rem;font-size:.83rem}
.pmsg .ph{font-size:.72rem;margin-bottom:.15rem}
.pmsg .ph .ar{color:var(--muted)}
.pmsg .ph .w{color:var(--muted);font-variant-numeric:tabular-nums;margin-left:.3rem}
.pmsg .body{white-space:pre-wrap;word-break:break-word}
.preqs{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.6rem}
.preq{font-size:.74rem;border:1px solid #fbcfe8;background:#fdf2f8;color:#9d174d;padding:.12rem .5rem;border-radius:99px}
"""

APP_JS = r"""
const $ = (s,r=document)=>r.querySelector(s);
const el = (t,c,h)=>{const e=document.createElement(t);if(c)e.className=c;if(h!=null)e.innerHTML=h;return e;};
const esc = s => (s==null?"":String(s)).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const fmtDur = s => {if(!s)return "—";s=Math.round(s);const m=Math.floor(s/60);return m?`${m}m${String(s%60).padStart(2,"0")}s`:`${s}s`;};
const fmtRel = s => {if(s==null)return "";if(s<0)s=0;const m=Math.floor(s/60);return m?`+${m}m${String(Math.round(s%60)).padStart(2,"0")}s`:`+${Math.round(s)}s`;};
const fmtTok = n => n?(n>=1000?(n/1000).toFixed(n>=100000?0:1)+"k":""+n):"0";
const fmtCost = c => (c==null)?"—":("$"+Number(c).toFixed(3));
const agentColor = aid => {const n=parseInt((aid||"").replace(/\D/g,""))||1;return ["--a1","--a2","--a3","--a4"][(n-1)%4];};
const fmtClock = s => {s=Math.max(0,s||0);const m=Math.floor(s/60);return m?`${m}m${String(Math.round(s%60)).padStart(2,"0")}s`:`${Math.round(s)}s`;};

const FEATS = ["task_list","scratchpad","mcp","auto_refresh","protocol"];
const FEAT_COLOR = {task_list:"#0ea5e9",scratchpad:"#d97706",mcp:"#8b5cf6",auto_refresh:"#16a34a",protocol:"#db2777"};

// task_list + protocol lanes come from the embedded timeline; the other three
// from feat_events (scanned from trajectories).
function buildLanes(p){
  const lanes={task_list:[],scratchpad:[],mcp:[],auto_refresh:[],protocol:[]};
  (p.timeline||[]).forEach(e=>{
    if(e.rel==null) return;
    const feat=(e.kind==="request"||e.kind==="respond"||e.kind==="message")?"protocol":"task_list";
    lanes[feat].push({t:e.rel,by:e.by,label:(e.status||e.kind),detail:(e.title||e.note||"")});
  });
  (p.feat_events||[]).forEach(e=>{ if(e.t!=null && lanes[e.feat]) lanes[e.feat].push({t:e.t,by:e.by,label:e.label,detail:""}); });
  FEATS.forEach(f=>lanes[f].sort((a,b)=>a.t-b.t));
  return lanes;
}

let replayRAF=null;
function stopReplay(){ if(replayRAF){cancelAnimationFrame(replayRAF);replayRAF=null;} }

function buildReplay(p, host){
  const lanes=buildLanes(p);
  let maxT=p.duration||0;
  FEATS.forEach(f=>lanes[f].forEach(e=>{ if(e.t>maxT)maxT=e.t; }));
  const total=FEATS.reduce((n,f)=>n+lanes[f].length,0);
  if(!total && !maxT){ host.appendChild(el("div","muted","No timed activity to replay for this pair.")); return; }
  maxT=Math.max(maxT,1);
  const all=[]; FEATS.forEach(f=>lanes[f].forEach(e=>all.push({...e,feat:f}))); all.sort((a,b)=>a.t-b.t);

  const sec=el("div","replay");
  const ctl=el("div","rctl");
  const playBtn=el("button","play","▶ play");
  const range=el("input"); range.type="range"; range.min="0"; range.max=String(maxT); range.step=String(Math.max(0.1,maxT/1000)); range.value="0";
  const tlab=el("span","rtime");
  const spd=el("div","spd");
  ctl.append(playBtn,range,tlab,spd); sec.appendChild(ctl);

  const dotEls={}; const playheads=[];
  FEATS.forEach(f=>{
    const lane=el("div","lane");
    const tf=p.team_features||{}; const dis=(f in tf)&&!tf[f];
    const lbl=el("div","lbl"+(dis?" dis":""));
    lbl.innerHTML=`<span class="sw" style="background:${FEAT_COLOR[f]}"></span>${f}`+
      `<span class="muted" style="font-weight:400">${dis?"off":lanes[f].length}</span>`;
    const track=el("div","track");
    const ph=el("div","playhead"); track.appendChild(ph); playheads.push(ph);
    dotEls[f]=[];
    lanes[f].forEach(e=>{
      const d=el("div","dot"); d.style.left=Math.min(100,e.t/maxT*100)+"%"; d.style.background=FEAT_COLOR[f];
      d.title=`+${fmtClock(e.t)} ${e.by||""}: ${e.label||""}`;
      track.appendChild(d); dotEls[f].push({el:d,t:e.t});
    });
    lane.append(lbl,track); sec.appendChild(lane);
  });
  if(p.ts_approx) sec.appendChild(el("div","rnote","⚠ scratchpad / mcp / auto_refresh times are approximate — this run's trajectory has no per-step timestamps, so they're placed by step order."));
  const feed=el("div","rfeed"); sec.appendChild(feed);
  host.appendChild(sec);

  const SPEEDS=[1,5,20,60];
  let speed=SPEEDS.reduce((b,s)=>Math.abs(s-maxT/12)<Math.abs(b-maxT/12)?s:b,20);
  SPEEDS.forEach(s=>{ const b=el("button",s===speed?"on":null,s+"×"); b.onclick=()=>{speed=s;[...spd.children].forEach(c=>c.classList.toggle("on",+c.textContent.replace("×","")===s));setUI();}; spd.appendChild(b); });

  let curT=0, playing=false, last=0;
  function setUI(){
    const pct=Math.min(100,curT/maxT*100);
    playheads.forEach(ph=>ph.style.left=pct+"%");
    range.value=String(curT);
    tlab.textContent=`+${fmtClock(curT)} / ${fmtClock(maxT)}`;
    FEATS.forEach(f=>dotEls[f].forEach(o=>o.el.classList.toggle("past",o.t<=curT)));
    const seen=all.filter(e=>e.t<=curT).slice(-7).reverse();
    feed.innerHTML="";
    if(!seen.length) feed.appendChild(el("div","muted","Press play or drag the slider to replay feature usage over the run."));
    seen.forEach(e=>{
      const row=el("div","row");
      row.innerHTML=`<span class="tt">+${fmtClock(e.t)}</span>`+
        `<span class="ft" style="background:${FEAT_COLOR[e.feat]}">${e.feat}</span>`+
        `<span><b style="color:var(${agentColor(e.by)})">${esc(e.by||"")}</b> ${esc(e.label||"")}`+
        `${e.detail?` — <span class="muted">${esc(e.detail)}</span>`:""}</span>`;
      feed.appendChild(row);
    });
  }
  function frame(ts){
    if(!playing) return;
    if(!last) last=ts;
    curT+=(ts-last)/1000*speed; last=ts;
    if(curT>=maxT){ curT=maxT; playing=false; playBtn.textContent="↻ replay"; }
    setUI();
    if(playing) replayRAF=requestAnimationFrame(frame);
  }
  playBtn.onclick=()=>{
    if(playing){ playing=false; playBtn.textContent="▶ play"; stopReplay(); return; }
    if(curT>=maxT) curT=0;
    playing=true; last=0; playBtn.textContent="⏸ pause"; replayRAF=requestAnimationFrame(frame);
  };
  range.oninput=()=>{ playing=false; stopReplay(); playBtn.textContent="▶ play"; curT=+range.value; setUI(); };
  setUI();
}

let FILTER = "all";          // status filter
let QUERY = "";

function runSummaries(){
  const m = new Map();
  DATA.forEach((p,i)=>{
    if(!m.has(p.run)) m.set(p.run,{run:p.run,framework:p.framework,model:p.model,idx:[],pass:0});
    const r=m.get(p.run); r.idx.push(i); if(p.status==="pass") r.pass++;
  });
  return [...m.values()];
}

function pairVisible(p){
  if(FILTER!=="all" && p.status!==FILTER) return false;
  if(QUERY){
    const hay = `${p.run} ${p.repo} ${p.task_id} ${(p.features||[]).join("_")} ${p.framework}`.toLowerCase();
    if(!hay.includes(QUERY)) return false;
  }
  return true;
}

function renderSide(){
  const wrap = $("#runs"); wrap.innerHTML="";
  let shown=0;
  runSummaries().forEach(r=>{
    const vis = r.idx.filter(i=>pairVisible(DATA[i]));
    if(!vis.length) return;
    shown+=vis.length;
    const d = el("details","run"); d.open = !!QUERY || FILTER!=="all";
    const sm = el("summary");
    sm.appendChild(el("span","",esc(r.run)));
    sm.appendChild(el("span","rate",`${r.pass}/${r.idx.length}`));
    d.appendChild(sm);
    vis.forEach(i=>{
      const p = DATA[i];
      const row = el("div","pair"); row.dataset.i=i;
      row.appendChild(el("span","dot "+p.status));
      row.appendChild(el("span","",`${esc(p.repo||"?").replace(/_task$/,"")} <span class="meta">#${esc(p.task_id)} · f${(p.features||[]).join("·f")}</span>`));
      row.onclick=()=>select(i);
      d.appendChild(row);
    });
    wrap.appendChild(d);
  });
  $("#count").textContent = `${shown} pair${shown!==1?"s":""} shown`;
  if(!shown) wrap.appendChild(el("div","empty","No pairs match."));
}

function kv(obj){
  const g = el("div","kv");
  for(const [k,v] of obj){ g.appendChild(el("span","k",esc(k))); g.appendChild(el("span","v",v)); }
  return g;
}

function agentCard(aid,a){
  const c = el("div","card");
  c.style.borderTopColor = `var(${agentColor(aid)})`; c.style.borderTopWidth="3px";
  const top = el("div","top");
  top.appendChild(el("span","aid",esc(aid)));
  if(a.role) top.appendChild(el("span","role "+a.role,esc(a.role)));
  top.appendChild(el("span","muted",`· feature ${esc(a.feature_id)}`));
  c.appendChild(top);
  const rows=[["status",esc(a.status||"—")],["steps",esc(a.steps??"—")],
    ["cost",fmtCost(a.cost)],["tokens",`${fmtTok(a.input_tokens)} in / ${fmtTok(a.output_tokens)} out`],
    ["patch lines",esc(a.patch_lines??"—")]];
  c.appendChild(kv(rows));
  if(a.error) c.appendChild(el("div","err",esc(a.error)));
  return c;
}

function timelineEl(tl){
  const box = el("div","tl");
  if(!tl||!tl.length){ box.appendChild(el("div","muted","No task-list activity recorded.")); return box; }
  tl.forEach(e=>{
    const ev = el("div","ev");
    const dotColor = agentColor(e.by);
    ev.style.setProperty("--m",`var(${dotColor})`);
    let h = "";
    if(e.rel!=null) h += `<span class="when">${fmtRel(e.rel)}</span>`;
    h += `<span class="who" style="color:var(${dotColor})">${esc(e.by)}</span>`;
    h += `<span class="kind kind-${esc(e.kind)}">${esc(e.status||e.kind)}</span>`;
    if(e.kind==="message" && e.to) h += `<span class="muted">→ ${esc(e.to)}</span> `;
    if(e.task) h += `<span class="tid">${esc(e.task)}</span>`;
    ev.innerHTML = h;
    const note = e.title || e.note;
    if(note) ev.appendChild(el("span","note",esc(note)));
    box.appendChild(ev);
  });
  return box;
}

function taskListEl(p){
  const box=el("div","tasklist");
  const tasks=p.tasks||[];
  if(!tasks.length){ box.appendChild(el("div","muted","No tasks recorded.")); return box; }
  const byTask={};
  (p.timeline||[]).forEach(e=>{
    if(!e.task || e.kind==="message" || e.kind==="request" || e.kind==="respond") return;
    (byTask[e.task]||(byTask[e.task]=[])).push(e);
  });
  const firstT=t=>{ const ev=byTask[t.id]; return (ev&&ev.length&&ev[0].rel!=null)?ev[0].rel:1e9; };
  [...tasks].sort((a,b)=>firstT(a)-firstT(b)).forEach(t=>{
    const card=el("div","tk");
    const head=el("div","head");
    let h=`<span class="st ${esc(t.status)}">${esc((t.status||"").replace("_"," "))}</span>`+
      `<span class="tt">${esc(t.title||"(untitled)")}</span>`;
    if(t.owner) h+=`<span class="meta">owner <b style="color:var(${agentColor(t.owner)})">${esc(t.owner)}</b></span>`;
    h+=`<span class="meta tid">${esc(t.id)}</span>`;
    head.innerHTML=h; card.appendChild(head);
    const evs=byTask[t.id]||[];
    if(evs.length){
      const life=el("div","life");
      evs.forEach(e=>{
        const l=el("div","l");
        l.innerHTML=`<span class="w">${e.rel!=null?fmtRel(e.rel):""}</span>`+
          `<span class="vb" style="background:var(${agentColor(e.by)})">${esc(e.status||e.kind)}</span>`+
          `<span><b style="color:var(${agentColor(e.by)})">${esc(e.by)}</b>`+
          `${e.note?` <span class="nt">${esc(e.note)}</span>`:""}</span>`;
        life.appendChild(l);
      });
      card.appendChild(life);
    }
    box.appendChild(card);
  });
  return box;
}

function scratchpadEl(p){
  const box=el("div","sp");
  box.appendChild(el("div","path","/workspace/shared/"));
  let any=false;
  if(p.plan){
    any=true;
    const d=el("details","spfile box");
    d.appendChild(el("summary",null,`<span class="tag plan">plan</span><span class="fn">PLAN.md</span> <span class="muted">lead decomposition</span>`));
    const b=el("div","boxbody"); b.appendChild(el("pre","plan",esc(p.plan))); d.appendChild(b);
    box.appendChild(d);
  }
  Object.entries(p.patches||{}).forEach(([aid,diff])=>{
    any=true;
    const d=el("details","spfile box");
    const lines=diff.split("\n").length;
    d.appendChild(el("summary",null,`<span class="tag patch">patch</span><span class="fn">${esc(aid)}.patch</span> <span class="muted">${lines} lines · ${esc(aid)}</span>`));
    const b=el("div","boxbody"); b.appendChild(el("pre","diff",highlightDiff(diff))); d.appendChild(b);
    box.appendChild(d);
  });
  if(!any){
    const tf=p.team_features||{};
    box.appendChild(el("div","muted",("scratchpad" in tf && !tf.scratchpad)?"Scratchpad disabled for this run.":"No shared-folder artifacts captured in the log."));
  }
  return box;
}

function protocolEl(p){
  const box=el("div","proto");
  const pr=p.protocol||{messages:[],requests:[]};
  if(!(pr.messages||[]).length && !(pr.requests||[]).length){
    const tf=p.team_features||{};
    box.appendChild(el("div","muted",("protocol" in tf && !tf.protocol)?"Protocol disabled; no inter-agent messages recorded.":"No inter-agent messages recorded."));
    return box;
  }
  if((pr.messages||[]).length){
    const msgs=el("div","msgs");
    pr.messages.forEach(mm=>{
      const m=el("div","pmsg");
      const col=agentColor(mm.from);
      m.style.borderColor=`var(${col})`;
      if(mm.from && p.lead && mm.from!==p.lead) m.style.marginLeft="auto";
      m.innerHTML=`<div class="ph"><b style="color:var(${col})">${esc(mm.from||"?")}</b>`+
        `<span class="ar"> → ${esc(mm.to||"?")}</span>`+
        `<span class="w">${mm.t!=null?fmtRel(mm.t):""}</span></div>`+
        `<div class="body">${esc(mm.content||"")}</div>`;
      msgs.appendChild(m);
    });
    box.appendChild(msgs);
  }
  if((pr.requests||[]).length){
    const lbl=el("div","mline"); lbl.style.margin=".7rem 0 .2rem"; lbl.innerHTML="<b>Typed requests</b> <span class='muted'>(protocol verbs)</span>";
    box.appendChild(lbl);
    const reqs=el("div","preqs");
    pr.requests.forEach(r=>reqs.appendChild(el("span","preq",
      `${esc(r.by||"?")} → ${esc(r.to||"?")}: ${esc(r.kind||"")}${r.t!=null?" · "+fmtRel(r.t):""}`)));
    box.appendChild(reqs);
  }
  return box;
}

function highlightDiff(text){
  return esc(text).split("\n").map(l=>{
    let cls="";
    if(/^\+(?!\+\+)/.test(l))cls="a";else if(/^-(?!--)/.test(l))cls="d";
    else if(/^@@/.test(l))cls="h";else if(/^(diff |index |\+\+\+|---)/.test(l))cls="m";
    return cls?`<span class="${cls}">${l}</span>`:l;
  }).join("\n");
}

function trajBox(aid,t){
  const det=el("details","box");
  const role = (currentPair.agents[aid]||{}).role||"";
  let label = `${aid}${role?" ("+role+")":""} — trajectory`;
  if(t.source==="full_traj") label+=` · ${t.api_calls??"?"} API calls · ${fmtCost(t.cost)}`;
  else if(t.source==="traj") label+=" · summary";
  else label+=" · none";
  det.appendChild(el("summary",null,esc(label)));
  const body=el("div","boxbody");
  if(!t.messages||!t.messages.length){ body.appendChild(el("div","muted","No trajectory messages captured (CLI adapters may only store a raw stream).")); }
  else {
    t.messages.forEach(m=>{
      const mm=el("div","msg "+(m.role||""));
      let head=`<span class="role">${esc(m.role)}</span>`;
      if(m.cost!=null) head+=` <span class="muted" style="font-size:.72rem">${fmtCost(m.cost)}</span>`;
      mm.innerHTML=head;
      mm.appendChild(el("pre",null,esc(m.content||"")));
      body.appendChild(mm);
    });
    if(t.omitted) body.appendChild(el("div","muted",`… ${t.omitted} more messages omitted`));
  }
  det.appendChild(body);
  return det;
}

let currentPair=null;
function select(i){
  stopReplay();
  currentPair = DATA[i];
  document.querySelectorAll(".pair").forEach(r=>r.classList.toggle("sel",+r.dataset.i===i));
  const p = DATA[i];
  const m = $("#main"); m.innerHTML=""; m.scrollTop=0;

  const hdr=el("div","hdr");
  hdr.appendChild(el("h2",null,`${esc((p.repo||"").replace(/_task$/,""))} <span class="muted">#${esc(p.task_id)}</span>`));
  hdr.appendChild(el("span","badge "+p.status,p.status));
  m.appendChild(hdr);
  m.appendChild(el("div","crumbs",
    `run <code>${esc(p.run)}</code> · features <code>${(p.features||[]).join(", ")}</code> · `+
    `<code>${esc(p.framework||"?")}</code> / <code>${esc(p.model||"?")}</code> · ${fmtDur(p.duration)} · lead ${esc(p.lead||"?")}`));

  // agents
  let s=el("section"); s.appendChild(el("h3",null,"Agents"));
  const cards=el("div","cards");
  Object.entries(p.agents).forEach(([aid,a])=>cards.appendChild(agentCard(aid,a)));
  s.appendChild(cards); m.appendChild(s);

  // coordination metrics + features
  s=el("section"); s.appendChild(el("h3",null,"Coordination"));
  const mt=p.metrics||{};
  const claims=Object.entries(mt.claims_per_agent||{}).map(([k,v])=>`${k}:${v}`).join(", ")||"—";
  const upd=Object.entries(mt.updates_per_agent||{}).map(([k,v])=>`${k}:${v}`).join(", ")||"—";
  s.appendChild(el("div","mline",
    `tasks done <b>${mt.tasks_done??"—"}/${mt.tasks_total??"—"}</b> · `+
    `time to first claim <b>${mt.time_to_first_claim_seconds!=null?fmtRel(mt.time_to_first_claim_seconds).slice(1):"—"}</b> · `+
    `unowned at end <b>${mt.unowned_at_end??"—"}</b> · claims <b>${claims}</b> · updates <b>${upd}</b>`));
  const chips=el("div","chips");
  Object.entries(p.team_features||{}).forEach(([k,v])=>chips.appendChild(el("span","chip "+(v?"on":"off"),esc(k))));
  s.appendChild(chips); m.appendChild(s);

  // feature usage replay
  s=el("section"); s.appendChild(el("h3",null,"Feature usage (replay)"));
  buildReplay(p,s); m.appendChild(s);

  // task list (per-task lifecycle)
  s=el("section"); s.appendChild(el("h3",null,"Task list"));
  s.appendChild(taskListEl(p)); m.appendChild(s);

  // scratchpad (/workspace/shared)
  s=el("section"); s.appendChild(el("h3",null,"Scratchpad"));
  s.appendChild(scratchpadEl(p)); m.appendChild(s);

  // protocol (inter-agent messages + typed requests)
  s=el("section"); s.appendChild(el("h3",null,"Protocol"));
  s.appendChild(protocolEl(p)); m.appendChild(s);

  // timeline
  s=el("section"); s.appendChild(el("h3",null,"Coordination timeline"));
  s.appendChild(timelineEl(p.timeline)); m.appendChild(s);

  // eval
  if(p.eval){
    s=el("section"); s.appendChild(el("h3",null,"Evaluation"));
    s.appendChild(evalEl(p)); m.appendChild(s);
  }

  // trajectories (secondary)
  s=el("section"); s.appendChild(el("h3",null,"Agent trajectories"));
  Object.entries(p.trajectories||{}).forEach(([aid,t])=>s.appendChild(trajBox(aid,t)));
  if(!Object.keys(p.trajectories||{}).length)
    s.appendChild(el("div","muted","No trajectories captured."));
  m.appendChild(s);

  location.hash = i;
}

function evalEl(p){
  const e=p.eval; const box=el("div");
  box.appendChild(el("div","mline",
    `both passed <b style="color:${e.both_passed?"var(--pass)":"var(--fail)"}">${e.both_passed?"yes":"no"}</b>`+
    (e.merge?` · merge <b>${esc(e.merge)}</b>`:"")+
    (e.apply_status?` · apply ${esc(Object.entries(e.apply_status).map(([k,v])=>k+":"+v).join(", "))}`:"")));
  (e.features||[]).forEach(f=>{
    const det=el("details","box");
    const ok=f.passed;
    det.appendChild(el("summary",null,
      `feature ${esc(f.feature_id)} — <b style="color:${ok?"var(--pass)":"var(--fail)"}">${ok?"PASS":"FAIL"}</b>`+
      ` <span class="muted">(${f.tests_passed??0} passed / ${f.tests_failed??0} failed)</span>`));
    const body=el("div","boxbody");
    if(f.test_output) body.appendChild(el("pre","testout",esc(f.test_output)));
    else body.appendChild(el("div","muted","No test output."));
    det.appendChild(body); box.appendChild(det);
  });
  if(e.error) box.appendChild(el("div","err",esc(e.error)));
  return box;
}

// filters
function setFilter(f,btn){
  FILTER=f; document.querySelectorAll(".filters button").forEach(b=>b.classList.toggle("on",b===btn));
  renderSide();
}
$("#q").addEventListener("input",e=>{QUERY=e.target.value.trim().toLowerCase();renderSide();});
document.querySelectorAll(".filters button").forEach(b=>b.onclick=()=>setFilter(b.dataset.f,b));

renderSide();
const start = parseInt(location.hash.slice(1));
if(!isNaN(start) && DATA[start]) {
  // open its run group then select
  renderSide(); select(start);
  document.querySelectorAll(".pair").forEach(r=>{ if(+r.dataset.i===start){ const d=r.closest("details"); if(d)d.open=true; r.scrollIntoView({block:"center"}); }});
} else {
  $("#main").innerHTML = '<div class="empty">Select a pair on the left to read its coordination story.</div>';
}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", nargs="+", metavar="SUBSTR", help="only include runs whose name contains any of these substrings"
    )
    ap.add_argument(
        "--study",
        action="store_true",
        help=f"curated study set only (runs matching {', '.join(STUDY_PREFIXES)})",
    )
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    patterns = list(STUDY_PREFIXES) if args.study else args.runs
    pairs = collect(patterns)
    if not pairs:
        raise SystemExit("no team pairs found under logs/")

    n_runs = len({p["run"] for p in pairs})
    counts: dict[str, int] = {}
    for p in pairs:
        counts[p["status"]] = counts.get(p["status"], 0) + 1

    data_json = json.dumps(pairs, ensure_ascii=False).replace("</", "<\\/")
    gen = date.today().isoformat()
    sub = f"{len(pairs)} pairs · {n_runs} runs · generated {gen}<br>" + " · ".join(
        f"{k} {v}" for k, v in sorted(counts.items())
    )

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CooperBench — Team Trajectory Viewer</title>
<style>{STYLE}</style></head><body>
<div id="app">
 <aside id="side">
  <div class="shead">
   <h1>Team Trajectory Viewer</h1>
   <div class="sub">{sub}</div>
   <input id="q" placeholder="filter: repo / run / task / feature…" autocomplete="off">
   <div class="filters">
    <button class="on" data-f="all">all</button>
    <button data-f="pass">pass</button>
    <button data-f="partial">partial</button>
    <button data-f="fail">fail</button>
    <button data-f="error">error</button>
   </div>
   <div class="sub" id="count" style="margin-top:.4rem"></div>
  </div>
  <div id="runs"></div>
 </aside>
 <main id="main"></main>
</div>
<script>const DATA={data_json};</script>
<script>{APP_JS}</script>
</body></html>
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    mb = len(html) / 1e6
    print(f"wrote {args.out} ({mb:.2f} MB, {len(pairs)} pairs, {n_runs} runs)")
    for k, v in sorted(counts.items()):
        print(f"  {k:8} {v}")


if __name__ == "__main__":
    main()
