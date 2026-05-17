#!/usr/bin/env python3
"""In-container CLI for the team task list.

Wraps ``TaskListClient`` so agents inside the container can interact
with the shared list via ``coop-task-*`` shell commands.  Reads four
env vars set by the adapter:

    CB_TEAM_REDIS_URL   redis://host[:port][/db][#run:<id>]
    CB_TEAM_RUN_ID      bench run id (Redis namespace)
    CB_TEAM_AGENT_ID    this agent's id
    CB_TEAM_AGENTS      comma-separated list of agent ids in the team
    CB_TEAM_ROLE        "lead" | "member" (optional; informational only)

Subcommands map 1:1 to ``TaskListClient`` methods.  Output is always
JSON or a bare id/integer so agents can parse it with their tools.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import redis

# When this script runs in the container, the in-tree _team package
# isn't installed.  Inline a minimal copy of the bits we need rather
# than pulling cooperbench as a dep into the container.
# (We mirror the public API of TaskListClient so the host-side tests
# exercise the same logic.)

VALID_STATUSES = frozenset({"open", "in_progress", "blocked", "done"})


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _client_and_ns():
    url = os.environ["CB_TEAM_REDIS_URL"]
    # Strip any ``#run:<id>`` fragment — our namespace comes from
    # CB_TEAM_RUN_ID explicitly so we don't depend on URL conventions.
    if "#" in url:
        url, _ = url.split("#", 1)
    return redis.from_url(url), f"cb:{os.environ['CB_TEAM_RUN_ID']}"


def _agent_id() -> str:
    return os.environ["CB_TEAM_AGENT_ID"]


def _log(client, ns: str, **event) -> None:
    import time

    event["ts"] = time.time()
    client.rpush(f"{ns}:task-log", json.dumps(event))


def cmd_create(args: argparse.Namespace) -> int:
    client, ns = _client_and_ns()
    import time
    import uuid

    task_id = uuid.uuid4().hex[:10]
    fields = {
        "id": task_id,
        "title": args.title,
        "owner": args.assign or "",
        "status": "open",
        "created_by": _agent_id(),
        "created_at": str(time.time()),
        "last_note": "",
        "metadata": "{}",
    }
    client.hset(f"{ns}:task:{task_id}", mapping=fields)
    client.sadd(f"{ns}:tasks:all", task_id)
    _log(client, ns, kind="create", task_id=task_id, by=_agent_id(), title=args.title)
    print(task_id)
    return 0


def cmd_claim(args: argparse.Namespace) -> int:
    client, ns = _client_and_ns()
    key = f"{ns}:task:{args.task_id}"
    existing = _decode(client.hget(key, "owner")) or ""
    if existing and existing != _agent_id():
        # Already owned by someone else — exit 2 so the caller can
        # tell "lost the race" from "real error".
        print(f"task {args.task_id} owned by {existing}", file=sys.stderr)
        return 2
    if not client.exists(key):
        print(f"task {args.task_id} does not exist", file=sys.stderr)
        return 1
    client.hset(key, mapping={"owner": _agent_id(), "status": "in_progress"})
    _log(client, ns, kind="claim", task_id=args.task_id, by=_agent_id())
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    client, ns = _client_and_ns()
    key = f"{ns}:task:{args.task_id}"
    owner = _decode(client.hget(key, "owner"))
    if owner is None:
        print(f"task {args.task_id} does not exist", file=sys.stderr)
        return 1
    if owner != _agent_id():
        print(f"task {args.task_id} owned by {owner!r}, not you", file=sys.stderr)
        return 3
    if args.status not in VALID_STATUSES:
        print(f"invalid status {args.status!r}; expected one of {sorted(VALID_STATUSES)}", file=sys.stderr)
        return 4
    updates = {"status": args.status}
    if args.note is not None:
        updates["last_note"] = args.note
    client.hset(key, mapping=updates)
    log_event = {"kind": "update", "task_id": args.task_id, "by": _agent_id(), "status": args.status}
    if args.note is not None:
        log_event["note"] = args.note
    _log(client, ns, **log_event)
    return 0


def _snapshot_to_directory(client, ns: str, target: str) -> None:
    """Mirror the full task list under ``target`` as one JSON file per task.

    Inlined here (rather than importing ``cooperbench.agents._team.fs_mirror``)
    because the in-container helper script must be self-contained — the
    cooperbench package isn't installed in the agent image.  The on-disk
    layout matches what ``fs_mirror.mirror_to_directory`` produces.
    """
    import os as _os
    import tempfile as _tempfile

    _os.makedirs(target, exist_ok=True)
    all_ids = sorted(_decode(m) for m in client.smembers(f"{ns}:tasks:all"))
    live = set(all_ids)

    def _atomic_write(path: str, content: str) -> None:
        dirname = _os.path.dirname(path) or "."
        _os.makedirs(dirname, exist_ok=True)
        tmp = _tempfile.NamedTemporaryFile(
            mode="w",
            dir=dirname,
            prefix=f".{_os.path.basename(path)}.",
            delete=False,
            encoding="utf-8",
        )
        try:
            tmp.write(content)
            tmp.flush()
            _os.fsync(tmp.fileno())
        finally:
            tmp.close()
        _os.replace(tmp.name, path)

    for tid in all_ids:
        raw = {_decode(k): _decode(v) for k, v in client.hgetall(f"{ns}:task:{tid}").items()}
        if not raw:
            continue
        try:
            raw["created_at"] = float(raw["created_at"])
        except (KeyError, ValueError):
            pass
        try:
            raw["metadata"] = json.loads(raw.get("metadata", "{}"))
        except (TypeError, json.JSONDecodeError):
            raw["metadata"] = {}
        _atomic_write(_os.path.join(target, f"{tid}.json"), json.dumps(raw, indent=2, default=str))

    # Remove stale per-task files (tasks that have been deleted).
    for fname in _os.listdir(target):
        if not fname.endswith(".json") or fname.startswith("_"):
            continue
        if fname[:-5] not in live:
            try:
                _os.remove(_os.path.join(target, fname))
            except OSError:
                pass

    import time as _time

    _atomic_write(
        _os.path.join(target, "_index.json"),
        json.dumps({"updated_at": _time.time(), "ids": all_ids}, indent=2),
    )

    log_raw = client.lrange(f"{ns}:task-log", 0, -1)
    log_lines = []
    for line in log_raw:
        try:
            log_lines.append(json.loads(_decode(line)))
        except json.JSONDecodeError:
            continue
    log_text = "\n".join(json.dumps(e) for e in log_lines)
    if log_text:
        log_text += "\n"
    _atomic_write(_os.path.join(target, "_log.jsonl"), log_text)


def cmd_request(args: argparse.Namespace) -> int:
    """Send a typed request to ``recipient``; optionally block for a response."""
    client, ns = _client_and_ns()
    import time as _time
    import uuid as _uuid

    body = args.body if args.body is not None else sys.stdin.read()
    rid = _uuid.uuid4().hex[:10]
    fields = {
        "id": rid,
        "from": _agent_id(),
        "to": args.recipient,
        "kind": args.kind,
        "body": body,
        "ts": str(_time.time()),
    }
    client.hset(f"{ns}:requests:open:{rid}", mapping=fields)
    client.sadd(f"{ns}:requests:by_recipient:{args.recipient}", rid)
    _log(client, ns, kind="request", request_id=rid, by=_agent_id(), to=args.recipient, request_kind=args.kind)

    if args.wait is None:
        print(rid)
        return 0

    # Block on response.
    timeout = max(0.05, float(args.wait))
    result = client.blpop([f"{ns}:responses:{rid}"], timeout=timeout)
    if result is None:
        print(f"timed out waiting for response to {rid}", file=sys.stderr)
        return 5
    _key, raw = result
    print(_decode(raw))
    return 0


def cmd_respond(args: argparse.Namespace) -> int:
    """Reply to an open request."""
    client, ns = _client_and_ns()
    key = f"{ns}:requests:open:{args.request_id}"
    if not client.exists(key):
        print(f"unknown request {args.request_id}", file=sys.stderr)
        return 6
    req = {_decode(k): _decode(v) for k, v in client.hgetall(key).items()}
    if req.get("to") != _agent_id():
        print(f"request {args.request_id} not addressed to {_agent_id()}", file=sys.stderr)
        return 7

    import time as _time

    body = args.body if args.body is not None else sys.stdin.read()
    response = {
        "id": args.request_id,
        "from": _agent_id(),
        "to": req.get("from"),
        "body": body,
        "ts": _time.time(),
    }
    client.rpush(f"{ns}:responses:{args.request_id}", json.dumps(response))
    client.delete(key)
    client.srem(f"{ns}:requests:by_recipient:{_agent_id()}", args.request_id)
    _log(client, ns, kind="response", request_id=args.request_id, by=_agent_id(), to=req.get("from"))
    return 0


def cmd_pending(_args: argparse.Namespace) -> int:
    """List open requests addressed to this agent."""
    client, ns = _client_and_ns()
    ids = sorted(_decode(m) for m in client.smembers(f"{ns}:requests:by_recipient:{_agent_id()}"))
    out = []
    for rid in ids:
        raw = {_decode(k): _decode(v) for k, v in client.hgetall(f"{ns}:requests:open:{rid}").items()}
        if not raw:
            continue
        try:
            raw["ts"] = float(raw["ts"])
        except (KeyError, ValueError):
            pass
        out.append(raw)
    print(json.dumps(out, indent=2))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    client, ns = _client_and_ns()
    ids = sorted(_decode(m) for m in client.smembers(f"{ns}:tasks:all"))
    out = []
    me = _agent_id()
    for tid in ids:
        raw = {_decode(k): _decode(v) for k, v in client.hgetall(f"{ns}:task:{tid}").items()}
        if not raw:
            continue
        if args.mine and raw.get("owner") != me:
            continue
        if args.open_only and raw.get("status") != "open":
            continue
        # Coerce known fields.
        try:
            raw["created_at"] = float(raw["created_at"])
        except (KeyError, ValueError):
            pass
        try:
            raw["metadata"] = json.loads(raw.get("metadata", "{}"))
        except (TypeError, json.JSONDecodeError):
            raw["metadata"] = {}
        out.append(raw)
    print(json.dumps(out, indent=2))

    # Side effect: snapshot the *full* list to the scratchpad so agents
    # can also browse it via ``ls /workspace/shared/tasks/`` without
    # needing this CLI.  Triggered on every list call (cheap; no inotify
    # required).
    mirror_target = os.environ.get("CB_TEAM_TASKS_DIR")
    if mirror_target:
        try:
            _snapshot_to_directory(client, ns, mirror_target)
        except OSError:
            pass  # mirror is best-effort
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coop-task")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create")
    p_create.add_argument("title")
    p_create.add_argument("--assign", default=None, help="pre-assign owner (still requires claim)")
    p_create.set_defaults(func=cmd_create)

    p_claim = sub.add_parser("claim")
    p_claim.add_argument("task_id")
    p_claim.set_defaults(func=cmd_claim)

    p_update = sub.add_parser("update")
    p_update.add_argument("task_id")
    p_update.add_argument("status")
    p_update.add_argument("-n", "--note", default=None)
    p_update.set_defaults(func=cmd_update)

    p_list = sub.add_parser("list")
    p_list.add_argument("--mine", action="store_true")
    p_list.add_argument("--open", dest="open_only", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_req = sub.add_parser("request")
    p_req.add_argument("recipient")
    p_req.add_argument("kind")
    p_req.add_argument("body", nargs="?", default=None)
    p_req.add_argument(
        "--wait",
        type=float,
        default=None,
        help="block up to N seconds for a response; if omitted, returns request_id and exits",
    )
    p_req.set_defaults(func=cmd_request)

    p_resp = sub.add_parser("respond")
    p_resp.add_argument("request_id")
    p_resp.add_argument("body", nargs="?", default=None)
    p_resp.set_defaults(func=cmd_respond)

    p_pending = sub.add_parser("pending")
    p_pending.set_defaults(func=cmd_pending)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
