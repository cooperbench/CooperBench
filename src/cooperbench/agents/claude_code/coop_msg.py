#!/usr/bin/env python3
"""Tiny messaging CLI for Claude Code to use inside the agent container.

Mirrors the semantics of CooperBench's host-side ``MessagingConnector``
(Redis lists, one inbox per agent, optional ``#run:<id>`` namespace
prefix) but lives in the container so the in-process LLM can invoke it
via Bash.

Usage:
    coop-send <recipient> <content>      # send to one agent
    coop-broadcast <content>             # send to every other agent
    coop-recv                            # drain this agent's inbox (JSON list)
    coop-peek                            # count unread messages
    coop-agents                          # list all agent ids

Config is read from environment variables (set by the adapter):
    COOP_REDIS_URL   redis://host[:port][/db][#run:<id>]
    COOP_AGENT_ID    this agent's id (e.g. "agent1")
    COOP_AGENTS      comma-separated list (e.g. "agent1,agent2")
    COOP_LOG_PATH    optional; if set, every successful send is appended
                     to this file as one JSON line for the host adapter
                     to harvest after the run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

import redis


def _client_and_prefix() -> tuple[redis.Redis, str]:
    url = os.environ["COOP_REDIS_URL"]
    if "#" in url:
        url, prefix = url.split("#", 1)
        prefix = prefix + ":"
    else:
        prefix = ""
    return redis.from_url(url), prefix


def _agent_id() -> str:
    return os.environ["COOP_AGENT_ID"]


def _agents() -> list[str]:
    raw = os.environ.get("COOP_AGENTS", "")
    return [a.strip() for a in raw.split(",") if a.strip()]


def _log_send(entry: dict[str, Any]) -> None:
    path = os.environ.get("COOP_LOG_PATH")
    if not path:
        return
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        # Logging failure shouldn't interrupt the send.
        pass


def _send(client: redis.Redis, prefix: str, recipient: str, content: str) -> None:
    entry = {
        "from": _agent_id(),
        "to": recipient,
        "content": content,
        "timestamp": time.time(),
        "timestamp_iso": datetime.now().isoformat(),
    }
    client.rpush(f"{prefix}{recipient}:inbox", json.dumps(entry))
    _log_send(entry)


def cmd_send(args: argparse.Namespace) -> int:
    client, prefix = _client_and_prefix()
    content = args.content if args.content is not None else sys.stdin.read()
    _send(client, prefix, args.recipient, content)
    print(f"sent to {args.recipient}", file=sys.stderr)
    return 0


def cmd_broadcast(args: argparse.Namespace) -> int:
    client, prefix = _client_and_prefix()
    content = args.content if args.content is not None else sys.stdin.read()
    me = _agent_id()
    for agent in _agents():
        if agent == me:
            continue
        _send(client, prefix, agent, content)
    return 0


def cmd_recv(_args: argparse.Namespace) -> int:
    client, prefix = _client_and_prefix()
    key = f"{prefix}{_agent_id()}:inbox"
    messages = []
    while True:
        raw = client.lpop(key)
        if raw is None:
            break
        try:
            messages.append(json.loads(raw))
        except json.JSONDecodeError:
            messages.append({"content": raw.decode() if isinstance(raw, bytes) else raw})
    print(json.dumps(messages, indent=2))
    return 0


def cmd_peek(_args: argparse.Namespace) -> int:
    client, prefix = _client_and_prefix()
    print(client.llen(f"{prefix}{_agent_id()}:inbox"))
    return 0


def cmd_agents(_args: argparse.Namespace) -> int:
    for agent in _agents():
        print(agent)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coop-msg")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_send = sub.add_parser("send")
    p_send.add_argument("recipient")
    p_send.add_argument("content", nargs="?", default=None)
    p_send.set_defaults(func=cmd_send)

    p_bcast = sub.add_parser("broadcast")
    p_bcast.add_argument("content", nargs="?", default=None)
    p_bcast.set_defaults(func=cmd_broadcast)

    sub.add_parser("recv").set_defaults(func=cmd_recv)
    sub.add_parser("peek").set_defaults(func=cmd_peek)
    sub.add_parser("agents").set_defaults(func=cmd_agents)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
