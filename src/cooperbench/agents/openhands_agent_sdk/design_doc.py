#!/usr/bin/env python3
"""In-sandbox CLI for the coop shared **design document** (Redis-backed).

openhands runs each agent in its own network-isolated Modal sandbox, so a
docker shared volume (what coop mode uses for ``mini_swe_agent_v2``) isn't
available.  Instead we back the shared design doc with the same Redis the
agents already use for messaging — a real concurrent store, so two agents
writing at once don't clobber each other (writes are atomic appends).

The doc is therefore an append-structured shared log: each ``design-note``
adds an attributed block that BOTH agents can read via ``design-show``.
Two commands are installed in the sandbox:

    design-show           # print the current shared design doc
    design-note <<'EOF'   # append an attributed block (reads stdin)
    ...
    EOF

Env (set by the adapter via a Modal secret):
    REDIS_URL      shared Redis URL (may carry a ``#run:<id>`` fragment)
    CB_DESIGN_KEY  Redis key holding this run's design doc
    AGENT_ID       this agent's id (used to attribute notes)
"""

from __future__ import annotations

import os
import sys
import time

SKELETON = (
    "# Shared Design Document\n\n"
    "This document is shared between both engineers on this codebase. Use it to\n"
    "agree on the design as you build: shared interfaces / function signatures,\n"
    "which files & symbols each of you owns, data formats passed between your\n"
    "features, and decisions that affect how your two patches will merge.\n"
    "It is NOT a scratchpad for throwaway notes. Each `design-note` you add is\n"
    "appended below and is visible to your colleague via `design-show`.\n"
)


def _client():
    url = os.environ.get("REDIS_URL")
    if not url:
        sys.stderr.write("shared design doc unavailable: REDIS_URL not set\n")
        sys.exit(1)
    import redis  # noqa: PLC0415 -- only needed when the CLI actually runs

    # The messaging layer namespaces the URL with a ``#run:<id>`` fragment
    # that redis.from_url() can't parse — strip it (the run is isolated by
    # CB_DESIGN_KEY instead).
    return redis.from_url(url.split("#run:")[0])


def main() -> int:
    key = os.environ.get("CB_DESIGN_KEY", "cb:design:default")
    agent = os.environ.get("AGENT_ID", "agent")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    r = _client()

    if cmd == "show":
        # Lazily seed the skeleton exactly once (SETNX → no clobber if a
        # peer already wrote).
        r.setnx(key, SKELETON)
        val = r.get(key)
        text = val.decode() if isinstance(val, (bytes, bytearray)) else str(val or "")
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    if cmd == "note":
        body = sys.stdin.read().strip()
        if not body:
            sys.stderr.write("design-note: nothing on stdin to append\n")
            return 2
        r.setnx(key, SKELETON)
        block = f"\n\n---\n### [{agent}] {time.strftime('%Y-%m-%d %H:%M:%S')}\n{body}\n"
        r.append(key, block)
        sys.stdout.write(f"appended {len(body)} chars to the shared design doc (visible to your colleague)\n")
        return 0

    sys.stderr.write("usage: design-show | design-note (reads stdin)\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
