"""In-loop task-list refresh helpers for Python-loop adapters.

Python-loop adapters (mini_swe_agent_v2, swe_agent, openhands_sdk) own
their agent loop and already poll the inbox between steps.  When team
mode is active we want the same automatic surfacing for the task list:
just before each LLM call, the adapter calls ``poll_team_state()``,
gets back a short summary string, and prepends it as a user-role
message.  The LLM sees the live state of the shared list without
needing to remember to call ``coop-task-list``.

The helpers are designed to be cheap and failure-tolerant: when
team-mode env vars are unset (solo / coop), or Redis is unreachable,
they return ``None`` and the adapter skips the injection.  Nothing
about this should ever crash an agent loop.
"""

from __future__ import annotations

import os
from typing import Any

import redis


def _client_from_env() -> Any:
    """Build a Redis client from the in-container env vars.

    Module-level seam so tests can monkey-patch in a fakeredis instance.
    Raises any redis error the caller is expected to swallow.
    """
    url = os.environ["CB_TEAM_REDIS_URL"]
    if "#" in url:
        url, _ = url.split("#", 1)
    return redis.from_url(url, socket_timeout=2)


def format_task_summary(tasks: list[dict[str, Any]], *, viewer: str) -> str:
    """Render a compact, LLM-friendly summary of the task list.

    The format is intentionally terse: the LLM will see this before
    every step, so we don't want to burn tokens on a verbose table.
    Status counts come first, then one line per task with the viewer's
    own tasks called out.
    """
    if not tasks:
        return "[Team task list]: no tasks yet."

    by_status: dict[str, int] = {}
    for t in tasks:
        s = t.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    header = ", ".join(f"{k}: {v}" for k, v in sorted(by_status.items()))
    lines = [f"[Team task list] {header}"]
    for t in tasks:
        owner = t.get("owner") or "?"
        marker = " (you own)" if owner == viewer else ""
        lines.append(f"  - {t.get('id', '?')} [{t.get('status', '?')}] owner={owner}{marker}: {t.get('title', '')}")
    return "\n".join(lines)


def _read_tasks(client: Any, run_id: str) -> list[dict[str, Any]]:
    ns = f"cb:{run_id}"
    ids = sorted((m.decode() if isinstance(m, bytes) else m) for m in client.smembers(f"{ns}:tasks:all"))
    tasks: list[dict[str, Any]] = []
    for tid in ids:
        raw = client.hgetall(f"{ns}:task:{tid}")
        if not raw:
            continue
        tasks.append(
            {
                (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
                for k, v in raw.items()
            }
        )
    return tasks


def poll_team_state() -> str | None:
    """Read the current task list and return a summary, or ``None``.

    Environment-driven (in-container) variant.  Returns ``None`` when
    team mode isn't active (env vars missing) or when Redis is
    unreachable — never raises.  The caller is expected to prepend the
    returned string as a user-role message before its next LLM query.
    """
    if not os.environ.get("CB_TEAM_REDIS_URL") or not os.environ.get("CB_TEAM_AGENT_ID"):
        return None
    run_id = os.environ.get("CB_TEAM_RUN_ID", "")
    viewer = os.environ["CB_TEAM_AGENT_ID"]
    try:
        client = _client_from_env()
        return format_task_summary(_read_tasks(client, run_id), viewer=viewer)
    except (redis.exceptions.RedisError, OSError, KeyError):
        return None


class TeamPoller:
    """Host-side per-agent task-list poller for Python-loop adapters.

    Each adapter instance creates one (if team kwargs are set) and
    calls ``poll()`` between steps; failures are swallowed so the
    agent loop never crashes because Redis is down.
    """

    def __init__(self, *, redis_url: str, run_id: str, agent_id: str) -> None:
        self._url = redis_url
        self._run_id = run_id
        self._agent_id = agent_id
        self._client: Any | None = None

    def _ensure_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        try:
            url = self._url
            if "#" in url:
                url, _ = url.split("#", 1)
            self._client = redis.from_url(url, socket_timeout=2)
        except (redis.exceptions.RedisError, OSError):
            return None
        return self._client

    def poll(self) -> str | None:
        """Return a fresh task-list summary, or ``None`` on failure."""
        client = self._ensure_client()
        if client is None:
            return None
        try:
            return format_task_summary(_read_tasks(client, self._run_id), viewer=self._agent_id)
        except (redis.exceptions.RedisError, OSError, KeyError):
            return None
