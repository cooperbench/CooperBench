"""Runtime helpers for team mode.

Translates host-side team config into the env vars + docker args every
adapter needs.  All keeping the same shape as ``_coop/runtime.py`` so
the team-mode plumbing is a strict superset.
"""

from __future__ import annotations

CONTAINER_SCRATCHPAD_DIR = "/workspace/shared"
"""Where the team scratchpad volume is mounted inside every container."""

CONTAINER_TASKS_MIRROR_DIR = f"{CONTAINER_SCRATCHPAD_DIR}/tasks"
"""Filesystem mirror of the task list, populated by ``coop-task-list``."""


def build_team_env(
    *,
    redis_url: str,
    run_id: str,
    agent_id: str,
    agents: list[str],
    team_role: str | None,
) -> dict[str, str]:
    """Compose env vars consumed by the in-container ``coop-task-*`` CLI.

    All four are required.  ``team_role`` is omitted when None so the
    in-container CLI can tell "not in team mode" from "no role assigned".
    """
    env: dict[str, str] = {
        "CB_TEAM_REDIS_URL": redis_url,
        "CB_TEAM_RUN_ID": run_id,
        "CB_TEAM_AGENT_ID": agent_id,
        "CB_TEAM_AGENTS": ",".join(agents),
        "CB_TEAM_TASKS_DIR": CONTAINER_TASKS_MIRROR_DIR,
    }
    if team_role:
        env["CB_TEAM_ROLE"] = team_role
    return env


def scratchpad_mount_args(volume_name: str | None) -> list[str]:
    """Build ``docker run`` arguments for the shared scratchpad.

    The runner creates one named volume per run (``cb-team-<run_id>``)
    and passes it to every agent container.  Passing an empty/None
    volume name produces no args (so coop or solo runs are unaffected).
    """
    if not volume_name:
        return []
    return ["--volume", f"{volume_name}:{CONTAINER_SCRATCHPAD_DIR}"]
