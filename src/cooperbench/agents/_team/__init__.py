"""Team-mode coordination primitives shared by every CLI / Python-loop adapter.

Team mode (``--setting team``) sits beside solo and coop.  Conceptually,
it gives the agents three things coop doesn't:

  1. A typed, shared **task list** with atomic claim semantics
     (``TaskListClient``).  Tasks have an owner, status, optional
     metadata, and every mutation is logged so the bench can compute
     coordination metrics after the fact.
  2. A designated **lead** role with a different system-prompt block
     instructing them to organize work via the task list before writing
     code.  Other agents are **members** and look for open tasks to
     claim.
  3. A **shared scratchpad** directory backed by a Docker volume
     (``/workspace/shared`` in every container).  Members can drop
     design notes, partial outputs, or interface contracts there as
     concrete coordination artifacts.

The transport for messaging (``coop-send`` / ``coop-recv`` /
``coop-broadcast``) is reused from ``cooperbench.agents._coop`` — team
mode is messaging-plus-task-list, not messaging-replacement.

This module exports the host-side helpers; the in-container
``coop-task-*`` shell tools are wired up by ``install_snippet.sh``.
"""

from cooperbench.agents._team.fs_mirror import mirror_to_directory
from cooperbench.agents._team.loop_refresh import TeamPoller, format_task_summary, poll_team_state
from cooperbench.agents._team.metrics import compute_metrics
from cooperbench.agents._team.prompt import build_team_instruction, team_task_section
from cooperbench.agents._team.protocol import ProtocolClient
from cooperbench.agents._team.runtime import (
    CONTAINER_SCRATCHPAD_DIR,
    CONTAINER_TASKS_MIRROR_DIR,
    build_team_env,
    scratchpad_mount_args,
)
from cooperbench.agents._team.task_list import TaskListClient

__all__ = [
    "CONTAINER_SCRATCHPAD_DIR",
    "CONTAINER_TASKS_MIRROR_DIR",
    "ProtocolClient",
    "TaskListClient",
    "TeamPoller",
    "build_team_env",
    "build_team_instruction",
    "compute_metrics",
    "format_task_summary",
    "mirror_to_directory",
    "poll_team_state",
    "scratchpad_mount_args",
    "team_task_section",
]
