"""Multi-agent **team harness** — the coordination primitives that
back CooperBench's ``--setting team`` mode, packaged as a standalone
library so other benchmarks (e.g. long-horizon harnesses) can consume
it without depending on CooperBench-specific assumptions.

What's in the harness
---------------------

Five independently-toggleable coordination mechanisms, designed so an
ablation study can switch each one off and measure its contribution:

============   ==================================================
``task_list``  Redis-backed shared task list with atomic claim,
               owner-only updates, and a full audit log.
               ``TaskListClient`` is the host- and in-container API.

``scratchpad`` ``/workspace/shared`` Docker volume mounted in every
               agent container, used for partial diffs, design notes,
               and interface contracts.

``mcp``        Stdio MCP server exposing ``wait_for_message`` so CLI
               agents can long-poll the inbox without busy-looping.
               (``mcp_server.py``.)

``auto_refresh`` ``TeamPoller`` for Python-loop adapters — prepends a
               compact task-list summary as a user-role message before
               every LLM call.  (``loop_refresh.py``.)

``protocol``   Typed request/response shell verbs (``coop-request`` /
               ``coop-respond`` / ``coop-pending``) layered on the
               messaging transport.  (``protocol.py``.)
============   ==================================================

The lead/member **role split** is the always-on baseline.  Without it,
team mode collapses into coop (N peer agents, no organizer), so we
treat it as the defining property of the harness rather than a toggle.

Public API
----------

- ``TeamHarnessConfig`` — dataclass of per-feature booleans
- ``TeamSession``       — per-run object that bundles config + state
                          and exposes adapter-facing factories
                          (env vars, mount args, MCP config, etc.)

The lower-level primitives (``TaskListClient``, ``ProtocolClient``,
``TeamPoller``, ``build_team_instruction``, ...) remain importable for
callers that want to assemble their own session shape — e.g. a
long-horizon benchmark that drives task creation dynamically rather
than pre-seeding one task per feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cooperbench.team_harness.fs_mirror import mirror_to_directory
from cooperbench.team_harness.loop_refresh import TeamPoller, format_task_summary, poll_team_state
from cooperbench.team_harness.metrics import compute_metrics
from cooperbench.team_harness.prompt import build_team_instruction, team_task_section
from cooperbench.team_harness.protocol import ProtocolClient
from cooperbench.team_harness.runtime import (
    CONTAINER_SCRATCHPAD_DIR,
    CONTAINER_TASKS_MIRROR_DIR,
    build_team_env,
    scratchpad_mount_args,
)
from cooperbench.team_harness.task_list import TaskListClient

_PACKAGE_DIR = Path(__file__).resolve().parent

COOP_TASK_SCRIPT_PATH: Path = _PACKAGE_DIR / "coop_task.py"
"""In-container CLI helper that backs ``coop-task-*``, ``coop-request``,
``coop-respond``, and ``coop-pending``.  Adapters copy this into the
container at setup time."""

INSTALL_SNIPPET_PATH: Path = _PACKAGE_DIR / "install_snippet.sh"
"""Shell snippet that creates ``/usr/local/bin/coop-task-*`` wrappers
around ``coop_task.py``.  Sourced by each adapter's container setup."""

MCP_SERVER_SCRIPT_PATH: Path = _PACKAGE_DIR / "mcp_server.py"
"""Stdio MCP server exposing ``wait_for_message``.  CLI adapters copy
this into the container and register it in their CLI's MCP config."""

MCP_SERVER_NAME = "cooperbench-team"
"""Name under which the MCP server is registered in CLI configs."""


@dataclass(frozen=True)
class TeamHarnessConfig:
    """Per-feature on/off switches for ablation studies.

    Each flag gates one of the harness's five coordination mechanisms.
    Defaults are all ``True`` — the full harness.  Flip individual
    flags to measure the marginal contribution of that feature.

    The lead/member role split is *not* a flag: without it team mode
    is just coop mode, so we treat it as the defining property rather
    than a toggle.
    """

    task_list: bool = True
    scratchpad: bool = True
    mcp: bool = True
    auto_refresh: bool = True
    protocol: bool = True

    @staticmethod
    def with_only(*features: str) -> TeamHarnessConfig:
        """Return a config with only the named features enabled.

        Convenience for ablation runs — ``TeamHarnessConfig.with_only("task_list")``
        gives you a single-feature harness without enumerating four False flags.
        """
        valid = {"task_list", "scratchpad", "mcp", "auto_refresh", "protocol"}
        bad = set(features) - valid
        if bad:
            raise ValueError(f"unknown feature(s) {sorted(bad)}; expected subset of {sorted(valid)}")
        return TeamHarnessConfig(**{f: (f in features) for f in valid})

    def disabled(self) -> list[str]:
        """Names of the features that are currently off."""
        return [f for f in ("task_list", "scratchpad", "mcp", "auto_refresh", "protocol") if not getattr(self, f)]


@dataclass
class TeamSession:
    """A single team-mode coordination session.

    Bundles per-run state (run id, namespaced Redis URL, ordered agent
    list, scratchpad volume name) with the feature config and exposes
    adapter-facing factories.  Each factory returns ``None`` / ``{}`` /
    ``[]`` when its feature is disabled, so adapters can write a single
    code path:

        env_vars = session.env_for(agent_id)        # {} if all off
        mount_args = session.scratchpad_mount_args()  # [] if off
        mcp_cfg = session.mcp_config(...)            # None if off

    ``redis_url`` is the **host-side** URL (e.g. ``redis://localhost:6379#run:abc``).
    ``env_for()`` automatically rewrites ``localhost`` / ``127.0.0.1`` to
    ``host.docker.internal`` so the in-container CLI can reach back to the
    host's Redis without the adapter having to handle that itself.  Callers
    that already have a container-reachable URL (e.g. openhands_sdk's
    Modal-hosted Redis) get a no-op rewrite.

    The first agent in ``agents`` is the lead; the rest are members.
    """

    run_id: str
    redis_url: str  # host URL, already namespaced with #run:<id>
    agents: list[str]
    team_volume: str
    config: TeamHarnessConfig = field(default_factory=TeamHarnessConfig)

    @staticmethod
    def _rewrite_url_for_container(url: str) -> str:
        """Host→container Redis URL rewrite.

        Duplicated from ``_coop.runtime.rewrite_comm_url_for_container``
        to keep ``team_harness`` free of CooperBench-internal imports —
        the harness is meant to be portable to other benchmarks that
        don't ship the ``_coop`` package.
        """
        if not url:
            return url
        for needle in ("//localhost", "//127.0.0.1"):
            if needle in url:
                return url.replace(needle, "//host.docker.internal", 1)
        return url

    @property
    def lead(self) -> str:
        return self.agents[0]

    def role_for(self, agent_id: str) -> str:
        return "lead" if agent_id == self.lead else "member"

    def is_active(self) -> bool:
        """Team mode is active when there's >=2 agents.  A team of one
        collapses to solo (the prompts and primitives are no-ops)."""
        return len(self.agents) > 1

    # --- task list lifecycle (host-side) --------------------------------

    def task_list_client(self, *, redis_client: object) -> TaskListClient | None:
        """Construct a ``TaskListClient`` bound to this session's run-id.

        Returns ``None`` if the ``task_list`` feature is disabled — the
        caller skips pre-seeding and metric harvesting in that case.
        """
        if not self.config.task_list:
            return None
        return TaskListClient(redis_client=redis_client, run_id=self.run_id)  # type: ignore[arg-type]

    def harvest_metrics(self, client: TaskListClient | None) -> tuple[dict | None, list, list]:
        """Read the audit log and compute coordination metrics.

        Returns ``(metrics, events, final_tasks)``; all are
        empty/``None`` when the task list is disabled or the client
        couldn't be constructed (e.g. Redis dropped during the run).
        """
        if not self.config.task_list or client is None:
            return None, [], []
        events = client.log_events()
        final_tasks = client.list()
        return compute_metrics(events, final_tasks=final_tasks), events, final_tasks

    # --- adapter-facing factories ---------------------------------------

    def env_for(self, agent_id: str) -> dict[str, str]:
        """``CB_TEAM_*`` env vars for the agent's container.

        The in-container ``coop-task-*`` CLI, the MCP server, and the
        ``auto_refresh`` poller all read these.  Returns ``{}`` only
        when every feature that consumes the env is disabled.  The
        Redis URL is host→container rewritten automatically so the
        in-container processes can reach the host's Redis.
        """
        if not (self.config.task_list or self.config.auto_refresh or self.config.protocol or self.config.mcp):
            return {}
        return build_team_env(
            redis_url=self._rewrite_url_for_container(self.redis_url),
            run_id=self.run_id,
            agent_id=agent_id,
            agents=self.agents,
            team_role=self.role_for(agent_id),
        )

    def scratchpad_mount_args(self) -> list[str]:
        """``docker run`` args to mount the shared scratchpad volume.

        Empty list when the scratchpad feature is disabled — the volume
        simply isn't created or mounted.
        """
        if not self.config.scratchpad:
            return []
        return scratchpad_mount_args(self.team_volume)

    def prompt_for(self, *, task: str, agent_id: str, git_enabled: bool = False) -> str:
        """Full team-instruction prompt (used by CLI adapters)."""
        return build_team_instruction(
            task,
            agents=self.agents,
            agent_id=agent_id,
            team_role=self.role_for(agent_id),
            git_enabled=git_enabled,
        )

    def prompt_section(self, *, agent_id: str) -> str:
        """Just the role-specific team block (used by Python-loop
        adapters that already have their own coop prompts)."""
        return team_task_section(
            agents=self.agents,
            agent_id=agent_id,
            team_role=self.role_for(agent_id),
        )

    def loop_poller(self, *, agent_id: str) -> TeamPoller | None:
        """``TeamPoller`` instance for Python-loop adapters to attach
        to their agent.  ``None`` when ``auto_refresh`` is disabled."""
        if not self.config.auto_refresh:
            return None
        return TeamPoller(
            redis_url=self.redis_url,
            run_id=self.run_id,
            agent_id=agent_id,
        )

    def mcp_config(self, *, container_script_path: str) -> dict | None:
        """JSON config to write into ``~/.claude.json`` (Claude Code) or
        derive into ``config.toml`` for Codex.

        ``container_script_path`` is the in-container location the
        adapter has copied ``MCP_SERVER_SCRIPT_PATH`` to.  Returns
        ``None`` when the ``mcp`` feature is disabled.
        """
        if not self.config.mcp:
            return None
        return {
            "mcpServers": {
                MCP_SERVER_NAME: {
                    "type": "stdio",
                    "command": "python3",
                    "args": [container_script_path],
                }
            }
        }


__all__ = [
    "COOP_TASK_SCRIPT_PATH",
    "CONTAINER_SCRATCHPAD_DIR",
    "CONTAINER_TASKS_MIRROR_DIR",
    "INSTALL_SNIPPET_PATH",
    "MCP_SERVER_NAME",
    "MCP_SERVER_SCRIPT_PATH",
    "ProtocolClient",
    "TaskListClient",
    "TeamHarnessConfig",
    "TeamPoller",
    "TeamSession",
    "build_team_env",
    "build_team_instruction",
    "compute_metrics",
    "format_task_summary",
    "mirror_to_directory",
    "poll_team_state",
    "scratchpad_mount_args",
    "team_task_section",
]
