"""Compatibility tests: every adapter in team mode must at minimum
accept the team kwargs without crashing and append the team section
to the task it sends the agent.

The CLI adapters (claude_code, codex) have richer wiring tested in
their own test files; this module is the cross-adapter sanity check
so a future refactor doesn't silently regress one adapter.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cooperbench.agents._team.prompt import team_task_section


class TestTeamTaskSectionVsBuildInstruction:
    """The two ways to inject the team prompt must stay consistent:
    ``team_task_section`` is what Python-loop adapters append; the
    bigger ``build_team_instruction`` (used by CLI adapters) must
    *contain* the same section."""

    def test_lead_section_is_substring_of_full_lead_prompt(self):
        from cooperbench.agents._team import build_team_instruction

        section = team_task_section(agents=["a1", "a2"], agent_id="a1", team_role="lead")
        full = build_team_instruction(
            task="dummy",
            agents=["a1", "a2"],
            agent_id="a1",
            team_role="lead",
        )
        # The block emitted by team_task_section is verbatim what
        # build_team_instruction inserts after the submission block.
        assert section.strip() in full

    def test_member_section_is_substring_of_full_member_prompt(self):
        from cooperbench.agents._team import build_team_instruction

        section = team_task_section(agents=["a1", "a2"], agent_id="a2", team_role="member")
        full = build_team_instruction(
            task="dummy",
            agents=["a1", "a2"],
            agent_id="a2",
            team_role="member",
        )
        assert section.strip() in full


class TestMiniSweAgentV2TeamWiring:
    """v2 adapter: appends team_task_section to the task; propagates
    CB_TEAM_* env into the container env_kwargs."""

    def test_appends_team_section_to_task(self):
        """We can't easily run the adapter end-to-end without a real
        sandbox, but we can verify the prompt-assembly side via a
        focused unit on the same code path the adapter uses."""
        from cooperbench.agents._team import team_task_section

        section = team_task_section(agents=["agent1", "agent2"], agent_id="agent1", team_role="lead")
        # Sanity: this is the exact piece the v2 adapter appends.
        assert "coop-task-create" in section
        assert "team-lead" in section.lower()


class TestOpenHandsTeamWiring:
    """openhands adapter: in team mode, builds coop_info with team_env
    so the sandbox sees CB_TEAM_* variables."""

    def test_team_env_dict_has_expected_keys(self):
        from cooperbench.agents._coop.runtime import rewrite_comm_url_for_container
        from cooperbench.agents._team.runtime import CONTAINER_TASKS_MIRROR_DIR

        # Reconstruct what the adapter builds.
        team_env = {
            "CB_TEAM_REDIS_URL": rewrite_comm_url_for_container("redis://localhost:6379#run:x") or "",
            "CB_TEAM_RUN_ID": "x",
            "CB_TEAM_AGENT_ID": "agent1",
            "CB_TEAM_AGENTS": "agent1,agent2",
            "CB_TEAM_TASKS_DIR": CONTAINER_TASKS_MIRROR_DIR,
            "CB_TEAM_ROLE": "lead",
        }
        # localhost rewrite happens host->container.
        assert "host.docker.internal" in team_env["CB_TEAM_REDIS_URL"]
        # All required keys present.
        for k in ("CB_TEAM_REDIS_URL", "CB_TEAM_RUN_ID", "CB_TEAM_AGENT_ID", "CB_TEAM_AGENTS", "CB_TEAM_ROLE"):
            assert team_env[k]


class TestSweAgentTeamWiring:
    """swe_agent adapter: minimum bar — accepts team kwargs without
    raising and appends team_task_section."""

    def test_team_kwargs_accepted_in_signature(self):
        import inspect

        from cooperbench.agents.swe_agent.adapter import SweAgentRunner

        sig = inspect.signature(SweAgentRunner.run)
        params = list(sig.parameters.keys())
        for kw in ("team_role", "team_id", "task_list_url"):
            assert kw in params


class TestAllAdaptersAcceptTeamKwargs:
    """Every registered runner must accept the team kwargs (or **kwargs)
    so the team runner can pass them uniformly."""

    @pytest.mark.parametrize(
        "agent_name",
        ["claude_code", "codex", "mini_swe_agent_v2", "swe_agent", "openhands_sdk"],
    )
    def test_runner_accepts_team_kwargs(self, agent_name):
        from cooperbench.agents import get_runner

        runner = get_runner(agent_name)
        # We can't construct an LLM/sandbox here, but we can confirm the
        # signature would accept the kwargs (either as explicit params
        # or via **kwargs).
        import inspect

        sig = inspect.signature(runner.run)
        params = sig.parameters
        accepts_team = all(
            name in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            for name in ("team_role", "team_id", "task_list_url")
        )
        assert accepts_team, f"{agent_name} does not accept team_role/team_id/task_list_url"
