"""Tests for ``TeamHarnessConfig`` and ``TeamSession``.

These cover the public facade — both the dataclass-level config
ergonomics (``with_only`` / ``disabled``) and the session-level
factories that adapters consume, particularly that each factory
returns ``None`` / empty when its feature is disabled so callers
can write one code path.
"""

from __future__ import annotations

import fakeredis
import pytest

from cooperbench.team_harness import (
    CONTAINER_SCRATCHPAD_DIR,
    MCP_SERVER_NAME,
    TaskListClient,
    TeamHarnessConfig,
    TeamPoller,
    TeamSession,
)


class TestTeamHarnessConfig:
    def test_defaults_all_enabled(self):
        cfg = TeamHarnessConfig()
        assert cfg.task_list is True
        assert cfg.scratchpad is True
        assert cfg.mcp is True
        assert cfg.auto_refresh is True
        assert cfg.protocol is True

    def test_with_only_enables_named_features_disables_rest(self):
        cfg = TeamHarnessConfig.with_only("task_list", "mcp")
        assert cfg.task_list is True
        assert cfg.mcp is True
        assert cfg.scratchpad is False
        assert cfg.auto_refresh is False
        assert cfg.protocol is False

    def test_with_only_unknown_feature_raises(self):
        with pytest.raises(ValueError):
            TeamHarnessConfig.with_only("not_a_feature")

    def test_disabled_lists_off_flags(self):
        cfg = TeamHarnessConfig(scratchpad=False, mcp=False)
        assert set(cfg.disabled()) == {"scratchpad", "mcp"}

    def test_disabled_empty_when_all_on(self):
        assert TeamHarnessConfig().disabled() == []


@pytest.fixture
def session():
    """Default session with all features on."""
    return TeamSession(
        run_id="r1",
        redis_url="redis://localhost:6379#run:r1",
        agents=["agent1", "agent2", "agent3"],
        team_volume="cb-team-r1",
    )


class TestTeamSessionBasics:
    def test_lead_is_first_agent(self, session):
        assert session.lead == "agent1"

    def test_role_for_lead_is_lead(self, session):
        assert session.role_for("agent1") == "lead"

    def test_role_for_others_is_member(self, session):
        assert session.role_for("agent2") == "member"
        assert session.role_for("agent3") == "member"

    def test_is_active_requires_two_agents(self):
        single = TeamSession(run_id="r", redis_url="redis://", agents=["only"], team_volume="")
        assert single.is_active() is False
        pair = TeamSession(run_id="r", redis_url="redis://", agents=["a", "b"], team_volume="")
        assert pair.is_active() is True


class TestEnvFor:
    def test_default_returns_full_env(self, session):
        env = session.env_for("agent1")
        assert env["CB_TEAM_RUN_ID"] == "r1"
        assert env["CB_TEAM_AGENT_ID"] == "agent1"
        assert env["CB_TEAM_AGENTS"] == "agent1,agent2,agent3"
        assert env["CB_TEAM_ROLE"] == "lead"

    def test_rewrites_localhost_for_container(self, session):
        env = session.env_for("agent1")
        assert "host.docker.internal" in env["CB_TEAM_REDIS_URL"]
        assert "localhost" not in env["CB_TEAM_REDIS_URL"]

    def test_passes_through_non_localhost_urls(self):
        s = TeamSession(
            run_id="r",
            redis_url="redis://modal-tunnel.example:6379#run:r",
            agents=["a", "b"],
            team_volume="",
        )
        env = s.env_for("a")
        assert env["CB_TEAM_REDIS_URL"] == "redis://modal-tunnel.example:6379#run:r"

    def test_returns_empty_when_all_env_consumers_disabled(self):
        cfg = TeamHarnessConfig(task_list=False, auto_refresh=False, protocol=False, mcp=False)
        s = TeamSession(
            run_id="r",
            redis_url="redis://localhost:6379",
            agents=["a", "b"],
            team_volume="",
            config=cfg,
        )
        assert s.env_for("a") == {}

    def test_returns_full_env_when_only_mcp_remains(self):
        # Even with task_list off, env_for must populate the URL so the
        # MCP server can BLPOP the inbox.
        cfg = TeamHarnessConfig(task_list=False, auto_refresh=False, protocol=False, mcp=True)
        s = TeamSession(
            run_id="r",
            redis_url="redis://localhost:6379",
            agents=["a", "b"],
            team_volume="",
            config=cfg,
        )
        env = s.env_for("a")
        assert env["CB_TEAM_REDIS_URL"]
        assert env["CB_TEAM_AGENT_ID"] == "a"


class TestScratchpadMountArgs:
    def test_returns_volume_mount_when_enabled(self, session):
        args = session.scratchpad_mount_args()
        assert args == ["--volume", f"cb-team-r1:{CONTAINER_SCRATCHPAD_DIR}"]

    def test_empty_when_feature_off(self):
        cfg = TeamHarnessConfig(scratchpad=False)
        s = TeamSession(run_id="r", redis_url="redis://", agents=["a", "b"], team_volume="cb-team-r", config=cfg)
        assert s.scratchpad_mount_args() == []

    def test_empty_when_volume_name_empty_even_if_enabled(self):
        # No volume name → nothing to mount; the underlying helper
        # already returns [] for falsy volume names, and we honour that.
        s = TeamSession(run_id="r", redis_url="redis://", agents=["a", "b"], team_volume="")
        assert s.scratchpad_mount_args() == []


class TestMcpConfig:
    def test_default_returns_config_with_server_entry(self, session):
        cfg = session.mcp_config(container_script_path="/tmp/srv.py")
        assert cfg is not None
        assert MCP_SERVER_NAME in cfg["mcpServers"]
        assert cfg["mcpServers"][MCP_SERVER_NAME]["args"] == ["/tmp/srv.py"]
        assert cfg["mcpServers"][MCP_SERVER_NAME]["command"] == "python3"

    def test_none_when_feature_off(self):
        cfg = TeamHarnessConfig(mcp=False)
        s = TeamSession(run_id="r", redis_url="redis://", agents=["a", "b"], team_volume="", config=cfg)
        assert s.mcp_config(container_script_path="/tmp/srv.py") is None


class TestTaskListClient:
    def test_returns_client_when_enabled(self, session):
        client = session.task_list_client(redis_client=fakeredis.FakeRedis())
        assert isinstance(client, TaskListClient)

    def test_none_when_feature_off(self, session):
        session.config = TeamHarnessConfig(task_list=False)
        assert session.task_list_client(redis_client=fakeredis.FakeRedis()) is None


class TestLoopPoller:
    def test_returns_poller_when_enabled(self, session):
        poller = session.loop_poller(agent_id="agent1")
        assert isinstance(poller, TeamPoller)

    def test_none_when_feature_off(self, session):
        session.config = TeamHarnessConfig(auto_refresh=False)
        assert session.loop_poller(agent_id="agent1") is None


class TestHarvestMetrics:
    def test_returns_metrics_for_active_task_list(self, session):
        client = session.task_list_client(redis_client=fakeredis.FakeRedis())
        assert client is not None
        # Seed two tasks; one claimed.
        tid_a = client.create(title="t1", created_by="bench", owner="agent2")
        client.create(title="t2", created_by="bench", owner="agent3")
        client.claim(tid_a, by="agent2")
        metrics, events, final = session.harvest_metrics(client)
        assert metrics is not None
        assert metrics["tasks_total"] == 2
        assert metrics["claims_per_agent"] == {"agent2": 1}
        assert len(events) >= 3  # 2 creates + 1 claim
        assert len(final) == 2

    def test_returns_empty_when_task_list_off(self, session):
        session.config = TeamHarnessConfig(task_list=False)
        metrics, events, final = session.harvest_metrics(None)
        assert metrics is None
        assert events == []
        assert final == []

    def test_returns_empty_when_client_is_none(self, session):
        # Real case: task_list on but Redis dropped → client is None.
        metrics, events, final = session.harvest_metrics(None)
        assert metrics is None


class TestPromptFor:
    def test_full_prompt_contains_team_section(self, session):
        prompt = session.prompt_for(task="dummy", agent_id="agent1")
        assert "dummy" in prompt
        assert "team-lead" in prompt.lower() or "team lead" in prompt.lower()

    def test_member_prompt_is_member_block(self, session):
        prompt = session.prompt_for(task="dummy", agent_id="agent2")
        assert "team member" in prompt.lower() or "team-member" in prompt.lower()


class TestPromptSection:
    def test_returns_section_only(self, session):
        section = session.prompt_section(agent_id="agent1")
        assert section
        # Should NOT contain the task body (which prompt_for would wrap around it).
        assert "agent1" in section or "team-lead" in section.lower()

    def test_empty_for_single_agent_team(self):
        # Team of one collapses to solo; section should be empty.
        s = TeamSession(run_id="r", redis_url="redis://", agents=["solo"], team_volume="")
        assert s.prompt_section(agent_id="solo") == ""
