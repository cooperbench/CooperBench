"""Unit tests for team-mode runtime helpers.

The runtime layer translates host-side team config into the env vars
that the in-container ``coop-task-*`` CLIs read, and into the docker
arg list (scratchpad mount).
"""

from cooperbench.agents._team.runtime import (
    CONTAINER_SCRATCHPAD_DIR,
    build_team_env,
    scratchpad_mount_args,
)


class TestBuildTeamEnv:
    def test_full_team_env(self):
        env = build_team_env(
            redis_url="redis://host.docker.internal:6379#run:abc",
            run_id="abc",
            agent_id="agent1",
            agents=["agent1", "agent2"],
            team_role="lead",
        )
        assert env["CB_TEAM_REDIS_URL"] == "redis://host.docker.internal:6379#run:abc"
        assert env["CB_TEAM_RUN_ID"] == "abc"
        assert env["CB_TEAM_AGENT_ID"] == "agent1"
        assert env["CB_TEAM_AGENTS"] == "agent1,agent2"
        assert env["CB_TEAM_ROLE"] == "lead"

    def test_missing_role_omitted_not_empty(self):
        """Empty values would cause the CLI to think it has a role."""
        env = build_team_env(
            redis_url="redis://x:6379",
            run_id="r",
            agent_id="a",
            agents=["a"],
            team_role=None,
        )
        assert "CB_TEAM_ROLE" not in env


class TestScratchpadMountArgs:
    def test_default_mount_string(self):
        args = scratchpad_mount_args(volume_name="cb-team-abc")
        assert "--volume" in args
        # Bind volume to the canonical container path.
        host_to_container = args[args.index("--volume") + 1]
        assert host_to_container == f"cb-team-abc:{CONTAINER_SCRATCHPAD_DIR}"

    def test_volume_name_required(self):
        """Empty volume name should produce no args, not a broken mount."""
        assert scratchpad_mount_args(volume_name="") == []
        assert scratchpad_mount_args(volume_name=None) == []
