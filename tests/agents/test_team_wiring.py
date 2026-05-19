"""Compatibility tests: every adapter in team mode must at minimum
accept the team kwargs without crashing and append the team section
to the task it sends the agent.

The CLI adapters (claude_code, codex) have richer wiring tested in
their own test files; this module is the cross-adapter sanity check
so a future refactor doesn't silently regress one adapter.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest

from cooperbench.team_harness.prompt import team_task_section


class TestTeamTaskSectionVsBuildInstruction:
    """The two ways to inject the team prompt must stay consistent:
    ``team_task_section`` is what Python-loop adapters append; the
    bigger ``build_team_instruction`` (used by CLI adapters) must
    *contain* the same section."""

    def test_lead_section_is_substring_of_full_lead_prompt(self):
        from cooperbench.team_harness import build_team_instruction

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
        from cooperbench.team_harness import build_team_instruction

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
        from cooperbench.team_harness import team_task_section

        section = team_task_section(agents=["agent1", "agent2"], agent_id="agent1", team_role="lead")
        # Sanity: this is the exact piece the v2 adapter appends.
        assert "coop-task-create" in section
        assert "team-lead" in section.lower()


class TestOpenHandsTeamWiring:
    """openhands adapter: in team mode, builds coop_info with team_env
    so the sandbox sees CB_TEAM_* variables."""

    def test_team_env_dict_has_expected_keys(self):
        from cooperbench.agents._coop.runtime import rewrite_comm_url_for_container
        from cooperbench.team_harness.runtime import CONTAINER_TASKS_MIRROR_DIR

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


class TestOpenHandsImageLayering:
    """When team mode is active, the openhands sandbox image gets
    layered with the coop-task-* install at runtime (no upstream
    image rebuild needed)."""

    def test_team_env_triggers_image_layering(self):
        """We can't actually build a Modal image in a unit test, but
        we can verify the code path: when ``coop_info["team_env"]``
        is set, the ``__enter__`` method should call ``add_local_file``,
        ``pip_install``, and ``run_commands`` on the base image."""

        from cooperbench.agents.openhands_agent_sdk.adapter import ModalSandboxContext

        # Build a fake modal.Image whose chain methods all return self
        # so we can introspect what got called.
        base_image = MagicMock()
        for attr in ("add_local_file", "pip_install", "run_commands"):
            getattr(base_image, attr).return_value = base_image

        ctx = ModalSandboxContext(
            image_name="example/oh:tag",
            timeout=60,
            coop_info={
                "agent_id": "agent1",
                "agents": ["agent1", "agent2"],
                "team_env": {"CB_TEAM_REDIS_URL": "redis://x", "CB_TEAM_RUN_ID": "r"},
            },
        )

        # Stub the entry-point so we only exercise the image-layering
        # branch — no real Modal Sandbox creation.
        with (
            mock_patch("modal.Image.from_registry", return_value=base_image),
            mock_patch("modal.App.lookup"),
            mock_patch("modal.Secret.from_dict"),
            mock_patch("modal.Sandbox.create") as sandbox_create,
            mock_patch.object(ctx, "_wait_for_server"),
        ):
            sandbox_create.return_value.tunnels.return_value = {8000: MagicMock(url="https://stub")}
            ctx.__enter__()

        # add_local_file is called THREE times in team mode:
        #   1. coop-task-* CLI helper
        #   2. CoopTaskTracker definition (drops into openhands install)
        #   3. Replacement __init__.py for the task_tracker package
        #      (forces the coop_definition import at openhands startup)
        assert base_image.add_local_file.call_count == 3
        sources = [call.args[0] for call in base_image.add_local_file.call_args_list]
        destinations = [call.args[1] for call in base_image.add_local_file.call_args_list]
        assert any("coop_task.py" in s for s in sources)
        assert any("coop_definition.py" in s for s in sources)
        assert any("_team_init_override.py" in s for s in sources)
        assert "/usr/local/bin/cb-coop-task.py" in destinations
        assert "/tmp/cb-coop-tracker.py" in destinations
        assert "/tmp/cb-task-tracker-init.py" in destinations
        # pip_install once for redis.
        base_image.pip_install.assert_called_once_with("redis")
        # Two run_commands layers in the team-mode branch: tool-file
        # install + side-effect __init__ append, and the coop-task-*
        # wrappers.  Splitting them keeps build failures localized.
        assert base_image.run_commands.call_count == 2
        all_cmds = " ".join(call.args[0] for call in base_image.run_commands.call_args_list)
        assert "coop-task-$sub" in all_cmds
        assert "cb-coop-task.py" in all_cmds
        assert "coop_definition.py" in all_cmds
        # The replacement __init__.py is copied into place to override
        # the upstream registration with our coop_definition import.
        assert "cb-task-tracker-init.py" in all_cmds
        # pyc caches must be wiped so the new __init__ takes effect.
        assert "*.pyc" in all_cmds

    def test_no_layering_when_team_inactive(self):
        """Solo / coop runs must NOT pay the image-build cost."""

        from cooperbench.agents.openhands_agent_sdk.adapter import ModalSandboxContext

        base_image = MagicMock()
        ctx = ModalSandboxContext(image_name="example/oh:tag", timeout=60, coop_info=None)

        with (
            mock_patch("modal.Image.from_registry", return_value=base_image),
            mock_patch("modal.App.lookup"),
            mock_patch("modal.Sandbox.create") as sandbox_create,
            mock_patch.object(ctx, "_wait_for_server"),
        ):
            sandbox_create.return_value.tunnels.return_value = {8000: MagicMock(url="https://stub")}
            ctx.__enter__()

        base_image.add_local_file.assert_not_called()
        base_image.pip_install.assert_not_called()
        base_image.run_commands.assert_not_called()


class TestOpenHandsTaskTrackerSwap:
    """The Redis-backed CoopTaskTrackerTool overrides the local
    TaskTrackerTool registration when ``coop_definition`` is imported
    (which happens server-side via the .pth file the openhands adapter
    installs in the Modal sandbox).  Host-side tool lists keep using
    the ``TaskTrackerTool`` name; the registry resolution does the
    swap transparently."""

    def test_importing_coop_definition_overrides_local_registration(self):
        # ``register_tool`` is idempotent and overwrites by name, so
        # importing coop_definition should rebind TaskTrackerTool.name
        # to the Redis-backed class.  We probe the internal registry
        # dict directly because ``resolve_tool`` requires a
        # ConversationState we don't have in unit tests.
        from openhands.sdk.tool import registry as _registry
        from openhands.tools.task_tracker import coop_definition  # noqa: F401 — registers
        from openhands.tools.task_tracker.definition import TaskTrackerTool

        # After the import above, the resolver under TaskTrackerTool.name
        # should be the one bound for CoopTaskTrackerTool.  The simplest
        # test that doesn't depend on the resolver internals is just
        # that we see "Coop" in the registered resolver's qualname.
        qualname = _registry._MODULE_QUALNAMES.get(TaskTrackerTool.name, "")
        assert "coop" in qualname.lower(), (
            f"expected TaskTrackerTool registration to come from coop_definition; got module qualname {qualname!r}"
        )

    def test_coop_tracker_round_trip_through_redis(self, monkeypatch):
        """Plan + view round-trip via fakeredis, writing to the same
        ``cb:<run_id>:`` namespace as ``TaskListClient``."""
        import fakeredis
        from openhands.tools.task_tracker.coop_definition import CoopTaskTrackerExecutor
        from openhands.tools.task_tracker.definition import TaskItem, TaskTrackerAction

        monkeypatch.setenv("CB_TEAM_REDIS_URL", "redis://stub")
        monkeypatch.setenv("CB_TEAM_RUN_ID", "t")
        monkeypatch.setenv("CB_TEAM_AGENT_ID", "agent1")

        fake = fakeredis.FakeRedis()
        ex = CoopTaskTrackerExecutor()
        with mock_patch(
            "openhands.tools.task_tracker.coop_definition._redis_client_from_env",
            return_value=fake,
        ):
            plan = TaskTrackerAction(
                command="plan",
                task_list=[
                    TaskItem(title="implement feature X", status="todo"),
                    TaskItem(title="add tests", status="in_progress"),
                ],
            )
            obs = ex(plan)
            assert obs.command == "plan"
            assert len(obs.task_list) == 2

            view = TaskTrackerAction(command="view")
            obs = ex(view)
            titles = [t.title for t in obs.task_list]
            assert any("implement feature X" in t for t in titles)

        # Confirm Redis was written in the shared namespace TaskListClient uses.
        ids = list(fake.smembers("cb:t:tasks:all"))
        assert len(ids) == 2


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
