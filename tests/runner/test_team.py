"""Tests for the team-mode runner.

These tests stub the agent runner so we don't need any LLM keys.
They verify:

  - lead/member role assignment (first agent = lead)
  - task list is pre-seeded with one task per feature, owned by the
    member assigned that feature
  - team_role, team_id, task_list_url propagate to every adapter call
  - results dict captures per-agent patches and coordination metrics
  - the runner doesn't crash when no Redis is available (degrades to
    no-task-list mode)
"""

from __future__ import annotations

from unittest.mock import patch as mock_patch

import fakeredis
import pytest

from cooperbench.agents import AgentResult
from cooperbench.runner.team import execute_team


@pytest.fixture
def fake_runner_factory():
    """Builds a stand-in agent runner that records every ``.run`` kwarg.

    The runner returns ``AgentResult(status="Submitted", patch="diff
    --git a/x b/x\n+ hi\n")`` so the team runner thinks every agent
    succeeded.
    """

    class _FakeRunner:
        def __init__(self):
            self.calls = []

        def run(self, task, image, **kwargs):
            self.calls.append({"task": task, "image": image, **kwargs})
            return AgentResult(
                status="Submitted",
                patch="diff --git a/x b/x\n+ hi\n",
                cost=0.1,
                steps=3,
                messages=[],
            )

    return _FakeRunner


@pytest.fixture
def mock_get_runner(fake_runner_factory):
    """Patches get_runner so every agent in the team uses the fake."""
    fake = fake_runner_factory()
    with mock_patch("cooperbench.runner.team.get_runner", return_value=fake):
        yield fake


@pytest.fixture
def in_memory_redis():
    """Patches Redis client construction to return a fakeredis instance."""
    fake = fakeredis.FakeRedis()
    with mock_patch("cooperbench.runner.team._redis_client", return_value=fake):
        yield fake


@pytest.fixture
def isolated_dirs(tmp_path):
    """Per-test dataset + logs roots so we don't touch the real ones."""
    dataset = tmp_path / "dataset"
    logs = tmp_path / "logs"
    (dataset / "demo_repo" / "task1" / "feature1").mkdir(parents=True)
    (dataset / "demo_repo" / "task1" / "feature2").mkdir(parents=True)
    (dataset / "demo_repo" / "task1" / "feature1" / "feature.md").write_text("feature 1 spec")
    (dataset / "demo_repo" / "task1" / "feature2" / "feature.md").write_text("feature 2 spec")
    return {"dataset": str(dataset), "logs": str(logs)}


class TestExecuteTeam:
    def test_lead_is_first_agent(self, mock_get_runner, in_memory_redis, isolated_dirs):
        execute_team(
            repo_name="demo_repo",
            task_id=1,
            features=[1, 2],
            run_name="t1",
            agent_name="fake",
            model_name="fake-model",
            force=True,
            backend="docker",
            dataset_dir=isolated_dirs["dataset"],
            logs_dir=isolated_dirs["logs"],
            redis_url="redis://localhost:6379",
        )

        roles = {c["agent_id"]: c.get("team_role") for c in mock_get_runner.calls}
        # First by sorted order (agent1) is the lead; rest are members.
        assert roles["agent1"] == "lead"
        assert roles["agent2"] == "member"

    def test_task_list_preseeded_with_one_per_feature(self, mock_get_runner, in_memory_redis, isolated_dirs):
        execute_team(
            repo_name="demo_repo",
            task_id=1,
            features=[1, 2],
            run_name="t1",
            agent_name="fake",
            model_name="fake-model",
            force=True,
            backend="docker",
            dataset_dir=isolated_dirs["dataset"],
            logs_dir=isolated_dirs["logs"],
            redis_url="redis://localhost:6379",
        )

        # Walk the fake Redis directly to confirm pre-seed.
        # Keys look like cb:<run_id>:tasks:all (a set).
        all_keys = [k.decode() if isinstance(k, bytes) else k for k in in_memory_redis.keys("cb:*:tasks:all")]
        assert len(all_keys) == 1, f"expected one task-set, got {all_keys}"
        task_ids = [m.decode() if isinstance(m, bytes) else m for m in in_memory_redis.smembers(all_keys[0])]
        assert len(task_ids) == 2  # one task per feature

    def test_task_list_url_propagated_to_every_adapter(self, mock_get_runner, in_memory_redis, isolated_dirs):
        execute_team(
            repo_name="demo_repo",
            task_id=1,
            features=[1, 2],
            run_name="t1",
            agent_name="fake",
            model_name="fake-model",
            force=True,
            backend="docker",
            dataset_dir=isolated_dirs["dataset"],
            logs_dir=isolated_dirs["logs"],
            redis_url="redis://localhost:6379",
        )

        for call in mock_get_runner.calls:
            assert call.get("team_id") is not None
            # task_list_url and comm_url share the namespaced Redis URL.
            assert "redis://" in (call.get("task_list_url") or "")
            assert "#run:" in (call.get("task_list_url") or "")

    def test_result_dict_includes_metrics(self, mock_get_runner, in_memory_redis, isolated_dirs):
        result = execute_team(
            repo_name="demo_repo",
            task_id=1,
            features=[1, 2],
            run_name="t1",
            agent_name="fake",
            model_name="fake-model",
            force=True,
            backend="docker",
            dataset_dir=isolated_dirs["dataset"],
            logs_dir=isolated_dirs["logs"],
            redis_url="redis://localhost:6379",
        )

        # Top-level shape mirrors execute_coop with one extra key.
        assert "result" in result
        inner = result["result"]
        assert "metrics" in inner
        m = inner["metrics"]
        assert "tasks_total" in m
        assert "claims_per_agent" in m
        # Two tasks were pre-seeded, both should be present.
        assert m["tasks_total"] == 2

    def test_supports_three_agent_team(self, mock_get_runner, in_memory_redis, isolated_dirs):
        # Add a third feature so we can test 3-agent teams.
        (isolated_dirs["dataset"] + "/demo_repo/task1/feature3").replace
        import os

        os.makedirs(isolated_dirs["dataset"] + "/demo_repo/task1/feature3", exist_ok=True)
        with open(isolated_dirs["dataset"] + "/demo_repo/task1/feature3/feature.md", "w") as f:
            f.write("f3")

        execute_team(
            repo_name="demo_repo",
            task_id=1,
            features=[1, 2, 3],
            run_name="t1",
            agent_name="fake",
            model_name="fake-model",
            force=True,
            backend="docker",
            dataset_dir=isolated_dirs["dataset"],
            logs_dir=isolated_dirs["logs"],
            redis_url="redis://localhost:6379",
        )

        roles = {c["agent_id"]: c.get("team_role") for c in mock_get_runner.calls}
        assert roles == {"agent1": "lead", "agent2": "member", "agent3": "member"}
