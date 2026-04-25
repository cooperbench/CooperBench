"""Smoke tests - quick validation that core functionality works.

These tests should run fast and catch obvious breakages.
"""

import subprocess
import sys


class TestSmoke:
    """Quick smoke tests for basic functionality."""

    def test_package_importable(self):
        """Test that package can be imported."""
        import cooperbench

        assert cooperbench.__version__

    def test_cli_executable(self):
        """Test that CLI is executable."""
        result = subprocess.run(
            [sys.executable, "-m", "cooperbench.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "cooperbench" in result.stdout.lower()

    def test_run_function_exists(self):
        """Test that run function is exported."""
        from cooperbench import run

        assert callable(run)

    def test_evaluate_function_exists(self):
        """Test that evaluate function is exported."""
        from cooperbench import evaluate

        assert callable(evaluate)

    def test_agent_registry_works(self):
        """Test that agent registry is functional."""
        from cooperbench.agents import list_agents

        agents = list_agents()
        assert isinstance(agents, list)
        assert "mini_swe_agent_v2" in agents

    def test_mini_swe_agent_loadable(self):
        """Test that mini_swe_agent_v2 can be loaded."""
        from cooperbench.agents import get_runner

        runner = get_runner("mini_swe_agent_v2")
        assert runner is not None
        assert hasattr(runner, "run")

    def test_sandbox_functions_exist(self):
        """Test that sandbox functions are exported."""
        from cooperbench import evaluate_merge, test_merged, test_solo

        assert callable(test_merged)
        assert callable(test_solo)
        assert callable(evaluate_merge)

    def test_discover_tasks_callable(self):
        """Test that discover_tasks works."""
        from cooperbench import discover_tasks

        # Should return list (possibly empty if no dataset)
        tasks = discover_tasks()
        assert isinstance(tasks, list)

    def test_discover_runs_callable(self):
        """Test that discover_runs works."""
        from cooperbench import discover_runs

        # Should return list (possibly empty if no logs)
        runs = discover_runs(run_name="nonexistent")
        assert isinstance(runs, list)


class TestDependencies:
    """Test that required dependencies are available."""

    def test_modal_importable(self):
        """Test that modal is importable."""
        import modal

        assert modal

    def test_redis_importable(self):
        """Test that redis is importable."""
        import redis

        assert redis

    def test_litellm_importable(self):
        """Test that litellm is importable."""
        import litellm

        assert litellm

    def test_pydantic_importable(self):
        """Test that pydantic is importable."""
        import pydantic

        assert pydantic

    def test_rich_importable(self):
        """Test that rich is importable."""
        import rich

        assert rich
