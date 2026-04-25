"""Tests for package structure and imports."""

import importlib


class TestPackageImports:
    """Tests for package imports and structure."""

    def test_version_exists(self):
        """Test that version is defined."""
        from cooperbench import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)
        # Should be semver-ish
        parts = __version__.split(".")
        assert len(parts) >= 2

    def test_main_exports(self):
        """Test that main functions are exported."""
        from cooperbench import (
            discover_runs,
            discover_tasks,
            evaluate,
            evaluate_merge,
            run,
            test_merged,
            test_solo,
        )

        assert callable(run)
        assert callable(evaluate)
        assert callable(discover_tasks)
        assert callable(discover_runs)
        assert callable(test_merged)
        assert callable(test_solo)
        assert callable(evaluate_merge)

    def test_agents_module(self):
        """Test that agents module is importable."""
        from cooperbench.agents import AgentResult, AgentRunner, get_runner, list_agents, register

        assert AgentResult is not None
        assert AgentRunner is not None
        assert callable(get_runner)
        assert callable(list_agents)
        assert callable(register)

    def test_all_submodules_importable(self):
        """Test that all submodules can be imported."""
        modules = [
            "cooperbench",
            "cooperbench.cli",
            "cooperbench.utils",
            # Runner package
            "cooperbench.runner",
            "cooperbench.runner.tasks",
            "cooperbench.runner.core",
            "cooperbench.runner.solo",
            "cooperbench.runner.coop",
            # Eval package
            "cooperbench.eval",
            "cooperbench.eval.runs",
            "cooperbench.eval.evaluate",
            "cooperbench.eval.sandbox",
            "cooperbench.eval.backends",
            "cooperbench.eval.backends.base",
            "cooperbench.eval.backends.modal",
            # Infra package
            "cooperbench.infra",
            "cooperbench.infra.redis",
            # Agents
            "cooperbench.agents",
            "cooperbench.agents.registry",
            "cooperbench.agents.mini_swe_agent_v2",
            "cooperbench.agents.mini_swe_agent_v2.adapter",
            "cooperbench.agents.mini_swe_agent_v2.agents.default",
            "cooperbench.agents.mini_swe_agent_v2.environments.modal",
            "cooperbench.agents.mini_swe_agent_v2.models.litellm_model",
            "cooperbench.agents.mini_swe_agent_v2.connectors.messaging",
            "cooperbench.agents.mini_swe_agent_v2.connectors.git",
        ]

        for module_name in modules:
            module = importlib.import_module(module_name)
            assert module is not None, f"Failed to import {module_name}"


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_create_agent_result(self):
        """Test creating an AgentResult."""
        from cooperbench.agents import AgentResult

        result = AgentResult(
            status="Submitted",
            patch="diff --git a/test.py",
            cost=0.05,
            steps=10,
        )

        assert result.status == "Submitted"
        assert result.patch == "diff --git a/test.py"
        assert result.cost == 0.05
        assert result.steps == 10
        assert result.messages == []
        assert result.error is None

    def test_agent_result_with_error(self):
        """Test AgentResult with error."""
        from cooperbench.agents import AgentResult

        result = AgentResult(
            status="Error",
            patch="",
            cost=0.01,
            steps=1,
            error="Something went wrong",
        )

        assert result.status == "Error"
        assert result.error == "Something went wrong"

    def test_agent_result_with_messages(self):
        """Test AgentResult with messages."""
        from cooperbench.agents import AgentResult

        messages = [
            {"role": "system", "content": "You are an agent"},
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "content": "I'll fix it"},
        ]

        result = AgentResult(
            status="Submitted",
            patch="...",
            cost=0.1,
            steps=5,
            messages=messages,
        )

        assert len(result.messages) == 3
        assert result.messages[0]["role"] == "system"
