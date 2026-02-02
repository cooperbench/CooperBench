"""Tests for cooperbench.agents.registry module."""

import inspect
import warnings

import pytest

from cooperbench.agents import AgentResult, get_runner, list_agents, register


class TestAgentRegistry:
    """Tests for agent registry."""

    def test_list_agents_returns_list(self):
        """Test that list_agents returns a list."""
        agents = list_agents()
        assert isinstance(agents, list)

    def test_mini_swe_agent_registered(self):
        """Test that mini_swe_agent is registered."""
        agents = list_agents()
        assert "mini_swe_agent" in agents

    def test_get_runner_returns_instance(self):
        """Test that get_runner returns an instance."""
        runner = get_runner("mini_swe_agent")
        assert runner is not None
        assert hasattr(runner, "run")

    def test_get_unknown_agent_raises(self):
        """Test that getting an unknown agent raises ValueError."""
        with pytest.raises(ValueError, match="Unknown agent"):
            get_runner("nonexistent_agent")

    def test_register_decorator(self):
        """Test the register decorator."""

        @register("test_agent_temp")
        class TestAgentRunner:
            def run(self, task, image, **kwargs) -> AgentResult:
                return AgentResult(status="Test", patch="", cost=0, steps=0)

        assert "test_agent_temp" in list_agents()
        runner = get_runner("test_agent_temp")
        assert runner is not None


class TestMiniSweAgentAdapter:
    """Tests for MiniSweAgent adapter (unit tests, no Modal)."""

    def test_adapter_has_run_method(self):
        """Test that adapter has run method."""
        runner = get_runner("mini_swe_agent")
        assert hasattr(runner, "run")
        assert callable(runner.run)

    def test_adapter_run_signature(self):
        """Test that run method has correct parameters."""
        runner = get_runner("mini_swe_agent")
        sig = inspect.signature(runner.run)
        params = list(sig.parameters.keys())

        # Required params
        assert "task" in params
        assert "image" in params

        # Optional params
        assert "agent_id" in params
        assert "model_name" in params
        assert "agents" in params
        assert "comm_url" in params


class TestExternalAgentRegistration:
    """Tests for external agent registration via COOPERBENCH_EXTERNAL_AGENTS."""

    def test_external_agent_registration(self, tmp_path, monkeypatch):
        """Test that external agents can be registered via environment variable."""
        # Create a temporary external agent module
        external_pkg = tmp_path / "external_test_agent"
        external_pkg.mkdir()
        (external_pkg / "__init__.py").write_text("")

        adapter_code = """
from cooperbench.agents.registry import register
from cooperbench.agents import AgentResult

@register("external_test_agent")
class ExternalTestAgentRunner:
    def run(self, task, image, **kwargs) -> AgentResult:
        return AgentResult(
            status="Submitted",
            patch="# External test patch",
            cost=0.0,
            steps=1,
            messages=[],
            error=None,
        )
"""
        (external_pkg / "adapter.py").write_text(adapter_code)

        # Add tmp_path to sys.path
        monkeypatch.syspath_prepend(str(tmp_path))

        # Set environment variable
        monkeypatch.setenv("COOPERBENCH_EXTERNAL_AGENTS", "external_test_agent.adapter")

        # Reload the registry module to pick up the env var
        import cooperbench.agents.registry as registry_module

        # Import the external agent
        registry_module._auto_register()

        # Check that the external agent was registered
        assert "external_test_agent" in list_agents()
        runner = get_runner("external_test_agent")
        assert runner is not None

    def test_multiple_external_agents(self, tmp_path, monkeypatch):
        """Test registering multiple external agents (comma-separated)."""
        # Create two external agent modules
        for i in [1, 2]:
            external_pkg = tmp_path / f"external_agent_{i}"
            external_pkg.mkdir()
            (external_pkg / "__init__.py").write_text("")

            adapter_code = f"""
from cooperbench.agents.registry import register
from cooperbench.agents import AgentResult

@register("external_agent_{i}")
class ExternalAgent{i}Runner:
    def run(self, task, image, **kwargs) -> AgentResult:
        return AgentResult(
            status="Submitted",
            patch="# Test patch {i}",
            cost=0.0,
            steps=1,
            messages=[],
            error=None,
        )
"""
            (external_pkg / "adapter.py").write_text(adapter_code)

        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setenv("COOPERBENCH_EXTERNAL_AGENTS", "external_agent_1.adapter,external_agent_2.adapter")

        import cooperbench.agents.registry as registry_module

        registry_module._auto_register()

        agents = list_agents()
        assert "external_agent_1" in agents
        assert "external_agent_2" in agents

    def test_external_agent_with_whitespace(self, tmp_path, monkeypatch):
        """Test that whitespace in env var is handled correctly."""
        external_pkg = tmp_path / "whitespace_agent"
        external_pkg.mkdir()
        (external_pkg / "__init__.py").write_text("")

        adapter_code = """
from cooperbench.agents.registry import register
from cooperbench.agents import AgentResult

@register("whitespace_agent")
class WhitespaceAgentRunner:
    def run(self, task, image, **kwargs) -> AgentResult:
        return AgentResult(
            status="Submitted",
            patch="",
            cost=0.0,
            steps=1,
            messages=[],
            error=None,
        )
"""
        (external_pkg / "adapter.py").write_text(adapter_code)

        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setenv("COOPERBENCH_EXTERNAL_AGENTS", " whitespace_agent.adapter ")

        import cooperbench.agents.registry as registry_module

        registry_module._auto_register()

        assert "whitespace_agent" in list_agents()

    def test_invalid_external_agent_warns(self, monkeypatch):
        """Test that invalid external agents produce warnings but don't crash."""
        monkeypatch.setenv("COOPERBENCH_EXTERNAL_AGENTS", "nonexistent.module.adapter")

        import cooperbench.agents.registry as registry_module

        # Should warn but not crash
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            registry_module._auto_register()

            # Check that a warning was issued
            assert len(w) >= 1
            assert any("Could not import external agent module" in str(warning.message) for warning in w)

        # Built-in agents should still work
        agents = list_agents()
        assert "mini_swe_agent" in agents or "swe_agent" in agents

    def test_empty_env_var(self, monkeypatch):
        """Test that empty COOPERBENCH_EXTERNAL_AGENTS is handled correctly."""
        monkeypatch.setenv("COOPERBENCH_EXTERNAL_AGENTS", "")

        import cooperbench.agents.registry as registry_module

        registry_module._auto_register()

        # Should work without errors, built-in agents should be available
        agents = list_agents()
        assert isinstance(agents, list)

    def test_builtin_agents_still_work(self):
        """Test that built-in agents work regardless of external agent config."""
        # This test ensures backward compatibility
        agents = list_agents()
        assert len(agents) >= 1
        # At least one built-in agent should be registered
        assert "mini_swe_agent" in agents or "swe_agent" in agents
