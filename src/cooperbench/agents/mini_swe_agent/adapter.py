"""Mini-SWE-Agent adapter for CooperBench.

This adapter wraps the mini-swe-agent framework to conform to the
AgentRunner interface used by CooperBench.
"""

from typing import TYPE_CHECKING

import yaml

from cooperbench.agents import AgentResult

if TYPE_CHECKING:
    from cooperbench.agents.mini_swe_agent.environments.docker import DockerEnvironment
from cooperbench.agents.mini_swe_agent.agents.default import DefaultAgent
from cooperbench.agents.mini_swe_agent.config import get_config_path
from cooperbench.agents.mini_swe_agent.connectors.git import GitConnector
from cooperbench.agents.mini_swe_agent.connectors.messaging import MessagingConnector
from cooperbench.agents.mini_swe_agent.environments.modal import ModalEnvironment
from cooperbench.agents.mini_swe_agent.models.litellm_model import LitellmModel
from cooperbench.agents.registry import register


@register("mini_swe_agent")
class MiniSweAgentRunner:
    """Adapter for mini-swe-agent framework."""

    def run(
        self,
        task: str,
        image: str,
        *,
        agent_id: str = "agent",
        model_name: str = "gpt-4o",
        agents: list[str] | None = None,
        comm_url: str | None = None,
        git_server_url: str | None = None,
        git_enabled: bool = False,
        messaging_enabled: bool = True,
        config: dict | None = None,
    ) -> AgentResult:
        """Run mini-swe-agent on a task.

        Args:
            task: The task description
            image: Docker image with the codebase
            agent_id: Unique identifier for this agent
            model_name: LLM model to use
            agents: List of all agent IDs (for collaboration)
            comm_url: Redis URL for inter-agent messaging
            git_server_url: Git server URL for code sharing
            git_enabled: Whether git collaboration is enabled
            messaging_enabled: Whether messaging is enabled
            config: Agent configuration (loaded from mini.yaml if not provided)

        Returns:
            AgentResult with status, patch, cost, steps, messages
        """
        # Always load default config, then merge with any overrides
        config_path = get_config_path("mini")
        with open(config_path) as f:
            default_config = yaml.safe_load(f)

        # Merge passed config overrides into default config
        if config is not None:
            default_config.update(config)

        agent_config = default_config.get("agent", {})
        backend = default_config.get("backend", "modal")

        # Create sandbox environment based on backend
        if backend == "docker":
            # Lazy import to avoid requiring docker package when not used
            from cooperbench.agents.mini_swe_agent.environments.docker import DockerEnvironment

            env = DockerEnvironment(
                image=image,
                cwd="/workspace/repo",
                timeout=3600,
            )
        else:
            env = ModalEnvironment(
                image=image,
                cwd="/workspace/repo",
                timeout=3600,
            )

        # Capture base commit for patch generation
        base_commit_result = env.execute("git rev-parse HEAD", timeout=10)
        base_commit = base_commit_result.get("output", "").strip()

        # Create LLM model
        model = LitellmModel(model_name=model_name)

        # Setup messaging connector if enabled
        comm = None
        if messaging_enabled and comm_url and agents and len(agents) > 1:
            comm = MessagingConnector(agent_id=agent_id, agents=agents, url=comm_url)

        # Setup git connector if enabled
        if git_enabled and git_server_url and agents:
            git_connector = GitConnector(
                agent_id=agent_id,
                agents=agents,
                server_url=git_server_url,
            )
            git_connector.setup(env)

        # Create agent with template variables for collaboration
        extra_vars = {
            "agent_id": agent_id if (agents and len(agents) > 1) else None,
            "agents": agents if agents else [],
            "git_enabled": git_enabled,
            "messaging_enabled": messaging_enabled,
        }

        agent = DefaultAgent(
            model=model,
            env=env,
            comm=comm,
            agent_id=agent_id,
            **agent_config,
        )
        agent.extra_template_vars.update(extra_vars)

        # Run agent
        error_msg = None
        try:
            status, _ = agent.run(task=task)
        except Exception as e:
            status = "Error"
            error_msg = str(e)

        # Extract patch (committed + uncommitted changes)
        patch = self._get_patch(env, base_commit)

        # Cleanup
        env.cleanup()

        return AgentResult(
            status=status,
            patch=patch,
            cost=model.cost,
            steps=model.n_calls,
            messages=agent.messages,
            error=error_msg,
        )

    def _get_patch(self, env: "ModalEnvironment | DockerEnvironment", base_commit: str) -> str:
        """Extract git diff from base commit to current working tree state."""
        try:
            # Single diff from base commit to working tree (includes both
            # committed and uncommitted changes)
            result = env.execute(f"git diff {base_commit}", timeout=30)
            return result.get("output", "").strip()
        except Exception:
            return ""
