"""Mini-SWE-Agent v2 adapter for CooperBench.

This adapter wraps the mini-swe-agent v2 framework (tool-calling version)
to conform to the AgentRunner interface used by CooperBench.
"""

import yaml

from cooperbench.agents import AgentResult
from cooperbench.agents.mini_swe_agent_v2.agents.default import DefaultAgent
from cooperbench.agents.mini_swe_agent_v2.config import get_config_path
from cooperbench.agents.mini_swe_agent_v2.connectors import GitConnector
from cooperbench.agents.mini_swe_agent_v2.connectors.messaging import MessagingConnector
from cooperbench.agents.mini_swe_agent_v2.models.litellm_model import LitellmModel
from cooperbench.agents.mini_swe_agent_v2.models.utils.actions_toolcall import SEND_MESSAGE_TOOL
from cooperbench.agents.registry import register


@register("mini_swe_agent_v2")
class MiniSweAgentV2Runner:
    """Adapter for mini-swe-agent v2 framework (tool-calling)."""

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
        agent_config: str | None = None,
        log_dir: str | None = None,
    ) -> AgentResult:
        """Run mini-swe-agent v2 on a task."""
        # Always load default config, then merge with any overrides
        config_path = get_config_path("mini")
        with open(config_path) as f:
            default_config = yaml.safe_load(f)

        # Merge passed config overrides into default config
        if config is not None:
            default_config.update(config)

        agent_cfg = default_config.get("agent", {})
        model_cfg = default_config.get("model", {})
        env_cfg = default_config.get("environment", {})
        backend = default_config.get("backend", "modal")

        # Create environment based on backend
        env_kwargs = {
            "image": image,
            "cwd": "/workspace/repo",
            "timeout": 3600,
        }
        if env_cfg.get("env"):
            env_kwargs["env"] = env_cfg["env"]

        if backend == "docker":
            from cooperbench.agents.mini_swe_agent_v2.environments.docker import DockerEnvironment

            if config and config.get("git_network"):
                env_kwargs["network"] = config["git_network"]
            env = DockerEnvironment(**env_kwargs)
        else:
            from cooperbench.agents.mini_swe_agent_v2.environments.modal import ModalEnvironment

            env = ModalEnvironment(**env_kwargs)

        # Capture base commit for patch generation
        base_commit_result = env.execute({"command": "git rev-parse HEAD"})
        base_commit = base_commit_result.get("output", "").strip()

        # Setup messaging connector if enabled
        comm = None
        use_messaging = messaging_enabled and comm_url and agents and len(agents) > 1
        if use_messaging:
            comm = MessagingConnector(agent_id=agent_id, agents=agents, url=comm_url)

        # Create LLM model with send_message tool if messaging is enabled
        extra_tools = [SEND_MESSAGE_TOOL] if use_messaging else None
        model = LitellmModel(model_name=model_name, extra_tools=extra_tools, **model_cfg)

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
            **agent_cfg,
        )
        agent.extra_template_vars.update(extra_vars)

        # Run agent
        error_msg = None
        try:
            result = agent.run(task=task)
            status = result.get("exit_status", "Submitted")
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
            cost=agent.cost,
            steps=agent.n_calls,
            messages=agent.messages,
            sent_messages=agent.sent_messages,
            error=error_msg,
        )

    def _get_patch(self, env, base_commit: str) -> str:
        """Extract git diff from base commit to current working tree state."""
        try:
            result = env.execute({"command": f"git diff {base_commit}"})
            return result.get("output", "").strip()
        except Exception:
            return ""
