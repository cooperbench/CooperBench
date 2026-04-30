"""Mini-SWE-Agent v2 adapter for CooperBench.

This adapter wraps the mini-swe-agent v2 framework (tool-calling version)
to conform to the AgentRunner interface used by CooperBench.
"""

import logging
from pathlib import Path

import yaml

from cooperbench.agents import AgentResult
from cooperbench.agents.mini_swe_agent_v2.agents.default import DefaultAgent
from cooperbench.agents.mini_swe_agent_v2.config import get_config_path
from cooperbench.agents.mini_swe_agent_v2.connectors import GitConnector
from cooperbench.agents.mini_swe_agent_v2.connectors.messaging import MessagingConnector
from cooperbench.agents.mini_swe_agent_v2.models.litellm_model import LitellmModel
from cooperbench.agents.mini_swe_agent_v2.utils.serialize import recursive_merge
from cooperbench.agents.registry import register

logger = logging.getLogger(__name__)


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
        **kwargs,
    ) -> AgentResult:
        """Run mini-swe-agent v2 on a task."""
        # Load coop config when multiple agents, otherwise solo config.
        is_coop = bool(agents) and len(agents) > 1
        config_name = "coop" if is_coop else "solo"
        config_path = get_config_path(config_name)
        with open(config_path) as f:
            default_config = yaml.safe_load(f)

        # If the caller passed an agent_config YAML path, deep-merge its
        # `config:` block into the defaults.  This is what CooperBench's
        # ``--agent-config`` flag forwards to the adapter.
        if agent_config:
            try:
                with open(agent_config) as f:
                    overrides = yaml.safe_load(f) or {}
                default_config = recursive_merge(default_config, overrides.get("config", overrides))
            except FileNotFoundError:
                logger.error(f"agent_config file not found: {agent_config}")
            except Exception as e:
                logger.error(f"Error loading agent_config {agent_config}: {e}")

        # Deep-merge passed config overrides into default config so that partial
        # overrides (e.g. only agent.compaction_enabled) don't clobber sibling keys.
        if config is not None:
            default_config = recursive_merge(default_config, config)

        agent_cfg = default_config.get("agent", {})
        model_cfg = default_config.get("model", {})
        env_cfg = default_config.get("environment", {})
        backend = default_config.get("backend", "docker")

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

        # Setup messaging connector if enabled
        comm = None
        use_messaging = messaging_enabled and comm_url and agents and len(agents) > 1
        if use_messaging:
            comm = MessagingConnector(agent_id=agent_id, agents=agents, url=comm_url)

        # Register only the bash tool with the model.  send_message is
        # intercepted by DefaultAgent.execute_actions from inside the bash
        # command string (``send_message <recipient> <<'MSG' ... MSG``).
        # Exposing a separate send_message tool confuses smaller models
        # into alternating between tools unreliably.
        model = LitellmModel(model_name=model_name, **model_cfg)

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
        result = {}
        try:
            result = agent.run(task=task)
            status = result.get("exit_status", "Submitted")
        except Exception as e:
            status = "Error"
            error_msg = str(e)

        patch = ""
        try:
            r = env.execute({"command": "cat patch.txt 2>/dev/null"})
            if r.get("returncode") == 0:
                patch = (r.get("output") or "").strip()
        except Exception:
            pass

        # Save full trajectory (includes segments when compaction occurred)
        if log_dir and agent._compaction_count > 0:
            traj_path = Path(log_dir) / f"{agent_id}_full_traj.json"
            agent.save(traj_path)
            logger.info(
                f"[{agent_id}] Full trajectory with segments saved to {traj_path} "
                f"({agent._compaction_count} compaction(s))"
            )

        # Cleanup
        env.cleanup()

        # Tool-calling assistant turns leave content=None (the body lives in
        # tool_calls).  CooperBench's downstream conversation extractor does
        # ``"send_message" in content`` which raises TypeError on None — coerce
        # to "" before returning.
        sanitized_messages = []
        for msg in agent.messages:
            if msg.get("content") is None:
                msg = {**msg, "content": ""}
            sanitized_messages.append(msg)

        return AgentResult(
            status=status,
            patch=patch,
            cost=agent.cost,
            steps=agent.n_calls,
            messages=sanitized_messages,
            sent_messages=agent.sent_messages,
            error=error_msg,
        )
