"""SWE-agent adapter for CooperBench.

This adapter wraps the SWE-agent framework to conform to the
AgentRunner interface used by CooperBench.

SWE-agent 1.0+ supports Modal execution through SWE-ReX.
"""

import os

os.environ.setdefault("SWE_AGENT_LOG_STREAM_LEVEL", "ERROR")
os.environ.setdefault("SWE_REX_LOG_STREAM_LEVEL", "ERROR")

import asyncio
import tempfile
from pathlib import Path

from cooperbench.agents import AgentResult
from cooperbench.agents.registry import register


@register("swe_agent")
class SweAgentRunner:
    """Adapter for SWE-agent framework."""

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
        """Run SWE-agent on a task.

        Args:
            task: The task description
            image: Docker image with the codebase
            agent_id: Unique identifier for this agent
            model_name: LLM model to use (passed to litellm)
            agents: List of all agent IDs (for collaboration)
            comm_url: Redis URL for inter-agent messaging
            git_server_url: Git server URL (ignored for now)
            git_enabled: Whether git collaboration is enabled (ignored for now)
            messaging_enabled: Whether messaging is enabled
            config: Optional SWE-agent configuration overrides

        Returns:
            AgentResult with status, patch, cost, steps, messages
        """
        import litellm
        import modal
        import swerex.deployment.modal as swerex_modal
        import yaml
        from swerex.deployment import get_deployment
        from swerex.deployment.config import ModalDeploymentConfig

        # Tell litellm to drop unsupported params (e.g., top_p for some models)
        litellm.drop_params = True

        # Import from vendored SWE-agent code
        from cooperbench.agents.swe_agent import CONFIG_DIR
        from cooperbench.agents.swe_agent.agent.agents import (
            DefaultAgentConfig,
            get_agent_from_config,
        )
        from cooperbench.agents.swe_agent.agent.models import GenericAPIModelConfig
        from cooperbench.agents.swe_agent.agent.problem_statement import TextProblemStatement
        from cooperbench.agents.swe_agent.environment.swe_env import SWEEnv

        # Monkey-patch SWE-ReX to clear entrypoint (required for CooperBench images)
        _original_from_registry = swerex_modal._ImageBuilder.from_registry

        def _patched_from_registry(self, image_name: str) -> modal.Image:
            result = _original_from_registry(self, image_name)
            return result.entrypoint([])  # Clear entrypoint

        swerex_modal._ImageBuilder.from_registry = _patched_from_registry

        # Determine if we're in collaboration mode
        is_coop = (messaging_enabled or git_enabled) and agents and len(agents) > 1

        # Setup messaging connector if in collaboration mode
        comm = None
        if is_coop and messaging_enabled and comm_url:
            from cooperbench.agents.mini_swe_agent.connectors.messaging import MessagingConnector

            comm = MessagingConnector(agent_id=agent_id, agents=agents, url=comm_url)

        # Setup git connector if enabled
        git_connector = None
        if is_coop and git_enabled and git_server_url:
            from cooperbench.agents.swe_agent.connectors.git import GitConnector

            git_connector = GitConnector(agent_id=agent_id, agents=agents, server_url=git_server_url)

        # Load config - use coop.yaml for collaboration, default.yaml otherwise
        config_file = "coop.yaml" if is_coop else "default.yaml"
        config_path = CONFIG_DIR / config_file
        with open(config_path) as f:
            default_config = yaml.safe_load(f)

        # Get agent config from default, merge with overrides
        agent_yaml_config = default_config.get("agent", {})
        agent_config = config or {}

        # Customize templates for CooperBench (hardcode working_dir since we use repo=None)
        working_dir = "/workspace/repo"
        if "templates" in agent_yaml_config:
            templates = agent_yaml_config["templates"]
            # Replace {{working_dir}} with actual path (Jinja2 can't handle this one)
            for key in ["system_template", "instance_template"]:
                if key in templates and templates[key]:
                    templates[key] = templates[key].replace("{{working_dir}}", working_dir)

        # Set ROOT in registry_variables (written to /root/.swe-agent-env, required by submit tool)
        if "tools" in agent_yaml_config:
            if "registry_variables" not in agent_yaml_config["tools"]:
                agent_yaml_config["tools"]["registry_variables"] = {}
            agent_yaml_config["tools"]["registry_variables"]["ROOT"] = working_dir

        # Gemini doesn't support cache_control with function calling, so remove it
        if "gemini" in model_name.lower():
            if "history_processors" in agent_yaml_config:
                agent_yaml_config["history_processors"] = [
                    hp for hp in agent_yaml_config["history_processors"]
                    if hp.get("type") != "cache_control"
                ]

        # Configure the model (SWE-agent uses litellm internally)
        model_config = GenericAPIModelConfig(
            name=model_name,
            per_instance_cost_limit=agent_config.get("cost_limit", 0.5),
            total_cost_limit=agent_config.get("total_cost_limit", 0.0),
            temperature=agent_config.get("temperature", 0.0),
        )

        # Update the YAML config with our model
        agent_yaml_config["model"] = model_config

        # Create agent config from YAML (includes templates, tools, etc.)
        agent_cfg = DefaultAgentConfig(**agent_yaml_config)

        # Configure Modal deployment via SWE-ReX
        deployment_config = ModalDeploymentConfig(
            image=image,
            deployment_timeout=agent_config.get("timeout", 3600),
            runtime_timeout=agent_config.get("runtime_timeout", 300),
        )
        deployment = get_deployment(deployment_config)

        # Create the agent with full config (templates, tools, etc.)
        # Pass collaboration parameters if in coop mode
        agent = get_agent_from_config(
            agent_cfg,
            comm=comm,
            agent_id=agent_id,
            agents=agents or [],
            git_enabled=git_enabled and git_server_url is not None,
        )

        # Create problem statement
        problem = TextProblemStatement(text=task)

        # Create environment - repo=None since the repo already exists in the image
        # We set working_dir via environment variable for the template
        env = SWEEnv(
            deployment=deployment,
            repo=None,
            post_startup_commands=[
                # Set ROOT env var (required by SWE-agent's submit tool)
                "export ROOT=/workspace/repo",
                # Set git config globally
                "git config --global user.email 'agent@cooperbench.dev'",
                "git config --global user.name 'CooperBench Agent'",
                # Initialize git repo in /workspace/repo (needed for SWE-agent's submit tool)
                "cd /workspace/repo && git init && git add -A && git commit -m 'Initial commit' || true",
            ],
        )

        # Run the agent
        error_msg = None
        status = "Submitted"
        patch = ""
        messages: list[dict] = []
        cost = 0.0
        steps = 0

        # Use temp dir for SWE-agent's internal .traj files (we save our own)
        with tempfile.TemporaryDirectory() as traj_dir:
            try:
                # Start the environment
                env.start()

                # Capture base commit for patch generation (before any agent changes)
                base_commit = env.communicate(
                    f"cd {working_dir} && git rev-parse HEAD", timeout=10
                ).strip()

                # Setup git collaboration if enabled
                if git_connector:
                    git_connector.setup(env, working_dir=working_dir)

                # Set working_dir for autosubmission (since repo=None)
                agent.working_dir = working_dir

                # Run agent on the problem
                result = agent.run(
                    problem_statement=problem,
                    env=env,
                    output_dir=Path(traj_dir),
                )

                # Extract results - handle both dict and object access
                info = getattr(result, "info", {})
                if isinstance(info, dict):
                    exit_status = info.get("exit_status", "")
                    model_stats = info.get("model_stats", {})
                else:
                    exit_status = getattr(info, "exit_status", "")
                    model_stats = getattr(info, "model_stats", {})

                if exit_status == "submitted":
                    status = "Submitted"
                elif exit_status in ("max_steps", "cost_limit"):
                    status = "LimitsExceeded"
                elif exit_status:
                    status = "Error"
                    error_msg = f"Exit status: {exit_status}"

                # Generate patch ourselves (works even when changes are committed for git collab)
                # This diffs from base commit to current HEAD + working tree
                try:
                    # First add any uncommitted changes to staging
                    env.communicate(f"cd {working_dir} && git add -A", timeout=10)
                    # Diff from base to HEAD (committed changes) + staged changes
                    patch = env.communicate(
                        f"cd {working_dir} && git diff {base_commit}", timeout=30
                    ).strip()
                except Exception:
                    # Fall back to submission from SWE-agent if our method fails
                    if isinstance(info, dict):
                        patch = info.get("submission", "") or ""
                    else:
                        patch = getattr(info, "submission", "") or ""

                # Get cost/steps from model_stats
                if isinstance(model_stats, dict):
                    cost = model_stats.get("instance_cost", 0.0)
                    steps = model_stats.get("api_calls", 0)
                else:
                    cost = getattr(model_stats, "instance_cost", 0.0)
                    steps = getattr(model_stats, "api_calls", 0)

                # Get trajectory for messages
                trajectory = getattr(result, "trajectory", []) or []
                for step in trajectory:
                    if isinstance(step, dict):
                        # Assistant message: combine thought + action (response may be empty with function_calling)
                        thought = step.get("thought", "")
                        action = step.get("action", "")
                        response = step.get("response", "")
                        assistant_content = response or f"{thought}\n{action}".strip()
                        if assistant_content:
                            messages.append({"role": "assistant", "content": assistant_content})
                        # User message: observation
                        if "observation" in step:
                            messages.append({"role": "user", "content": str(step["observation"])})
                    else:
                        thought = getattr(step, "thought", "")
                        action = getattr(step, "action", "")
                        response = getattr(step, "response", "")
                        assistant_content = response or f"{thought}\n{action}".strip()
                        if assistant_content:
                            messages.append({"role": "assistant", "content": assistant_content})
                        if hasattr(step, "observation"):
                            messages.append({"role": "user", "content": str(step.observation)})

            except Exception as e:
                status = "Error"
                error_msg = str(e)

            finally:
                # Cleanup
                try:
                    env.close()
                except Exception:
                    pass
                try:
                    asyncio.run(deployment.stop())
                except RuntimeError:
                    # Event loop already running - schedule it
                    asyncio.get_event_loop().create_task(deployment.stop())
                except Exception:
                    pass

        return AgentResult(
            status=status,
            patch=patch,
            cost=cost,
            steps=steps,
            messages=messages,
            error=error_msg,
        )
