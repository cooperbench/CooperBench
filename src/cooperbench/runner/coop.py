"""Coop mode execution - multiple agents collaborate on separate features."""

import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

import modal
import yaml

from cooperbench.agents import get_runner
from cooperbench.agents.mini_swe_agent.connectors import create_git_server
from cooperbench.config import ConfigManager
from cooperbench.utils import console, get_image_name


def execute_coop(
    repo_name: str,
    task_id: int,
    features: list[int],
    run_name: str,
    agent_name: str = "mini_swe_agent",
    model_name: str = "vertex_ai/gemini-3-flash-preview",
    redis_url: str = "redis://localhost:6379",
    force: bool = False,
    quiet: bool = False,
    git_enabled: bool = False,
    messaging_enabled: bool = True,
    backend: str = "modal",
    agent_config: str | None = None,
) -> dict | None:
    """Execute a cooperative task (two agents, separate features).

    Args:
        agent_config: Path to agent-specific configuration file (optional)
    """
    n_agents = len(features)
    agents = [f"agent{i + 1}" for i in range(n_agents)]
    run_id = uuid.uuid4().hex[:8]
    start_time = datetime.now()

    feature_str = "_".join(f"f{f}" for f in sorted(features))
    log_dir = Path("logs") / run_name / "coop" / repo_name / str(task_id) / feature_str
    result_file = log_dir / "result.json"

    if result_file.exists() and not force:
        with open(result_file) as f:
            prev_result = json.load(f)
        # Re-run if any agent had an error
        agents_had_error = any(a.get("status") == "Error" for a in prev_result.get("agents", {}).values())
        if not agents_had_error:
            return {"skipped": True, **prev_result}

    namespaced_redis = f"{redis_url}#run:{run_id}"

    # Create git server if enabled
    # Note: openhands_sdk manages its own git server internally, so we skip creation here
    git_server = None
    git_server_url = None
    git_network = None
    if git_enabled and agent_name != "openhands_sdk":
        if not quiet:
            console.print("  [dim]git[/dim] creating shared server...")
        app = modal.App.lookup("cooperbench", create_if_missing=True) if backend == "modal" else None

        # Build git server kwargs based on backend
        git_server_kwargs = {"backend": backend, "run_id": run_id, "app": app}
        if backend == "gcp":
            config = ConfigManager()
            if project_id := config.get("gcp_project_id"):
                git_server_kwargs["project_id"] = project_id
            if zone := config.get("gcp_zone"):
                git_server_kwargs["zone"] = zone

        git_server = create_git_server(**git_server_kwargs)
        git_server_url = git_server.url
        git_network = getattr(git_server, "network_name", None)
        if not quiet:
            console.print(f"  [dim]git[/dim] [green]ready[/green] {git_server_url}")

    results = {}
    threads = []

    def run_thread(agent_id: str, feature_id: int):
        try:
            results[agent_id] = _spawn_agent(
                repo_name=repo_name,
                task_id=task_id,
                feature_id=feature_id,
                agent_name=agent_name,
                model_name=model_name,
                agent_id=agent_id,
                agents=agents,
                redis_url=namespaced_redis if messaging_enabled and n_agents > 1 else None,
                git_server_url=git_server_url,
                git_enabled=git_enabled,
                git_network=git_network,
                messaging_enabled=messaging_enabled,
                quiet=quiet,
                backend=backend,
                agent_config=agent_config,
            )
        except Exception as e:
            results[agent_id] = {
                "feature_id": feature_id,
                "agent_id": agent_id,
                "status": "Error",
                "patch": "",
                "cost": 0,
                "steps": 0,
                "messages": [],
                "error": str(e),
            }

    try:
        # Sort features to ensure agent assignment matches sorted directory name
        sorted_features = sorted(features)
        for agent_id, feature_id in zip(agents, sorted_features):
            t = threading.Thread(target=run_thread, args=(agent_id, feature_id))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
    finally:
        # Cleanup git server
        if git_server:
            git_server.cleanup()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    total_cost = sum(r.get("cost", 0) for r in results.values())
    total_steps = sum(r.get("steps", 0) for r in results.values())

    # Save files
    log_dir.mkdir(parents=True, exist_ok=True)

    # Extract conversation (inter-agent messages)
    conversation = _extract_conversation(results, agents)

    # Sort by timestamp and dedupe (keep only sent messages, not received)
    sent_msgs = [m for m in conversation if not m.get("received")]
    sent_msgs.sort(key=lambda x: x.get("timestamp") or 0)

    # Save conversation
    with open(log_dir / "conversation.json", "w") as f:
        json.dump(sent_msgs, f, indent=2, default=str)

    for agent_id in agents:
        r = results[agent_id]
        fid = r["feature_id"]

        patch_file = log_dir / f"agent{fid}.patch"
        patch_file.write_text(r.get("patch", ""))

        traj_file = log_dir / f"agent{fid}_traj.json"
        with open(traj_file, "w") as f:
            json.dump(
                {
                    "repo": repo_name,
                    "task_id": task_id,
                    "feature_id": fid,
                    "agent_id": agent_id,
                    "model": model_name,
                    "status": r.get("status"),
                    "cost": r.get("cost"),
                    "steps": r.get("steps"),
                    "messages": r.get("messages", []),
                },
                f,
                indent=2,
                default=str,
            )

    result_data = {
        "repo": repo_name,
        "task_id": task_id,
        "features": sorted_features,
        "setting": "coop",
        "run_id": run_id,
        "run_name": run_name,
        "agent_framework": agent_name,
        "model": model_name,
        "started_at": start_time.isoformat(),
        "ended_at": end_time.isoformat(),
        "duration_seconds": duration,
        "agents": {
            agent_id: {
                "feature_id": r["feature_id"],
                "status": r.get("status"),
                "cost": r.get("cost", 0),
                "steps": r.get("steps", 0),
                "patch_lines": len(r.get("patch", "").splitlines()),
                "error": r.get("error"),
            }
            for agent_id, r in results.items()
        },
        "total_cost": total_cost,
        "total_steps": total_steps,
        "messages_sent": len(sent_msgs),
        "log_dir": str(log_dir),
    }

    with open(log_dir / "result.json", "w") as f:
        json.dump(result_data, f, indent=2)

    return {
        "results": results,
        "total_cost": total_cost,
        "total_steps": total_steps,
        "duration": duration,
        "run_id": run_id,
        "log_dir": str(log_dir),
    }


def _spawn_agent(
    repo_name: str,
    task_id: int,
    feature_id: int,
    agent_name: str,
    model_name: str,
    agent_id: str | None = None,
    agents: list[str] | None = None,
    redis_url: str | None = None,
    git_server_url: str | None = None,
    git_enabled: bool = False,
    git_network: str | None = None,
    messaging_enabled: bool = True,
    quiet: bool = False,
    backend: str = "modal",
    agent_config: str | None = None,
) -> dict:
    """Spawn a single agent on a feature using the agent framework adapter.

    Args:
        agent_config: Path to agent-specific configuration file (optional)
    """
    task_dir = Path("dataset") / repo_name / f"task{task_id}"
    feature_file = task_dir / f"feature{feature_id}" / "feature.md"

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    task = feature_file.read_text()
    image = get_image_name(repo_name, task_id)

    if not quiet:
        console.print(f"  [dim]{agent_id}[/dim] starting...")

    # Load agent config file if provided
    # run_id is passed for agents that need to coordinate shared infrastructure
    config = {"backend": backend, "run_id": redis_url.split("#run:")[1] if redis_url and "#run:" in redis_url else None}
    if git_network:
        config["git_network"] = git_network
    if agent_config:
        config_path = Path(agent_config)
        if config_path.exists():
            with open(config_path) as f:
                agent_config_dict = yaml.safe_load(f)
                if agent_config_dict:
                    config.update(agent_config_dict)
        else:
            raise FileNotFoundError(f"Agent config file not found: {agent_config}")

    # Use the agent framework adapter
    runner = get_runner(agent_name)
    result = runner.run(
        task=task,
        image=image,
        agent_id=agent_id or "agent",
        model_name=model_name,
        agents=agents,
        comm_url=redis_url,
        git_server_url=git_server_url,
        git_enabled=git_enabled,
        messaging_enabled=messaging_enabled,
        config=config,
    )

    return {
        "feature_id": feature_id,
        "agent_id": agent_id,
        "status": result.status,
        "patch": result.patch,
        "cost": result.cost,
        "steps": result.steps,
        "messages": result.messages,
        "sent_messages": result.sent_messages,  # For tool-based agents
        "error": result.error,
    }


def _extract_conversation(results: dict, agents: list[str]) -> list[dict]:
    """Extract inter-agent messages from results."""
    conversation = []

    for agent_id in agents:
        r = results[agent_id]
        fid = r["feature_id"]

        # Method 1: Check for sent_messages field (OpenHands SDK adapter)
        # This is the preferred method for tool-based agents
        for sent_msg in r.get("sent_messages", []):
            conversation.append(
                {
                    "from": agent_id,
                    "to": sent_msg.get("to", sent_msg.get("recipient")),
                    "message": sent_msg.get("message", sent_msg.get("content", "")),
                    "timestamp": sent_msg.get("timestamp"),
                    "feature_id": fid,
                }
            )

        # Method 2: Parse from messages list (mini_swe_agent bash commands + OpenHands events)
        for msg in r.get("messages", []):
            content = msg.get("content", "")
            ts = msg.get("timestamp")

            # Outgoing: agent sent a message via send_message command (bash format)
            if msg.get("role") == "assistant" and "send_message" in content:
                # Extract: send_message agentX "message"
                match = re.search(r'send_message\s+(\w+)\s+"([^"]+)"', content)
                if match:
                    to_agent, message = match.groups()
                    conversation.append(
                        {
                            "from": agent_id,
                            "to": to_agent,
                            "message": message,
                            "timestamp": ts,
                            "feature_id": fid,
                        }
                    )

            # Outgoing: OpenHands tool-based format (message_recipient/message_content fields)
            if msg.get("message_recipient") and msg.get("message_content"):
                conversation.append(
                    {
                        "from": agent_id,
                        "to": msg["message_recipient"],
                        "message": msg["message_content"],
                        "timestamp": ts,
                        "feature_id": fid,
                    }
                )

            # Incoming: received message from another agent
            if msg.get("role") == "user" and "[Message from" in content:
                match = re.search(r"\[Message from (\w+)\]:\s*(.+)", content)
                if match:
                    from_agent, message = match.groups()
                    conversation.append(
                        {
                            "from": from_agent,
                            "to": agent_id,
                            "message": message.strip(),
                            "timestamp": ts,
                            "feature_id": fid,
                            "received": True,
                        }
                    )

    return conversation
