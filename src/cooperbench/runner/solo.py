"""Solo mode execution - one agent implements multiple features."""

import json
import uuid
from datetime import datetime
from pathlib import Path

from cooperbench.agents import get_runner
from cooperbench.utils import console, get_image_name


def execute_solo(
    repo_name: str,
    task_id: int,
    features: list[int],
    run_name: str,
    agent_name: str = "mini_swe_agent",
    model_name: str = "gemini/gemini-3-flash-preview",
    force: bool = False,
    quiet: bool = False,
    backend: str = "modal",
    agent_config: str | None = None,
) -> dict | None:
    """Execute a solo task (one agent, multiple features).

    Args:
        agent_config: Path to agent-specific configuration file (optional)
    """
    run_id = uuid.uuid4().hex[:8]
    start_time = datetime.now()

    feature_str = "_".join(f"f{f}" for f in sorted(features))
    log_dir = Path("logs") / run_name / "solo" / repo_name / str(task_id) / feature_str
    result_file = log_dir / "result.json"

    if result_file.exists() and not force:
        with open(result_file) as f:
            prev_result = json.load(f)
        # Re-run if previous result was an error
        if prev_result.get("agent", {}).get("status") != "Error":
            return {"skipped": True, **prev_result}

    try:
        result = _spawn_solo_agent(
            repo_name=repo_name,
            task_id=task_id,
            features=features,
            agent_name=agent_name,
            model_name=model_name,
            quiet=quiet,
            backend=backend,
            agent_config=agent_config,
        )
    except Exception as e:
        result = {
            "features": features,
            "agent_id": "solo",
            "status": "Error",
            "patch": "",
            "cost": 0,
            "steps": 0,
            "messages": [],
            "error": str(e),
        }

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Save files
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save patch
    patch_file = log_dir / "solo.patch"
    patch_file.write_text(result.get("patch", ""))

    # Save trajectory
    traj_file = log_dir / "solo_traj.json"
    with open(traj_file, "w") as f:
        json.dump(
            {
                "repo": repo_name,
                "task_id": task_id,
                "features": features,
                "agent_id": "solo",
                "model": model_name,
                "status": result.get("status"),
                "cost": result.get("cost"),
                "steps": result.get("steps"),
                "messages": result.get("messages", []),
            },
            f,
            indent=2,
            default=str,
        )

    result_data = {
        "repo": repo_name,
        "task_id": task_id,
        "features": features,
        "setting": "solo",
        "run_id": run_id,
        "run_name": run_name,
        "agent_framework": agent_name,
        "model": model_name,
        "started_at": start_time.isoformat(),
        "ended_at": end_time.isoformat(),
        "duration_seconds": duration,
        "agent": {
            "status": result.get("status"),
            "cost": result.get("cost", 0),
            "steps": result.get("steps", 0),
            "patch_lines": len(result.get("patch", "").splitlines()),
            "error": result.get("error"),
        },
        "total_cost": result.get("cost", 0),
        "total_steps": result.get("steps", 0),
        "log_dir": str(log_dir),
    }

    with open(log_dir / "result.json", "w") as f:
        json.dump(result_data, f, indent=2)

    return {
        "result": result,
        "total_cost": result.get("cost", 0),
        "total_steps": result.get("steps", 0),
        "duration": duration,
        "run_id": run_id,
        "log_dir": str(log_dir),
    }


def _spawn_solo_agent(
    repo_name: str,
    task_id: int,
    features: list[int],
    agent_name: str,
    model_name: str,
    quiet: bool = False,
    backend: str = "modal",
    agent_config: str | None = None,
) -> dict:
    """Spawn a single agent on multiple features (solo mode).

    Args:
        agent_config: Path to agent-specific configuration file (optional)
    """
    task_dir = Path("dataset") / repo_name / f"task{task_id}"

    # Combine feature specs
    combined_task = []
    for fid in features:
        feature_file = task_dir / f"feature{fid}" / "feature.md"
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")
        combined_task.append(f"## Feature {fid}\n\n{feature_file.read_text()}")

    task = "\n\n---\n\n".join(combined_task)
    image = get_image_name(repo_name, task_id)

    if not quiet:
        console.print("  [dim]solo[/dim] starting...")

    # Use the agent framework adapter
    runner = get_runner(agent_name)
    result = runner.run(
        task=task,
        image=image,
        agent_id="solo",
        model_name=model_name,
        # Solo mode: no collaboration
        agents=None,
        comm_url=None,
        git_server_url=None,
        git_enabled=False,
        messaging_enabled=False,
        config={"backend": backend},
        agent_config=agent_config,
    )

    return {
        "features": features,
        "agent_id": "solo",
        "status": result.status,
        "patch": result.patch,
        "cost": result.cost,
        "steps": result.steps,
        "messages": result.messages,
        "error": result.error,
    }
