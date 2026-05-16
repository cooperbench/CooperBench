"""Team-mode execution: N agents (one lead, the rest members) with a
pre-seeded shared task list and a shared scratchpad volume.

Sits beside ``execute_solo`` and ``execute_coop``.  Behaviorally the
nearest neighbor is coop — the same per-agent patch + per-agent
trajectory + Redis messaging — with three additions:

  1. A ``TaskListClient`` namespace pre-seeded with one task per
     feature.  Each task is pre-assigned to the agent expected to
     implement that feature, so members can find their work via
     ``coop-task-list --mine`` even if the lead never speaks first.
  2. A designated lead agent (the first one by id) gets ``team_role=
     "lead"``; the rest get ``"member"``.  Adapters pass this through
     to ``build_team_instruction`` so the lead and members see
     different prompt blocks.
  3. A named Docker volume (``cb-team-<run_id>``) mounted at
     ``/workspace/shared`` in every container.  Free coordination
     artifact; the prompt encourages agents to use it for partial
     diffs and design notes.

Post-run, the audit log on Redis is read and ``compute_metrics``
turns it into a small dict appended to ``result.json``.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

import modal
import redis
import yaml

from cooperbench.agents import get_runner
from cooperbench.agents._team import TaskListClient, compute_metrics
from cooperbench.agents.mini_swe_agent_v2.connectors import create_git_server
from cooperbench.config import ConfigManager
from cooperbench.runner.coop import _extract_conversation, _message_timestamp_key
from cooperbench.runner.tasks import DEFAULT_DATASET_DIR, DEFAULT_LOGS_DIR
from cooperbench.utils import console, get_image_name

TEAM_VOLUME_PREFIX = "cb-team"


def _redis_client(url: str) -> redis.Redis:
    """Factory so tests can monkeypatch this to return ``fakeredis``."""
    # Strip the ``#run:<id>`` fragment we use for messaging namespacing
    # before handing the URL to redis.from_url, which doesn't speak it.
    if "#" in url:
        url, _ = url.split("#", 1)
    return redis.from_url(url)


def execute_team(  # noqa: PLR0915  (long but linear; refactor lands in a follow-up)
    repo_name: str,
    task_id: int,
    features: list[int],
    run_name: str,
    agent_name: str = "claude_code",
    model_name: str = "claude-sonnet-4-6",
    redis_url: str = "redis://localhost:6379",
    force: bool = False,
    quiet: bool = False,
    git_enabled: bool = False,
    messaging_enabled: bool = True,
    backend: str = "docker",
    agent_config: str | None = None,
    dataset_dir: Path | str | None = None,
    logs_dir: Path | str | None = None,
) -> dict | None:
    """Execute a team-mode task.

    Same signature as ``execute_coop`` plus implicit lead/member role
    assignment.  Returns a dict with ``result`` (per-agent + metrics),
    ``total_cost``, ``total_steps``, ``duration``, ``run_id``,
    ``log_dir`` — matching the coop runner so the top-level core
    runner can treat them uniformly.
    """
    n_agents = len(features)
    agents = [f"agent{i + 1}" for i in range(n_agents)]
    run_id = uuid.uuid4().hex[:8]
    start_time = datetime.now()

    logs_root = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR
    feature_str = "_".join(f"f{f}" for f in sorted(features))
    log_dir = logs_root / run_name / "team" / repo_name / str(task_id) / feature_str
    result_file = log_dir / "result.json"

    if result_file.exists() and not force:
        with open(result_file) as f:
            prev_result = json.load(f)
        agents_had_error = any(a.get("status") == "Error" for a in prev_result.get("agents", {}).values())
        if not agents_had_error:
            return {"skipped": True, **prev_result}

    namespaced_redis = f"{redis_url}#run:{run_id}"
    team_volume = f"{TEAM_VOLUME_PREFIX}-{run_id}"

    # Pre-seed task list.
    task_list = None
    try:
        client = _redis_client(redis_url)
        task_list = TaskListClient(redis_client=client, run_id=run_id)
        sorted_features = sorted(features)
        for agent_id, feature_id in zip(agents[1:], sorted_features[1:]):
            # Members get pre-assigned tasks for the features they own.
            task_list.create(
                title=f"Implement feature {feature_id}",
                created_by="bench-runner",
                owner=agent_id,
                metadata={"feature_id": feature_id, "assigned_to": agent_id},
            )
        # Lead gets a meta-task to organize integration.
        task_list.create(
            title=f"Lead-only: integrate and submit feature {sorted_features[0]}",
            created_by="bench-runner",
            owner=agents[0],
            metadata={"feature_id": sorted_features[0], "assigned_to": agents[0], "lead_task": True},
        )
    except (redis.exceptions.RedisError, OSError) as e:
        if not quiet:
            console.print(f"  [yellow]task-list[/yellow] degraded: {e}")
        task_list = None

    git_server = None
    git_server_url = None
    git_network = None
    if git_enabled and agent_name != "openhands_sdk":
        if not quiet:
            console.print("  [dim]git[/dim] creating shared server...")
        app = modal.App.lookup("cooperbench", create_if_missing=True) if backend == "modal" else None
        git_server_kwargs = {"backend": backend, "run_id": run_id, "app": app}
        if backend == "gcp":
            cfg = ConfigManager()
            if pid := cfg.get("gcp_project_id"):
                git_server_kwargs["project_id"] = pid
            if zone := cfg.get("gcp_zone"):
                git_server_kwargs["zone"] = zone
        git_server = create_git_server(**git_server_kwargs)
        git_server_url = git_server.url
        git_network = getattr(git_server, "network_name", None)
        if not quiet:
            console.print(f"  [dim]git[/dim] [green]ready[/green] {git_server_url}")

    results: dict = {}
    threads = []

    def run_thread(agent_id: str, feature_id: int, role: str):
        try:
            results[agent_id] = _spawn_team_agent(
                repo_name=repo_name,
                task_id=task_id,
                feature_id=feature_id,
                agent_name=agent_name,
                model_name=model_name,
                agent_id=agent_id,
                agents=agents,
                team_role=role,
                team_id=run_id,
                team_volume=team_volume,
                task_list_url=namespaced_redis,
                redis_url=namespaced_redis if messaging_enabled and n_agents > 1 else None,
                git_server_url=git_server_url,
                git_enabled=git_enabled,
                git_network=git_network,
                messaging_enabled=messaging_enabled,
                quiet=quiet,
                backend=backend,
                agent_config=agent_config,
                run_name=run_name,
                features=features,
                dataset_dir=dataset_dir,
                logs_dir=logs_dir,
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
        sorted_features = sorted(features)
        for idx, (agent_id, feature_id) in enumerate(zip(agents, sorted_features)):
            role = "lead" if idx == 0 else "member"
            t = threading.Thread(target=run_thread, args=(agent_id, feature_id, role))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
    finally:
        if git_server:
            git_server.cleanup()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    total_cost = sum(r.get("cost", 0) for r in results.values())
    total_steps = sum(r.get("steps", 0) for r in results.values())

    log_dir.mkdir(parents=True, exist_ok=True)

    conversation = _extract_conversation(results, agents)
    sent_msgs = [m for m in conversation if not m.get("received")]
    sent_msgs.sort(key=_message_timestamp_key)
    with open(log_dir / "conversation.json", "w") as f:
        json.dump(sent_msgs, f, indent=2, default=str)

    # Per-agent patches + trajectories.
    for agent_id in agents:
        r = results[agent_id]
        fid = r["feature_id"]
        (log_dir / f"agent{fid}.patch").write_text(r.get("patch", ""))
        with open(log_dir / f"agent{fid}_traj.json", "w") as f:
            json.dump(
                {
                    "repo": repo_name,
                    "task_id": task_id,
                    "feature_id": fid,
                    "agent_id": agent_id,
                    "team_role": "lead" if agent_id == agents[0] else "member",
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

    # Harvest task list audit + metrics.
    metrics: dict | None = None
    if task_list is not None:
        try:
            events = task_list.log_events()
            final_tasks = task_list.list()
            metrics = compute_metrics(events, final_tasks=final_tasks)
            with open(log_dir / "task_log.json", "w") as f:
                json.dump(events, f, indent=2, default=str)
            with open(log_dir / "tasks.json", "w") as f:
                json.dump(final_tasks, f, indent=2, default=str)
        except (redis.exceptions.RedisError, OSError) as e:
            if not quiet:
                console.print(f"  [yellow]task-list[/yellow] harvest degraded: {e}")

    result_data = {
        "repo": repo_name,
        "task_id": task_id,
        "features": sorted_features,
        "setting": "team",
        "run_id": run_id,
        "run_name": run_name,
        "agent_framework": agent_name,
        "model": model_name,
        "started_at": start_time.isoformat(),
        "ended_at": end_time.isoformat(),
        "duration_seconds": duration,
        "lead_agent": agents[0],
        "agents": {
            agent_id: {
                "feature_id": r["feature_id"],
                "team_role": "lead" if agent_id == agents[0] else "member",
                "status": r.get("status"),
                "cost": r.get("cost", 0),
                "steps": r.get("steps", 0),
                "input_tokens": r.get("input_tokens", 0),
                "output_tokens": r.get("output_tokens", 0),
                "cache_read_tokens": r.get("cache_read_tokens", 0),
                "cache_write_tokens": r.get("cache_write_tokens", 0),
                "patch_lines": len(r.get("patch", "").splitlines()),
                "error": r.get("error"),
            }
            for agent_id, r in results.items()
        },
        "total_cost": total_cost,
        "total_steps": total_steps,
        "messages_sent": len(sent_msgs),
        "metrics": metrics or {},
        "log_dir": str(log_dir),
    }

    with open(log_dir / "result.json", "w") as f:
        json.dump(result_data, f, indent=2)

    return {
        "result": result_data,
        "results": result_data["agents"],
        "total_cost": total_cost,
        "total_steps": total_steps,
        "duration": duration,
        "run_id": run_id,
        "log_dir": str(log_dir),
    }


def _spawn_team_agent(
    *,
    repo_name: str,
    task_id: int,
    feature_id: int,
    agent_name: str,
    model_name: str,
    agent_id: str,
    agents: list[str],
    team_role: str,
    team_id: str,
    team_volume: str,
    task_list_url: str,
    redis_url: str | None,
    git_server_url: str | None,
    git_enabled: bool,
    git_network: str | None,
    messaging_enabled: bool,
    quiet: bool,
    backend: str,
    agent_config: str | None,
    run_name: str | None,
    features: list[int] | None,
    dataset_dir: Path | str | None,
    logs_dir: Path | str | None,
) -> dict:
    root = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    task_dir = root / repo_name / f"task{task_id}"
    logs_root = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR
    feature_file = task_dir / f"feature{feature_id}" / "feature.md"
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    task = feature_file.read_text()
    image = get_image_name(repo_name, task_id)

    log_dir_path = None
    if run_name and features:
        feature_str = "_".join(f"f{f}" for f in sorted(features))
        log_dir_path = str(logs_root / run_name / "team" / repo_name / str(task_id) / feature_str)

    if not quiet:
        console.print(f"  [dim]{agent_id}[/dim] [{team_role}] starting...")

    config = {
        "backend": backend,
        "run_id": run_id_from_url(task_list_url),
        "team_volume": team_volume,
    }
    if git_network:
        config["git_network"] = git_network
    if agent_config:
        config_path = Path(agent_config)
        if config_path.exists():
            with open(config_path) as f:
                agent_cfg = yaml.safe_load(f)
                if agent_cfg:
                    config.update(agent_cfg)
        else:
            raise FileNotFoundError(f"Agent config file not found: {agent_config}")

    runner = get_runner(agent_name)
    result = runner.run(
        task=task,
        image=image,
        agent_id=agent_id,
        model_name=model_name,
        agents=agents,
        comm_url=redis_url,
        git_server_url=git_server_url,
        git_enabled=git_enabled,
        messaging_enabled=messaging_enabled,
        config=config,
        agent_config=agent_config,
        log_dir=log_dir_path,
        team_role=team_role,
        team_id=team_id,
        task_list_url=task_list_url,
    )

    return {
        "feature_id": feature_id,
        "agent_id": agent_id,
        "team_role": team_role,
        "status": result.status,
        "patch": result.patch,
        "cost": result.cost,
        "steps": result.steps,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cache_read_tokens": result.cache_read_tokens,
        "cache_write_tokens": result.cache_write_tokens,
        "messages": result.messages,
        "error": result.error,
    }


def run_id_from_url(task_list_url: str) -> str | None:
    """Extract the ``#run:<id>`` fragment from a namespaced Redis URL."""
    if "#run:" in task_list_url:
        return task_list_url.split("#run:")[1]
    return None
