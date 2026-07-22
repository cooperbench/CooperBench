"""Oracle-mode runners — inject ground-truth solutions before spawning agents.

``execute_oracle_coop``  — two agents, each receives their own solution.
``execute_oracle_solo``  — one agent receives both solutions.
``execute_oracle_coop_full`` — two agents, each receives BOTH solutions
                               (information upper-bound within a two-agent setting).

All three produce the same result.json / patch files as the standard runners so
that the existing evaluator (``cooperbench.eval``) works without modification.
They add an extra ``oracle`` key to result.json that records which mode was used.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

import yaml

from cooperbench.agents import get_runner
from cooperbench.agents.mini_swe_agent_v2.connectors import create_git_server
from cooperbench.config import ConfigManager
from cooperbench.oracle.prompt import OracleMode, build_oracle_task
from cooperbench.runner.coop import _extract_conversation, _message_timestamp_key
from cooperbench.runner.tasks import DEFAULT_DATASET_DIR, DEFAULT_LOGS_DIR
from cooperbench.utils import console, get_image_name


def execute_oracle_coop(
    repo_name: str,
    task_id: int,
    features: list[int],
    run_name: str,
    agent_name: str = "mini_swe_agent_v2",
    model_name: str = "vertex_ai/gemini-3-flash-preview",
    redis_url: str = "redis://localhost:6379",
    force: bool = False,
    quiet: bool = False,
    git_enabled: bool = False,
    messaging_enabled: bool = True,
    backend: str = "docker",
    agent_config: str | None = None,
    dataset_dir: Path | str | None = None,
    logs_dir: Path | str | None = None,
    oracle_mode: OracleMode = OracleMode.PATCH,
    oracle_full: bool = False,
) -> dict | None:
    """Execute a cooperative task where each agent receives its own ground-truth solution.

    This is the primary oracle mode.  Compared to ``execute_coop``, each agent's task
    string is prefixed with the corresponding ground-truth patch from
    ``dataset/<repo>/<task>/feature<id>/feature.patch``.

    Args:
        oracle_mode:  How the solution is presented (patch / code / intent).
        oracle_full:  If True, each agent receives BOTH solutions (oracle_coop_full).
                      Default False: each agent only sees its own solution.
        Other args:   Same as ``execute_coop``.
    """
    n_agents = len(features)
    agents = [f"agent{i + 1}" for i in range(n_agents)]
    run_id = uuid.uuid4().hex[:8]
    start_time = datetime.now()

    logs_root = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR
    feature_str = "_".join(f"f{f}" for f in sorted(features))
    setting_name = "oracle_coop_full" if oracle_full else "oracle_coop"
    log_dir = logs_root / run_name / setting_name / repo_name / str(task_id) / feature_str
    result_file = log_dir / "result.json"

    if result_file.exists() and not force:
        with open(result_file) as f:
            prev_result = json.load(f)
        agents_had_error = any(a.get("status") == "Error" for a in prev_result.get("agents", {}).values())
        if not agents_had_error:
            return {"skipped": True, **prev_result}

    namespaced_redis = f"{redis_url}#run:{run_id}"

    # Create git server if enabled (same logic as execute_coop)
    git_server = None
    git_server_url = None
    git_network = None
    if git_enabled and agent_name != "openhands_sdk":
        if not quiet:
            console.print("  [dim]git[/dim] creating shared server...")
        app = None
        import modal

        app = modal.App.lookup("cooperbench", create_if_missing=True) if backend == "modal" else None

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
            results[agent_id] = _spawn_oracle_agent(
                repo_name=repo_name,
                task_id=task_id,
                feature_id=feature_id,
                all_features=features,
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
                run_name=run_name,
                features=features,
                dataset_dir=dataset_dir,
                logs_dir=logs_dir,
                oracle_mode=oracle_mode,
                oracle_full=oracle_full,
                setting_name=setting_name,
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
        for agent_id, feature_id in zip(agents, sorted_features):
            t = threading.Thread(target=run_thread, args=(agent_id, feature_id))
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
        "log_dir": str(log_dir),
        "oracle": {
            "mode": oracle_mode.value,
            "full": oracle_full,
        },
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


def execute_oracle_solo(
    repo_name: str,
    task_id: int,
    features: list[int],
    run_name: str,
    agent_name: str = "mini_swe_agent_v2",
    model_name: str = "vertex_ai/gemini-3-flash-preview",
    force: bool = False,
    quiet: bool = False,
    backend: str = "docker",
    agent_config: str | None = None,
    dataset_dir: Path | str | None = None,
    logs_dir: Path | str | None = None,
    oracle_mode: OracleMode = OracleMode.PATCH,
) -> dict | None:
    """Execute a solo task where the agent receives ground-truth patches for ALL features.

    This is the oracle upper-bound: one agent, full information.
    AUC here represents the ceiling achievable when implementation is solved but
    coordination is trivially free (single agent).

    Args:
        oracle_mode:  How solutions are presented.
        Other args:   Same as ``execute_solo``.
    """
    run_id = uuid.uuid4().hex[:8]
    start_time = datetime.now()

    logs_root = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR
    feature_str = "_".join(f"f{f}" for f in sorted(features))
    log_dir = logs_root / run_name / "oracle_solo" / repo_name / str(task_id) / feature_str
    result_file = log_dir / "result.json"

    if result_file.exists() and not force:
        with open(result_file) as f:
            prev_result = json.load(f)
        if prev_result.get("agent", {}).get("status") != "Error":
            return {"skipped": True, **prev_result}

    try:
        result = _spawn_oracle_solo_agent(
            repo_name=repo_name,
            task_id=task_id,
            features=features,
            agent_name=agent_name,
            model_name=model_name,
            quiet=quiet,
            backend=backend,
            agent_config=agent_config,
            run_name=run_name,
            dataset_dir=dataset_dir,
            logs_dir=logs_dir,
            oracle_mode=oracle_mode,
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

    log_dir.mkdir(parents=True, exist_ok=True)

    patch_file = log_dir / "solo.patch"
    patch_file.write_text(result.get("patch", ""))

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
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "cache_read_tokens": result.get("cache_read_tokens", 0),
            "cache_write_tokens": result.get("cache_write_tokens", 0),
            "patch_lines": len(result.get("patch", "").splitlines()),
            "error": result.get("error"),
        },
        "total_cost": result.get("cost", 0),
        "total_steps": result.get("steps", 0),
        "log_dir": str(log_dir),
        "oracle": {
            "mode": oracle_mode.value,
        },
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _spawn_oracle_agent(
    repo_name: str,
    task_id: int,
    feature_id: int,
    all_features: list[int],
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
    backend: str = "docker",
    agent_config: str | None = None,
    run_name: str | None = None,
    features: list[int] | None = None,
    dataset_dir: Path | str | None = None,
    logs_dir: Path | str | None = None,
    oracle_mode: OracleMode = OracleMode.PATCH,
    oracle_full: bool = False,
    setting_name: str = "oracle_coop",
) -> dict:
    """Spawn a single oracle-mode coop agent."""
    root = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    task_dir = root / repo_name / f"task{task_id}"
    logs_root = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR

    feature_file = task_dir / f"feature{feature_id}" / "feature.md"
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    base_task = feature_file.read_text()

    if oracle_full:
        # Inject own solution first, then append partner solution as context.
        own_feature_dir = task_dir / f"feature{feature_id}"
        task = build_oracle_task(base_task, own_feature_dir, mode=oracle_mode)

        # Append partner solution(s) as additional context (read-only reference).
        partner_ids = [f for f in sorted(all_features) if f != feature_id]
        if partner_ids:
            partner_blocks = []
            for pid in partner_ids:
                partner_dir = task_dir / f"feature{pid}"
                partner_base = (partner_dir / "feature.md").read_text() if (partner_dir / "feature.md").exists() else ""
                partner_task = build_oracle_task(partner_base, partner_dir, mode=oracle_mode)
                partner_blocks.append(f"## Partner agent's solution (feature {pid})\n\n{partner_task}")
            task = task + "\n\n" + "\n\n".join(partner_blocks)
    else:
        own_feature_dir = task_dir / f"feature{feature_id}"
        task = build_oracle_task(base_task, own_feature_dir, mode=oracle_mode)

    image = get_image_name(repo_name, task_id)

    log_dir_path = None
    if run_name and features:
        feature_str = "_".join(f"f{f}" for f in sorted(features))
        log_dir_path = str(logs_root / run_name / setting_name / repo_name / str(task_id) / feature_str)

    if not quiet:
        console.print(f"  [dim]{agent_id}[/dim] starting (oracle/{oracle_mode.value})...")

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
        agent_config=agent_config,
        log_dir=log_dir_path,
    )

    return {
        "feature_id": feature_id,
        "agent_id": agent_id,
        "status": result.status,
        "patch": result.patch,
        "cost": result.cost,
        "steps": result.steps,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cache_read_tokens": result.cache_read_tokens,
        "cache_write_tokens": result.cache_write_tokens,
        "messages": result.messages,
        "sent_messages": result.sent_messages,
        "error": result.error,
    }


def _spawn_oracle_solo_agent(
    repo_name: str,
    task_id: int,
    features: list[int],
    agent_name: str,
    model_name: str,
    quiet: bool = False,
    backend: str = "docker",
    agent_config: str | None = None,
    run_name: str | None = None,
    dataset_dir: Path | str | None = None,
    logs_dir: Path | str | None = None,
    oracle_mode: OracleMode = OracleMode.PATCH,
) -> dict:
    """Spawn a single oracle-mode solo agent (receives all solutions)."""
    root = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    task_dir = root / repo_name / f"task{task_id}"
    logs_root = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR

    # Combine feature specs, each injected with its own oracle solution.
    combined_parts = []
    for fid in features:
        feature_file = task_dir / f"feature{fid}" / "feature.md"
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")
        base = feature_file.read_text()
        feature_dir = task_dir / f"feature{fid}"
        oracle_task = build_oracle_task(base, feature_dir, mode=oracle_mode)
        combined_parts.append(f"## Feature {fid}\n\n{oracle_task}")

    task = "\n\n---\n\n".join(combined_parts)
    image = get_image_name(repo_name, task_id)

    log_dir_path = None
    if run_name:
        feature_str = "_".join(f"f{f}" for f in sorted(features))
        log_dir_path = str(logs_root / run_name / "oracle_solo" / repo_name / str(task_id) / feature_str)

    if not quiet:
        console.print(f"  [dim]oracle_solo[/dim] starting ({oracle_mode.value})...")

    config = {"backend": backend}
    if agent_config:
        config_path = Path(agent_config)
        if config_path.exists():
            with open(config_path) as f:
                agent_config_dict = yaml.safe_load(f)
                if agent_config_dict:
                    config.update(agent_config_dict)
        else:
            raise FileNotFoundError(f"Agent config file not found: {agent_config}")

    runner = get_runner(agent_name)
    result = runner.run(
        task=task,
        image=image,
        agent_id="solo",
        model_name=model_name,
        agents=None,
        comm_url=None,
        git_server_url=None,
        git_enabled=False,
        messaging_enabled=False,
        config=config,
        agent_config=agent_config,
        log_dir=log_dir_path,
    )

    return {
        "features": features,
        "agent_id": "solo",
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
