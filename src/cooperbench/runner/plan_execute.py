"""plan_execute mode — two-phase coop.

Phase 1 (plan): both agents see the feature spec and are instructed to plan
only, writing free-form ``plan.txt`` files. They have the full coop toolset
(messaging + git) so they can coordinate to avoid eventual merge conflicts.

Phase 2 (execute): two fresh agent containers. Each agent's task message is
**its own ``plan.txt`` verbatim** — no feature spec, no teammate plan, no
Phase 1 conversation log. They still have the full coop toolset and write
``patch.txt`` exactly like the default coop flow. Eval runs against Phase 2
patches only.

Shares all per-pair primitives with ``execute_coop`` via the helpers in
``coop.py``; this module is the orchestrator that wires the two phases.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from cooperbench.runner.coop import _run_pair_phase, setup_pair_infra
from cooperbench.runner.tasks import DEFAULT_LOGS_DIR


def execute_plan_execute(
    repo_name: str,
    task_id: int,
    features: list[int],
    run_name: str,
    agent_name: str = "openhands_sdk",
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
) -> dict | None:
    """Two-phase coop: plan → execute. Same signature as ``execute_coop``."""
    if agent_name != "openhands_sdk":
        raise NotImplementedError(
            f"--setting plan_execute is only supported with -a openhands_sdk "
            f"in v1 (got -a {agent_name}). The other adapters need to honour "
            f"config['submission_template'] and config['submission_artifact'] "
            f"first."
        )

    n_agents = len(features)
    agents = [f"agent{i + 1}" for i in range(n_agents)]
    sorted_features = sorted(features)
    run_id = uuid.uuid4().hex[:8]
    start_time = datetime.now()

    logs_root = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR
    feature_str = "_".join(f"f{f}" for f in sorted_features)
    pair_log_dir = logs_root / run_name / "plan_execute" / repo_name / str(task_id) / feature_str
    phase1_log_dir = pair_log_dir / "phase1"
    result_file = pair_log_dir / "result.json"

    if result_file.exists() and not force:
        with open(result_file) as f:
            prev_result = json.load(f)
        agents_had_error = any(a.get("status") == "Error" for a in prev_result.get("agents", {}).values())
        if not agents_had_error:
            return {"skipped": True, **prev_result}

    # Build the plan-phase submission template once. Coop-aware so the
    # template references the teammate when there's more than one agent.
    from cooperbench.agents.openhands_agent_sdk.adapter import (
        _plan_submission_instructions,
    )

    is_coop = messaging_enabled and n_agents > 1
    plan_template = _plan_submission_instructions(is_coop=is_coop)

    namespaced_redis, git_server, git_server_url, git_network = setup_pair_infra(
        redis_url=redis_url,
        run_id=run_id,
        agent_name=agent_name,
        backend=backend,
        git_enabled=git_enabled,
        quiet=quiet,
    )

    try:
        # ─── Phase 1: plan ───────────────────────────────────────────────
        phase1 = _run_pair_phase(
            repo_name=repo_name,
            task_id=task_id,
            features=features,
            agents=agents,
            agent_name=agent_name,
            model_name=model_name,
            redis_url=namespaced_redis,
            git_server_url=git_server_url,
            git_enabled=git_enabled,
            git_network=git_network,
            messaging_enabled=messaging_enabled,
            backend=backend,
            agent_config=agent_config,
            run_name=run_name,
            dataset_dir=dataset_dir,
            logs_dir=logs_dir,
            log_dir=phase1_log_dir,
            quiet=quiet,
            extra_config={
                "submission_template": plan_template,
                "submission_artifact": "plan.txt",
            },
            artifact_suffix="plan",
        )

        # Persist phase 1's own result.json for inspection
        _write_phase_result(
            log_dir=phase1_log_dir,
            phase_name="plan",
            repo_name=repo_name,
            task_id=task_id,
            sorted_features=sorted_features,
            run_id=run_id,
            run_name=run_name,
            agent_name=agent_name,
            model_name=model_name,
            phase=phase1,
        )

        # The plan content for each agent lives in result.patch (the adapter
        # cat's whatever ``submission_artifact`` it was told to read).
        plan_per_agent = {agent_id: r.get("patch") or "" for agent_id, r in phase1["results"].items()}

        # If a plan came back empty, the executor will get an empty task.
        # Warn loudly — but don't abort; let the run produce evidence.
        for agent_id, plan in plan_per_agent.items():
            if not plan.strip():
                from cooperbench.utils import console

                console.print(f"  [yellow]warning[/yellow] phase 1 produced empty plan for {agent_id}")

        # ─── Phase 2: execute ────────────────────────────────────────────
        phase2 = _run_pair_phase(
            repo_name=repo_name,
            task_id=task_id,
            features=features,
            agents=agents,
            agent_name=agent_name,
            model_name=model_name,
            redis_url=namespaced_redis,
            git_server_url=git_server_url,
            git_enabled=git_enabled,
            git_network=git_network,
            messaging_enabled=messaging_enabled,
            backend=backend,
            agent_config=agent_config,
            run_name=run_name,
            dataset_dir=dataset_dir,
            logs_dir=logs_dir,
            log_dir=pair_log_dir,
            quiet=quiet,
            task_override_per_agent=plan_per_agent,
            # No extra_config → adapter falls back to the default patch.txt
            # submission template and reads patch.txt.
            artifact_suffix="patch",
        )
    finally:
        if git_server:
            git_server.cleanup()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Combined cost/steps roll up both phases. Agent-level fields come from
    # phase 2 (the executor — that's what eval looks at), but cost / steps
    # / tokens sum both phases.
    p1_results = phase1["results"]
    p2_results = phase2["results"]

    def combined_agent(agent_id: str) -> dict:
        p1 = p1_results.get(agent_id, {})
        p2 = p2_results.get(agent_id, {})
        return {
            "feature_id": p2.get("feature_id"),
            "status": p2.get("status"),
            "cost": (p1.get("cost", 0) or 0) + (p2.get("cost", 0) or 0),
            "steps": (p1.get("steps", 0) or 0) + (p2.get("steps", 0) or 0),
            "input_tokens": (p1.get("input_tokens", 0) or 0) + (p2.get("input_tokens", 0) or 0),
            "output_tokens": (p1.get("output_tokens", 0) or 0) + (p2.get("output_tokens", 0) or 0),
            "cache_read_tokens": (p1.get("cache_read_tokens", 0) or 0) + (p2.get("cache_read_tokens", 0) or 0),
            "cache_write_tokens": (p1.get("cache_write_tokens", 0) or 0) + (p2.get("cache_write_tokens", 0) or 0),
            "patch_lines": len((p2.get("patch", "") or "").splitlines()),
            "plan_lines": len((p1.get("patch", "") or "").splitlines()),
            "error": p2.get("error") or p1.get("error"),
            "phase1_cost": p1.get("cost", 0),
            "phase2_cost": p2.get("cost", 0),
            "phase1_steps": p1.get("steps", 0),
            "phase2_steps": p2.get("steps", 0),
        }

    total_cost = phase1["total_cost"] + phase2["total_cost"]
    total_steps = phase1["total_steps"] + phase2["total_steps"]

    result_data = {
        "repo": repo_name,
        "task_id": task_id,
        "features": sorted_features,
        "setting": "plan_execute",
        "run_id": run_id,
        "run_name": run_name,
        "agent_framework": agent_name,
        "model": model_name,
        "started_at": start_time.isoformat(),
        "ended_at": end_time.isoformat(),
        "duration_seconds": duration,
        "agents": {agent_id: combined_agent(agent_id) for agent_id in agents},
        "total_cost": total_cost,
        "total_steps": total_steps,
        "messages_sent": len(phase2["conversation"]),
        "phase1_messages_sent": len(phase1["conversation"]),
        "log_dir": str(pair_log_dir),
    }

    pair_log_dir.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)

    return {
        "results": p2_results,
        "total_cost": total_cost,
        "total_steps": total_steps,
        "duration": duration,
        "run_id": run_id,
        "log_dir": str(pair_log_dir),
    }


def _write_phase_result(
    *,
    log_dir: Path,
    phase_name: str,
    repo_name: str,
    task_id: int,
    sorted_features: list[int],
    run_id: str,
    run_name: str,
    agent_name: str,
    model_name: str,
    phase: dict,
) -> None:
    """Per-phase result.json so each phase's cost / steps / statuses are
    introspectable on disk without parsing the top-level combined result."""
    log_dir.mkdir(parents=True, exist_ok=True)
    results = phase["results"]
    data = {
        "repo": repo_name,
        "task_id": task_id,
        "features": sorted_features,
        "setting": "plan_execute",
        "phase": phase_name,
        "run_id": run_id,
        "run_name": run_name,
        "agent_framework": agent_name,
        "model": model_name,
        "agents": {
            agent_id: {
                "feature_id": r.get("feature_id"),
                "status": r.get("status"),
                "cost": r.get("cost", 0),
                "steps": r.get("steps", 0),
                "input_tokens": r.get("input_tokens", 0),
                "output_tokens": r.get("output_tokens", 0),
                "cache_read_tokens": r.get("cache_read_tokens", 0),
                "cache_write_tokens": r.get("cache_write_tokens", 0),
                "artifact_lines": len((r.get("patch", "") or "").splitlines()),
                "error": r.get("error"),
            }
            for agent_id, r in results.items()
        },
        "total_cost": phase["total_cost"],
        "total_steps": phase["total_steps"],
        "messages_sent": len(phase["conversation"]),
    }
    with open(log_dir / "result.json", "w") as f:
        json.dump(data, f, indent=2)
