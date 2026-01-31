"""Run benchmark tasks.

This module runs agents on CooperBench tasks. Each task involves implementing
features in a codebase, with tests validating the implementation.

Two modes:
    coop: Two agents work on separate features in parallel, communicating
          via Redis. Their patches are later merged and tested together.
    solo: One agent implements both features, producing a single patch.

Directory structure for logs:
    logs/{run_name}/{setting}/{repo}/{task_id}/{features}/
        - result.json: Run metadata (timing, cost, status)
        - agent{N}.patch or solo.patch: Generated patches
        - agent{N}_traj.json or solo_traj.json: Full agent trajectories
        - conversation.json: Inter-agent messages (coop mode only)

Key functions:
    run: Main entry point - runs tasks with progress display
    discover_tasks: Find tasks from dataset/ directory
"""

import json
import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

os.environ["MSWEA_SILENT_STARTUP"] = "1"
os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"

# Imports below must be after env var setup  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table  # noqa: E402

# Import agent framework components
from cooperbench.agents import get_runner  # noqa: E402
from cooperbench.agents.mini_swe_agent.connectors.git import GitServer  # noqa: E402
from cooperbench.utils import console, get_image_name  # noqa: E402


def run(
    run_name: str,
    repo: str | None = None,
    task_id: int | None = None,
    features: list[int] | None = None,
    model_name: str = "gemini/gemini-3-flash-preview",
    agent: str = "mini_swe_agent",
    concurrency: int = 20,
    force: bool = False,
    redis_url: str = "redis://localhost:6379",
    setting: str = "coop",
    git_enabled: bool = False,
    messaging_enabled: bool = True,
) -> None:
    """Run benchmark tasks.

    Args:
        run_name: Experiment name (used for log directory)
        repo: Filter by repository (e.g., "llama_index_task")
        task_id: Filter by specific task ID
        features: Specific feature pair [f1, f2] to run
        model_name: LLM model (e.g., "gpt-4o", "gemini/gemini-3-flash-preview")
        agent: Agent framework to use (default: "mini_swe")
        concurrency: Max parallel tasks
        force: Rerun even if results exist
        redis_url: Redis URL for agent communication (coop mode)
        setting: "coop" (2 agents) or "solo" (1 agent)
        git_enabled: Enable git collaboration (agents can push/pull/merge)
        messaging_enabled: Enable messaging (send_message command)
    """
    tasks = discover_tasks(repo_filter=repo, task_filter=task_id, features_filter=features)

    if not tasks:
        console.print("[yellow]no tasks found[/yellow]")
        return

    bench_start_time = time.time()
    is_single = len(tasks) == 1
    is_solo = setting == "solo"

    # Build tools string for display
    tools = []
    if messaging_enabled:
        tools.append("messaging")
    if git_enabled:
        tools.append("git")
    tools_str = ", ".join(tools) if tools else "none"

    console.print()
    console.print(f"[bold]cooperbench[/bold] [dim]{run_name}[/dim] [cyan]({setting})[/cyan]")
    if is_single:
        t = tasks[0]
        console.print(f"[dim]task:[/dim] {t['repo']}/{t['task_id']} [dim]features:[/dim] {t['features']}")
    else:
        console.print(f"[dim]tasks:[/dim] {len(tasks)} [dim]concurrency:[/dim] {concurrency}")
    console.print(f"[dim]agent:[/dim] {agent}")
    console.print(f"[dim]model:[/dim] {model_name}")
    if not is_solo:
        console.print(f"[dim]tools:[/dim] {tools_str}")
    console.print()

    # Solo mode doesn't need Redis or git server
    if not is_solo:
        if messaging_enabled:
            _ensure_redis(redis_url)

    log_dir = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_name": run_name,
        "agent_framework": agent,
        "model": model_name,
        "setting": setting,
        "concurrency": concurrency,
        "total_tasks": len(tasks),
        "started_at": datetime.now().isoformat(),
    }
    with open(log_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    results_list = []
    completed = 0
    failed = 0
    skipped = 0
    total_cost = 0

    def execute_task(task_info):
        if is_solo:
            return _execute_solo(
                repo_name=task_info["repo"],
                task_id=task_info["task_id"],
                features=task_info["features"],
                run_name=run_name,
                agent_name=agent,
                model_name=model_name,
                force=force,
                quiet=not is_single,
            )
        else:
            return _execute_coop(
                repo_name=task_info["repo"],
                task_id=task_info["task_id"],
                features=task_info["features"],
                run_name=run_name,
                agent_name=agent,
                model_name=model_name,
                redis_url=redis_url,
                force=force,
                quiet=not is_single,
                git_enabled=git_enabled,
                messaging_enabled=messaging_enabled,
            )

    if is_single:
        # Single task - show detailed output
        result = execute_task(tasks[0])
        if result:
            if result.get("skipped"):
                skipped = 1
                console.print("  [dim]skipped (already completed)[/dim]")
            else:
                completed = 1
                total_cost = result.get("total_cost", 0)

                # Show results table
                console.print()
                table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
                table.add_column("agent")
                table.add_column("feature")
                table.add_column("status")
                table.add_column("cost", justify="right")
                table.add_column("steps", justify="right")
                table.add_column("lines", justify="right")

                if is_solo:
                    r = result.get("result", {})
                    status = r.get("status", "Error")
                    status_style = "green" if status == "Submitted" else "red"
                    table.add_row(
                        "solo",
                        ",".join(str(f) for f in tasks[0]["features"]),
                        f"[{status_style}]{status}[/{status_style}]",
                        f"${r.get('cost', 0):.2f}",
                        str(r.get("steps", 0)),
                        str(len(r.get("patch", "").splitlines())),
                    )
                else:
                    for agent_id, r in result.get("results", {}).items():
                        status = r.get("status", "Error")
                        status_style = "green" if status == "Submitted" else "red"
                        table.add_row(
                            agent_id,
                            str(r.get("feature_id", "?")),
                            f"[{status_style}]{status}[/{status_style}]",
                            f"${r.get('cost', 0):.2f}",
                            str(r.get("steps", 0)),
                            str(len(r.get("patch", "").splitlines())),
                        )

                console.print(table)
                console.print()
                console.print(f"[dim]total:[/dim] ${total_cost:.2f} [dim]time:[/dim] {result.get('duration', 0):.0f}s")
    else:
        # Multiple tasks - show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]eta[/dim]"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_progress = progress.add_task("running", total=len(tasks))

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_task = {executor.submit(execute_task, t): t for t in tasks}

                for future in as_completed(future_to_task):
                    task_info = future_to_task[future]
                    feat_str = ",".join(str(f) for f in task_info["features"])
                    task_name = f"{task_info['repo']}/{task_info['task_id']}"

                    try:
                        result = future.result()
                        if result is None:
                            failed += 1
                            status = "failed"
                            cost = 0
                        elif result.get("skipped"):
                            skipped += 1
                            status = "skip"
                            cost = result.get("total_cost", 0)
                        else:
                            completed += 1
                            cost = result.get("total_cost", 0)
                            status = "done"

                        total_cost += cost
                        results_list.append({"task": f"{task_name}/{feat_str}", "status": status, "cost": cost})

                        status_color = {"done": "green", "skip": "dim", "failed": "red"}[status]
                        progress.console.print(
                            f"  [{status_color}]{status}[/{status_color}] {task_name} [dim][{feat_str}][/dim]"
                        )

                    except Exception as e:
                        failed += 1
                        results_list.append({"task": f"{task_name}/{feat_str}", "status": "error", "error": str(e)})
                        progress.console.print(f"  [red]error[/red] {task_name} [dim]{e}[/dim]")

                    progress.update(task_progress, advance=1)

    # Summary
    total_time = time.time() - bench_start_time
    summary = {
        "run_name": run_name,
        "completed_at": datetime.now().isoformat(),
        "total_tasks": len(tasks),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "total_cost": total_cost,
        "total_time_seconds": total_time,
        "results": results_list,
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    console.print()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("completed", f"[green]{completed}[/green]")
    if skipped:
        table.add_row("skipped", f"[dim]{skipped}[/dim]")
    if failed:
        table.add_row("failed", f"[red]{failed}[/red]")
    table.add_row("cost", f"${total_cost:.2f}")

    # Format time nicely
    mins, secs = divmod(int(total_time), 60)
    if mins > 0:
        table.add_row("time", f"{mins}m {secs}s")
    else:
        table.add_row("time", f"{secs}s")

    console.print(table)
    # Show the actual output directory (with setting subdirectory)
    actual_log_dir = log_dir / setting
    console.print()
    console.print(f"[dim]logs:[/dim] {actual_log_dir}")


def discover_tasks(
    repo_filter: str | None = None,
    task_filter: int | None = None,
    features_filter: list[int] | None = None,
) -> list[dict]:
    """Discover benchmark tasks from dataset/.

    Args:
        repo_filter: Filter by repository name
        task_filter: Filter by task ID
        features_filter: Specific feature pair to use

    Returns:
        List of task dicts with repo, task_id, features
    """
    dataset_dir = Path("dataset")
    tasks = []

    for repo_dir in sorted(dataset_dir.iterdir()):
        if not repo_dir.is_dir() or repo_dir.name == "README.md":
            continue
        if repo_filter and repo_filter != repo_dir.name:
            continue

        for task_dir in sorted(repo_dir.iterdir()):
            if not task_dir.is_dir() or not task_dir.name.startswith("task"):
                continue

            task_id = int(task_dir.name.replace("task", ""))
            if task_filter and task_filter != task_id:
                continue

            feature_ids = []
            for feature_dir in sorted(task_dir.iterdir()):
                if feature_dir.is_dir() and feature_dir.name.startswith("feature"):
                    fid = int(feature_dir.name.replace("feature", ""))
                    feature_ids.append(fid)

            if len(feature_ids) < 2:
                continue

            if features_filter:
                if all(f in feature_ids for f in features_filter):
                    tasks.append(
                        {
                            "repo": repo_dir.name,
                            "task_id": task_id,
                            "features": features_filter,
                        }
                    )
            else:
                # All pairwise combinations: nC2
                feature_ids.sort()
                for f1, f2 in combinations(feature_ids, 2):
                    tasks.append(
                        {
                            "repo": repo_dir.name,
                            "task_id": task_id,
                            "features": [f1, f2],
                        }
                    )

    return tasks


# === Internal functions ===


def _ensure_redis(redis_url: str = "redis://localhost:6379") -> None:
    """Ensure Redis is running, auto-start via Docker if needed."""
    import redis as redis_lib

    client = redis_lib.from_url(redis_url)
    try:
        client.ping()
        console.print("  [dim]redis[/dim] [green]connected[/green]")
        return
    except redis_lib.ConnectionError:
        pass

    console.print("  [dim]redis[/dim] [yellow]starting via docker...[/yellow]")
    try:
        subprocess.run(
            ["docker", "run", "-d", "--name", "cooperbench-redis", "-p", "6379:6379", "redis:alpine"],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(["docker", "start", "cooperbench-redis"], capture_output=True)
    except FileNotFoundError:
        console.print("[red]error:[/red] Docker not found. Install Docker or Redis.")
        raise SystemExit(1)

    for _ in range(10):
        time.sleep(0.5)
        try:
            client.ping()
            console.print("  [dim]redis[/dim] [green]started[/green]")
            return
        except redis_lib.ConnectionError:
            pass

    console.print("[red]error:[/red] Failed to start Redis")
    raise SystemExit(1)


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
    messaging_enabled: bool = True,
    quiet: bool = False,
) -> dict:
    """Spawn a single agent on a feature using the agent framework adapter."""

    task_dir = Path("dataset") / repo_name / f"task{task_id}"
    feature_file = task_dir / f"feature{feature_id}" / "feature.md"

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    task = feature_file.read_text()
    image = get_image_name(repo_name, task_id)

    if not quiet:
        console.print(f"  [dim]{agent_id}[/dim] starting...")

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
    )

    return {
        "feature_id": feature_id,
        "agent_id": agent_id,
        "status": result.status,
        "patch": result.patch,
        "cost": result.cost,
        "steps": result.steps,
        "messages": result.messages,
        "error": result.error,
    }


def _spawn_solo_agent(
    repo_name: str,
    task_id: int,
    features: list[int],
    agent_name: str,
    model_name: str,
    quiet: bool = False,
) -> dict:
    """Spawn a single agent on multiple features (solo mode) using the agent framework adapter."""

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


def _execute_solo(
    repo_name: str,
    task_id: int,
    features: list[int],
    run_name: str,
    agent_name: str = "mini_swe_agent",
    model_name: str = "gemini/gemini-3-flash-preview",
    force: bool = False,
    quiet: bool = False,
) -> dict | None:
    """Execute a solo task (one agent, multiple features)."""

    run_id = uuid.uuid4().hex[:8]
    start_time = datetime.now()

    feature_str = "_".join(f"f{f}" for f in sorted(features))
    log_dir = Path("logs") / run_name / "solo" / repo_name / str(task_id) / feature_str
    result_file = log_dir / "result.json"

    if result_file.exists() and not force:
        with open(result_file) as f:
            return {"skipped": True, **json.load(f)}

    try:
        result = _spawn_solo_agent(
            repo_name=repo_name,
            task_id=task_id,
            features=features,
            agent_name=agent_name,
            model_name=model_name,
            quiet=quiet,
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


def _execute_coop(
    repo_name: str,
    task_id: int,
    features: list[int],
    run_name: str,
    agent_name: str = "mini_swe_agent",
    model_name: str = "gemini/gemini-3-flash-preview",
    redis_url: str = "redis://localhost:6379",
    force: bool = False,
    quiet: bool = False,
    git_enabled: bool = False,
    messaging_enabled: bool = True,
) -> dict | None:
    """Execute a cooperative task (two agents, separate features)."""
    import modal

    n_agents = len(features)
    agents = [f"agent{i + 1}" for i in range(n_agents)]
    run_id = uuid.uuid4().hex[:8]
    start_time = datetime.now()

    feature_str = "_".join(f"f{f}" for f in sorted(features))
    log_dir = Path("logs") / run_name / "coop" / repo_name / str(task_id) / feature_str
    result_file = log_dir / "result.json"

    if result_file.exists() and not force:
        with open(result_file) as f:
            return {"skipped": True, **json.load(f)}

    namespaced_redis = f"{redis_url}#run:{run_id}"

    # Create git server if enabled
    git_server = None
    git_server_url = None
    if git_enabled:
        if not quiet:
            console.print("  [dim]git[/dim] creating shared server...")
        app = modal.App.lookup("cooperbench", create_if_missing=True)
        git_server = GitServer.create(app=app, run_id=run_id)
        git_server_url = git_server.url
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
                messaging_enabled=messaging_enabled,
                quiet=quiet,
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
    conversation = []
    for agent_id in agents:
        r = results[agent_id]
        fid = r["feature_id"]
        for msg in r.get("messages", []):
            content = msg.get("content", "")
            ts = msg.get("timestamp")

            # Outgoing: agent sent a message via send_message command
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


__all__ = ["run", "discover_tasks"]
