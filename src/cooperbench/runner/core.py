"""Core runner for benchmark task execution."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from cooperbench.infra.redis import ensure_redis
from cooperbench.runner.coop import execute_coop
from cooperbench.runner.solo import execute_solo
from cooperbench.runner.tasks import discover_tasks
from cooperbench.utils import console

load_dotenv()

os.environ["MSWEA_SILENT_STARTUP"] = "1"
os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"


def run(
    run_name: str,
    subset: str | None = None,
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
        subset: Use a predefined subset (e.g., 'lite')
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
    tasks = discover_tasks(subset=subset, repo_filter=repo, task_filter=task_id, features_filter=features)

    if not tasks:
        console.print("[yellow]no tasks found[/yellow]")
        return

    bench_start_time = time.time()
    is_single = len(tasks) == 1
    is_solo = setting == "solo"

    _print_header(
        run_name, setting, tasks, agent, model_name, concurrency, is_single, is_solo, git_enabled, messaging_enabled
    )

    # Solo mode doesn't need Redis or git server
    if not is_solo:
        if messaging_enabled:
            ensure_redis(redis_url)

    log_dir = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    _save_config(log_dir, run_name, agent, model_name, setting, concurrency, len(tasks))

    results_list = []
    completed = 0
    failed = 0
    skipped = 0
    total_cost = 0

    def execute_task(task_info):
        if is_solo:
            return execute_solo(
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
            return execute_coop(
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
                console.print("[dim]→ skip[/dim] (already completed)")
            else:
                completed = 1
                total_cost = result.get("total_cost", 0)
                _print_single_result(result, tasks[0], is_solo)
    else:
        # Multiple tasks - show progress
        completed, skipped, failed, total_cost, results_list = _run_with_progress(tasks, execute_task, concurrency)

    # Summary
    total_time = time.time() - bench_start_time
    _save_summary(log_dir, run_name, len(tasks), completed, skipped, failed, total_cost, total_time, results_list)
    _print_summary(completed, skipped, failed, total_cost, total_time, log_dir / setting)


def _print_header(
    run_name: str,
    setting: str,
    tasks: list,
    agent: str,
    model_name: str,
    concurrency: int,
    is_single: bool,
    is_solo: bool,
    git_enabled: bool,
    messaging_enabled: bool,
) -> None:
    """Print run header information."""
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


def _save_config(
    log_dir: Path, run_name: str, agent: str, model_name: str, setting: str, concurrency: int, total_tasks: int
) -> None:
    """Save run configuration."""
    run_config = {
        "run_name": run_name,
        "agent_framework": agent,
        "model": model_name,
        "setting": setting,
        "concurrency": concurrency,
        "total_tasks": total_tasks,
        "started_at": datetime.now().isoformat(),
    }
    with open(log_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)


def _print_single_result(result: dict, task: dict, is_solo: bool) -> None:
    """Print detailed result for a single task."""
    total_cost = result.get("total_cost", 0)

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
            ",".join(str(f) for f in task["features"]),
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


def _run_with_progress(tasks: list, execute_task, concurrency: int) -> tuple:
    """Run multiple tasks with progress display."""
    results_list = []
    completed = 0
    failed = 0
    skipped = 0
    total_cost = 0

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

                    status_display = {
                        "done": "[green]✓ done[/green]",
                        "skip": "[dim]→ done[/dim]",
                        "failed": "[red]✗ failed[/red]",
                    }[status]
                    progress.console.print(f"{status_display} {task_name} [dim][{feat_str}][/dim]")

                except Exception as e:
                    failed += 1
                    results_list.append({"task": f"{task_name}/{feat_str}", "status": "error", "error": str(e)})
                    progress.console.print(f"[red]✗ error[/red] {task_name} [dim]{e}[/dim]")

                progress.update(task_progress, advance=1)

    return completed, skipped, failed, total_cost, results_list


def _save_summary(
    log_dir: Path,
    run_name: str,
    total_tasks: int,
    completed: int,
    skipped: int,
    failed: int,
    total_cost: float,
    total_time: float,
    results_list: list,
) -> None:
    """Save run summary."""
    summary = {
        "run_name": run_name,
        "completed_at": datetime.now().isoformat(),
        "total_tasks": total_tasks,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "total_cost": total_cost,
        "total_time_seconds": total_time,
        "results": results_list,
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def _print_summary(
    completed: int, skipped: int, failed: int, total_cost: float, total_time: float, log_dir: Path
) -> None:
    """Print run summary."""
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
    console.print()
    console.print(f"[dim]logs:[/dim] {log_dir}")
