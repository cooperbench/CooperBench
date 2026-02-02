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

# Optional import for cleanup handler (may not exist in all versions)
try:
    from cooperbench.agents.mini_swe_agent.environments.modal import install_cleanup_handler
except ImportError:
    install_cleanup_handler = None
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
    auto_eval: bool = True,
    eval_concurrency: int = 10,
    backend: str = "modal",
    agent_config: str | None = None,
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
        auto_eval: Automatically evaluate runs after completion
        eval_concurrency: Max parallel evaluations (default: 10)
        backend: Execution backend ("modal" or "docker")
        agent_config: Path to agent-specific configuration file (optional)
    """
    # Install cleanup handler to terminate Modal sandboxes on Ctrl+C
    if install_cleanup_handler:
        install_cleanup_handler()

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
                backend=backend,
                agent_config=agent_config,
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
                backend=backend,
                agent_config=agent_config,
            )

    eval_stats = None
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
            # Evaluate single task if auto_eval enabled (runs for skipped too, _evaluate_single handles existing evals)
            if auto_eval:
                run_info = _build_run_info(result, tasks[0], setting, run_name)
                if run_info:
                    from cooperbench.eval.evaluate import _evaluate_single

                    eval_result = _evaluate_single(run_info, force=force)
                    if eval_result:
                        stats = _process_eval_result(eval_result, tasks[0])
                        if stats:
                            eval_passed, eval_failed, eval_errors, eval_skipped = stats
                            eval_stats = (eval_passed, eval_failed, eval_errors, eval_skipped, [])
                        else:
                            eval_stats = None
    else:
        # Multiple tasks - show progress
        completed, skipped, failed, total_cost, results_list, eval_stats = _run_with_progress(
            tasks, execute_task, concurrency, auto_eval, eval_concurrency, setting, run_name, force
        )

    # Summary
    session_time = time.time() - bench_start_time
    _save_summary(log_dir, run_name, len(tasks), completed, skipped, failed, total_cost, session_time, results_list)

    # Get aggregate totals from all result.json files (includes previous sessions)
    from cooperbench.utils import get_run_totals

    run_totals = get_run_totals(run_name, setting)

    # Use session time if no skipped (exact), otherwise aggregate (approximate)
    time_info = {
        "wall": session_time if skipped == 0 else run_totals["wall_time"],
        "run": run_totals["run_time"],
        "approximate": skipped > 0,
    }

    _print_summary(completed, skipped, failed, run_totals["total_cost"], time_info, log_dir / setting, eval_stats)


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


def _build_run_info(result: dict, task_info: dict, setting: str, run_name: str) -> dict | None:
    """Build run_info dict for evaluation from run result."""
    log_dir = result.get("log_dir")
    if not log_dir:
        # Reconstruct log_dir for older results that don't have it
        feature_str = "_".join(f"f{f}" for f in sorted(task_info["features"]))
        log_dir = str(Path("logs") / run_name / setting / task_info["repo"] / str(task_info["task_id"]) / feature_str)

    return {
        "log_dir": log_dir,
        "setting": setting,
        "repo": task_info["repo"],
        "task_id": task_info["task_id"],
        "features": task_info["features"],
    }


def _process_eval_result(eval_result: dict | None, task_info: dict) -> tuple | None:
    """Process eval result and return stats tuple (passed, failed, errors, skipped).

    For skipped evals (eval.json already exists), we read the actual result
    so pass/fail counts reflect the true state across all tasks.
    """
    if not eval_result:
        return None

    # For skipped evals, extract actual pass/fail from the loaded data
    is_skipped = eval_result.get("skipped", False)

    if eval_result.get("error"):
        return (0, 0, 1, 0)
    elif eval_result.get("both_passed"):
        return (1, 0, 0, 1 if is_skipped else 0)
    else:
        return (0, 1, 0, 1 if is_skipped else 0)


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


def _run_with_progress(
    tasks: list,
    execute_task,
    concurrency: int,
    auto_eval: bool,
    eval_concurrency: int,
    setting: str,
    run_name: str,
    force: bool,
) -> tuple:
    """Run multiple tasks with progress display and optional inline evaluation."""
    from cooperbench.eval.evaluate import _evaluate_single

    results_list = []
    completed = 0
    failed = 0
    skipped = 0
    total_cost = 0

    # Eval tracking
    eval_executor = None
    eval_futures = {}  # {future: (task_info, result, task_name, feat_str)}
    eval_passed = 0
    eval_failed = 0
    eval_errors = 0
    eval_skipped = 0
    eval_results = []

    if auto_eval:
        eval_executor = ThreadPoolExecutor(max_workers=eval_concurrency)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
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

                        # Submit eval if enabled (for done or skipped - _evaluate_single handles existing evals)
                        if auto_eval and status in ("done", "skip") and eval_executor:
                            run_info = _build_run_info(result, task_info, setting, run_name)
                            if run_info:
                                eval_future = eval_executor.submit(_evaluate_single, run_info, force)
                                eval_futures[eval_future] = (task_info, result, task_name, feat_str)
                            progress.console.print(f"{status_display} {task_name} [dim][{feat_str}][/dim]")
                        else:
                            progress.console.print(f"{status_display} {task_name} [dim][{feat_str}][/dim]")

                        # Check for completed evals (non-blocking)
                        if eval_futures:
                            completed_evals = [f for f in list(eval_futures.keys()) if f.done()]
                            for eval_future in completed_evals:
                                task_info, result, task_name, feat_str = eval_futures.pop(eval_future)
                                try:
                                    eval_result = eval_future.result()
                                    eval_stats = _process_eval_result(eval_result, task_info)
                                    if eval_stats:
                                        ep, ef, ee, es = eval_stats[:4]
                                        eval_passed += ep
                                        eval_failed += ef
                                        eval_errors += ee
                                        eval_skipped += es

                                        # Store eval result
                                        eval_results.append(
                                            {
                                                "task": f"{task_name}/{feat_str}",
                                                "status": "pass"
                                                if eval_result.get("both_passed")
                                                else "fail"
                                                if not eval_result.get("error")
                                                else "error",
                                            }
                                        )

                                        # Print eval result indented to show it's a test result
                                        # For skipped evals (already existed), show actual result with dim indicator
                                        is_skipped = eval_result.get("skipped", False)
                                        if eval_result.get("error"):
                                            eval_status = "[yellow]✗ error[/yellow]"
                                        elif eval_result.get("both_passed"):
                                            eval_status = "[dim]→ pass[/dim]" if is_skipped else "[green]✓ pass[/green]"
                                        else:
                                            eval_status = "[dim]→ fail[/dim]" if is_skipped else "[red]✗ fail[/red]"

                                        progress.console.print(f"  {eval_status} {task_name} [dim][{feat_str}][/dim]")
                                except Exception as e:
                                    eval_errors += 1
                                    progress.console.print(f"  → [yellow]✗ eval error[/yellow] [dim]{e}[/dim]")

                    except Exception as e:
                        failed += 1
                        results_list.append({"task": f"{task_name}/{feat_str}", "status": "error", "error": str(e)})
                        progress.console.print(f"[red]✗ error[/red] {task_name} [dim]{e}[/dim]")

                    progress.update(task_progress, advance=1)

            # Wait for all remaining evals to complete
            if eval_futures:
                for eval_future in as_completed(eval_futures.keys()):
                    task_info, result, task_name, feat_str = eval_futures.pop(eval_future)
                    try:
                        eval_result = eval_future.result()
                        eval_stats = _process_eval_result(eval_result, task_info)
                        if eval_stats:
                            ep, ef, ee, es = eval_stats[:4]
                            eval_passed += ep
                            eval_failed += ef
                            eval_errors += ee
                            eval_skipped += es

                            # Store eval result
                            eval_results.append(
                                {
                                    "task": f"{task_name}/{feat_str}",
                                    "status": "pass"
                                    if eval_result.get("both_passed")
                                    else "fail"
                                    if not eval_result.get("error")
                                    else "error",
                                }
                            )

                            # For skipped evals (already existed), show actual result with dim indicator
                            is_skipped = eval_result.get("skipped", False)
                            if eval_result.get("error"):
                                eval_status = "[yellow]✗ error[/yellow]"
                            elif eval_result.get("both_passed"):
                                eval_status = "[dim]→ pass[/dim]" if is_skipped else "[green]✓ pass[/green]"
                            else:
                                eval_status = "[dim]→ fail[/dim]" if is_skipped else "[red]✗ fail[/red]"

                            progress.console.print(f"  {eval_status} {task_name} [dim][{feat_str}][/dim]")
                    except Exception as e:
                        eval_errors += 1
                        progress.console.print(f"  → [yellow]✗ eval error[/yellow] {task_name} [dim]{e}[/dim]")

    finally:
        if eval_executor:
            eval_executor.shutdown(wait=True)

    eval_stats = (eval_passed, eval_failed, eval_errors, eval_skipped, eval_results) if auto_eval else None
    return completed, skipped, failed, total_cost, results_list, eval_stats


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
    completed: int,
    skipped: int,
    failed: int,
    total_cost: float,
    time_info: dict,
    log_dir: Path,
    eval_stats: tuple | None = None,
) -> None:
    """Print run summary with optional eval stats."""
    console.print()

    # Build run stats line
    run_parts = [f"[green]{completed}[/green] completed"]
    if skipped:
        run_parts.append(f"[dim]{skipped}[/dim] skipped")
    if failed:
        run_parts.append(f"[red]{failed}[/red] failed")
    run_line = f"runs:  {', '.join(run_parts)}"

    # Build eval stats line if available
    eval_line = None
    if eval_stats:
        eval_passed, eval_failed, eval_errors, eval_skipped, _ = eval_stats
        # total_eval counts actual pass/fail/error (skipped evals have their results counted in pass/fail)
        total_eval = eval_passed + eval_failed + eval_errors
        if total_eval > 0:
            eval_parts = [f"{total_eval} evaluated"]
            if eval_passed > 0:
                eval_parts.append(f"[green]{eval_passed}[/green] passed")
            if eval_failed > 0:
                eval_parts.append(f"[red]{eval_failed}[/red] failed")
            if eval_errors > 0:
                eval_parts.append(f"[yellow]{eval_errors}[/yellow] errors")

            # Calculate pass rate from all evaluated (pass + fail)
            pass_rate = eval_passed / max(total_eval, 1) * 100
            eval_line = f"evals: {', '.join(eval_parts)} ({pass_rate:.1f}%)"

    # Format time nicely
    def fmt_time(seconds: float) -> str:
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

    wall_str = fmt_time(time_info["wall"])
    run_str = fmt_time(time_info["run"])

    # Add ~ prefix if time is approximate (aggregated across sessions)
    if time_info.get("approximate"):
        wall_str = f"~{wall_str}"

    # Print summary
    console.print(run_line)
    if eval_line:
        console.print(eval_line)
    console.print(f"cost:  ${total_cost:.2f}")
    console.print(f"time:  {wall_str} [dim](agent: {run_str})[/dim]")
    console.print()
    console.print(f"[dim]logs:[/dim] {log_dir}")
