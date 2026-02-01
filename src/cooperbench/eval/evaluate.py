"""Evaluation harness for benchmark runs."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from cooperbench.eval.runs import discover_runs
from cooperbench.eval.sandbox import _sanitize_patch, test_merged, test_solo
from cooperbench.utils import console


def evaluate(
    run_name: str,
    subset: str | None = None,
    repo: str | None = None,
    task_id: int | None = None,
    features: list[int] | None = None,
    concurrency: int = 10,
    force: bool = False,
    backend: str = "modal",
) -> None:
    """Evaluate completed runs.

    Args:
        run_name: Name of the run to evaluate
        subset: Filter to a predefined subset (e.g., 'lite')
        repo: Filter by repository name
        task_id: Filter by task ID
        features: Specific feature pair to evaluate
        concurrency: Number of parallel evaluations
        force: Force re-evaluation even if eval.json exists
        backend: Execution backend ("modal", "docker", "gcp")
    """
    runs = discover_runs(
        run_name=run_name,
        subset=subset,
        repo_filter=repo,
        task_filter=task_id,
        features_filter=features,
    )

    if not runs:
        console.print("[yellow]no runs found to evaluate[/yellow]")
        return

    # Filter already-evaluated runs if not forcing
    if not force:
        original_count = len(runs)
        runs = [r for r in runs if not (Path(r["log_dir"]) / "eval.json").exists()]
        skipped_count = original_count - len(runs)
        if skipped_count > 0:
            console.print(f"[dim]skipping {skipped_count} already evaluated[/dim]")
        if not runs:
            console.print("[dim]all runs already evaluated[/dim]")
            return

    is_single = len(runs) == 1

    # Header
    console.print()
    console.print(f"[bold]cooperbench eval[/bold] [dim]{run_name}[/dim]")
    console.print(f"[dim]runs:[/dim] {len(runs)}")
    console.print(f"[dim]backend:[/dim] {backend}")
    console.print()

    # For GCP with multiple runs, use batch mode for efficiency
    if backend in ("gcp", "gcp_batch") and len(runs) > 1:
        passed, failed, errors, skipped, results = _run_gcp_batch(runs, concurrency, force)
    else:
        # Docker/Modal: run interactively
        results = []
        passed = 0
        failed = 0
        errors = 0
        skipped = 0

        def eval_run(run_info: dict) -> dict | None:
            return _evaluate_single(run_info, force=force, backend=backend)

        if is_single:
            # Single run - show detailed output
            run_info = runs[0]
            feat_str = ",".join(str(f) for f in run_info["features"])
            console.print(f"  [dim]evaluating[/dim] {run_info['repo']}/{run_info['task_id']} [{feat_str}]")

            result = eval_run(run_info)
            if result:
                if result.get("skipped"):
                    skipped = 1
                    console.print("[dim]→ skip[/dim] (already evaluated)")
                elif result.get("error"):
                    errors = 1
                    console.print(f"[red]✗ error[/red]: {result['error']}")
                elif result.get("both_passed"):
                    passed = 1
                    console.print("[green]✓ pass[/green] both features")
                else:
                    failed = 1
                    f1 = "[green]✓[/green]" if result.get("feature1", {}).get("passed") else "[red]✗[/red]"
                    f2 = "[green]✓[/green]" if result.get("feature2", {}).get("passed") else "[red]✗[/red]"
                    console.print(f"[yellow]✗ partial[/yellow] f1:{f1} f2:{f2}")
        else:
            # Multiple runs - show progress
            passed, failed, errors, skipped, results = _run_with_progress(runs, eval_run, concurrency)

    # Save summary
    log_dir = Path("logs") / run_name
    _save_summary(log_dir, run_name, len(runs), passed, failed, errors, skipped, results)
    _print_summary(passed, failed, errors, skipped, len(runs))


def _run_gcp_batch(runs: list[dict], parallelism: int, force: bool) -> tuple:
    """Run evaluations using GCP Batch (all tasks submitted at once).

    This is much more efficient for large-scale evaluation because:
    - Single VM startup cost amortized across all tasks
    - Tasks run in parallel within the batch job
    - Auto-cleanup after completion

    Args:
        runs: List of run_info dicts from discover_runs
        parallelism: Max parallel tasks in batch job
        force: Force re-evaluation (unused here, filtering done earlier)

    Returns:
        Tuple of (passed, failed, errors, skipped, results)
    """
    from cooperbench.eval.backends import get_batch_evaluator
    from cooperbench.eval.backends.gcp import EvalTask
    from cooperbench.eval.sandbox import _filter_test_files, _load_patch

    # Convert runs to EvalTask objects
    tasks = []
    for i, run_info in enumerate(runs):
        task_dir = Path("dataset") / run_info["repo"] / f"task{run_info['task_id']}"
        f1, f2 = run_info["features"]

        tests1_path = task_dir / f"feature{f1}" / "tests.patch"
        tests2_path = task_dir / f"feature{f2}" / "tests.patch"

        # Load test patches (with sanitization for newlines etc)
        tests1_patch = _sanitize_patch(tests1_path.read_text()) if tests1_path.exists() else ""
        tests2_patch = _sanitize_patch(tests2_path.read_text()) if tests2_path.exists() else ""

        setting = run_info["setting"]
        log_dir = run_info["log_dir"]

        if setting == "solo":
            # Solo mode: single patch for both features
            patch_file = Path(log_dir) / "solo.patch"
            patch1 = _load_patch(patch_file) if patch_file.exists() else ""
            patch1 = _filter_test_files(patch1) if patch1 else ""
            patch2 = ""
        else:
            # Coop mode: separate patches from each agent
            patch1_file = Path(log_dir) / f"agent{f1}.patch"
            patch2_file = Path(log_dir) / f"agent{f2}.patch"
            patch1 = _load_patch(patch1_file) if patch1_file.exists() else ""
            patch2 = _load_patch(patch2_file) if patch2_file.exists() else ""
            patch1 = _filter_test_files(patch1) if patch1 else ""
            patch2 = _filter_test_files(patch2) if patch2 else ""

        task = EvalTask(
            task_index=i,
            repo_name=run_info["repo"],
            task_id=run_info["task_id"],
            feature1_id=f1,
            feature2_id=f2,
            setting=setting,
            log_dir=log_dir,
            patch1=patch1,
            patch2=patch2,
            tests1_patch=tests1_patch,
            tests2_patch=tests2_patch,
        )
        tasks.append(task)

    # Submit batch job with progress display
    evaluator = get_batch_evaluator("gcp")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        batch_task = progress.add_task("submitting", total=len(tasks))

        def on_progress(status: str, completed: int, total: int):
            status_text = {
                "submitting": "submitting to GCP Batch",
                "queued": "queued",
                "provisioning": "provisioning VMs",
                "running": "evaluating",
                "collecting": "collecting results",
            }.get(status, status)
            progress.update(batch_task, description=status_text, completed=completed)

        batch_results = evaluator.run_batch(tasks, parallelism=parallelism, on_progress=on_progress)
        progress.update(batch_task, completed=len(tasks))

    # Process results
    passed = 0
    failed = 0
    errors = 0
    skipped = 0
    results = []

    for batch_result in batch_results:
        run_info = runs[batch_result.task_index]
        feat_str = ",".join(str(f) for f in run_info["features"])
        task_name = f"{run_info['repo']}/{run_info['task_id']}"

        # Build eval_result for saving
        eval_result = {
            "repo": batch_result.repo_name,
            "task_id": batch_result.task_id,
            "features": batch_result.features,
            "setting": batch_result.setting,
            "merge": {
                "status": batch_result.merge_status,
                "strategy": batch_result.merge_strategy,
            }
            if batch_result.setting == "coop"
            else None,
            "feature1": {
                "passed": batch_result.feature1_passed,
                "test_output": batch_result.feature1_output or "",
            },
            "feature2": {
                "passed": batch_result.feature2_passed,
                "test_output": batch_result.feature2_output or "",
            },
            "both_passed": batch_result.both_passed,
            "error": batch_result.error,
            "evaluated_at": datetime.now().isoformat(),
        }

        # Save eval.json
        log_dir = Path(run_info["log_dir"])
        with open(log_dir / "eval.json", "w") as f:
            json.dump(eval_result, f, indent=2)

        # Update counters
        if batch_result.error:
            errors += 1
            status = "error"
            console.print(f"[yellow]✗ error[/yellow] {task_name} [dim]{batch_result.error}[/dim]")
        elif batch_result.both_passed:
            passed += 1
            status = "pass"
            console.print(f"[green]✓ pass[/green] {task_name} [dim][{feat_str}][/dim]")
        else:
            failed += 1
            status = "fail"
            f1 = "[green]✓[/green]" if batch_result.feature1_passed else "[red]✗[/red]"
            f2 = "[green]✓[/green]" if batch_result.feature2_passed else "[red]✗[/red]"
            console.print(f"[red]✗ fail[/red] {task_name} [dim][{feat_str}][/dim] f1:{f1} f2:{f2}")

        results.append({"run": f"{task_name}/{feat_str}", "status": status})

    return passed, failed, errors, skipped, results


def _evaluate_single(run_info: dict, force: bool = False, backend: str = "modal") -> dict | None:
    """Evaluate a single run."""
    log_dir = Path(run_info["log_dir"])
    eval_file = log_dir / "eval.json"

    if eval_file.exists() and not force:
        with open(eval_file) as f:
            return {"skipped": True, **json.load(f)}

    setting = run_info["setting"]
    repo = run_info["repo"]
    task_id = run_info["task_id"]
    features = run_info["features"]
    f1, f2 = features[0], features[1]

    if setting == "solo":
        # Solo evaluation
        patch_file = log_dir / "solo.patch"
        patch = patch_file.read_text() if patch_file.exists() else ""

        result = test_solo(
            repo_name=repo,
            task_id=task_id,
            feature1_id=f1,
            feature2_id=f2,
            patch=patch,
            backend=backend,
        )

        eval_result = {
            "repo": repo,
            "task_id": task_id,
            "features": features,
            "setting": "solo",
            "merge": None,
            "feature1": result.get("feature1", {}),
            "feature2": result.get("feature2", {}),
            "both_passed": result.get("both_passed", False),
            "error": result.get("error"),
            "evaluated_at": datetime.now().isoformat(),
        }
    else:
        # Coop evaluation - merge two agent patches
        patch1_file = log_dir / f"agent{f1}.patch"
        patch2_file = log_dir / f"agent{f2}.patch"

        patch1 = patch1_file.read_text() if patch1_file.exists() else ""
        patch2 = patch2_file.read_text() if patch2_file.exists() else ""

        result = test_merged(
            repo_name=repo,
            task_id=task_id,
            feature1_id=f1,
            feature2_id=f2,
            patch1=patch1,
            patch2=patch2,
            backend=backend,
        )

        eval_result = {
            "repo": repo,
            "task_id": task_id,
            "features": features,
            "setting": "coop",
            "merge": result.get("merge", {}),
            "feature1": result.get("feature1", {}),
            "feature2": result.get("feature2", {}),
            "both_passed": result.get("both_passed", False),
            "error": result.get("error"),
            "evaluated_at": datetime.now().isoformat(),
        }

    # Save result
    with open(eval_file, "w") as f:
        json.dump(eval_result, f, indent=2)

    return eval_result


def _run_with_progress(runs: list, eval_run, concurrency: int) -> tuple:
    """Run evaluations with progress display."""
    results = []
    passed = 0
    failed = 0
    errors = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        eval_progress = progress.add_task("evaluating", total=len(runs))

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_run = {executor.submit(eval_run, r): r for r in runs}

            for future in as_completed(future_to_run):
                run_info = future_to_run[future]
                feat_str = ",".join(str(f) for f in run_info["features"])
                task_name = f"{run_info['repo']}/{run_info['task_id']}"

                try:
                    result = future.result()
                    if result is None:
                        errors += 1
                        status = "error"
                    elif result.get("skipped"):
                        skipped += 1
                        status = "skip"
                    elif result.get("error"):
                        errors += 1
                        status = "error"
                    elif result.get("both_passed"):
                        passed += 1
                        status = "pass"
                    else:
                        failed += 1
                        status = "fail"

                    results.append({"run": f"{task_name}/{feat_str}", "status": status})

                    status_display = {
                        "pass": "[green]✓ pass[/green]",
                        "fail": "[red]✗ fail[/red]",
                        "skip": "[dim]→ skip[/dim]",
                        "error": "[yellow]✗ error[/yellow]",
                    }[status]
                    progress.console.print(f"{status_display} {task_name} [dim][{feat_str}][/dim]")

                except Exception as e:
                    errors += 1
                    results.append({"run": f"{task_name}/{feat_str}", "status": "error", "error": str(e)})
                    progress.console.print(f"[yellow]✗ error[/yellow] {task_name} [dim]{e}[/dim]")

                progress.update(eval_progress, advance=1)

    return passed, failed, errors, skipped, results


def _save_summary(
    log_dir: Path,
    run_name: str,
    total_runs: int,
    passed: int,
    failed: int,
    errors: int,
    skipped: int,
    results: list,
) -> None:
    """Save evaluation summary."""
    summary = {
        "run_name": run_name,
        "evaluated_at": datetime.now().isoformat(),
        "total_runs": total_runs,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "pass_rate": passed / max(passed + failed, 1),
        "results": results,
    }
    with open(log_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def _print_summary(passed: int, failed: int, errors: int, skipped: int, total: int) -> None:
    """Print evaluation summary."""
    console.print()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("passed", f"[green]{passed}[/green]")
    table.add_row("failed", f"[red]{failed}[/red]")
    if errors:
        table.add_row("errors", f"[yellow]{errors}[/yellow]")
    if skipped:
        table.add_row("skipped", f"[dim]{skipped}[/dim]")
    table.add_row("pass rate", f"{passed / max(passed + failed, 1):.1%}")
    console.print(table)
    console.print()
