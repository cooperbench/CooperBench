"""Evaluation harness for benchmark runs."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from cooperbench.eval.runs import discover_runs
from cooperbench.eval.sandbox import test_merged, test_solo
from cooperbench.utils import console


def evaluate(
    run_name: str,
    subset: str | None = None,
    repo: str | None = None,
    task_id: int | None = None,
    features: list[int] | None = None,
    concurrency: int = 10,
    force: bool = False,
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

    is_single = len(runs) == 1

    # Header
    console.print()
    console.print(f"[bold]cooperbench eval[/bold] [dim]{run_name}[/dim]")
    console.print(f"[dim]runs:[/dim] {len(runs)}")
    console.print()

    results = []
    passed = 0
    failed = 0
    errors = 0
    skipped = 0

    def eval_run(run_info: dict) -> dict | None:
        return _evaluate_single(run_info, force=force)

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


def _evaluate_single(run_info: dict, force: bool = False) -> dict | None:
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
