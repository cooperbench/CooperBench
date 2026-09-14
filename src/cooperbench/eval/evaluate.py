"""Evaluation harness for benchmark runs."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from cooperbench.eval.runs import discover_runs
from cooperbench.eval.sandbox import _sanitize_patch, test_merged, test_merged_n, test_solo, test_solo_n
from cooperbench.runner.tasks import DEFAULT_DATASET_DIR, DEFAULT_LOGS_DIR
from cooperbench.utils import console


def evaluate(
    run_name: str,
    subset: str | None = None,
    repo: str | None = None,
    task_id: int | None = None,
    features: list[int] | None = None,
    concurrency: int = 10,
    force: bool = False,
    backend: str = "docker",
    dataset_dir: str | None = None,
    logs_dir: str | None = None,
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
        dataset_dir: Root of the dataset tree.  Defaults to ``./dataset``.
        logs_dir: Root of the logs tree.  Defaults to ``./logs``.
    """
    runs = discover_runs(
        run_name=run_name,
        subset=subset,
        repo_filter=repo,
        task_filter=task_id,
        features_filter=features,
        logs_dir=logs_dir,
        dataset_dir=dataset_dir,
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

    # Guardrail: openhands_sdk produces patches that include a committed
    # ``patch.txt`` in the working tree and relies on the same Modal/Redis
    # tunnel the agent used; running eval through Docker (or any non-modal
    # backend) silently changes the test environment.  Read the run's
    # config.json once and bail with a clear warning so users don't burn
    # an eval pass for no reason.
    run_config_path = (Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR) / run_name / "config.json"
    agent_framework: str | None = None
    if run_config_path.exists():
        try:
            agent_framework = json.loads(run_config_path.read_text()).get("agent_framework")
        except Exception:
            agent_framework = None
    if agent_framework == "openhands_sdk" and backend != "modal":
        console.print(
            f"[yellow]warning:[/yellow] run [bold]{run_name}[/bold] was produced by "
            f"agent_framework=openhands_sdk, which requires --backend modal. "
            f"Refusing to evaluate with --backend {backend}. "
            f"Rerun with --backend modal."
        )
        return

    # Header
    console.print()
    console.print(f"[bold]cooperbench eval[/bold] [dim]{run_name}[/dim]")
    console.print(f"[dim]runs:[/dim] {len(runs)}")
    console.print(f"[dim]backend:[/dim] {backend}")
    console.print()

    # For GCP with multiple runs, use batch mode for efficiency
    if backend in ("gcp", "gcp_batch") and len(runs) > 1:
        passed, failed, errors, skipped, results = _run_gcp_batch(runs, concurrency, force, dataset_dir=dataset_dir)
    else:
        # Docker/Modal: run interactively
        results = []
        passed = 0
        failed = 0
        errors = 0
        skipped = 0

        def eval_run(run_info: dict) -> dict | None:
            return _evaluate_single(run_info, force=force, backend=backend, dataset_dir=dataset_dir)

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
                elif result.get("both_passed") or result.get("all_passed"):
                    passed = 1
                    console.print("[green]✓ pass[/green] all features")
                else:
                    failed = 1
                    # Support both legacy feature1/feature2 and new features_result dict
                    if result.get("features_result"):
                        parts = []
                        for fid, fr in result["features_result"].items():
                            icon = "[green]✓[/green]" if fr.get("passed") else "[red]✗[/red]"
                            parts.append(f"f{fid}:{icon}")
                        console.print(f"[yellow]✗ partial[/yellow] {' '.join(parts)}")
                    else:
                        f1 = "[green]✓[/green]" if result.get("feature1", {}).get("passed") else "[red]✗[/red]"
                        f2 = "[green]✓[/green]" if result.get("feature2", {}).get("passed") else "[red]✗[/red]"
                        console.print(f"[yellow]✗ partial[/yellow] f1:{f1} f2:{f2}")
        else:
            # Multiple runs - show progress
            passed, failed, errors, skipped, results = _run_with_progress(runs, eval_run, concurrency)

    # Save summary
    logs_root = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR
    log_dir = logs_root / run_name
    _save_summary(log_dir, run_name, len(runs), passed, failed, errors, skipped, results)
    _print_summary(passed, failed, errors, skipped, len(runs))


def _run_gcp_batch(
    runs: list[dict],
    parallelism: int,
    force: bool,
    dataset_dir: Path | str | None = None,
) -> tuple:
    """Run evaluations using GCP Batch (all tasks submitted at once).

    This is much more efficient for large-scale evaluation because:
    - Single VM startup cost amortized across all tasks
    - Tasks run in parallel within the batch job
    - Auto-cleanup after completion

    Args:
        runs: List of run_info dicts from discover_runs
        parallelism: Max parallel tasks in batch job
        force: Force re-evaluation (unused here, filtering done earlier)
        dataset_dir: Root of the dataset tree.  Defaults to ``./dataset``.

    Returns:
        Tuple of (passed, failed, errors, skipped, results)
    """
    from cooperbench.eval.backends import get_batch_evaluator
    from cooperbench.eval.backends.gcp import EvalTask
    from cooperbench.eval.sandbox import _filter_test_files, _load_patch

    root = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR

    # Convert runs to EvalTask objects
    tasks = []
    for i, run_info in enumerate(runs):
        task_dir = root / run_info["repo"] / f"task{run_info['task_id']}"
        features = run_info["features"]

        # Load test patches (with sanitization for newlines etc)
        tests_patches = []
        for fid in features:
            tests_path = task_dir / f"feature{fid}" / "tests.patch"
            tests_patches.append(_sanitize_patch(tests_path.read_text()) if tests_path.exists() else "")

        setting = run_info["setting"]
        log_dir = run_info["log_dir"]

        if setting == "solo":
            # Solo mode: single patch covering all features
            patch_file = Path(log_dir) / "solo.patch"
            patch = _load_patch(patch_file) if patch_file.exists() else ""
            patches = [_filter_test_files(patch) if patch else ""]
        else:
            # Coop/team mode: separate patch from each agent
            patches = []
            for fid in features:
                patch_file = Path(log_dir) / f"agent{fid}.patch"
                patch = _load_patch(patch_file) if patch_file.exists() else ""
                patches.append(_filter_test_files(patch) if patch else "")

        task = EvalTask(
            task_index=i,
            repo_name=run_info["repo"],
            task_id=run_info["task_id"],
            setting=setting,
            log_dir=log_dir,
            feature_ids=features,
            patches=patches,
            tests_patches=tests_patches,
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

        features = batch_result.features
        features_passed = batch_result.features_passed or []
        features_output = batch_result.features_output or []

        # Build eval_result for saving
        features_result = {
            str(fid): {
                "feature_id": fid,
                "passed": features_passed[i] if i < len(features_passed) else False,
                "test_output": features_output[i] if i < len(features_output) else "",
            }
            for i, fid in enumerate(features)
        }
        eval_result = {
            "repo": batch_result.repo_name,
            "task_id": batch_result.task_id,
            "features": features,
            "setting": batch_result.setting,
            "merge": {
                "status": batch_result.merge_status,
                "strategy": batch_result.merge_strategy,
            }
            if batch_result.setting in ("coop", "team")
            else None,
            "features_result": features_result,
            "all_passed": batch_result.all_passed,
            "error": batch_result.error,
            "evaluated_at": datetime.now().isoformat(),
        }
        # Dual-write legacy keys for 2-agent runs
        if len(features) == 2:
            eval_result["feature1"] = {
                "passed": batch_result.feature1_passed,
                "test_output": batch_result.feature1_output or "",
            }
            eval_result["feature2"] = {
                "passed": batch_result.feature2_passed,
                "test_output": batch_result.feature2_output or "",
            }
            eval_result["both_passed"] = batch_result.both_passed

        # Save eval.json
        log_dir = Path(run_info["log_dir"])
        with open(log_dir / "eval.json", "w") as f:
            json.dump(eval_result, f, indent=2)

        # Update counters
        if batch_result.error:
            errors += 1
            status = "error"
            console.print(f"[yellow]✗ error[/yellow] {task_name} [dim]{batch_result.error}[/dim]")
        elif batch_result.all_passed:
            passed += 1
            status = "pass"
            console.print(f"[green]✓ pass[/green] {task_name} [dim][{feat_str}][/dim]")
        else:
            failed += 1
            status = "fail"
            icons = " ".join(
                f"f{fid}:" + ("[green]✓[/green]" if features_result[str(fid)]["passed"] else "[red]✗[/red]")
                for fid in features
            )
            console.print(f"[red]✗ fail[/red] {task_name} [dim][{feat_str}][/dim] {icons}")

        results.append({"run": f"{task_name}/{feat_str}", "status": status})

    return passed, failed, errors, skipped, results


def _evaluate_single(
    run_info: dict,
    force: bool = False,
    backend: str = "docker",
    dataset_dir: str | None = None,
) -> dict | None:
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

    if setting == "solo" and len(features) == 2:
        # Solo evaluation (legacy 2-feature path)
        patch_file = log_dir / "solo.patch"
        patch = patch_file.read_text() if patch_file.exists() else ""

        result = test_solo(
            repo_name=repo,
            task_id=task_id,
            feature1_id=f1,
            feature2_id=f2,
            patch=patch,
            backend=backend,
            dataset_dir=dataset_dir,
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
    elif setting == "solo":
        # Solo evaluation — one patch, N features
        patch_file = log_dir / "solo.patch"
        patch = patch_file.read_text() if patch_file.exists() else ""

        result = test_solo_n(
            repo_name=repo,
            task_id=task_id,
            feature_ids=features,
            patch=patch,
            backend=backend,
            dataset_dir=dataset_dir,
        )

        eval_result = {
            "repo": repo,
            "task_id": task_id,
            "features": features,
            "setting": "solo",
            "merge": None,
            "features_result": result.get("features", {}),
            "all_passed": result.get("all_passed", False),
            "error": result.get("error"),
            "evaluated_at": datetime.now().isoformat(),
        }
    elif setting == "coop" and len(features) == 2:
        # Coop evaluation (legacy 2-agent path) - merge two agent patches
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
            dataset_dir=dataset_dir,
        )

        eval_result = {
            "repo": repo,
            "task_id": task_id,
            "features": features,
            "setting": "coop",
            "apply_status": result.get("apply_status"),
            "merge": result.get("merge", {}),
            "feature1": result.get("feature1", {}),
            "feature2": result.get("feature2", {}),
            "both_passed": result.get("both_passed", False),
            "error": result.get("error"),
            "evaluated_at": datetime.now().isoformat(),
        }
    else:
        # Team (any N) or coop with N>2 — N agents, one patch per feature
        agent_patches = []
        missing_patch_fids = []
        for fid in features:
            pf = log_dir / f"agent{fid}.patch"
            if pf.exists():
                agent_patches.append(pf.read_text())
            else:
                agent_patches.append("")
                missing_patch_fids.append(fid)

        result = test_merged_n(
            repo_name=repo,
            task_id=task_id,
            feature_ids=features,
            patches=agent_patches,
            backend=backend,
            dataset_dir=dataset_dir,
        )

        # If any agent produced no patch file at all, the merge result
        # should reflect missing input rather than a spuriously clean merge.
        if missing_patch_fids:
            merge_dict = dict(result.get("merge") or {})
            if merge_dict.get("status") == "clean":
                merge_dict["status"] = "missing_input"
                result = {**result, "merge": merge_dict, "all_passed": False}

        eval_result = {
            "repo": repo,
            "task_id": task_id,
            "features": features,
            "setting": setting,
            "apply_status": result.get("apply_status"),
            "merge": result.get("merge", {}),
            "features_result": result.get("features", {}),
            "all_passed": result.get("all_passed", False),
            "error": result.get("error"),
            "evaluated_at": datetime.now().isoformat(),
        }
        # Dual-write legacy keys for 2-agent runs
        if len(features) == 2:
            eval_result["feature1"] = result.get("feature1", {})
            eval_result["feature2"] = result.get("feature2", {})
            eval_result["both_passed"] = result.get("both_passed", False)

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
                    elif result.get("both_passed") or result.get("all_passed"):
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
