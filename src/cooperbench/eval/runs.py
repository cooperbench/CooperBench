"""Run discovery from logs/ directory."""

import json
from pathlib import Path

from cooperbench.runner.tasks import load_subset


def discover_runs(
    run_name: str,
    subset: str | None = None,
    repo_filter: str | None = None,
    task_filter: int | None = None,
    features_filter: list[int] | None = None,
) -> list[dict]:
    """Discover completed runs from logs/ directory.

    Supports both new structure (logs/{run_name}/{setting}/{repo}/)
    and legacy structure (logs/{run_name}/{repo}/).

    Args:
        run_name: Name of the run
        subset: Filter to a predefined subset (e.g., 'lite')
        repo_filter: Filter by repository name
        task_filter: Filter by task ID
        features_filter: Specific feature pair to find

    Returns:
        List of run dicts with repo, task_id, features, log_dir, setting
    """
    runs = []
    log_dir = Path("logs") / run_name

    if not log_dir.exists():
        return runs

    # Load subset filter if specified
    subset_tasks = None
    if subset:
        subset_tasks = set(load_subset(subset))

    # Check for new structure (solo/, coop/)
    for setting in ["solo", "coop"]:
        setting_dir = log_dir / setting
        if setting_dir.exists():
            runs.extend(
                _discover_runs_in_dir(
                    setting_dir,
                    setting=setting,
                    subset_tasks=subset_tasks,
                    repo_filter=repo_filter,
                    task_filter=task_filter,
                    features_filter=features_filter,
                )
            )

    # Check legacy structure (direct repo dirs)
    if not runs:
        runs.extend(
            _discover_runs_in_dir(
                log_dir,
                setting=None,  # Will be inferred from result.json
                subset_tasks=subset_tasks,
                repo_filter=repo_filter,
                task_filter=task_filter,
                features_filter=features_filter,
            )
        )

    return runs


def _discover_runs_in_dir(
    base_dir: Path,
    setting: str | None,
    subset_tasks: set | None,
    repo_filter: str | None,
    task_filter: int | None,
    features_filter: list[int] | None,
) -> list[dict]:
    """Discover runs in a specific directory."""
    runs = []

    for repo_dir in sorted(base_dir.iterdir()):
        if not repo_dir.is_dir() or not repo_dir.name.endswith("_task"):
            continue
        if repo_filter and repo_filter != repo_dir.name:
            continue

        for task_dir in sorted(repo_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            try:
                task_id = int(task_dir.name)
            except ValueError:
                continue
            if task_filter and task_filter != task_id:
                continue

            # Filter by subset if specified
            if subset_tasks and (repo_dir.name, task_id) not in subset_tasks:
                continue

            for feature_dir in sorted(task_dir.iterdir()):
                if not feature_dir.is_dir():
                    continue

                # Parse feature string: f1_f2 or f1_f5
                parts = feature_dir.name.split("_")
                try:
                    features = [int(p[1:]) for p in parts if p.startswith("f")]
                except ValueError:
                    continue

                if len(features) < 2:
                    continue

                if features_filter:
                    if set(features_filter) != set(features):
                        continue

                result_file = feature_dir / "result.json"
                if not result_file.exists():
                    continue

                # Infer setting from result.json or solo.patch presence
                run_setting = setting
                if run_setting is None:
                    with open(result_file) as f:
                        result_data = json.load(f)
                    run_setting = result_data.get("setting")
                    if run_setting is None:
                        # Legacy: check for solo.patch
                        run_setting = "solo" if (feature_dir / "solo.patch").exists() else "coop"

                runs.append(
                    {
                        "repo": repo_dir.name,
                        "task_id": task_id,
                        "features": features,
                        "log_dir": str(feature_dir),
                        "setting": run_setting,
                    }
                )

    return runs
