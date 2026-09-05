"""Task discovery from dataset/ directory."""

import json
from itertools import combinations
from pathlib import Path

DEFAULT_DATASET_DIR = Path("dataset")
DEFAULT_LOGS_DIR = Path("logs")


def load_subset(subset_name: str, dataset_dir: Path | str | None = None) -> dict:
    """Load a subset definition from ``<dataset_dir>/subsets/``.

    Args:
        subset_name: Name of the subset (e.g., 'lite')
        dataset_dir: Root of the dataset tree.  Defaults to ``./dataset``.

    Returns:
        Dict with:
          - tasks: set of (repo, task_id) tuples
          - pairs: dict mapping (repo, task_id) to list of [f1, f2] pairs (if specified)
    """
    root = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    subset_path = root / "subsets" / f"{subset_name}.json"
    if not subset_path.exists():
        raise ValueError(f"Subset '{subset_name}' not found at {subset_path}")

    with open(subset_path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Subset '{subset_name}' at {subset_path} is not valid JSON: {e}") from e

    try:
        task_list = data["tasks"]
    except KeyError:
        raise ValueError(f"Subset '{subset_name}' at {subset_path} is missing required key 'tasks'")

    tasks = set()
    pairs = {}
    for t in task_list:
        key = (t["repo"], t["task_id"])
        tasks.add(key)
        # If pairs are specified, store them
        if "pairs" in t:
            pairs[key] = [tuple(p) for p in t["pairs"]]
        # "groups" supports N-element tuples for team-mode tasks
        if "groups" in t:
            pairs[key] = [tuple(g) for g in t["groups"]]

    return {"tasks": tasks, "pairs": pairs}


def discover_tasks(
    subset: str | None = None,
    repo_filter: str | None = None,
    task_filter: int | None = None,
    features_filter: list[int] | None = None,
    dataset_dir: Path | str | None = None,
    n_agents: int = 2,
) -> list[dict]:
    """Discover benchmark tasks from ``dataset_dir``.

    Args:
        subset: Use a predefined subset (e.g., 'lite')
        repo_filter: Filter by repository name
        task_filter: Filter by task ID
        features_filter: Specific feature set to use
        dataset_dir: Root of the dataset tree.  Defaults to ``./dataset``.
        n_agents: Number of agents per task; controls combination size in the
            default (no subset, no features_filter) path.  Defaults to 2.

    Returns:
        List of task dicts with repo, task_id, features
    """
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    tasks = []

    # Load subset filter if specified
    subset_data = None
    if subset:
        subset_data = load_subset(subset, dataset_dir=dataset_dir)

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

            # Filter by subset if specified
            task_key = (repo_dir.name, task_id)
            if subset_data and task_key not in subset_data["tasks"]:
                continue

            feature_ids = []
            for feature_dir in sorted(task_dir.iterdir()):
                if feature_dir.is_dir() and feature_dir.name.startswith("feature"):
                    fid = int(feature_dir.name.replace("feature", ""))
                    feature_ids.append(fid)

            if len(feature_ids) < n_agents:
                continue

            if features_filter:
                # Command-line filter takes precedence
                if all(f in feature_ids for f in features_filter):
                    tasks.append(
                        {
                            "repo": repo_dir.name,
                            "task_id": task_id,
                            "features": features_filter,
                        }
                    )
            elif subset_data and task_key in subset_data["pairs"]:
                # Use specific groups/pairs from subset (supports N-element groups)
                for group in subset_data["pairs"][task_key]:
                    group_list = list(group)
                    if all(f in feature_ids for f in group_list):
                        tasks.append(
                            {
                                "repo": repo_dir.name,
                                "task_id": task_id,
                                "features": group_list,
                            }
                        )
            else:
                # All nCk combinations where k == n_agents
                feature_ids.sort()
                for combo in combinations(feature_ids, n_agents):
                    tasks.append(
                        {
                            "repo": repo_dir.name,
                            "task_id": task_id,
                            "features": list(combo),
                        }
                    )

    return tasks
