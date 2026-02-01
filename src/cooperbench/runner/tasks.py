"""Task discovery from dataset/ directory."""

import json
from itertools import combinations
from pathlib import Path


def load_subset(subset_name: str) -> dict:
    """Load a subset definition from dataset/subsets/.

    Args:
        subset_name: Name of the subset (e.g., 'lite')

    Returns:
        Dict with:
          - tasks: set of (repo, task_id) tuples
          - pairs: dict mapping (repo, task_id) to list of [f1, f2] pairs (if specified)
    """
    subset_path = Path("dataset/subsets") / f"{subset_name}.json"
    if not subset_path.exists():
        raise ValueError(f"Subset '{subset_name}' not found at {subset_path}")

    with open(subset_path) as f:
        data = json.load(f)

    tasks = set()
    pairs = {}
    for t in data["tasks"]:
        key = (t["repo"], t["task_id"])
        tasks.add(key)
        # If pairs are specified, store them
        if "pairs" in t:
            pairs[key] = [tuple(p) for p in t["pairs"]]

    return {"tasks": tasks, "pairs": pairs}


def discover_tasks(
    subset: str | None = None,
    repo_filter: str | None = None,
    task_filter: int | None = None,
    features_filter: list[int] | None = None,
) -> list[dict]:
    """Discover benchmark tasks from dataset/.

    Args:
        subset: Use a predefined subset (e.g., 'lite')
        repo_filter: Filter by repository name
        task_filter: Filter by task ID
        features_filter: Specific feature pair to use

    Returns:
        List of task dicts with repo, task_id, features
    """
    dataset_dir = Path("dataset")
    tasks = []

    # Load subset filter if specified
    subset_data = None
    if subset:
        subset_data = load_subset(subset)

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

            if len(feature_ids) < 2:
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
                # Use specific pairs from subset
                for pair in subset_data["pairs"][task_key]:
                    f1, f2 = pair
                    if f1 in feature_ids and f2 in feature_ids:
                        tasks.append(
                            {
                                "repo": repo_dir.name,
                                "task_id": task_id,
                                "features": [f1, f2],
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
