"""Task discovery from dataset/ directory."""

import json
from itertools import combinations
from pathlib import Path


def load_subset(subset_name: str) -> list[tuple[str, int]]:
    """Load a subset definition from dataset/subsets/.

    Args:
        subset_name: Name of the subset (e.g., 'lite')

    Returns:
        List of (repo, task_id) tuples in the subset
    """
    subset_path = Path("dataset/subsets") / f"{subset_name}.json"
    if not subset_path.exists():
        raise ValueError(f"Subset '{subset_name}' not found at {subset_path}")

    with open(subset_path) as f:
        data = json.load(f)

    return [(t["repo"], t["task_id"]) for t in data["tasks"]]


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
    subset_tasks = None
    if subset:
        subset_tasks = set(load_subset(subset))

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
            if subset_tasks and (repo_dir.name, task_id) not in subset_tasks:
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
