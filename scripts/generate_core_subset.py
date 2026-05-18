"""Generate the 10-pair `core` subset via stratified diversity sampling.

The goal is a small core set whose composition mirrors the full dataset's
shape (repo distribution + language coverage), so its pass-rate roughly
tracks the overall dataset pass-rate without any per-task eval data.

Stratification:
  - Compute per-repo pair counts from the full dataset (sum of C(n_features, 2)
    across each task in the repo).
  - Allocate the 10 slots across repos using largest-remainder proportional
    allocation, with a floor of 1 slot per primary language (Python, Go,
    Rust, JS/TS) so each language is represented.
  - Within each selected repo, sample pairs uniformly at random (seed=42),
    spreading across tasks where possible (no task gets a 2nd pair until
    every task in the repo has 1).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
OUTPUT_PATH = DATASET_DIR / "subsets" / "core.json"
TARGET_PAIRS = 10
SEED = 42

# One primary language per repo. Each listed language gets at least one slot.
REPO_LANGUAGE: dict[str, str] = {
    "dottxt_ai_outlines_task": "python",
    "dspy_task": "python",
    "go_chi_task": "go",
    "huggingface_datasets_task": "python",
    "llama_index_task": "python",
    "openai_tiktoken_task": "python",
    "pallets_click_task": "python",
    "pallets_jinja_task": "python",
    "pillow_task": "python",
    "react_hook_form_task": "ts",
    "samuelcolvin_dirty_equals_task": "python",
    "typst_task": "rust",
}
LANGUAGE_FLOOR = {"python", "go", "rust", "ts"}


def enumerate_all_pairs() -> dict[str, dict[int, list[tuple[int, int]]]]:
    """Walk the dataset and return {repo: {task_id: [(f1, f2), ...]}}."""
    out: dict[str, dict[int, list[tuple[int, int]]]] = defaultdict(dict)
    for repo_dir in sorted(DATASET_DIR.iterdir()):
        if not repo_dir.is_dir() or not repo_dir.name.endswith("_task"):
            continue
        for task_dir in sorted(repo_dir.iterdir()):
            if not task_dir.is_dir() or not task_dir.name.startswith("task"):
                continue
            task_id = int(task_dir.name.removeprefix("task"))
            feature_ids = sorted(
                int(f.name.removeprefix("feature"))
                for f in task_dir.iterdir()
                if f.is_dir() and f.name.startswith("feature")
            )
            if len(feature_ids) < 2:
                continue
            out[repo_dir.name][task_id] = list(combinations(feature_ids, 2))
    return out


def allocate_slots(repo_pair_counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder proportional allocation with a per-language floor."""
    # 1. Floor: one slot per language (assigned to the largest repo in that language).
    by_language: dict[str, list[str]] = defaultdict(list)
    for repo, lang in REPO_LANGUAGE.items():
        if repo in repo_pair_counts:
            by_language[lang].append(repo)

    allocation: dict[str, int] = {repo: 0 for repo in repo_pair_counts}
    for lang in LANGUAGE_FLOOR:
        repos = sorted(by_language.get(lang, []), key=lambda r: -repo_pair_counts[r])
        if repos:
            allocation[repos[0]] = 1

    remaining = total - sum(allocation.values())
    if remaining <= 0:
        return allocation

    # 2. Proportional remainder allocation across all repos.
    grand_total = sum(repo_pair_counts.values())
    quotas = {
        repo: count * total / grand_total for repo, count in repo_pair_counts.items()
    }
    # Already-assigned floor counts are credited toward each repo's quota.
    deficits = sorted(
        ((quotas[r] - allocation[r], r) for r in repo_pair_counts),
        key=lambda x: (-x[0], x[1]),
    )
    for _, repo in deficits:
        if remaining == 0:
            break
        allocation[repo] += 1
        remaining -= 1

    return allocation


def sample_repo_pairs(
    tasks: dict[int, list[tuple[int, int]]],
    n: int,
    rng: random.Random,
) -> list[tuple[int, tuple[int, int]]]:
    """Sample n pairs from a repo, spreading across tasks before repeating one."""
    task_ids = sorted(tasks.keys())
    rng.shuffle(task_ids)
    picked: list[tuple[int, tuple[int, int]]] = []
    used: dict[int, set[tuple[int, int]]] = defaultdict(set)
    round_idx = 0
    while len(picked) < n:
        progress = False
        for tid in task_ids:
            if len(picked) >= n:
                break
            available = [p for p in tasks[tid] if p not in used[tid]]
            if not available:
                continue
            choice = rng.choice(available)
            used[tid].add(choice)
            picked.append((tid, choice))
            progress = True
        if not progress:
            break  # all pairs exhausted
        round_idx += 1
    return picked


def main() -> None:
    rng = random.Random(SEED)
    repo_to_tasks = enumerate_all_pairs()
    repo_pair_counts = {repo: sum(len(v) for v in tasks.values()) for repo, tasks in repo_to_tasks.items()}

    allocation = allocate_slots(repo_pair_counts, TARGET_PAIRS)

    repo_picks: dict[str, dict[int, list[list[int]]]] = {}
    for repo in sorted(repo_to_tasks):
        n = allocation.get(repo, 0)
        if n == 0:
            continue
        picks = sample_repo_pairs(repo_to_tasks[repo], n, rng)
        by_task: dict[int, list[list[int]]] = defaultdict(list)
        for tid, (f1, f2) in picks:
            by_task[tid].append([f1, f2])
        repo_picks[repo] = {tid: sorted(prs) for tid, prs in by_task.items()}

    tasks_out = []
    for repo in sorted(repo_picks):
        for tid in sorted(repo_picks[repo]):
            tasks_out.append({
                "repo": repo,
                "task_id": tid,
                "pairs": repo_picks[repo][tid],
            })

    total_pairs = sum(len(t["pairs"]) for t in tasks_out)
    subset = {
        "name": "core",
        "description": (
            f"{TARGET_PAIRS}-pair core subset via stratified diversity sampling "
            f"(proportional repo allocation with language floor, seed={SEED}). "
            "Designed so the core's pass-rate roughly tracks the overall dataset's "
            "without per-task eval data."
        ),
        "stats": {
            "tasks": len(tasks_out),
            "pairs": total_pairs,
            "repos": len(repo_picks),
        },
        "allocation": {repo: allocation[repo] for repo in sorted(allocation) if allocation[repo]},
        "tasks": tasks_out,
    }

    OUTPUT_PATH.write_text(json.dumps(subset, indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH} ({total_pairs} pairs, {len(tasks_out)} tasks, {len(repo_picks)} repos)")
    print("Allocation:", subset["allocation"])


if __name__ == "__main__":
    main()
