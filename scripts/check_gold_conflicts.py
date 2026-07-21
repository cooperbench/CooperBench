"""Check which gold patch pairs (or N-tuples) have merge conflicts across all CooperBench tasks.

Spins up one Modal sandbox per feature group, running up to 75 in parallel.
652 pairs total => ~9 rounds of 75.

Results are saved to dataset/gold_conflict_report.json.

Usage:
    python scripts/check_gold_conflicts.py
    python scripts/check_gold_conflicts.py --repo pallets_click_task
    python scripts/check_gold_conflicts.py --max-workers 75
    python scripts/check_gold_conflicts.py --group-size 3
    python scripts/check_gold_conflicts.py --group-size 3 --output-subset dataset/subsets/triples_clean.json
    python scripts/check_gold_conflicts.py --group-size 3 --output-slight dataset/subsets/triples_slight.json
    python scripts/check_gold_conflicts.py --group-size 3 --output-slight dataset/subsets/triples_slight.json --slight-n 30
"""

import argparse
import base64
import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import modal

# Rate limiter: Modal allows 5 sandbox creations/sec, we target 4/sec
_creation_lock = threading.Lock()
_last_creation_time = 0.0
_CREATION_INTERVAL = 0.3  # seconds between sandbox creations


def _rate_limited_create_sandbox(modal_image, timeout, app):
    """Create a Modal sandbox with rate limiting."""
    global _last_creation_time
    with _creation_lock:
        now = time.time()
        wait = _CREATION_INTERVAL - (now - _last_creation_time)
        if wait > 0:
            time.sleep(wait)
        _last_creation_time = time.time()

    return modal.Sandbox.create(
        "sleep",
        "infinity",
        image=modal_image,
        timeout=timeout,
        workdir="/workspace",
        app=app,
    )


REGISTRY = "akhatua"
IMAGE_PREFIX = "cooperbench"
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


def get_image_name(repo_name: str, task_id: int) -> str:
    repo_clean = repo_name.replace("_task", "").replace("_", "-")
    return f"{REGISTRY}/{IMAGE_PREFIX}-{repo_clean}:task{task_id}"


def filter_test_files(patch_content: str) -> str:
    """Filter test files from patch content."""
    if not patch_content:
        return patch_content

    filtered_lines = []
    skip = False
    for line in patch_content.split("\n"):
        if line.startswith("diff --git"):
            skip = any(p in line for p in ["/test_", "/tests/", "_test.py", "/test/", "tests.py"])
        if not skip:
            filtered_lines.append(line)

    result = "\n".join(filtered_lines)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


def discover_all_pairs(repo_filter: str | None = None, task_filter: int | None = None) -> list[dict]:
    """Discover all feature pairs, each with their gold patches pre-loaded.

    Kept for backwards compatibility. Internally delegates to discover_all_groups.
    """
    return discover_all_groups(
        group_size=2,
        repo_filter=repo_filter,
        task_filter=task_filter,
    )


def discover_all_groups(
    group_size: int,
    repo_filter: str | None = None,
    task_filter: int | None = None,
) -> list[dict]:
    """Discover all N-tuple feature groups with their gold patches pre-loaded.

    When group_size == 2, each record also carries 'f1', 'f2', 'patch1', and
    'patch2' keys so callers that rely on the old schema continue to work unchanged.
    """
    groups = []

    for repo_dir in sorted(DATASET_DIR.iterdir()):
        if not repo_dir.is_dir() or repo_dir.name in ("README.md", "subsets"):
            continue
        if repo_filter and repo_filter != repo_dir.name:
            continue

        for task_dir in sorted(repo_dir.iterdir()):
            if not task_dir.is_dir() or not task_dir.name.startswith("task"):
                continue

            task_id = int(task_dir.name.replace("task", ""))
            if task_filter is not None and task_filter != task_id:
                continue

            feature_ids = []
            patches = {}
            for feature_dir in sorted(task_dir.iterdir()):
                if feature_dir.is_dir() and feature_dir.name.startswith("feature"):
                    fid = int(feature_dir.name.replace("feature", ""))
                    patch_path = feature_dir / "feature.patch"
                    if patch_path.exists():
                        feature_ids.append(fid)
                        patches[fid] = patch_path.read_text()

            if len(feature_ids) < group_size:
                continue

            feature_ids.sort()
            for combo in combinations(feature_ids, group_size):
                record = {
                    "repo": repo_dir.name,
                    "task_id": task_id,
                    "features": list(combo),
                    "patches": [patches[fid] for fid in combo],
                }
                # Backwards compat keys for group_size == 2
                if group_size == 2:
                    record["f1"] = combo[0]
                    record["f2"] = combo[1]
                    record["patch1"] = patches.get(combo[0], "")
                    record["patch2"] = patches.get(combo[1], "")
                groups.append(record)

    return groups


def _build_merge_script(n: int, features: list[int]) -> str:
    """Return a bash script that sets up N branches and attempts a sequential fold merge.

    Branch names use integer indices (agent1..agentN), not raw feature IDs,
    matching the convention in _setup_branches_n / _merge_fold in sandbox.py.

    Outputs one of:
        MERGE_RESULT=clean
        MERGE_RESULT=conflict   (first step that fails, then exits)
    Also outputs PATCH{i}_APPLY_FAILED for each patch that could not be applied.
    When a conflict is detected, also outputs:
        CONFLICT_HUNKS=<int>   number of conflict marker blocks
        CONFLICT_LINES=<int>   number of lines between <<<<<<< and >>>>>>> markers
        CONFLICT_FILES=<int>   number of files with conflicts
    """
    # Build branch-setup block: one stanza per agent (uses index i, not feature ID)
    branch_setup_parts = []
    for i, fid in enumerate(features, start=1):
        branch_setup_parts.append(f"""\
git checkout $BASE_SHA 2>&1
git checkout -b agent{i} 2>&1
if [ -s /patches/patch{i}.patch ]; then
    git apply --ignore-whitespace /patches/patch{i}.patch 2>&1 \\
        || git apply --3way /patches/patch{i}.patch 2>&1 \\
        || echo "PATCH{i}_APPLY_FAILED"
fi
git add -A
git commit -m "Feature {fid}" --allow-empty 2>&1
""")
    branch_setup = "\n".join(branch_setup_parts)

    # Build sequential fold block: merge agent2..agentN into agent1
    # On conflict, measure size before aborting (only first conflict matters).
    fold_parts = []
    for i in range(2, n + 1):
        fold_parts.append(f"""\
if git merge agent{i} --no-commit --no-ff 2>&1; then
    git commit -m "Fold agent{i}" --allow-empty 2>&1
else
    echo "MERGE_RESULT=conflict"
    git diff HEAD > /tmp/conflict_diff.txt 2>/dev/null
    HUNKS=$(grep -c '^<<<<<<< ' /tmp/conflict_diff.txt 2>/dev/null || echo 0)
    LINES=$(grep -c '^\\+' /tmp/conflict_diff.txt 2>/dev/null || echo 0)
    FILES=$(grep -c '^diff --git' /tmp/conflict_diff.txt 2>/dev/null || echo 0)
    echo "CONFLICT_HUNKS=$HUNKS"
    echo "CONFLICT_LINES=$LINES"
    echo "CONFLICT_FILES=$FILES"
    git merge --abort 2>/dev/null || true
    exit 0
fi
""")
    fold_block = "\n".join(fold_parts)

    return f"""\
cd /workspace/repo
git config user.email "eval@cooperbench.local"
git config user.name "CooperBench Eval"

BASE_SHA=$(git rev-parse HEAD)

{branch_setup}
# Sequential fold: merge agent2..agent{n} into agent1
git checkout agent1 2>&1
{fold_block}
echo "MERGE_RESULT=clean"
"""


def check_one_pair(pair: dict, timeout: int = 300) -> dict:
    """Check merge conflict for a single feature pair in its own Modal sandbox.

    Kept for backwards compatibility. Internally delegates to check_one_group.
    """
    return check_one_group(pair, timeout=timeout)


def check_one_group(group: dict, timeout: int = 300) -> dict:
    """Check merge conflict for a single feature group in its own Modal sandbox.

    Supports any group size >= 2.  For group_size == 2 the returned dict
    also carries 'f1', 'f2', 'patch1_apply_failed', 'patch2_apply_failed'
    so callers using the old schema continue to work without modification.

    The returned dict always includes 'conflict_hunks', 'conflict_lines', and
    'conflict_files' (all 0 when has_conflict=False or an error occurred).
    """
    repo = group["repo"]
    task_id = group["task_id"]
    features = group["features"]  # e.g. [1, 2] or [1, 2, 3]
    n = len(features)
    patches = [filter_test_files(p) for p in group["patches"]]
    image = get_image_name(repo, task_id)

    sb = None
    try:
        modal_image = modal.Image.from_registry(image).entrypoint([])
        app = modal.App.lookup("cooperbench-gold-conflicts", create_if_missing=True)
        sb = _rate_limited_create_sandbox(modal_image, timeout, app)

        # Setup
        r = sb.exec("mkdir", "-p", "/patches")
        r.wait()

        # Write each patch file
        for i, content in enumerate(patches, start=1):
            encoded = base64.b64encode(content.encode()).decode()
            r = sb.exec(
                "bash",
                "-c",
                f"echo '{encoded}' | base64 -d > /patches/patch{i}.patch",
            )
            r.wait()

        # Setup branches + try sequential fold merge
        merge_script = _build_merge_script(n, features)
        r = sb.exec("bash", "-c", merge_script)
        # Read stdout/stderr eagerly before calling wait() so we capture
        # output even if the sandbox shuts down immediately after the script
        # exits (Modal terminates containers quickly after the last exec).
        stdout = r.stdout.read()
        stderr = r.stderr.read()
        r.wait()
        output = stdout + stderr

        has_conflict = "MERGE_RESULT=conflict" in output
        patch_apply_failed = [f"PATCH{i}_APPLY_FAILED" in output for i in range(1, n + 1)]

        # Parse conflict size metrics (present only when has_conflict=True)
        conflict_hunks = 0
        conflict_lines = 0
        conflict_files = 0
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("CONFLICT_HUNKS="):
                try:
                    conflict_hunks = int(line.split("=", 1)[1] or "0")
                except ValueError:
                    conflict_hunks = 0
            elif line.startswith("CONFLICT_LINES="):
                try:
                    conflict_lines = int(line.split("=", 1)[1] or "0")
                except ValueError:
                    conflict_lines = 0
            elif line.startswith("CONFLICT_FILES="):
                try:
                    conflict_files = int(line.split("=", 1)[1] or "0")
                except ValueError:
                    conflict_files = 0

        result = {
            "repo": repo,
            "task_id": task_id,
            "features": features,
            "has_conflict": has_conflict,
            "patch_apply_failed": patch_apply_failed,  # list, index i-1 -> patch i
            "conflict_hunks": conflict_hunks,
            "conflict_lines": conflict_lines,
            "conflict_files": conflict_files,
            "error": None,
        }
        # Backwards compat keys for group_size == 2
        if n == 2:
            result["f1"] = features[0]
            result["f2"] = features[1]
            result["patch1_apply_failed"] = patch_apply_failed[0]
            result["patch2_apply_failed"] = patch_apply_failed[1]
        return result

    except Exception as e:
        result = {
            "repo": repo,
            "task_id": task_id,
            "features": features,
            "has_conflict": None,
            "patch_apply_failed": None,
            "conflict_hunks": 0,
            "conflict_lines": 0,
            "conflict_files": 0,
            "error": str(e),
        }
        if n == 2:
            result["f1"] = features[0]
            result["f2"] = features[1]
            result["patch1_apply_failed"] = None
            result["patch2_apply_failed"] = None
        return result

    finally:
        if sb is not None:
            try:
                sb.terminate()
            except Exception:
                pass  # sandbox already shut down — not an error


def build_report(all_results: list[dict], elapsed: float, group_size: int) -> dict:
    """Build the full report dict from raw results.

    For group_size == 2 the schema is identical to the original script output
    (total_pairs, conflict_pairs with f1/f2 keys, etc.).  For group_size > 2
    it uses total_groups and conflict_groups instead.
    """
    total_checked = len(all_results)
    conflicts = [r for r in all_results if r.get("has_conflict")]
    errors = [r for r in all_results if r.get("error")]
    clean = [r for r in all_results if r.get("has_conflict") is False]

    # Per-task breakdown (identical structure regardless of group_size)
    task_breakdown: dict[str, dict] = {}
    for r in all_results:
        key = f"{r['repo']}/task{r['task_id']}"
        if key not in task_breakdown:
            task_breakdown[key] = {"total": 0, "conflicts": 0, "clean": 0, "errors": 0}
        task_breakdown[key]["total"] += 1
        if r.get("error"):
            task_breakdown[key]["errors"] += 1
        elif r["has_conflict"]:
            task_breakdown[key]["conflicts"] += 1
        else:
            task_breakdown[key]["clean"] += 1

    summary: dict = {
        "group_size": group_size,
        "conflicts": len(conflicts),
        "clean_merges": len(clean),
        "errors": len(errors),
        "conflict_rate": (f"{len(conflicts) / total_checked * 100:.1f}%" if total_checked else "N/A"),
        "elapsed_seconds": round(elapsed, 1),
    }
    if group_size == 2:
        summary["total_pairs"] = total_checked
    else:
        summary["total_groups"] = total_checked

    conflict_records = [{"repo": r["repo"], "task_id": r["task_id"], "features": r["features"]} for r in conflicts]
    # Backwards compat: add f1/f2 to conflict records for group_size == 2
    if group_size == 2:
        for rec, r in zip(conflict_records, conflicts):
            rec["f1"] = r["f1"]
            rec["f2"] = r["f2"]

    report: dict = {
        "summary": summary,
        "per_task": task_breakdown,
        "all_results": all_results,
    }
    if group_size == 2:
        report["conflict_pairs"] = conflict_records
    else:
        report["conflict_groups"] = conflict_records

    return report


def write_subset_json(clean_results: list[dict], group_size: int, output_path: Path) -> None:
    """Write a dataset/subsets-compatible JSON for the clean groups.

    Format mirrors core.json / flash.json but uses a 'groups' key
    (list[list[int]]) rather than 'pairs' (list[list[int, int]]).
    Only called when --output-subset is given (requires --group-size > 2).
    """
    # Aggregate clean groups by (repo, task_id)
    by_task: dict[tuple[str, int], list[list[int]]] = defaultdict(list)
    for r in clean_results:
        by_task[(r["repo"], r["task_id"])].append(sorted(r["features"]))

    tasks_out = []
    for repo, task_id in sorted(by_task):
        tasks_out.append(
            {
                "repo": repo,
                "task_id": task_id,
                "groups": sorted(by_task[(repo, task_id)]),
            }
        )

    total_groups = sum(len(t["groups"]) for t in tasks_out)
    repos = len({t["repo"] for t in tasks_out})

    subset = {
        "name": output_path.stem,
        "description": (
            f"Clean {group_size}-tuple groups from gold conflict check. "
            f"{total_groups} groups across {len(tasks_out)} tasks."
        ),
        "stats": {
            "tasks": len(tasks_out),
            "groups": total_groups,
            "group_size": group_size,
            "repos": repos,
        },
        "tasks": tasks_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(subset, indent=2) + "\n")


def write_slight_subset_json(
    conflict_results: list[dict],
    group_size: int,
    output_path: Path,
    top_n: int,
) -> None:
    """Write a dataset/subsets-compatible JSON for the N smallest conflicts.

    Only includes triples where has_conflict=True AND conflict_lines > 0.
    Entries with conflict_lines == 0 (measurement failed) are excluded entirely
    rather than ranked last, since their true size is unknown.

    Sorted ascending by (conflict_lines, conflict_hunks). The top_n entries
    are written. Each task entry carries a 'group_conflict_info' dict keyed by
    the string representation of the feature list, e.g. "[1, 2, 3]".
    """
    # Filter: must have has_conflict=True and a measurable conflict_lines > 0
    measurable = [
        r for r in conflict_results
        if r.get("has_conflict") is True and r.get("conflict_lines", 0) > 0
    ]

    # Sort ascending by (conflict_lines, conflict_hunks)
    measurable.sort(key=lambda r: (r.get("conflict_lines", 0), r.get("conflict_hunks", 0)))

    selected = measurable[:top_n]

    # Aggregate by (repo, task_id), preserving sort order for group list
    # Use an ordered structure: track insertion order via a list of keys.
    by_task: dict[tuple[str, int], dict] = {}
    task_key_order: list[tuple[str, int]] = []
    for r in selected:
        key = (r["repo"], r["task_id"])
        if key not in by_task:
            by_task[key] = {"groups": [], "group_conflict_info": {}}
            task_key_order.append(key)
        features_sorted = sorted(r["features"])
        by_task[key]["groups"].append(features_sorted)
        info_key = str(features_sorted)
        by_task[key]["group_conflict_info"][info_key] = {
            "conflict_lines": r.get("conflict_lines", 0),
            "conflict_hunks": r.get("conflict_hunks", 0),
            "conflict_files": r.get("conflict_files", 0),
        }

    tasks_out = []
    for repo, task_id in task_key_order:
        entry = by_task[(repo, task_id)]
        tasks_out.append(
            {
                "repo": repo,
                "task_id": task_id,
                "groups": entry["groups"],
                "group_conflict_info": entry["group_conflict_info"],
            }
        )

    total_groups = sum(len(t["groups"]) for t in tasks_out)
    repos = len({t["repo"] for t in tasks_out})
    max_conflict_lines = max((r.get("conflict_lines", 0) for r in selected), default=0)

    subset = {
        "name": output_path.stem,
        "description": (
            f"Smallest-conflict {group_size}-tuple groups from gold conflict check "
            f"(top {top_n} by conflict_lines). "
            f"{total_groups} groups across {len(tasks_out)} tasks."
        ),
        "stats": {
            "tasks": len(tasks_out),
            "groups": total_groups,
            "group_size": group_size,
            "repos": repos,
            "max_conflict_lines": max_conflict_lines,
        },
        "tasks": tasks_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(subset, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Check gold patch merge conflicts")
    parser.add_argument("--repo", type=str, default=None, help="Filter by repo name")
    parser.add_argument("--task", type=int, default=None, help="Filter by task ID")
    parser.add_argument("--max-workers", type=int, default=75, help="Max parallel sandboxes")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATASET_DIR / "gold_conflict_report.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=2,
        metavar="N",
        help="Check N-tuple feature groups instead of pairs (default: 2)",
    )
    parser.add_argument(
        "--output-subset",
        type=str,
        default=None,
        help=(
            "When --group-size > 2, write a dataset/subsets-compatible JSON "
            "containing only clean (conflict-free) groups to this path."
        ),
    )
    parser.add_argument(
        "--output-slight",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "When --group-size > 2, write a dataset/subsets-compatible JSON "
            "containing the --slight-n triples with the smallest conflicts "
            "(has_conflict=True, ranked by conflict_lines ASC then conflict_hunks ASC) "
            "to this path."
        ),
    )
    parser.add_argument(
        "--slight-n",
        type=int,
        default=50,
        metavar="N",
        help="Number of slight-conflict triples to select for --output-slight (default: 50)",
    )
    args = parser.parse_args()

    if args.group_size < 2:
        parser.error("--group-size must be >= 2")
    if args.output_subset and args.group_size == 2:
        parser.error("--output-subset is only valid when --group-size > 2")
    if args.output_slight and args.group_size == 2:
        parser.error("--output-slight is only valid when --group-size > 2")
    if args.slight_n < 1:
        parser.error("--slight-n must be >= 1")

    noun = "pairs" if args.group_size == 2 else f"{args.group_size}-tuples"
    print(f"Discovering {noun}...")
    groups = discover_all_groups(
        group_size=args.group_size,
        repo_filter=args.repo,
        task_filter=args.task,
    )
    print(f"Found {len(groups)} feature {noun} to check")
    print(f"Running with {args.max_workers} parallel sandboxes")

    start = time.time()
    all_results = []
    done_count = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(check_one_group, g): g for g in groups}

        for future in as_completed(futures):
            done_count += 1
            g = futures[future]
            features_str = "+".join(f"f{f}" for f in g["features"])
            try:
                result = future.result()
                all_results.append(result)
                status = "CONFLICT" if result["has_conflict"] else "clean"
                if result["has_conflict"] and result.get("conflict_lines", 0) > 0:
                    status = (
                        f"CONFLICT (lines={result['conflict_lines']} "
                        f"hunks={result['conflict_hunks']} "
                        f"files={result['conflict_files']})"
                    )
                if result["error"]:
                    status = f"ERROR: {result['error'][:60]}"
                print(f"  [{done_count}/{len(groups)}] {g['repo']}/task{g['task_id']} {features_str}: {status}")
            except Exception as e:
                print(f"  [{done_count}/{len(groups)}] FATAL: {e}")

    elapsed = time.time() - start
    report = build_report(all_results, elapsed, args.group_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    s = report["summary"]
    total_checked = s.get("total_pairs") or s.get("total_groups", 0)
    conflicts_count = s["conflicts"]
    clean_count = s["clean_merges"]
    errors_count = s["errors"]

    print("\n" + "=" * 60)
    print("GOLD PATCH MERGE CONFLICT REPORT")
    print("=" * 60)
    print(f"Group size:           {args.group_size}")
    print(f"Total {noun} checked: {total_checked}")
    print(f"Clean merges:         {clean_count}")
    print(f"Conflicts:            {conflicts_count}")
    print(f"Errors:               {errors_count}")
    print(f"Conflict rate:        {s['conflict_rate']}")
    print(f"Time elapsed:         {elapsed:.1f}s")
    print()

    print("Per-task breakdown:")
    print(f"{'Task':<50} {'Total':>6} {'Conflict':>9} {'Clean':>6} {'Error':>6}")
    print("-" * 80)
    task_breakdown = report["per_task"]
    for key in sorted(task_breakdown.keys()):
        b = task_breakdown[key]
        print(f"{key:<50} {b['total']:>6} {b['conflicts']:>9} {b['clean']:>6} {b['errors']:>6}")

    conflict_records = report.get("conflict_pairs") or report.get("conflict_groups", [])
    if conflict_records:
        print(f"\nConflicting {noun} ({len(conflict_records)}):")
        for rec in conflict_records:
            features_str = " + ".join(f"feature{f}" for f in rec["features"])
            print(f"  {rec['repo']}/task{rec['task_id']}: {features_str}")

    print(f"\nFull report saved to: {output_path}")

    # Optional subset JSON for clean groups (only valid with --group-size > 2)
    if args.output_subset:
        clean_results = [r for r in all_results if r.get("has_conflict") is False]
        write_subset_json(clean_results, args.group_size, Path(args.output_subset))
        print(f"Clean-groups subset saved to: {args.output_subset}")

    # Optional subset JSON for slight (smallest) conflicts (only valid with --group-size > 2)
    if args.output_slight:
        conflict_results = [r for r in all_results if r.get("has_conflict") is True]
        write_slight_subset_json(conflict_results, args.group_size, Path(args.output_slight), args.slight_n)
        print(f"Slight-conflicts subset saved to: {args.output_slight}")


if __name__ == "__main__":
    main()
