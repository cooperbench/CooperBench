"""Check which gold patch pairs have merge conflicts across all CooperBench tasks.

Spins up one Modal sandbox per feature pair, running up to 75 in parallel.
652 pairs total => ~9 rounds of 75.

Results are saved to dataset/gold_conflict_report.json.

Usage:
    python scripts/check_gold_conflicts.py
    python scripts/check_gold_conflicts.py --repo pallets_click_task
    python scripts/check_gold_conflicts.py --max-workers 75
"""

import argparse
import base64
import json
import threading
import time
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
            skip = any(
                p in line
                for p in ["/test_", "/tests/", "_test.py", "/test/", "tests.py"]
            )
        if not skip:
            filtered_lines.append(line)

    result = "\n".join(filtered_lines)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


def discover_all_pairs(
    repo_filter: str | None = None, task_filter: int | None = None
) -> list[dict]:
    """Discover all feature pairs, each with their gold patches pre-loaded."""
    pairs = []

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

            if len(feature_ids) < 2:
                continue

            feature_ids.sort()
            for f1, f2 in combinations(feature_ids, 2):
                pairs.append(
                    {
                        "repo": repo_dir.name,
                        "task_id": task_id,
                        "f1": f1,
                        "f2": f2,
                        "patch1": patches.get(f1, ""),
                        "patch2": patches.get(f2, ""),
                    }
                )

    return pairs


def check_one_pair(pair: dict, timeout: int = 300) -> dict:
    """Check merge conflict for a single feature pair in its own Modal sandbox."""
    repo = pair["repo"]
    task_id = pair["task_id"]
    f1, f2 = pair["f1"], pair["f2"]
    patch1 = filter_test_files(pair["patch1"])
    patch2 = filter_test_files(pair["patch2"])
    image = get_image_name(repo, task_id)

    sb = None
    try:
        modal_image = modal.Image.from_registry(image).entrypoint([])
        app = modal.App.lookup("cooperbench-gold-conflicts", create_if_missing=True)
        sb = _rate_limited_create_sandbox(modal_image, timeout, app)

        # Setup
        r = sb.exec("mkdir", "-p", "/patches")
        r.wait()

        # Write patches
        for fname, content in [("patch1.patch", patch1), ("patch2.patch", patch2)]:
            encoded = base64.b64encode(content.encode()).decode()
            r = sb.exec("bash", "-c", f"echo '{encoded}' | base64 -d > /patches/{fname}")
            r.wait()

        # Setup branches + try naive merge
        merge_script = f"""
cd /workspace/repo
git config user.email "eval@cooperbench.local"
git config user.name "CooperBench Eval"

BASE_SHA=$(git rev-parse HEAD)

# Branch 1: apply gold patch for feature {f1}
git checkout -b agent1 2>&1
if [ -s /patches/patch1.patch ]; then
    git apply --ignore-whitespace /patches/patch1.patch 2>&1 || git apply --3way /patches/patch1.patch 2>&1 || echo "PATCH1_APPLY_FAILED"
fi
git add -A
git commit -m "Feature {f1}" --allow-empty 2>&1

# Branch 2: apply gold patch for feature {f2}
git checkout $BASE_SHA 2>&1
git checkout -b agent2 2>&1
if [ -s /patches/patch2.patch ]; then
    git apply --ignore-whitespace /patches/patch2.patch 2>&1 || git apply --3way /patches/patch2.patch 2>&1 || echo "PATCH2_APPLY_FAILED"
fi
git add -A
git commit -m "Feature {f2}" --allow-empty 2>&1

# Try naive merge (agent1 into agent2)
if git merge agent1 --no-commit --no-ff 2>&1; then
    echo "MERGE_RESULT=clean"
else
    echo "MERGE_RESULT=conflict"
    git merge --abort 2>/dev/null || true
fi
"""
        r = sb.exec("bash", "-c", merge_script)
        r.wait()
        output = r.stdout.read() + r.stderr.read()

        has_conflict = "MERGE_RESULT=conflict" in output
        patch1_failed = "PATCH1_APPLY_FAILED" in output
        patch2_failed = "PATCH2_APPLY_FAILED" in output

        return {
            "repo": repo,
            "task_id": task_id,
            "f1": f1,
            "f2": f2,
            "has_conflict": has_conflict,
            "patch1_apply_failed": patch1_failed,
            "patch2_apply_failed": patch2_failed,
            "error": None,
        }

    except Exception as e:
        return {
            "repo": repo,
            "task_id": task_id,
            "f1": f1,
            "f2": f2,
            "has_conflict": None,
            "patch1_apply_failed": None,
            "patch2_apply_failed": None,
            "error": str(e),
        }
    finally:
        if sb is not None:
            sb.terminate()


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
    args = parser.parse_args()

    print("Discovering pairs...")
    pairs = discover_all_pairs(repo_filter=args.repo, task_filter=args.task)
    print(f"Found {len(pairs)} feature pairs to check")
    print(f"Running with {args.max_workers} parallel sandboxes")

    start = time.time()
    all_results = []
    done_count = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(check_one_pair, p): p for p in pairs}

        for future in as_completed(futures):
            done_count += 1
            p = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                status = "CONFLICT" if result["has_conflict"] else "clean"
                if result["error"]:
                    status = f"ERROR: {result['error'][:60]}"
                print(
                    f"  [{done_count}/{len(pairs)}] "
                    f"{p['repo']}/task{p['task_id']} f{p['f1']}+f{p['f2']}: {status}"
                )
            except Exception as e:
                print(f"  [{done_count}/{len(pairs)}] FATAL: {e}")

    elapsed = time.time() - start

    # Compute summary
    total_checked = len(all_results)
    conflicts = [r for r in all_results if r.get("has_conflict")]
    errors = [r for r in all_results if r.get("error")]
    clean = [r for r in all_results if r.get("has_conflict") is False]

    # Per-task breakdown
    task_breakdown = {}
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

    report = {
        "summary": {
            "total_pairs": total_checked,
            "conflicts": len(conflicts),
            "clean_merges": len(clean),
            "errors": len(errors),
            "conflict_rate": f"{len(conflicts)/total_checked*100:.1f}%" if total_checked else "N/A",
            "elapsed_seconds": round(elapsed, 1),
        },
        "per_task": task_breakdown,
        "conflict_pairs": [
            {"repo": r["repo"], "task_id": r["task_id"], "f1": r["f1"], "f2": r["f2"]}
            for r in conflicts
        ],
        "all_results": all_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("GOLD PATCH MERGE CONFLICT REPORT")
    print("=" * 60)
    print(f"Total pairs checked:  {total_checked}")
    print(f"Clean merges:         {len(clean)}")
    print(f"Conflicts:            {len(conflicts)}")
    print(f"Errors:               {len(errors)}")
    print(f"Conflict rate:        {report['summary']['conflict_rate']}")
    print(f"Time elapsed:         {elapsed:.1f}s")
    print()

    print("Per-task breakdown:")
    print(f"{'Task':<50} {'Total':>6} {'Conflict':>9} {'Clean':>6} {'Error':>6}")
    print("-" * 80)
    for key in sorted(task_breakdown.keys()):
        b = task_breakdown[key]
        print(f"{key:<50} {b['total']:>6} {b['conflicts']:>9} {b['clean']:>6} {b['errors']:>6}")

    if conflicts:
        print(f"\nConflicting pairs ({len(conflicts)}):")
        for r in conflicts:
            print(f"  {r['repo']}/task{r['task_id']}: feature{r['f1']} vs feature{r['f2']}")

    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()
