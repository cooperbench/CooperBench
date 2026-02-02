"""Validation - check conflicts and test results using Modal sandboxes."""

import re
from pathlib import Path

from cooperbench.eval.backends import get_backend
from cooperbench.utils import get_image_name
from cooperbench.eval.sandbox import _parse_results, _write_patch


def _extract_feature_title(feature_md_path: Path) -> str | None:
    """Extract title from a feature.md file."""
    if not feature_md_path.exists():
        return None

    content = feature_md_path.read_text()
    # Look for **Title**: pattern
    match = re.search(r"\*\*Title\*\*:\s*(.+?)(?:\n|$)", content)
    if match:
        return match.group(1).strip()
    return None


def check_conflicts_in_sandbox(
    repo_name: str,
    task_id: int,
    new_feature_patch: str,
    existing_feature_ids: list[int],
    timeout: int = 300,
    backend: str = "modal",
) -> dict:
    """Check which existing features conflict with a new feature.

    Runs inside a Modal sandbox with the task's Docker image.

    Args:
        repo_name: Repository name (e.g., "dspy_task")
        task_id: Task ID
        new_feature_patch: The new feature patch as a string
        existing_feature_ids: List of existing feature IDs to check against
        timeout: Sandbox timeout in seconds
        backend: Execution backend ("modal", "docker")

    Returns:
        Dict with:
        - conflicts: list[int] - feature IDs that conflict
        - clean: list[int] - feature IDs that merge cleanly
        - errors: list[str] - any errors encountered
        - output: str - raw output from sandbox
    """
    task_dir = Path("dataset") / repo_name / f"task{task_id}"

    if not task_dir.exists():
        return {"conflicts": [], "clean": [], "errors": [f"Task dir not found: {task_dir}"], "output": ""}

    image = get_image_name(repo_name, task_id)
    eval_backend = get_backend(backend)
    sb = eval_backend.create_sandbox(image, timeout)

    try:
        # Write the new patch to sandbox
        _write_patch(sb, "new_feature.patch", new_feature_patch)

        # Extract feature titles for commit messages
        feature_titles = {}
        for fid in existing_feature_ids:
            feature_md_path = task_dir / f"feature{fid}" / "feature.md"
            title = _extract_feature_title(feature_md_path)
            feature_titles[fid] = title or f"Feature {fid}"

        # Write existing feature patches
        for fid in existing_feature_ids:
            feature_patch_path = task_dir / f"feature{fid}" / "feature.patch"
            if feature_patch_path.exists():
                content = feature_patch_path.read_text()
                _write_patch(sb, f"feature{fid}.patch", content)

        # Run conflict checking script
        script = _build_conflict_check_script(existing_feature_ids, feature_titles)

        result = sb.exec("bash", "-c", script)
        output = result.stdout_read() + result.stderr_read()

        # Parse results and collect feature info
        conflicts = []
        conflicts_info = []
        clean = []
        errors = []

        # Parse output line by line, capturing conflict content
        lines = output.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("CONFLICT_START:"):
                # Format: CONFLICT_START:fid then content until CONFLICT_END:fid
                fid = int(line.split(":")[1].strip())
                conflict_content = []
                i += 1
                while i < len(lines) and not lines[i].startswith(f"CONFLICT_END:{fid}"):
                    conflict_content.append(lines[i])
                    i += 1
                conflicts.append(fid)
                # Get title from feature_titles we extracted earlier
                title = feature_titles.get(fid, f"Feature {fid}")
                conflicts_info.append({
                    "id": fid,
                    "title": title,
                    "conflict_diff": "\n".join(conflict_content),
                })
            elif line.startswith("CLEAN:"):
                # Format: CLEAN:fid
                fid = int(line.split(":")[1].strip())
                clean.append(fid)
            elif line.startswith("ERROR:"):
                errors.append(line)
            i += 1

        return {
            "conflicts": conflicts,
            "conflicts_info": conflicts_info,
            "clean": clean,
            "errors": errors,
            "output": output,
        }

    except Exception as e:
        return {"conflicts": [], "clean": [], "errors": [str(e)], "output": ""}
    finally:
        sb.terminate()


def run_tests_in_sandbox(
    repo_name: str,
    task_id: int,
    feature_patch: str,
    tests_patch: str,
    timeout: int = 600,
    backend: str = "modal",
) -> dict:
    """Run the NEW tests for a generated feature in a Modal sandbox.

    Uses runner.sh which handles task-specific environment setup (deps, etc).
    Passes the new test files as the 3rd param (requires updated runner.sh).

    Args:
        repo_name: Repository name
        task_id: Task ID
        feature_patch: The feature implementation patch
        tests_patch: The tests patch
        timeout: Sandbox timeout
        backend: Execution backend

    Returns:
        Dict with: passed, tests_passed, tests_failed, output, error
    """
    import logging
    logger = logging.getLogger(__name__)

    image = get_image_name(repo_name, task_id)
    logger.debug(f"Creating sandbox with image: {image}")
    eval_backend = get_backend(backend)
    sb = eval_backend.create_sandbox(image, timeout)
    logger.debug("Sandbox created successfully")

    try:
        # Write patches to /patches/ directory
        logger.debug(f"Writing tests.patch ({len(tests_patch)} bytes)")
        _write_patch(sb, "tests.patch", tests_patch)
        logger.debug(f"Writing feature.patch ({len(feature_patch)} bytes)")
        _write_patch(sb, "feature.patch", feature_patch)

        # Extract NEW test function names from the patch (not just files)
        # This ensures we only run tests added by the agent, not pre-existing tests
        new_test_specs = _extract_new_test_functions(tests_patch)
        test_path = " ".join(new_test_specs) if new_test_specs else ""
        logger.debug(f"New test functions to run: {test_path}")

        # Use runner.sh with: tests.patch feature.patch [test_path]
        # - Old images: 3rd param ignored, runs default tests
        # - New images: runs the specific new test files
        if test_path:
            logger.debug(f"Running: bash /usr/local/bin/runner.sh tests.patch feature.patch {test_path}")
            result = sb.exec("bash", "/usr/local/bin/runner.sh", "tests.patch", "feature.patch", test_path)
        else:
            logger.debug("Running: bash /usr/local/bin/runner.sh tests.patch feature.patch")
            result = sb.exec("bash", "/usr/local/bin/runner.sh", "tests.patch", "feature.patch")
        logger.debug(f"Runner completed with exit code: {result.returncode}")

        output = result.stdout_read() + result.stderr_read()
        exit_code = result.returncode

        # Parse test results (reuse existing parser that handles pytest, go, cargo, jest)
        parsed = _parse_results(output)

        return {
            "passed": exit_code == 0 and parsed["passed"] > 0,
            "tests_passed": parsed["passed"],
            "tests_failed": parsed["failed"],
            "output": output,
            "error": None,
        }
    except Exception as e:
        return {
            "passed": False,
            "tests_passed": 0,
            "tests_failed": 0,
            "output": "",
            "error": str(e),
        }
    finally:
        sb.terminate()


def _extract_test_files_from_patch(patch: str) -> list[str]:
    """Extract new/modified file paths from a patch."""
    import re
    files = []
    for match in re.finditer(r"^\+\+\+ b/(.+)$", patch, re.MULTILINE):
        path = match.group(1)
        if path and not path.startswith("/dev/null"):
            files.append(path)
    return files


def _extract_new_test_functions(patch: str) -> list[str]:
    """Extract new test function names from a patch with their file paths.

    Returns paths in pytest format: path/to/test.py::test_function_name
    """
    import re

    test_specs = []
    current_file = None

    for line in patch.split("\n"):
        # Track which file we're in
        if line.startswith("+++ b/"):
            current_file = line[6:]  # Remove "+++ b/" prefix
        # Find new test function definitions (lines starting with +def test_)
        elif line.startswith("+def test_") and current_file:
            # Extract function name: +def test_foo(args): -> test_foo
            match = re.match(r"\+def (test_\w+)\s*\(", line)
            if match:
                func_name = match.group(1)
                test_specs.append(f"{current_file}::{func_name}")

    return test_specs


def validate_generated_feature(
    repo_name: str,
    task_id: int,
    feature_patch: str,
    tests_patch: str,
    min_conflicts: int = 1,
    timeout: int = 600,
    backend: str = "modal",
) -> dict:
    """Full validation of a generated feature.

    Checks:
    1. Tests pass
    2. Conflicts with at least min_conflicts existing features

    Args:
        repo_name: Repository name
        task_id: Task ID
        feature_patch: The feature implementation patch
        tests_patch: The tests patch
        min_conflicts: Minimum required conflicts (default: 1)
        timeout: Sandbox timeout
        backend: Execution backend

    Returns:
        Dict with validation results
    """
    task_dir = Path("dataset") / repo_name / f"task{task_id}"

    # Get existing feature IDs
    existing_ids = _get_existing_feature_ids(task_dir)

    # Step 1: Run tests
    test_result = run_tests_in_sandbox(
        repo_name=repo_name,
        task_id=task_id,
        feature_patch=feature_patch,
        tests_patch=tests_patch,
        timeout=timeout,
        backend=backend,
    )

    if not test_result["passed"]:
        return {
            "valid": False,
            "reason": "tests_failed",
            "test_result": test_result,
            "conflict_result": None,
        }

    # Step 2: Check conflicts
    conflict_result = check_conflicts_in_sandbox(
        repo_name=repo_name,
        task_id=task_id,
        new_feature_patch=feature_patch,
        existing_feature_ids=existing_ids,
        timeout=timeout,
        backend=backend,
    )

    num_conflicts = len(conflict_result["conflicts"])

    if num_conflicts < min_conflicts:
        return {
            "valid": False,
            "reason": f"insufficient_conflicts ({num_conflicts} < {min_conflicts})",
            "test_result": test_result,
            "conflict_result": conflict_result,
        }

    return {
        "valid": True,
        "reason": None,
        "test_result": test_result,
        "conflict_result": conflict_result,
    }


def _build_conflict_check_script(feature_ids: list[int], feature_titles: dict[int, str]) -> str:
    """Build bash script for checking REAL git merge conflicts.

    For each existing feature:
    1. Create branch A from base, apply existing feature, commit
    2. Create branch B from base, apply new feature, commit
    3. Try git merge --no-commit from A
    4. Check if merge has conflicts (git merge --abort needed)
    """
    def _build_feature_check(fid: int, title: str) -> str:
        # Escape title for shell - replace : with space to avoid parsing issues
        safe_title = title.replace(":", " -").replace("'", "\\'").replace('"', '\\"')
        return f'''
# Check feature {fid} for REAL merge conflicts
echo "Checking feature {fid}..."
git checkout --quiet $BASE_SHA
git clean -fd >/dev/null 2>&1

# Branch A: existing feature {fid}
git checkout --quiet -b __existing_{fid}
if ! git apply /patches/feature{fid}.patch 2>/dev/null; then
    echo "ERROR:feature{fid} patch failed to apply"
    git checkout --quiet $BASE_SHA 2>/dev/null || true
    git branch -D __existing_{fid} 2>/dev/null || true
    continue
fi
git add -A
git commit -qm "{safe_title}" --allow-empty

# Branch B: new feature (from base)
git checkout --quiet $BASE_SHA
git checkout --quiet -b __new_{fid}
if ! git apply /patches/new_feature.patch 2>/dev/null; then
    echo "ERROR:new_feature patch failed to apply for check {fid}"
    git checkout --quiet $BASE_SHA 2>/dev/null || true
    git branch -D __existing_{fid} 2>/dev/null || true
    git branch -D __new_{fid} 2>/dev/null || true
    continue
fi
git add -A
git commit -qm "new feature" --allow-empty

# Try to merge existing feature into new feature branch
# --no-commit so we can check for conflicts without auto-commit
if git merge --no-commit --no-ff __existing_{fid} 2>/dev/null; then
    # Merge succeeded cleanly
    echo "CLEAN:{fid}"
    git reset --hard HEAD >/dev/null 2>&1
else
    # Merge has conflicts! Capture the actual conflict content
    echo "CONFLICT_START:{fid}"
    # Show files with conflict markers (<<<<<<< ======= >>>>>>>)
    for f in $(git diff --name-only --diff-filter=U 2>/dev/null); do
        echo "--- $f ---"
        cat "$f" | grep -A 50 -B 5 "<<<<<<" | head -100
    done
    echo "CONFLICT_END:{fid}"
    git merge --abort 2>/dev/null || git reset --hard HEAD >/dev/null 2>&1
fi

# Cleanup branches
git checkout --quiet $BASE_SHA 2>/dev/null || true
git branch -D __existing_{fid} 2>/dev/null || true
git branch -D __new_{fid} 2>/dev/null || true
'''

    feature_checks = "\n".join(
        _build_feature_check(fid, feature_titles.get(fid, f"Feature {fid}"))
        for fid in feature_ids
    )

    return f'''
cd /workspace/repo

# Get base commit
BASE_SHA=$(git rev-parse HEAD)

# Ensure clean state
git reset --hard HEAD >/dev/null 2>&1
git clean -fd >/dev/null 2>&1

# Configure git for commits
git config user.email "test@test.com" 2>/dev/null || true
git config user.name "Test" 2>/dev/null || true

{feature_checks}

# Final cleanup
git checkout --quiet $BASE_SHA 2>/dev/null || true
git reset --hard HEAD >/dev/null 2>&1
'''


def _get_existing_feature_ids(task_dir: Path) -> list[int]:
    """Get IDs of existing features in a task."""
    ids = []
    for d in task_dir.iterdir():
        if d.is_dir() and d.name.startswith("feature"):
            try:
                fid = int(d.name.replace("feature", ""))
                ids.append(fid)
            except ValueError:
                pass
    return sorted(ids)
