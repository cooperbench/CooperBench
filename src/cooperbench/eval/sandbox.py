"""Sandbox execution for patch testing."""

import base64
import re
from pathlib import Path

from cooperbench.eval.backends import get_backend
from cooperbench.eval.backends.base import Sandbox
from cooperbench.utils import get_image_name


def run_patch_test(
    repo_name: str,
    task_id: int,
    feature_id: int,
    agent_patch: str | Path | None = None,
    timeout: int = 600,
    backend: str = "modal",
) -> dict:
    """Test a single patch against one feature's tests.

    Args:
        repo_name: Repository name (e.g., "llama_index_task")
        task_id: Task ID from dataset/
        feature_id: Which feature's tests to run
        agent_patch: Patch content (str) or path to .patch file
        timeout: Max seconds for sandbox execution
        backend: Evaluation backend ("modal", "docker", "gcp_batch")

    Returns:
        Dict with keys: passed, tests_passed, tests_failed, output, error
    """
    task_dir = Path("dataset") / repo_name / f"task{task_id}"
    feature_dir = task_dir / f"feature{feature_id}"
    tests_patch_path = feature_dir / "tests.patch"
    gold_patch_path = feature_dir / "feature.patch"

    if not tests_patch_path.exists():
        return _error_result(f"Tests patch not found: {tests_patch_path}")

    tests_patch = tests_patch_path.read_text()

    # If no agent patch provided, use the gold patch from dataset
    if agent_patch is None and gold_patch_path.exists():
        agent_patch = gold_patch_path

    agent_patch_content = _load_patch(agent_patch)

    # Filter test files from agent patch
    if agent_patch_content:
        agent_patch_content = _filter_test_files(agent_patch_content)

    if agent_patch is not None and not agent_patch_content:
        return _error_result("Agent patch is empty")

    image = get_image_name(repo_name, task_id)
    eval_backend = get_backend(backend)
    sb = eval_backend.create_sandbox(image, timeout)

    try:
        _write_patch(sb, "tests.patch", tests_patch)
        if agent_patch_content:
            _write_patch(sb, "agent.patch", agent_patch_content)

        # Use runner.sh with [tests.patch, feature.patch]
        if agent_patch_content:
            result = sb.exec("bash", "/usr/local/bin/runner.sh", "tests.patch", "agent.patch")
        else:
            result = sb.exec("bash", "/usr/local/bin/runner.sh", "tests.patch")

        output = result.stdout_read() + result.stderr_read()
        exit_code = result.returncode
        parsed = _parse_results(output)

        return {
            "passed": exit_code == 0 and parsed["passed"] > 0,
            "tests_passed": parsed["passed"],
            "tests_failed": parsed["failed"],
            "tests_total": parsed["passed"] + parsed["failed"],
            "output": output,
            "error": None,
        }
    except Exception as e:
        return _error_result(str(e))
    finally:
        sb.terminate()


def test_merged(
    repo_name: str,
    task_id: int,
    feature1_id: int,
    feature2_id: int,
    patch1: str | Path | None = None,
    patch2: str | Path | None = None,
    timeout: int = 600,
    backend: str = "modal",
) -> dict:
    """Test merged patches from two agents (coop mode).

    Creates two git branches, applies each agent's patch, merges them,
    then tests the merged result against both feature test suites.

    Args:
        repo_name: Repository name
        task_id: Task ID
        feature1_id: First feature ID (agent1's task)
        feature2_id: Second feature ID (agent2's task)
        patch1: First agent's patch
        patch2: Second agent's patch
        timeout: Max seconds for sandbox execution
        backend: Evaluation backend

    Returns:
        Dict with keys: merge (status/strategy/diff), feature1, feature2,
        both_passed, error
    """
    task_dir = Path("dataset") / repo_name / f"task{task_id}"

    tests1_path = task_dir / f"feature{feature1_id}" / "tests.patch"
    tests2_path = task_dir / f"feature{feature2_id}" / "tests.patch"

    if not tests1_path.exists():
        return _merged_error_result(f"Tests patch not found: {tests1_path}")
    if not tests2_path.exists():
        return _merged_error_result(f"Tests patch not found: {tests2_path}")

    patch1_content = _load_patch(patch1) or ""
    patch2_content = _load_patch(patch2) or ""

    # Filter test files from patches
    patch1_content = _filter_test_files(patch1_content)
    patch2_content = _filter_test_files(patch2_content)

    tests1_content = tests1_path.read_text()
    tests2_content = tests2_path.read_text()

    image = get_image_name(repo_name, task_id)
    eval_backend = get_backend(backend)
    sb = eval_backend.create_sandbox(image, timeout)

    try:
        # Write all patches
        _write_patch(sb, "patch1.patch", patch1_content)
        _write_patch(sb, "patch2.patch", patch2_content)
        _write_patch(sb, "tests1.patch", tests1_content)
        _write_patch(sb, "tests2.patch", tests2_content)

        # Step 1: Apply patches to branches
        setup_result = _setup_branches(sb)
        if setup_result.get("error"):
            return _merged_error_result(setup_result["error"])

        base_sha = setup_result.get("base_sha")
        if not base_sha:
            return _merged_error_result("Failed to get base commit SHA")

        # Step 2: Try naive merge
        naive_result = _merge_naive(sb, base_sha)

        merge_status = "clean" if not naive_result["conflict"] else "conflicts"
        strategy_used = "naive"
        merged_diff = naive_result["diff"]

        # Step 3: If conflicts, try union merge
        if naive_result["conflict"]:
            union_result = _merge_union(sb, base_sha)
            if not union_result.get("error"):
                strategy_used = "union"
                merged_diff = union_result["diff"]
            else:
                # Both naive and union failed - cannot proceed
                return _merged_error_result(
                    f"Both naive and union merge strategies failed. "
                    f"Naive: conflicts. Union: {union_result.get('error')}"
                )

        # Step 4: Copy the right diff file to merged.patch
        if strategy_used == "naive":
            sb.exec("cp", "/patches/naive_diff.patch", "/patches/merged.patch")
        else:
            sb.exec("cp", "/patches/union_diff.patch", "/patches/merged.patch")

        # Verify merged.patch was created
        verify = sb.exec("test", "-f", "/patches/merged.patch")
        if verify.returncode != 0:
            return _merged_error_result(f"Failed to create merged.patch (strategy: {strategy_used})")

        # Test feature 1
        test1_result = _run_tests(sb, "tests1.patch", "merged.patch", base_sha)

        # Test feature 2
        test2_result = _run_tests(sb, "tests2.patch", "merged.patch", base_sha)

        return {
            "merge": {
                "status": merge_status,
                "strategy": strategy_used,
                "diff": merged_diff[:5000] if merged_diff else "",  # Truncate for storage
            },
            "feature1": {
                "passed": test1_result["passed"],
                "test_output": test1_result["output"],
            },
            "feature2": {
                "passed": test2_result["passed"],
                "test_output": test2_result["output"],
            },
            "both_passed": test1_result["passed"] and test2_result["passed"],
            "error": None,
        }
    except Exception as e:
        return _merged_error_result(str(e))
    finally:
        sb.terminate()


def test_solo(
    repo_name: str,
    task_id: int,
    feature1_id: int,
    feature2_id: int,
    patch: str | Path | None = None,
    timeout: int = 600,
    backend: str = "modal",
) -> dict:
    """Test a solo patch against both features' tests.

    In solo mode, one agent implements both features in a single patch.
    We test that patch against each feature's test suite separately.

    Args:
        repo_name: Repository name
        task_id: Task ID
        feature1_id: First feature ID
        feature2_id: Second feature ID
        patch: The solo agent's combined patch
        timeout: Max seconds for sandbox execution
        backend: Evaluation backend

    Returns:
        Dict with keys: setting, patch_lines, feature1, feature2,
        both_passed, error
    """
    task_dir = Path("dataset") / repo_name / f"task{task_id}"

    tests1_path = task_dir / f"feature{feature1_id}" / "tests.patch"
    tests2_path = task_dir / f"feature{feature2_id}" / "tests.patch"

    if not tests1_path.exists():
        return _solo_error_result(f"Tests patch not found: {tests1_path}")
    if not tests2_path.exists():
        return _solo_error_result(f"Tests patch not found: {tests2_path}")

    patch_content = _load_patch(patch) or ""

    # Filter test files from patch
    patch_content = _filter_test_files(patch_content)

    tests1_content = tests1_path.read_text()
    tests2_content = tests2_path.read_text()

    image = get_image_name(repo_name, task_id)
    eval_backend = get_backend(backend)
    sb = eval_backend.create_sandbox(image, timeout)

    try:
        # Get base SHA
        result = sb.exec("bash", "-c", "cd /workspace/repo && git rev-parse HEAD")
        base_sha = result.stdout_read().strip()

        if not base_sha:
            return _solo_error_result("Failed to get base commit SHA")

        # Write patches
        _write_patch(sb, "solo.patch", patch_content)
        _write_patch(sb, "tests1.patch", tests1_content)
        _write_patch(sb, "tests2.patch", tests2_content)

        # Test feature 1: runner.sh tests1.patch solo.patch
        test1_result = _run_tests(sb, "tests1.patch", "solo.patch", base_sha)

        # Test feature 2: runner.sh tests2.patch solo.patch
        test2_result = _run_tests(sb, "tests2.patch", "solo.patch", base_sha)

        return {
            "setting": "solo",
            "patch_lines": len(patch_content.splitlines()) if patch_content else 0,
            "feature1": {
                "passed": test1_result["passed"],
                "test_output": test1_result["output"],
            },
            "feature2": {
                "passed": test2_result["passed"],
                "test_output": test2_result["output"],
            },
            "both_passed": test1_result["passed"] and test2_result["passed"],
            "error": None,
        }
    except Exception as e:
        return _solo_error_result(str(e))
    finally:
        sb.terminate()


# Alias for training compatibility
def evaluate_merge(
    repo_name: str,
    task_id: int,
    feature1_id: int,
    feature2_id: int,
    patch1: str,
    patch2: str,
) -> dict:
    """Evaluate merged patches - wrapper for training compatibility.

    Returns dict with keys expected by training code:
        feature1_tests_passed, feature1_tests_total,
        feature2_tests_passed, feature2_tests_total, error
    """
    result = test_merged(
        repo_name=repo_name,
        task_id=task_id,
        feature1_id=feature1_id,
        feature2_id=feature2_id,
        patch1=patch1,
        patch2=patch2,
    )
    return {
        "feature1_tests_passed": 1 if result.get("feature1", {}).get("passed") else 0,
        "feature1_tests_total": 1,
        "feature2_tests_passed": 1 if result.get("feature2", {}).get("passed") else 0,
        "feature2_tests_total": 1,
        "error": result.get("error"),
    }


# === Helper functions ===


def _write_patch(sb: Sandbox, filename: str, content: str) -> None:
    """Write a patch file to the sandbox."""
    encoded = base64.b64encode(content.encode()).decode()
    result = sb.exec("bash", "-c", f"echo '{encoded}' | base64 -d > /patches/{filename}")
    if result.returncode != 0:
        raise RuntimeError(f"Failed to write {filename}: {result.stderr_read()}")


def _setup_branches(sb: Sandbox) -> dict:
    """Set up git branches for merge testing."""
    commands = """
cd /workspace/repo
git config user.email "eval@cooperbench.local"
git config user.name "CooperBench Eval"

# Save base commit SHA
BASE_SHA=$(git rev-parse HEAD)
echo "BASE_SHA=$BASE_SHA"

# Create agent1 branch and apply patch1
git checkout -b agent1 2>&1
if [ -s /patches/patch1.patch ]; then
    git apply /patches/patch1.patch 2>&1 || git apply --3way /patches/patch1.patch 2>&1 || echo "PATCH1_FAILED"
fi
git add -A
git commit -m "Agent 1 changes" --allow-empty 2>&1

# Create agent2 branch from base and apply patch2
git checkout $BASE_SHA 2>&1
git checkout -b agent2 2>&1
if [ -s /patches/patch2.patch ]; then
    git apply /patches/patch2.patch 2>&1 || git apply --3way /patches/patch2.patch 2>&1 || echo "PATCH2_FAILED"
fi
git add -A
git commit -m "Agent 2 changes" --allow-empty 2>&1

echo "SETUP_COMPLETE"
"""
    result = sb.exec("bash", "-c", commands)
    output = result.stdout_read() + result.stderr_read()

    if "SETUP_COMPLETE" not in output:
        return {"error": f"Branch setup failed: {output}"}

    # Extract base SHA
    base_sha = None
    for line in output.split("\n"):
        if line.startswith("BASE_SHA="):
            base_sha = line.split("=")[1].strip()
            break

    return {"output": output, "error": None, "base_sha": base_sha}


def _merge_naive(sb: Sandbox, base_sha: str) -> dict:
    """Try naive git merge."""
    commands = f"""
cd /workspace/repo
git checkout agent2 2>&1

# Try naive merge
if git merge agent1 --no-commit --no-ff 2>&1; then
    echo "MERGE_STATUS=clean"
    # Commit the merge temporarily to get proper diff
    git commit -m "Temp merge" 2>&1
    # Diff against BASE commit (not against agent2)
    git diff {base_sha} HEAD > /patches/naive_diff.patch
else
    echo "MERGE_STATUS=conflicts"
    git merge --abort 2>/dev/null || true
fi
"""
    result = sb.exec("bash", "-c", commands)
    output = result.stdout_read() + result.stderr_read()

    conflict = "MERGE_STATUS=conflicts" in output

    # Read diff from file if clean merge
    diff = ""
    if not conflict:
        diff_result = sb.exec("cat", "/patches/naive_diff.patch")
        diff = diff_result.stdout_read()

    return {"conflict": conflict, "diff": diff, "output": output}


def _merge_union(sb: Sandbox, base_sha: str) -> dict:
    """Try union merge strategy."""
    commands = f"""
cd /workspace/repo
git checkout agent2 2>&1
git reset --hard HEAD 2>&1

# Set up union merge strategy
echo "* merge=union" >> .gitattributes

# Try union merge
if git merge agent1 --no-commit --no-ff 2>&1; then
    echo "UNION_STATUS=clean"
    # Commit the merge temporarily to get proper diff
    git commit -m "Temp union merge" 2>&1
    # Diff against BASE commit
    git diff {base_sha} HEAD > /patches/union_diff.patch
else
    echo "UNION_STATUS=conflicts"
    git merge --abort 2>/dev/null || true
fi

# Restore gitattributes
git checkout .gitattributes 2>/dev/null || rm -f .gitattributes
"""
    result = sb.exec("bash", "-c", commands)
    output = result.stdout_read() + result.stderr_read()

    if "UNION_STATUS=conflicts" in output:
        return {"error": "Union merge still has conflicts", "diff": "", "output": output}

    # Read diff from file
    diff_result = sb.exec("cat", "/patches/union_diff.patch")
    diff = diff_result.stdout_read()

    return {"diff": diff, "output": output, "error": None}


def _run_tests(sb: Sandbox, tests_patch: str, feature_patch: str, base_sha: str) -> dict:
    """Run tests via runner.sh."""
    commands = f"""
cd /workspace/repo

# Reset to base commit
git checkout --force {base_sha} 2>&1
git reset --hard {base_sha} 2>&1
git clean -fdx 2>&1

echo "Reset to base: $(git rev-parse HEAD)"

# Run tests via runner.sh
bash /usr/local/bin/runner.sh {tests_patch} {feature_patch}
"""
    result = sb.exec("bash", "-c", commands)

    output = result.stdout_read() + result.stderr_read()
    exit_code = result.returncode
    parsed = _parse_results(output)

    return {
        "passed": exit_code == 0 and parsed["passed"] > 0,
        "output": output,
        "tests_passed": parsed["passed"],
        "tests_failed": parsed["failed"],
    }


def _parse_results(output: str) -> dict:
    """Parse test output to extract pass/fail counts.

    Supports: pytest, go test, cargo test, jest/vitest
    """
    passed = 0
    failed = 0

    # jest/vitest (TypeScript) - check first due to specific "Tests:" prefix
    # Format: "Tests:       2 failed, 15 passed, 17 total"
    jest_match = re.search(r"Tests:\s*(?:(\d+)\s*failed,\s*)?(\d+)\s*passed", output)
    if jest_match:
        failed = int(jest_match.group(1)) if jest_match.group(1) else 0
        passed = int(jest_match.group(2))
        return {"passed": passed, "failed": failed}

    # pytest - look for the summary line format "X passed in Y.YYs"
    pytest_passed = re.search(r"(\d+) passed", output)
    pytest_failed = re.search(r"(\d+) failed", output)
    pytest_error = re.search(r"(\d+) error", output)

    if pytest_passed:
        passed = int(pytest_passed.group(1))
    if pytest_failed:
        failed = int(pytest_failed.group(1))
    if pytest_error:
        failed += int(pytest_error.group(1))

    if passed > 0 or failed > 0:
        return {"passed": passed, "failed": failed}

    # go test - verbose output (--- PASS:/--- FAIL:)
    go_pass = len(re.findall(r"--- PASS:", output))
    go_fail = len(re.findall(r"--- FAIL:", output))
    if go_pass or go_fail:
        return {"passed": go_pass, "failed": go_fail}

    # go test - non-verbose output (ok/FAIL package lines)
    # Format: "ok  github.com/pkg  0.123s" or "FAIL github.com/pkg [build failed]"
    go_ok_packages = len(re.findall(r"^ok\s+\S+", output, re.MULTILINE))
    go_fail_packages = len(re.findall(r"^FAIL\s+\S+", output, re.MULTILINE))
    if go_ok_packages or go_fail_packages:
        # If any package failed, count it; otherwise count ok packages as passed
        return {"passed": go_ok_packages if go_fail_packages == 0 else 0, "failed": go_fail_packages}

    # cargo test
    cargo_match = re.search(r"test result:.*?(\d+) passed.*?(\d+) failed", output)
    if cargo_match:
        return {"passed": int(cargo_match.group(1)), "failed": int(cargo_match.group(2))}

    return {"passed": passed, "failed": failed}


def _filter_test_files(patch_content: str) -> str:
    """Filter test files from patch content."""
    if not patch_content:
        return patch_content

    filtered_lines = []
    skip_until_next_diff = False

    for line in patch_content.split("\n"):
        # Check if this is a new file diff header
        if line.startswith("diff --git"):
            # Check if it's a test file
            is_test_file = (
                "/test_" in line or "/tests/" in line or "_test.py" in line or "/test/" in line or "tests.py" in line
            )
            skip_until_next_diff = is_test_file

        if not skip_until_next_diff:
            filtered_lines.append(line)

    result = "\n".join(filtered_lines)
    # Ensure patch ends with newline (required by git)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


def _load_patch(patch: str | Path | None) -> str | None:
    """Load patch content from string or file."""
    if patch is None:
        return None
    if isinstance(patch, Path):
        content = patch.read_text()
    elif not patch or not patch.strip():
        # Empty string should return None, not try to read "." directory
        return None
    elif len(patch) < 500 and Path(patch).exists() and Path(patch).is_file():
        # If it looks like a file path (short, exists, is a file), read it
        content = Path(patch).read_text()
    else:
        content = patch

    # Sanitize patch content
    return _sanitize_patch(content)


def _sanitize_patch(content: str) -> str:
    """Sanitize patch content to fix common issues."""
    if not content:
        return content

    # Fix shell-escaped single quotes (e.g., won'\''t -> won't)
    content = content.replace("'\\''", "'")

    # Ensure patch ends with newline (required by git)
    if not content.endswith("\n"):
        content += "\n"

    return content


def _error_result(error: str) -> dict:
    return {
        "passed": False,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_total": 0,
        "output": "",
        "error": error,
    }


def _merged_error_result(error: str) -> dict:
    return {
        "merge": {"status": "error", "strategy": None, "diff": ""},
        "feature1": {"passed": False, "test_output": ""},
        "feature2": {"passed": False, "test_output": ""},
        "both_passed": False,
        "error": error,
    }


def _solo_error_result(error: str) -> dict:
    return {
        "setting": "solo",
        "patch_lines": 0,
        "feature1": {"passed": False, "test_output": ""},
        "feature2": {"passed": False, "test_output": ""},
        "both_passed": False,
        "error": error,
    }
