"""PR-to-Dataset adapter -- converts SWE-smith instance JSONL into
CooperBench task directories (Dockerfile, runner.sh, feature.md, patches),
builds Docker images, and validates that patches apply and tests pass.

The default invocation runs the **full pipeline** (convert → build →
validate).  Individual stages can be skipped with ``--skip-build`` /
``--skip-validate``.

Usage::

    # Full pipeline: convert + build + validate
    python -m cooperbench.generation.onboard \\
        --input logs/pr_collection/tasks/pallets-flask-insts.jsonl \\
        --repo-name flask_task \\
        --repo-url https://github.com/pallets/flask.git

    # Convert only (no Docker)
    python -m cooperbench.generation.onboard \\
        --input logs/pr_collection/tasks/pallets-flask-insts.jsonl \\
        --repo-name flask_task \\
        --repo-url https://github.com/pallets/flask.git \\
        --skip-build --skip-validate

    # Live collection + full pipeline
    python -m cooperbench.generation.onboard \\
        --collect pallets/flask \\
        --repo-name flask_task \\
        --repo-url https://github.com/pallets/flask.git
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-task result tracking
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Status of each pipeline stage for one PR / task."""
    pull_number: int
    task_dir: Path | None = None

    convert_ok: bool | None = None
    convert_error: str | None = None

    build_ok: bool | None = None
    build_error: str | None = None

    validate_ok: bool | None = None
    validate_output: str | None = None
    validate_error: str | None = None

    skipped_build: bool = False
    skipped_validate: bool = False
    duration_seconds: float = 0.0

    @property
    def overall_ok(self) -> bool:
        """True only when every non-skipped stage succeeded."""
        if self.convert_ok is False:
            return False
        if not self.skipped_build and self.build_ok is False:
            return False
        if not self.skipped_validate and self.validate_ok is False:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "pull_number": self.pull_number,
            "task_dir": str(self.task_dir) if self.task_dir else None,
            "convert_ok": self.convert_ok,
            "convert_error": self.convert_error,
            "build_ok": self.build_ok,
            "build_error": self.build_error,
            "validate_ok": self.validate_ok,
            "validate_output": self.validate_output,
            "validate_error": self.validate_error,
            "skipped_build": self.skipped_build,
            "skipped_validate": self.skipped_validate,
            "duration_seconds": self.duration_seconds,
            "overall_ok": self.overall_ok,
        }


# ---------------------------------------------------------------------------
# Filtering (thin wrapper — collect.py has the main logic)
# ---------------------------------------------------------------------------

def filter_candidates(
    jsonl_path: str | Path,
    min_test_lines: int = 5,
    max_patch_files: int = 10,
) -> list[dict]:
    """Load JSONL and keep only PRs that have a usable test_patch."""
    from cooperbench.generation.collect import filter_instances

    return filter_instances(
        Path(jsonl_path),
        min_test_lines=min_test_lines,
        max_patch_files=max_patch_files,
    )


# ---------------------------------------------------------------------------
# Dockerfile generation
# ---------------------------------------------------------------------------

DOCKERFILE_TEMPLATE = textwrap.dedent("""\
    FROM {base_image}

    # Install system dependencies
    RUN apt-get update && apt-get install -y \\
        {system_deps} \\
        && rm -rf /var/lib/apt/lists/*

    # Clone the repository and checkout the specific commit
    WORKDIR /workspace
    RUN git clone {repo_url} repo && \\
        cd repo && \\
        git checkout {commit_sha}

    # Set up Python environment
    WORKDIR /workspace/repo

    # Pre-install dependencies into system Python (cache layer)
    RUN pip install --upgrade pip && \\
        {install_cmd}

    # Copy the runner script
    COPY runner.sh /usr/local/bin/
    RUN chmod +x /usr/local/bin/runner.sh

    ENTRYPOINT ["/usr/local/bin/runner.sh"]
""")


def generate_dockerfile(
    repo_url: str,
    commit_sha: str,
    install_cmd: str = 'pip install -e ".[test]" || pip install -e . && pip install pytest',
    system_deps: str = "git curl build-essential",
    base_image: str = "python:3.11-slim",
) -> str:
    """Return a Dockerfile string matching the Pallets Jinja pattern."""
    return DOCKERFILE_TEMPLATE.format(
        base_image=base_image,
        system_deps=system_deps,
        repo_url=repo_url,
        commit_sha=commit_sha,
        install_cmd=install_cmd,
    )


# ---------------------------------------------------------------------------
# runner.sh generation
# ---------------------------------------------------------------------------

RUNNER_SH_TEMPLATE = textwrap.dedent("""\
    #!/bin/bash

    set -e

    # Cleanup function
    cleanup() {{
        echo "Cleaning up repository..."
        if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
            git reset --hard HEAD 2>/dev/null || true
            git clean -fdx 2>/dev/null || true
            echo "Repository cleaned."
        fi
    }}

    trap cleanup EXIT INT TERM

    # Get input params
    TEST_PATCH="$1"
    FEATURE_PATCH="$2"

    if [[ -z "$TEST_PATCH" ]]; then
        echo "Usage: docker run -v \\$(pwd):/patches <image> <test_patch> [feature_patch]"
        exit 1
    fi

    cd /workspace/repo

    # Ensure we start with a clean state
    echo "Ensuring clean repository state..."
    git reset --hard HEAD
    git clean -fdx

    # Apply feature patch if provided
    if [[ -n "$FEATURE_PATCH" ]]; then
        echo "Applying feature patch: $FEATURE_PATCH"
        if [[ -f "/patches/$FEATURE_PATCH" ]]; then
            git apply --ignore-whitespace --ignore-space-change "/patches/$FEATURE_PATCH" || git apply --3way "/patches/$FEATURE_PATCH"
            echo "Feature patch applied successfully."
        else
            echo "Error: Feature patch not found at /patches/$FEATURE_PATCH"
            exit 1
        fi
    fi

    # Apply test patch
    echo "Applying test patch: $TEST_PATCH"
    if [[ -f "/patches/$TEST_PATCH" ]]; then
        git apply --ignore-whitespace --ignore-space-change "/patches/$TEST_PATCH" || git apply --3way "/patches/$TEST_PATCH"
        echo "Test patch applied successfully."
    else
        echo "Error: Test patch not found at /patches/$TEST_PATCH"
        exit 1
    fi

    # Install dependencies (SYSTEM PYTHON)
    echo "Installing dependencies..."
    {reinstall_cmd}

    # Run tests
    echo "Running tests..."
    python -m pytest {test_files} -v

    echo "Test execution completed!"
""")


_NON_TEST_CODE_EXTENSIONS = {
    ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".md",
    ".rst", ".txt", ".lock", ".xml", ".html", ".css",
}


def extract_test_files(test_patch: str) -> list[str]:
    """Parse ``+++ b/…`` headers from a test patch and return only runnable
    test source files (Python, etc.), excluding CI configs and docs."""
    files: list[str] = []
    for match in re.finditer(r"^\+\+\+ b/(.+)$", test_patch, re.MULTILINE):
        path = match.group(1)
        if not path or path.startswith("/dev/null"):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in _NON_TEST_CODE_EXTENSIONS:
            continue
        if path.startswith(".github/"):
            continue
        files.append(path)
    return sorted(set(files))


def generate_runner_sh(
    test_files: list[str],
    reinstall_cmd: str = 'pip install -e ".[test]" || pip install -e .\npip install pytest pytest-xdist pytest_mock',
) -> str:
    """Return a runner.sh script with the test files baked in."""
    files_str = " ".join(test_files) if test_files else "tests/"
    return RUNNER_SH_TEMPLATE.format(
        test_files=files_str,
        reinstall_cmd=reinstall_cmd,
    )


# ---------------------------------------------------------------------------
# runner.sh generation (conda-aware, for SWE-smith converted images)
# ---------------------------------------------------------------------------

RUNNER_SH_CONDA_TEMPLATE = textwrap.dedent("""\
    #!/bin/bash
    set -e

    # Activate the conda environment used by SWE-smith images
    source /opt/miniconda3/etc/profile.d/conda.sh
    conda activate testbed

    cleanup() {{
        echo "Cleaning up repository..."
        cd /workspace/repo 2>/dev/null || true
        git reset --hard HEAD 2>/dev/null || true
        git clean -fdx 2>/dev/null || true
    }}
    trap cleanup EXIT INT TERM

    TEST_PATCH="$1"
    FEATURE_PATCH="$2"

    if [[ -z "$TEST_PATCH" ]]; then
        echo "Usage: runner.sh <test_patch> [feature_patch]"
        exit 1
    fi

    cd /workspace/repo
    git reset --hard HEAD
    git clean -fdx

    # Apply feature patch if provided
    if [[ -n "$FEATURE_PATCH" ]]; then
        echo "Applying feature patch: $FEATURE_PATCH"
        if [[ -f "/patches/$FEATURE_PATCH" ]]; then
            git apply --ignore-whitespace --ignore-space-change "/patches/$FEATURE_PATCH" \\
                || git apply --3way "/patches/$FEATURE_PATCH"
        else
            echo "Error: Feature patch not found at /patches/$FEATURE_PATCH"
            exit 1
        fi
    fi

    # Apply test patch
    echo "Applying test patch: $TEST_PATCH"
    if [[ -f "/patches/$TEST_PATCH" ]]; then
        git apply --ignore-whitespace --ignore-space-change "/patches/$TEST_PATCH" \\
            || git apply --3way "/patches/$TEST_PATCH"
    else
        echo "Error: Test patch not found at /patches/$TEST_PATCH"
        exit 1
    fi

    # Re-install package so code changes are picked up
    pip install -e . 2>/dev/null || true

    # Auto-detect test files from the test patch
    TEST_FILES=$(grep '^diff --git a/' "/patches/$TEST_PATCH" 2>/dev/null \\
        | sed 's|diff --git a/\\([^ ]*\\) b/.*|\\1|')
    if [[ -z "$TEST_FILES" ]]; then
        echo "Warning: Could not detect test files from patch, running tests/"
        TEST_FILES="tests/"
    fi

    echo "Running tests: $TEST_FILES"
    python -m pytest $TEST_FILES -x -v

    echo "Test execution completed!"
""")


def generate_runner_sh_conda() -> str:
    """Return a conda-aware runner.sh for SWE-smith converted images.

    Unlike the standard template, this:
    - Activates the conda ``testbed`` environment
    - Auto-detects test files from the test patch at runtime
    - Does not require test file paths baked in at generation time
    """
    return RUNNER_SH_CONDA_TEMPLATE


# ---------------------------------------------------------------------------
# feature.md generation via LLM
# ---------------------------------------------------------------------------

FEATURE_MD_SYSTEM_PROMPT = (
    "You are a technical writer creating feature specifications for a software "
    "engineering benchmark. Given information about a code change (patch diff and "
    "optionally a problem statement), produce a concise feature.md document.\n\n"
    "Follow this exact format:\n\n"
    "**Title**: <descriptive title>\n\n"
    "**Pull Request Details**\n\n"
    "**Description**:\n<2-3 sentences>\n\n"
    "**Technical Background**:\n<context about the codebase and why this change matters>\n\n"
    "**Solution**:\n<what was changed and how, key implementation details>\n\n"
    "**Files Modified**\n- <list of files>\n"
)

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Patch adds a `case_sensitive` parameter to the `groupby()` filter "
            "in Jinja2 so grouping can be case-insensitive, matching sort() behavior.\n\n"
            "Files changed: src/jinja2/filters.py"
        ),
    },
    {
        "role": "assistant",
        "content": textwrap.dedent("""\
            **Title**: Add case_sensitive parameter to groupby() filter

            **Pull Request Details**

            **Description**:
            This pull request adds a `case_sensitive` parameter to the `groupby()` filter to enable case-insensitive grouping, bringing consistency with the `sort()` filter behavior.

            **Technical Background**:
            The `groupby()` filter in Jinja2 performs automatic sorting under the hood before grouping items, but it only supported case-sensitive sorting. This created inconsistency with the `sort()` filter, which defaults to case-insensitive sorting. When both filters were used on the same dataset in a template, they would produce different sort orders for the same fields.

            **Solution**:
            The implementation adds a `case_sensitive` parameter (defaulting to `False` for case-insensitive behavior) to both the synchronous and asynchronous versions of the `groupby()` filter. When `case_sensitive=False`, the sorting key uses the `ignore_case` postprocessor to normalize case during comparison, items are grouped by their case-normalized keys, and the final output preserves the original case from the first item in each group.

            **Files Modified**
            - `src/jinja2/filters.py`"""),
    },
    {
        "role": "user",
        "content": (
            "Patch modifies click.edit() to accept an iterable of filenames so "
            "multiple files can be opened in the editor at once.\n\n"
            "Files changed: src/click/_termui_impl.py, src/click/termui.py"
        ),
    },
    {
        "role": "assistant",
        "content": textwrap.dedent("""\
            **Title**: Add Support for Editing Multiple Files with click.edit

            **Pull Request Details**

            **Description**:
            This enhancement extends the `click.edit()` function to support editing multiple files simultaneously in editors that support multiple tabs or buffers (such as vim, nano, vscode, etc.).

            **Technical Background**:
            Previously, the `click.edit()` function only supported editing a single file at a time through its `filename` parameter. Users who wanted to edit multiple files had to either call `click.edit()` multiple times or implement custom workarounds using internal Click APIs.

            **Solution**:
            The implementation modifies the `click.edit()` function to accept either a single filename string or an iterable of filename strings for the `filename` parameter. When multiple filenames are provided, they are passed as separate arguments to the editor command. The change maintains full backward compatibility.

            **Files Modified**
            - `src/click/_termui_impl.py`
            - `src/click/termui.py`"""),
    },
]


def generate_feature_md(
    problem_statement: str | None,
    patch: str,
    test_patch: str | None = None,
    model_name: str = "gemini/gemini-2.0-flash",
) -> str:
    """Use an LLM to produce a feature.md from PR data."""
    import litellm

    parts: list[str] = []
    if problem_statement and problem_statement.strip():
        parts.append(f"Problem statement / PR description:\n{problem_statement.strip()}")

    patch_files = re.findall(r"^\+\+\+ b/(.+)$", patch, re.MULTILINE)
    parts.append(f"Files changed: {', '.join(patch_files)}")

    patch_lines = patch.splitlines()
    if len(patch_lines) > 200:
        snippet = "\n".join(patch_lines[:150]) + "\n\n... (truncated) ...\n\n" + "\n".join(patch_lines[-30:])
    else:
        snippet = patch
    parts.append(f"Patch diff:\n```diff\n{snippet}\n```")

    if test_patch and test_patch.strip():
        test_snippet = test_patch if len(test_patch.splitlines()) <= 80 else "\n".join(test_patch.splitlines()[:60])
        parts.append(f"Test patch:\n```diff\n{test_snippet}\n```")

    user_msg = "\n\n".join(parts)

    messages = [
        {"role": "system", "content": FEATURE_MD_SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": user_msg},
    ]

    response = litellm.completion(
        model=model_name,
        messages=messages,
        temperature=0.3,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Docker image building
# ---------------------------------------------------------------------------

def _image_exists(image_name: str) -> bool:
    """Return True if a Docker image is available locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True,
    )
    return result.returncode == 0


def build_task_image(
    repo_name: str,
    task_id: int,
    dataset_dir: Path | None = None,
) -> tuple[bool, str]:
    """Build and tag a Docker image for a task directory.

    Returns ``(success, error_message)``.
    """
    from cooperbench.utils import get_image_name

    dataset_dir = dataset_dir or Path("dataset")
    task_dir = dataset_dir / repo_name / f"task{task_id}"
    image_name = get_image_name(repo_name, task_id)

    if not task_dir.exists():
        msg = f"Task directory not found: {task_dir}"
        logger.error(msg)
        return False, msg

    logger.info("Building image %s from %s …", image_name, task_dir)
    result = subprocess.run(
        ["docker", "build", "-t", image_name, str(task_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        msg = result.stderr[-2000:] if result.stderr else "unknown build error"
        logger.error("Docker build failed for task%d:\n%s", task_id, msg)
        return False, msg

    logger.info("Image built: %s", image_name)
    return True, ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_task(
    repo_name: str,
    task_id: int,
    dataset_dir: Path | None = None,
    auto_build: bool = False,
) -> dict:
    """Apply patches inside the built image and verify tests pass.

    If *auto_build* is True and the image doesn't exist locally, it will be
    built first.

    Returns a dict with keys: ``passed``, ``output``, ``error``.
    """
    from cooperbench.utils import get_image_name

    dataset_dir = dataset_dir or Path("dataset")
    task_dir = dataset_dir / repo_name / f"task{task_id}"
    feature_dir = task_dir / "feature1"
    image_name = get_image_name(repo_name, task_id)

    if not feature_dir.exists():
        return {"passed": False, "output": "", "error": f"feature1 dir not found: {feature_dir}"}

    if not _image_exists(image_name):
        if auto_build:
            logger.info("Image %s not found — building automatically …", image_name)
            ok, err = build_task_image(repo_name, task_id, dataset_dir=dataset_dir)
            if not ok:
                return {"passed": False, "output": "", "error": f"auto-build failed: {err}"}
        else:
            return {"passed": False, "output": "", "error": f"Image not found: {image_name}"}

    patches_dir = feature_dir.resolve()

    logger.info("Validating %s/task%d (image %s) …", repo_name, task_id, image_name)
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{patches_dir}:/patches",
                image_name,
                "tests.patch", "feature.patch",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": "", "error": "docker run timed out (600s)"}

    passed = result.returncode == 0
    output = result.stdout + result.stderr
    if passed:
        logger.info("Validation PASSED for task%d", task_id)
    else:
        logger.warning("Validation FAILED for task%d (exit %d)", task_id, result.returncode)

    return {"passed": passed, "output": output, "error": None}


# ---------------------------------------------------------------------------
# Single-task conversion
# ---------------------------------------------------------------------------

def convert_pr_to_task(
    instance: dict,
    repo_name: str,
    repo_url: str,
    dataset_dir: Path | None = None,
    install_cmd: str = 'pip install -e ".[test]" || pip install -e . && pip install pytest',
    reinstall_cmd: str = 'pip install -e ".[test]" || pip install -e .\npip install pytest pytest-xdist pytest_mock',
    system_deps: str = "git curl build-essential",
    base_image: str = "python:3.11-slim",
    model_name: str = "gemini/gemini-2.0-flash",
    use_swesmith: bool = False,
) -> Path:
    """Create a full CooperBench task directory from an SWE-smith instance.

    When *use_swesmith* is True, skips Dockerfile generation (the Docker image
    is created via :func:`convert_swesmith.convert_task` instead) and uses
    the conda-aware runner.sh template.

    Directory layout::

        dataset/{repo_name}/task{pull_number}/
            Dockerfile          # only when use_swesmith=False
            runner.sh
            feature1/
                feature.md
                feature.patch
                tests.patch

    Returns the path to the created task directory.
    """
    dataset_dir = dataset_dir or Path("dataset")
    task_id = instance["pull_number"]
    task_dir = dataset_dir / repo_name / f"task{task_id}"
    feature_dir = task_dir / "feature1"
    feature_dir.mkdir(parents=True, exist_ok=True)

    (feature_dir / "feature.patch").write_text(instance["patch"])
    (feature_dir / "tests.patch").write_text(instance["test_patch"])
    logger.info("Wrote patches for PR #%d", task_id)

    if use_swesmith:
        # SWE-smith path: no Dockerfile, conda-aware runner, and build
        # the per-task Docker image via the converter
        runner_sh = generate_runner_sh_conda()
        (task_dir / "runner.sh").write_text(runner_sh)
        os.chmod(task_dir / "runner.sh", 0o755)

        from cooperbench.generation.convert_swesmith import convert_task
        convert_task(
            repo_name=repo_name,
            task_id=task_id,
            base_commit=instance["base_commit"],
        )
    else:
        # Standard path: generate Dockerfile + standard runner.sh
        dockerfile = generate_dockerfile(
            repo_url=repo_url,
            commit_sha=instance["base_commit"],
            install_cmd=install_cmd,
            system_deps=system_deps,
            base_image=base_image,
        )
        (task_dir / "Dockerfile").write_text(dockerfile)

        test_files = extract_test_files(instance["test_patch"])
        runner_sh = generate_runner_sh(test_files, reinstall_cmd=reinstall_cmd)
        (task_dir / "runner.sh").write_text(runner_sh)
        os.chmod(task_dir / "runner.sh", 0o755)

    logger.info("Generating feature.md for PR #%d via %s …", task_id, model_name)
    feature_md = generate_feature_md(
        problem_statement=instance.get("problem_statement"),
        patch=instance["patch"],
        test_patch=instance.get("test_patch"),
        model_name=model_name,
    )
    (feature_dir / "feature.md").write_text(feature_md)
    logger.info("Wrote feature.md for PR #%d", task_id)

    return task_dir


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    candidates: list[dict],
    repo_name: str,
    repo_url: str,
    dataset_dir: Path | None = None,
    install_cmd: str = 'pip install -e ".[test]" || pip install -e . && pip install pytest',
    reinstall_cmd: str = 'pip install -e ".[test]" || pip install -e .\npip install pytest pytest-xdist pytest_mock',
    system_deps: str = "git curl build-essential",
    base_image: str = "python:3.11-slim",
    model_name: str = "gemini/gemini-2.0-flash",
    skip_build: bool = False,
    skip_validate: bool = False,
    use_swesmith: bool = False,
) -> list[TaskResult]:
    """Run the full onboarding pipeline: convert → build → validate.

    When *use_swesmith* is True, the Docker image is created via the
    SWE-smith converter (Tier 2) during the convert stage, so the
    separate build stage is skipped automatically.

    Each candidate is processed independently; a failure in one PR does not
    stop processing of subsequent PRs.

    Returns a list of :class:`TaskResult` objects (one per candidate).
    """
    dataset_dir = dataset_dir or Path("dataset")

    if use_swesmith:
        skip_build = True

    results: list[TaskResult] = []

    for i, inst in enumerate(candidates, 1):
        pr = inst["pull_number"]
        logger.info(
            "=== [%d/%d] PR #%d ===",
            i, len(candidates), pr,
        )
        t0 = time.time()
        tr = TaskResult(pull_number=pr, skipped_build=skip_build, skipped_validate=skip_validate)

        # --- Stage 1: Convert -----------------------------------------------
        try:
            task_dir = convert_pr_to_task(
                instance=inst,
                repo_name=repo_name,
                repo_url=repo_url,
                dataset_dir=dataset_dir,
                install_cmd=install_cmd,
                reinstall_cmd=reinstall_cmd,
                system_deps=system_deps,
                base_image=base_image,
                model_name=model_name,
                use_swesmith=use_swesmith,
            )
            tr.convert_ok = True
            tr.task_dir = task_dir
        except Exception as exc:
            logger.exception("Convert failed for PR #%d", pr)
            tr.convert_ok = False
            tr.convert_error = str(exc)
            tr.duration_seconds = time.time() - t0
            results.append(tr)
            continue  # skip build+validate for this PR

        # --- Stage 2: Build Docker image ------------------------------------
        if not skip_build:
            ok, err = build_task_image(repo_name, pr, dataset_dir=dataset_dir)
            tr.build_ok = ok
            if not ok:
                tr.build_error = err
                tr.duration_seconds = time.time() - t0
                results.append(tr)
                continue  # skip validate if build failed
        else:
            logger.info("Skipping Docker build for task%d (--skip-build)", pr)

        # --- Stage 3: Validate patches + tests ------------------------------
        if not skip_validate:
            vr = validate_task(
                repo_name, pr,
                dataset_dir=dataset_dir,
                auto_build=skip_build,  # if we skipped explicit build, let validate auto-build
            )
            tr.validate_ok = vr["passed"]
            tr.validate_output = vr.get("output", "")
            if vr.get("error"):
                tr.validate_error = vr["error"]
        else:
            logger.info("Skipping validation for task%d (--skip-validate)", pr)

        tr.duration_seconds = time.time() - t0
        results.append(tr)

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[TaskResult]) -> None:
    """Print a human-readable summary table to stdout."""
    passed = sum(1 for r in results if r.overall_ok)
    total = len(results)

    print(f"\n{'=' * 72}")
    print(f"  Pipeline Report: {passed}/{total} tasks fully succeeded")
    print(f"{'=' * 72}")
    print(f"  {'PR':>7s}  {'Convert':>8s}  {'Build':>8s}  {'Validate':>9s}  {'Time':>6s}  Status")
    print(f"  {'---':>7s}  {'-------':>8s}  {'-----':>8s}  {'--------':>9s}  {'----':>6s}  ------")

    for r in results:
        def _status(ok: bool | None, skipped: bool) -> str:
            if skipped:
                return "skip"
            if ok is None:
                return "-"
            return "ok" if ok else "FAIL"

        cvt = _status(r.convert_ok, False)
        bld = _status(r.build_ok, r.skipped_build)
        val = _status(r.validate_ok, r.skipped_validate)
        secs = f"{r.duration_seconds:.0f}s"
        overall = "PASS" if r.overall_ok else "FAIL"

        print(f"  #{r.pull_number:>6d}  {cvt:>8s}  {bld:>8s}  {val:>9s}  {secs:>6s}  {overall}")

        if r.convert_error:
            print(f"           convert error: {r.convert_error[:120]}")
        if r.build_error:
            print(f"           build error: {r.build_error[:120]}")
        if r.validate_error:
            print(f"           validate error: {r.validate_error[:120]}")

    print(f"{'=' * 72}\n")


def save_report(results: list[TaskResult], path: Path) -> None:
    """Write the full pipeline report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "total": len(results),
        "passed": sum(1 for r in results if r.overall_ok),
        "tasks": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Report saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Onboard GitHub PRs into CooperBench dataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            By default the full pipeline runs: filter → convert → build → validate.
            Use --skip-build and/or --skip-validate to omit stages.

            Examples:

              # Full pipeline from existing JSONL
              python -m cooperbench.generation.onboard \\
                  --input logs/pr_collection/tasks/pallets-flask-insts.jsonl \\
                  --repo-name flask_task \\
                  --repo-url https://github.com/pallets/flask.git

              # Collect live + full pipeline
              python -m cooperbench.generation.onboard \\
                  --collect pallets/flask \\
                  --repo-name flask_task \\
                  --repo-url https://github.com/pallets/flask.git

              # Convert only (no Docker)
              python -m cooperbench.generation.onboard \\
                  --input logs/pr_collection/tasks/pallets-flask-insts.jsonl \\
                  --repo-name flask_task \\
                  --repo-url https://github.com/pallets/flask.git \\
                  --skip-build --skip-validate
        """),
    )

    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        dest="input_path",
        help="Path to instances JSONL (from collect.py or SWE-smith)",
    )
    input_group.add_argument(
        "--collect",
        dest="collect_repo",
        help="Collect PRs live from this GitHub repo (e.g. pallets/flask)",
    )

    # Repo identity
    parser.add_argument("--repo-name", required=True, help="CooperBench repo name (e.g. flask_task)")
    parser.add_argument("--repo-url", required=True, help="Git clone URL for the repository")

    # Template parameters
    parser.add_argument(
        "--install-cmd",
        default='pip install -e ".[test]" || pip install -e . && pip install pytest',
        help="pip install command for Dockerfile",
    )
    parser.add_argument(
        "--reinstall-cmd",
        default='pip install -e ".[test]" || pip install -e .\npip install pytest pytest-xdist pytest_mock',
        help="pip reinstall command for runner.sh (after patch application)",
    )
    parser.add_argument("--base-image", default="python:3.11-slim")
    parser.add_argument("--system-deps", default="git curl build-essential")
    parser.add_argument("--model", default="gemini/gemini-2.0-flash", help="LLM model for feature.md generation")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--max-pulls", type=int, default=50, help="Max PRs to collect (with --collect)")

    # Stage control
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker image building")
    parser.add_argument("--skip-validate", action="store_true", help="Skip patch validation in containers")

    # Filtering
    parser.add_argument("--min-test-lines", type=int, default=5)
    parser.add_argument("--max-patch-files", type=int, default=10)

    # Output
    parser.add_argument(
        "--report",
        type=Path,
        help="Path to save JSON report (default: logs/onboard/{repo_name}_report.json)",
    )

    args = parser.parse_args()

    # --- Resolve candidates --------------------------------------------------
    if args.collect_repo:
        from cooperbench.generation.collect import collect_and_filter

        candidates = collect_and_filter(
            args.collect_repo,
            "logs/pr_collection",
            max_pulls=args.max_pulls,
            min_test_lines=args.min_test_lines,
            max_patch_files=args.max_patch_files,
        )
    else:
        candidates = filter_candidates(
            args.input_path,
            min_test_lines=args.min_test_lines,
            max_patch_files=args.max_patch_files,
        )

    if not candidates:
        logger.error("No candidates found after filtering.")
        return

    logger.info("Pipeline starting: %d candidates, stages: convert%s%s",
                len(candidates),
                "" if args.skip_build else " → build",
                "" if args.skip_validate else " → validate")

    # --- Run pipeline --------------------------------------------------------
    results = run_pipeline(
        candidates=candidates,
        repo_name=args.repo_name,
        repo_url=args.repo_url,
        dataset_dir=args.dataset_dir,
        install_cmd=args.install_cmd,
        reinstall_cmd=args.reinstall_cmd,
        system_deps=args.system_deps,
        base_image=args.base_image,
        model_name=args.model,
        skip_build=args.skip_build,
        skip_validate=args.skip_validate,
    )

    # --- Report --------------------------------------------------------------
    print_report(results)

    report_path = args.report or Path(f"logs/onboard/{args.repo_name}_report.json")
    save_report(results, report_path)

    passed = sum(1 for r in results if r.overall_ok)
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
