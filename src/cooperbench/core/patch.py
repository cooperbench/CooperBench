"""
Patch processing utilities for CooperBench.

This module provides functions for generating, applying, and filtering git patches,
as well as categorizing files in patches by type (code, tests, docs, dependencies).
"""

import re
import sys
from collections import deque
from pathlib import Path

from cooperbench.core.git import run_git_command
from cooperbench.core.logger import get_logger

logger = get_logger("cooperbench.patch")


TEST_PATTERNS: list[str] = [
    r"test[s]?/",
    r"__test[s]?__/",
    r"spec[s]?/",
    r"\.test\.",
    r"\.spec\.",
    r"_test\.",
    r"_spec\.",
    r"test_.*\.py$",
    r".*_test\.py$",
    r".*Test\.(java|kt|scala)$",
    r".*Tests\.(java|kt|scala)$",
]

DOC_PATTERNS: list[str] = [
    r"readme",
    r"\.md$",
    r"\.rst$",
    r"\.txt$",
    r"docs?/",
    r"documentation/",
    r"\.wiki",
    r"changelog",
    r"contributing",
    r"license",
    r"authors",
    r"install",
    r"usage",
]

PACKAGE_VERSION_PATTERNS: list[str] = [
    r"pyproject\.toml$",
    r"uv\.lock$",
    r"requirements.*\.txt$",
    r"Pipfile$",
    r"Pipfile\.lock$",
    r"poetry\.lock$",
    r"package\.json$",
    r"package-lock\.json$",
    r"yarn\.lock$",
    r"pnpm-lock\.yaml$",
    r"composer\.json$",
    r"composer\.lock$",
    r"Gemfile$",
    r"Gemfile\.lock$",
    r"go\.mod$",
    r"go\.sum$",
    r"Cargo\.toml$",
    r"Cargo\.lock$",
    r"setup\.py$",
    r"setup\.cfg$",
    r"environment\.ya?ml$",
    r"conda-environment\.ya?ml$",
    r"\.conda$",
    r"requirements\.in$",
    r"dev-requirements\.txt$",
    r"test-requirements\.txt$",
]


def generate_patch(agent_workspace_path: Path, base_commit: str) -> str:
    """Generate a diff from the current changes in the feature branch.
    
    Args:
        agent_workspace_path: Path to the agent workspace
        base_commit: Commit hash to diff against

    Returns:
        Diff output as string (empty string if no differences)
    """
    print(f"Generating cumulative patch against base commit {base_commit[:7]}")
    diff_output = str(run_git_command(agent_workspace_path, "diff", base_commit, capture_output=True))
    return diff_output if diff_output else ""


def apply_patch(
    agent_workspace_path: Path,
    local_patch_path: Path,
    filter_patch: bool = True,
) -> None:
    """Apply a patch file to the given agent workspace.
    
    Args:
        agent_workspace_path: Path to the agent workspace
        local_patch_path: Path to the local patch file
        filter_patch: Whether to filter non-code changes before applying
    """
    if filter_patch:
        local_patch_path = _filter_patch(local_patch_path)
    
    try:
        if not local_patch_path.exists():
            logger.error(f"Patch file not found at {local_patch_path}")
            sys.exit(1)
        
        if local_patch_path.stat().st_size == 0:
            logger.info(f"Patch file {local_patch_path.name} is empty. Skipping application.")
            return

        try:
            run_git_command(
                agent_workspace_path,
                "apply",
                "--whitespace=fix",
                str(local_patch_path.resolve()),
            )
            logger.log_patch_application(str(local_patch_path), True)
        except Exception:
            logger.log_patch_application(str(local_patch_path), False)
            raise

        try:
            run_git_command(agent_workspace_path, "add", ".", capture_output=True)
            logger.log_git_operation("add", ["."], True)
        except Exception as e:
            logger.log_git_operation("add", ["."], False, str(e))
            raise

        status_output = str(
            run_git_command(
                agent_workspace_path,
                "status",
                "--porcelain",
                capture_output=True,
            )
        )
        if status_output:
            commit_message = f"Applied patch ({local_patch_path.name})"
            try:
                run_git_command(agent_workspace_path, "commit", "-m", commit_message, capture_output=True)
                logger.log_git_operation("commit", ["-m", commit_message], True)
            except Exception as e:
                logger.log_git_operation("commit", ["-m", commit_message], False, str(e))
                raise

    except Exception as e:
        logger.error(f"Error applying patch: {e}")
        sys.exit(1)


def _filter_patch(local_patch_path: Path) -> Path:
    """Filter non-code changes from a patch file.
    
    Saves the filtered patch to a .filtered.patch file.
    """
    patch_content = local_patch_path.read_text()
    filtered_patch = split_patch_by_type(patch_content)
    tmp_path = local_patch_path.with_suffix(".filtered.patch")
    tmp_path.write_text(filtered_patch["code"])
    return tmp_path


def categorize_files(file_paths: list[str]) -> dict[str, list[str]]:
    """Categorize files into code, tests, docs, and dependencies."""
    categorized: dict[str, list[str]] = {"code": [], "docs": [], "tests": [], "dependencies": []}

    for file_path in file_paths:
        if not file_path or file_path == "/dev/null":
            continue

        file_lower = file_path.lower()
        is_package_version = any(re.search(pattern, file_lower) for pattern in PACKAGE_VERSION_PATTERNS)
        is_test = any(re.search(pattern, file_lower) for pattern in TEST_PATTERNS)
        is_doc = any(re.search(pattern, file_lower) for pattern in DOC_PATTERNS)

        if is_test:
            categorized["tests"].append(file_path)
        elif is_doc:
            categorized["docs"].append(file_path)
        elif is_package_version:
            categorized["dependencies"].append(file_path)
        else:
            categorized["code"].append(file_path)

    for category in categorized:
        categorized[category] = sorted(list(set(categorized[category])))

    return categorized


def parse_patch_file_paths(patch: str) -> dict[str, list[str]]:
    """Parse patch content to extract and categorize changed file paths."""
    if not patch:
        return {"code": [], "docs": [], "tests": [], "dependencies": []}

    file_patterns = [r"^diff --git a/(.+) b/(.+)$", r"^--- a/(.+)$", r"^\+\+\+ b/(.+)$"]

    files: set[str] = set()
    for line in patch.split("\n"):
        for pattern in file_patterns:
            match = re.match(pattern, line)
            if match:
                if pattern.startswith("^diff"):
                    files.add(match.group(2))
                else:
                    files.add(match.group(1))

    return categorize_files(list(files))


def split_patch(patch: str) -> list[str]:
    """Split a patch into individual file patches."""
    lines = patch.split("\n")
    file_patches: deque[str] = deque()
    curr_patch: deque[str] = deque()

    while lines:
        line = lines.pop()
        curr_patch.appendleft(line)
        if line.startswith("diff --git"):
            file_patches.appendleft("\n".join(curr_patch))
            curr_patch.clear()

    return list(file_patches)


def split_patch_by_type(base_patch: str) -> dict[str, str]:
    """Split a patch into separate patches based on file types.
    
    Returns a dict with keys: code, tests, docs, dependencies
    """
    if not base_patch:
        return {"code": "", "tests": "", "docs": "", "dependencies": ""}

    patches = split_patch(base_patch)
    grouped_patches: dict[str, str] = {"code": "", "tests": "", "docs": "", "dependencies": ""}
    
    for patch in patches:
        categorized_patch = parse_patch_file_paths(patch)
        category = next((k for k, v in categorized_patch.items() if v), None)
        if category:
            grouped_patches[category] += patch + "\n"

    return grouped_patches


def count_code_lines_changed(patch: str, code_files: list[str]) -> int:
    """Count the number of code lines changed in a patch."""
    if not patch or not code_files:
        return 0

    code_lines_changed = 0
    current_file = None
    in_code_file = False

    for line in patch.split("\n"):
        diff_match = re.match(r"^diff --git a/(.+) b/(.+)$", line)
        if diff_match:
            current_file = diff_match.group(2)
            in_code_file = current_file in code_files
            continue

        if in_code_file and (line.startswith("+") or line.startswith("-")):
            if not line.startswith("+++") and not line.startswith("---"):
                code_lines_changed += 1

    return code_lines_changed
