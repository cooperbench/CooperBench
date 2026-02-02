"""Patch splitting - separate feature changes from test changes."""

from unidiff import PatchSet

# Common patterns for test files across different languages/frameworks
DEFAULT_TEST_PATTERNS = [
    "test_",
    "_test.",
    "/tests/",
    "/test/",
    ".test.",
    ".spec.",
    "_spec.",
    "tests.py",
    "test.py",
    # Rust
    "#[cfg(test)]",
    "mod tests",
]

# Files to exclude from patches (agent helper scripts, junk files)
JUNK_FILE_PATTERNS = [
    "fix_",  # Helper scripts like fix_parquet.py
    "temp_",  # Temporary files
    "tmp_",
    "debug_",
    "scratch_",
    "helper_",
    ".pyc",
    "__pycache__",
    ".egg-info",
]


def split_patch(
    patch: str,
    test_patterns: list[str] | None = None,
) -> tuple[str, str]:
    """Split a patch into feature.patch and tests.patch.

    Args:
        patch: The full git diff as a string
        test_patterns: List of patterns to identify test files.
                      Defaults to common test file patterns.

    Returns:
        Tuple of (feature_patch, tests_patch) as strings.
        Either may be empty if no matching files found.
    """
    if test_patterns is None:
        test_patterns = DEFAULT_TEST_PATTERNS

    if not patch or not patch.strip():
        return "", ""

    try:
        patchset = PatchSet(patch)
    except Exception:
        # If we can't parse the patch, return it all as feature
        return patch, ""

    feature_hunks = []
    test_hunks = []

    for patched_file in patchset:
        path = patched_file.path

        # Skip junk/helper files
        if _is_junk_file(path):
            continue

        # Check if this is a test file
        is_test = _is_test_file(path, test_patterns)

        if is_test:
            test_hunks.append(str(patched_file))
        else:
            feature_hunks.append(str(patched_file))

    feature_patch = "\n".join(feature_hunks) if feature_hunks else ""
    tests_patch = "\n".join(test_hunks) if test_hunks else ""

    # Ensure patches end with newline (required for git apply)
    # Strip first to remove excess whitespace, then add exactly one newline
    feature_patch = feature_patch.strip() + "\n" if feature_patch.strip() else ""
    tests_patch = tests_patch.strip() + "\n" if tests_patch.strip() else ""

    return feature_patch, tests_patch


def _is_junk_file(path: str) -> bool:
    """Check if a file should be excluded from patches."""
    path_lower = path.lower()
    filename = path.split("/")[-1].lower()

    # Check filename patterns
    for pattern in JUNK_FILE_PATTERNS:
        if filename.startswith(pattern) or pattern in path_lower:
            return True

    # Exclude root-level Python scripts that aren't in src/ or proper package structure
    # These are usually helper scripts the agent created
    if "/" not in path and path.endswith(".py"):
        return True

    return False


def _is_test_file(path: str, patterns: list[str]) -> bool:
    """Check if a file path matches test file patterns."""
    path_lower = path.lower()

    for pattern in patterns:
        if pattern.lower() in path_lower:
            return True

    return False


def extract_feature_description(agent_output: str) -> str | None:
    """Extract the feature.md content from agent's output.

    The agent is instructed to output the feature description in a specific
    markdown format. This function extracts that content.

    Args:
        agent_output: The full agent conversation/output

    Returns:
        The extracted feature description, or None if not found.
    """
    # Look for the feature description block
    # The agent outputs it in markdown format starting with **Title**:

    # Only match structured feature description markers, not bash comments
    markers = [
        "**Title**:",
        "**Title:**",  # Without space variant
        "# Feature:",
        "## Feature",
    ]

    # Find the start of the feature description
    start_idx = -1
    for marker in markers:
        idx = agent_output.find(marker)
        if idx != -1:
            if start_idx == -1 or idx < start_idx:
                start_idx = idx

    if start_idx == -1:
        return None

    # Extract from the marker to end of that block
    # Look for common end markers or take until end
    content = agent_output[start_idx:]

    # Try to find where the description ends
    # Usually followed by code blocks or action outputs
    end_markers = [
        "\n```bash",
        "\n```python",
        "\n<action>",
        "\n## Steps",
        "\nBegin by",
    ]

    end_idx = len(content)
    for marker in end_markers:
        idx = content.find(marker)
        if idx != -1 and idx < end_idx:
            end_idx = idx

    description = content[:end_idx].strip()

    # Clean up any markdown code block wrappers
    if description.startswith("```markdown"):
        description = description[len("```markdown") :].strip()
    if description.startswith("```"):
        description = description[3:].strip()
    if description.endswith("```"):
        description = description[:-3].strip()

    # Remove agent submission markers
    cleanup_patterns = [
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        "SUBMIT_FINAL_OUTPUT",
        "END_OF_FEATURE",
        "<submit>",
        "</submit>",
    ]
    for pattern in cleanup_patterns:
        description = description.replace(pattern, "").strip()

    # Remove trailing whitespace on each line
    description = "\n".join(line.rstrip() for line in description.split("\n"))

    return description if description else None
