"""Prompt building for feature generation."""

from pathlib import Path

from unidiff import PatchSet


def _extract_patch_info(patch_path: Path) -> dict:
    """Extract file and line information from a patch file."""
    try:
        content = patch_path.read_text()
        patchset = PatchSet(content)
    except Exception:
        return {"files": [], "raw": "", "error": "Failed to parse patch"}

    files_info = []
    for patched_file in patchset:
        file_info = {
            "path": patched_file.path,
            "added": patched_file.added,
            "removed": patched_file.removed,
            "hunks": [],
        }
        for hunk in patched_file:
            file_info["hunks"].append(
                {
                    "source_start": hunk.source_start,
                    "source_length": hunk.source_length,
                    "target_start": hunk.target_start,
                    "target_length": hunk.target_length,
                }
            )
        files_info.append(file_info)

    return {"files": files_info, "raw": content}


def _read_feature_md(feature_dir: Path) -> str:
    """Read and return contents of feature.md."""
    feature_md = feature_dir / "feature.md"
    if feature_md.exists():
        return feature_md.read_text()
    return ""


def _extract_feature_title(feature_md_content: str) -> str | None:
    """Extract just the title from feature.md content."""
    import re

    # Look for **Title**: pattern
    match = re.search(r"\*\*Title\*\*:\s*(.+?)(?:\n|$)", feature_md_content)
    if match:
        return match.group(1).strip()

    # Fallback: look for # Title or ## Title
    match = re.search(r"^#+ (.+?)$", feature_md_content, re.MULTILINE)
    if match:
        return match.group(1).strip()

    return None


def _get_feature_info(task_dir: Path, feature_id: int) -> dict | None:
    """Get full information about a specific feature."""
    feature_dir = task_dir / f"feature{feature_id}"

    if not feature_dir.exists():
        return None

    feature_info = {
        "id": feature_id,
        "name": f"feature{feature_id}",
        "description": _read_feature_md(feature_dir),
        "patch_info": None,
    }

    # Extract patch information
    feature_patch = feature_dir / "feature.patch"
    if feature_patch.exists():
        feature_info["patch_info"] = _extract_patch_info(feature_patch)

    return feature_info


def _get_existing_feature_ids(task_dir: Path) -> list[int]:
    """Get IDs of all existing features in a task."""
    ids = []
    for d in task_dir.iterdir():
        if d.is_dir() and d.name.startswith("feature"):
            try:
                fid = int(d.name.replace("feature", ""))
                ids.append(fid)
            except ValueError:
                pass
    return sorted(ids)


def _get_test_command(task_dir: Path) -> str:
    """Extract the test command from runner.sh, resolving variables."""
    runner_sh = task_dir / "runner.sh"
    if not runner_sh.exists():
        return "# Test command not found - check runner.sh"

    content = runner_sh.read_text()

    # First, try to find variable definitions like TEST_PATH="..."
    variables = {}
    for line in content.split("\n"):
        line = line.strip()
        # Match patterns like: TEST_PATH="tests/io/test_parquet.py"
        if "=" in line and not line.startswith("#"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                var_name = parts[0].strip()
                var_value = parts[1].strip().strip('"').strip("'")
                variables[var_name] = var_value

    # Look for pytest or cargo test commands
    for line in content.split("\n"):
        line = line.strip()
        if "pytest" in line and not line.startswith("#"):
            if "python -m pytest" in line:
                # Resolve variables in the command
                resolved = line
                for var_name, var_value in variables.items():
                    resolved = resolved.replace(f"${var_name}", var_value)
                    resolved = resolved.replace(f"${{{var_name}}}", var_value)
                    resolved = resolved.replace(f'"${var_name}"', f'"{var_value}"')
                    resolved = resolved.replace(f'"${{{var_name}}}"', f'"{var_value}"')
                # Clean up timeout wrapper if present
                if resolved.startswith("timeout"):
                    # Extract just the pytest part
                    if "python -m pytest" in resolved:
                        idx = resolved.find("python -m pytest")
                        resolved = resolved[idx:]
                return resolved
        if "cargo test" in line and not line.startswith("#"):
            return line

    return "# See runner.sh for test commands"


def _extract_test_file(test_command: str) -> str:
    """Extract the test file path from a test command."""
    import re

    # Try to extract pytest test file path
    # Matches patterns like: pytest "tests/foo/test_bar.py" or pytest tests/foo/test_bar.py
    pytest_match = re.search(r'pytest\s+["\']?([^\s"\']+(?:test[^\s"\']*\.py|tests?/[^\s"\']+))["\']?', test_command)
    if pytest_match:
        return pytest_match.group(1)

    # Try to find any .py file path that looks like a test file
    test_file_match = re.search(r'([^\s"\']+(?:test[^\s"\']*\.py|tests?/[^\s"\']+\.py))', test_command)
    if test_file_match:
        return test_file_match.group(1)

    # For cargo test, return the tests directory
    if "cargo test" in test_command:
        return "tests/"

    return "the existing test file"


def _format_code_snippet(patch_content: str, max_lines: int = 80) -> str:
    """Format patch content for display, limiting length."""
    lines = patch_content.split("\n")
    if len(lines) <= max_lines:
        return patch_content

    # Take first half and last quarter
    first_part = lines[: max_lines // 2]
    last_part = lines[-(max_lines // 4) :]

    return "\n".join(first_part) + "\n\n... (truncated) ...\n\n" + "\n".join(last_part)


GENERATION_PROMPT_TEMPLATE = """Create a NEW feature that will CONFLICT with an existing feature during git merge.

## Existing Feature (your feature must conflict with this)

{feature_description}

### Code Changes:
{files_summary}

```diff
{code_snippet}
```
{other_features_section}
## Requirements

Your new feature must:
1. **Cause merge conflicts** - modify some of the same lines/regions (around lines {hot_lines})
2. **Be a real enhancement** - not random changes, but a legitimate useful feature
3. **MUST include tests** - write NEW test functions that verify your feature works
4. **Pass all tests** - run tests to verify everything works before submitting

**IMPORTANT TEST REQUIREMENTS**:
- You MUST write tests before submitting. A feature without tests is incomplete.
- **Add your tests to the SAME test file** that the existing tests use: `{test_file}`
- Do NOT create new test files. Add new test functions/classes to the existing test file.
- This ensures test changes can create merge conflicts with other features' tests.

Existing tests: `{test_command}`

You CAN modify other files too, but at least some changes must overlap with the existing feature to create conflicts.

## Output Format

**IMPORTANT**: Before submitting, you MUST create a feature description file at `.feature_description.md` in the repo root.

This description must be **detailed enough that another developer could implement the same feature** without seeing your code. Include:
- What the feature does and why it's useful
- The API/interface changes (new parameters, functions, classes)
- Key implementation details (algorithms, data structures, edge cases handled)
- How it integrates with the existing code

```bash
cat << 'FEATURE_EOF' > .feature_description.md
**Title**: [Descriptive feature title]

**Description**: [2-3 sentences explaining what the feature does and its purpose]

**API Changes**:
- [New function/method signatures with parameters]
- [New parameters added to existing functions]
- [New classes or data structures]

**Implementation Details**:
- [Key algorithms or logic used]
- [How it modifies existing behavior]
- [Edge cases handled]

**Files Modified**: [List each file and what was changed in it]
FEATURE_EOF
```

This file is required for the submission to be valid. Create it right before you submit.

Start by exploring the modified files to understand the code structure.
"""


def build_prompt(task_dir: Path, feature_id: int | None = None) -> str:
    """Build the generation prompt for a task, targeting a specific feature.

    Args:
        task_dir: Path to the task directory (e.g., dataset/dspy_task/task8394)
        feature_id: ID of the specific feature to target for conflicts.
                   If None, uses the first feature.

    Returns:
        The formatted prompt string
    """
    task_dir = Path(task_dir)

    # Get existing feature IDs
    existing_ids = _get_existing_feature_ids(task_dir)
    if not existing_ids:
        return "Error: No existing features found in task"

    # Default to first feature if not specified
    if feature_id is None:
        feature_id = existing_ids[0]

    if feature_id not in existing_ids:
        return f"Error: Feature {feature_id} not found. Available: {existing_ids}"

    # Get full feature info
    feature = _get_feature_info(task_dir, feature_id)
    if not feature:
        return f"Error: Could not load feature {feature_id}"

    # Format feature description
    feature_description = feature["description"] or f"(No description for feature {feature_id})"

    # Format files summary
    files_summary = ""
    hot_lines = []
    if feature["patch_info"] and feature["patch_info"].get("files"):
        for f in feature["patch_info"]["files"]:
            files_summary += f"- `{f['path']}` (+{f['added']}/-{f['removed']} lines)\n"
            for hunk in f["hunks"]:
                start = hunk["source_start"]
                end = start + hunk["source_length"]
                files_summary += f"  - Lines {start}-{end}\n"
                hot_lines.extend([start, end])

    # Format hot lines
    if hot_lines:
        hot_lines_str = ", ".join(str(line) for line in sorted(set(hot_lines))[:5])
    else:
        hot_lines_str = "the modified sections"

    # Get code snippet
    code_snippet = ""
    if feature["patch_info"] and feature["patch_info"].get("raw"):
        code_snippet = _format_code_snippet(feature["patch_info"]["raw"])
    else:
        code_snippet = "(patch content not available)"

    # Get test command and extract test file path
    test_command = _get_test_command(task_dir)
    test_file = _extract_test_file(test_command)

    # Collect titles from other features (to avoid duplicates)
    other_features_section = ""
    other_ids = [fid for fid in existing_ids if fid != feature_id]
    if other_ids:
        other_titles = []
        for fid in other_ids:
            feature_dir = task_dir / f"feature{fid}"
            md_content = _read_feature_md(feature_dir)
            title = _extract_feature_title(md_content) if md_content else None
            if title:
                other_titles.append(f'- Feature {fid}: "{title}"')
            else:
                other_titles.append(f"- Feature {fid}: (no title)")

        if other_titles:
            other_features_section = (
                "\n## Other Existing Features\n\n"
                "The following features already exist in this task. "
                "Make sure your proposed feature is different but compatible with these. \n"
                + "\n".join(other_titles)
                + "\n"
            )

    # Build final prompt
    prompt = GENERATION_PROMPT_TEMPLATE.format(
        feature_description=feature_description,
        files_summary=files_summary or "(no files info)",
        code_snippet=code_snippet,
        hot_lines=hot_lines_str,
        test_command=test_command,
        test_file=test_file,
        other_features_section=other_features_section,
    )

    return prompt


def list_features(task_dir: Path) -> list[int]:
    """List all feature IDs in a task.

    Args:
        task_dir: Path to the task directory

    Returns:
        List of feature IDs
    """
    return _get_existing_feature_ids(Path(task_dir))
