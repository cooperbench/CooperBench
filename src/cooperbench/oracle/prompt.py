"""Oracle prompt injection — prepends ground-truth solution to agent task text.

Three injection styles are supported (see OracleMode):

  patch        Raw unified diff — faithful to the ground truth, most literal.
  code         Post-patch file contents — shows the finished state, no diff noise.
  intent       Diff + one-paragraph plain-English explanation extracted from feature.md.

The agent is always told:
  - The provided solution is the correct implementation.
  - Its goal is to apply/coordinate the solution, not to re-implement from scratch.
  - If deviating, it must explain why in its final message.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class OracleMode(str, Enum):
    """How the ground-truth solution is presented to the agent."""

    PATCH = "patch"
    """Raw unified diff of the ground-truth implementation."""

    CODE = "code"
    """Modified file contents (post-patch state)."""

    INTENT = "intent"
    """Raw diff + plain-English description extracted from feature.md."""


_SOLUTION_BLOCK = """\
---
## Oracle Solution (ground-truth implementation)

You have been given the correct implementation of your assigned feature.
Your goal is NOT to re-implement this feature from scratch.
Your objectives are:
1. Understand what this solution does.
2. Communicate with the other agent about your solution's scope and approach.
3. Coordinate so that both features work correctly after merging.
4. Apply your solution (or a compatible adaptation) to the shared repository.

Do NOT deviate from the provided solution unless you discover an unavoidable
conflict with the other agent's feature.  If you do deviate, explain why in
your final message before submitting.

{solution_section}
---
"""

_PATCH_SECTION = """\
The solution is provided as a unified diff.  Apply it with `git apply` or
by making the equivalent edits manually:

```diff
{patch}
```
"""

_CODE_SECTION = """\
The solution has already been applied to the following file(s).
Review them to understand your implementation before coordinating:

{file_contents}
"""

_INTENT_SECTION = """\
### What this feature does

{description}

### Implementation diff

```diff
{patch}
```
"""


def _read_patch(feature_dir: Path) -> str:
    """Return raw patch text, or empty string if the file is missing."""
    patch_path = feature_dir / "feature.patch"
    if patch_path.exists():
        return patch_path.read_text()
    return ""


def _read_description(feature_dir: Path) -> str:
    """Return the feature.md description block, stripped of markdown headings."""
    md_path = feature_dir / "feature.md"
    if not md_path.exists():
        return ""
    content = md_path.read_text().strip()
    # Return full content — the agent benefits from the full spec.
    return content


def _apply_patch_to_temp(base_dir: Path, patch_text: str) -> dict[str, str]:
    """Apply a patch in a temp copy of base_dir and return {path: content} for modified files.

    Returns an empty dict if anything fails (missing git, bad patch, etc.).
    The caller falls back to returning the raw patch in that case.
    """
    import subprocess
    import tempfile

    if not patch_text.strip():
        return {}

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Copy base directory
            import shutil

            shutil.copytree(str(base_dir), str(tmp_path / "repo"), symlinks=True)
            repo = tmp_path / "repo"

            # Write patch to file
            patch_file = tmp_path / "patch.diff"
            patch_file.write_text(patch_text)

            result = subprocess.run(
                ["git", "apply", "--whitespace=fix", str(patch_file)],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return {}

            # Parse changed file paths from the patch header
            changed_files: dict[str, str] = {}
            for line in patch_text.splitlines():
                if line.startswith("+++ b/"):
                    rel_path = line[6:]
                    abs_path = repo / rel_path
                    if abs_path.exists():
                        try:
                            changed_files[rel_path] = abs_path.read_text()
                        except UnicodeDecodeError:
                            pass
            return changed_files
    except Exception:
        return {}


def _format_code_section(file_contents: dict[str, str]) -> str:
    """Format a dict of {path: content} into fenced code blocks."""
    if not file_contents:
        return "(Could not extract modified file contents; see the diff instead.)"

    parts = []
    for path, content in file_contents.items():
        lang = "python" if path.endswith(".py") else ""
        parts.append(f"**{path}**\n\n```{lang}\n{content}\n```")
    return "\n\n".join(parts)


def build_oracle_task(
    base_task: str,
    feature_dir: Path,
    mode: OracleMode = OracleMode.PATCH,
    repo_base_dir: Path | None = None,
) -> str:
    """Prepend the ground-truth solution to an agent's task description.

    Args:
        base_task:      The original task text (feature.md content).
        feature_dir:    Path to the feature directory (contains feature.patch, feature.md).
        mode:           How to present the solution (patch / code / intent).
        repo_base_dir:  Base directory of the repo image — required for ``code`` mode to
                        apply the patch and extract post-patch file contents.  If None,
                        ``code`` mode falls back to ``patch`` mode.

    Returns:
        Modified task string with the oracle solution block prepended.
    """
    feature_dir = Path(feature_dir)
    patch = _read_patch(feature_dir)

    if not patch:
        # No patch available — return original task unchanged; the agent runs blind.
        return base_task

    if mode == OracleMode.PATCH:
        solution_section = _PATCH_SECTION.format(patch=patch)

    elif mode == OracleMode.CODE:
        file_contents: dict[str, str] = {}
        if repo_base_dir is not None:
            file_contents = _apply_patch_to_temp(Path(repo_base_dir), patch)
        solution_section = _CODE_SECTION.format(file_contents=_format_code_section(file_contents))

    elif mode == OracleMode.INTENT:
        description = _read_description(feature_dir)
        solution_section = _INTENT_SECTION.format(description=description, patch=patch)

    else:
        solution_section = _PATCH_SECTION.format(patch=patch)

    oracle_block = _SOLUTION_BLOCK.format(solution_section=solution_section)
    return oracle_block + "\n" + base_task
