"""
Git merge operations and conflict analysis for CooperBench.

This module provides utilities for performing merges between feature branches,
analyzing merge conflicts, and generating merge reports.
"""

from pathlib import Path
from typing import Any

from cooperbench.core.git import run_git_command
from cooperbench.core.logger import get_logger

logger = get_logger("cooperbench.merge")


def merge(
    agent_workspace2_path: Path,
    feature1_branch: str,
) -> tuple[str, bool]:
    """Merge feature1 branch into feature2 workspace and capture diff output.

    Args:
        agent_workspace2_path: Path to feature2 agent workspace
        feature1_branch: Name of feature1 branch to merge

    Returns:
        Tuple of (diff output text, conflict boolean)
    """
    conflict = False
    diff_output: str = ""

    merge_output, returncode = run_git_command(
        agent_workspace2_path,
        "merge",
        feature1_branch,
        "--no-commit",
        "--no-ff",
        capture_output=True,
        check=False,
        return_returncode=True,
    )

    if returncode == 0:
        diff_output = str(run_git_command(agent_workspace2_path, "diff", "--cached", capture_output=True))
    elif (
        "CONFLICT (content): Merge conflict in" in merge_output
        and "Automatic merge failed; fix conflicts and then commit the result." in merge_output
    ):
        logger.debug("Merge resulted in conflicts!")
        conflict = True
        diff_output = str(run_git_command(agent_workspace2_path, "diff", capture_output=True))
        if not diff_output:
            diff_output = _capture_conflict_content(agent_workspace2_path)
    else:
        raise ValueError(f"Unexpected merge error: {merge_output}")

    return diff_output, conflict


def merge_union(
    agent_workspace2_path: Path,
    feature1_branch: str,
) -> tuple[str, bool]:
    """Merge with union strategy (concatenates conflicting changes).

    Args:
        agent_workspace2_path: Path to second feature agent workspace
        feature1_branch: Branch name for feature1

    Returns:
        Tuple of (diff output text, conflict boolean)
    """
    conflict = False
    diff_output: str = ""

    run_git_command(agent_workspace2_path, "merge", "--abort", check=False)

    gitattributes_path = agent_workspace2_path / ".gitattributes"
    had_existing_attrs = gitattributes_path.exists()
    original_attrs: str | None = None
    if had_existing_attrs:
        original_attrs = gitattributes_path.read_text(encoding="utf-8", errors="ignore")

    try:
        temp_marker = "# --- BEGIN TEMP UNION FOR ANALYSIS --- \n"
        union_line = "* merge=union\n"
        end_marker = "# --- END TEMP UNION FOR ANALYSIS --- \n"
        with gitattributes_path.open("a", encoding="utf-8") as f:
            f.write("\n" + temp_marker + union_line + end_marker)

        u_out, u_code = run_git_command(
            agent_workspace2_path,
            "merge",
            feature1_branch,
            "--no-commit",
            "--no-ff",
            capture_output=True,
            check=False,
            return_returncode=True,
        )

        if u_code == 0:
            diff_output = str(
                run_git_command(
                    agent_workspace2_path,
                    "diff",
                    "--cached",
                    capture_output=True,
                )
            )
        else:
            logger.debug("Merge resulted in conflicts!")
            conflict = True
            diff_output = str(run_git_command(agent_workspace2_path, "diff", capture_output=True))
            if not diff_output:
                diff_output = _capture_conflict_content(agent_workspace2_path)
            run_git_command(agent_workspace2_path, "merge", "--abort", check=False)

        return diff_output, conflict

    finally:
        try:
            if had_existing_attrs:
                gitattributes_path.write_text(original_attrs or "", encoding="utf-8")
            else:
                if gitattributes_path.exists():
                    gitattributes_path.unlink()
        except Exception as e:
            logger.warning(f"[union test] Failed to restore .gitattributes: {e}")


def analyze_merge(
    diff_output: str,
    conflict: bool,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Analyze merge between two features.

    Args:
        diff_output: Diff output text from the merge
        conflict: Whether the merge had conflicts
        report: Initial report dictionary to populate

    Returns:
        Dict containing merge analysis results
    """
    if conflict:
        report["merge_status"] = "conflicts"

    conflict_details = _analyze_diff_content(diff_output)
    report["conflict_details"] = conflict_details
    report["conflict_score"] = _calculate_conflict_score(conflict_details)

    return report


def analyze_merge_union(
    agent_workspace2_path: Path,
    feature1_branch: str,
    report: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Analyze merge with union strategy.

    Args:
        agent_workspace2_path: Path to second feature agent workspace
        feature1_branch: Branch name for feature1
        report: Initial report dictionary to populate

    Returns:
        Tuple of (Dict containing merge analysis results, diff content)
    """
    run_git_command(agent_workspace2_path, "merge", "--abort", check=False)

    gitattributes_path = agent_workspace2_path / ".gitattributes"
    had_existing_attrs = gitattributes_path.exists()
    original_attrs: str | None = None
    if had_existing_attrs:
        original_attrs = gitattributes_path.read_text(encoding="utf-8", errors="ignore")

    try:
        temp_marker = "# --- BEGIN TEMP UNION FOR ANALYSIS --- \n"
        union_line = "* merge=union\n"
        end_marker = "# --- END TEMP UNION FOR ANALYSIS --- \n"
        with gitattributes_path.open("a", encoding="utf-8") as f:
            f.write("\n" + temp_marker + union_line + end_marker)

        u_out, u_code = run_git_command(
            agent_workspace2_path,
            "merge",
            feature1_branch,
            "--no-commit",
            "--no-ff",
            capture_output=True,
            check=False,
            return_returncode=True,
        )

        if u_code == 0:
            diff_output = str(
                run_git_command(
                    agent_workspace2_path,
                    "diff",
                    "--cached",
                    capture_output=True,
                )
            )

            changed_names = (
                str(
                    run_git_command(
                        agent_workspace2_path,
                        "diff",
                        "--name-only",
                        "--cached",
                        capture_output=True,
                    )
                )
                .strip()
                .splitlines()
            )

            report["union_merge_status"] = "clean"
            report["union_changed_files"] = changed_names
            report["strategy"] = "union"

            conflict_details = _analyze_diff_content(diff_output)
            report["union_conflict_details"] = conflict_details
            report["union_conflict_score"] = _calculate_conflict_score(conflict_details)
            report["notes"] = (
                "Regular merge had conflicts; union driver concatenated overlapping changes. "
                "Review resulting files to ensure semantic correctness."
            )
            return report, diff_output

        else:
            logger.error("[union test] Union merge still resulted in conflicts; this should not happen.")
            union_conflict_diff = str(run_git_command(agent_workspace2_path, "diff", capture_output=True))
            if not union_conflict_diff:
                union_conflict_diff = _capture_conflict_content(agent_workspace2_path)

            report["union_merge_status"] = "conflicts"
            report["strategy"] = "union"
            report["notes"] = "Even with union driver, some paths remain conflicted."

            conflict_details = _analyze_diff_content(union_conflict_diff)
            report["union_conflict_details"] = conflict_details
            report["union_conflict_score"] = _calculate_conflict_score(conflict_details)

            run_git_command(agent_workspace2_path, "merge", "--abort", check=False)
            return report, union_conflict_diff

    finally:
        try:
            if had_existing_attrs:
                gitattributes_path.write_text(original_attrs or "", encoding="utf-8")
            else:
                if gitattributes_path.exists():
                    gitattributes_path.unlink()
        except Exception as e:
            logger.warning(f"[union test] Failed to restore .gitattributes: {e}")


def _capture_conflict_content(agent_workspace2_path: Path) -> str:
    """Capture content of conflicted files."""
    diff_output = ""
    try:
        conflicted_files = str(
            run_git_command(
                agent_workspace2_path,
                "diff",
                "--name-only",
                capture_output=True,
            )
        ).splitlines()
        for file_path in conflicted_files:
            try:
                file_content = str(
                    run_git_command(
                        agent_workspace2_path,
                        "show",
                        f":{file_path}",
                        capture_output=True,
                    )
                )
                if file_content:
                    diff_output += f"--- a/{file_path}\n+++ b/{file_path}\n{file_content}\n"
            except Exception as e:
                logger.warning(f"Could not read conflicted file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error capturing conflict content: {e}")

    return diff_output


def _analyze_diff_content(diff_output: str) -> dict[str, Any]:
    """Analyze diff content for conflict details."""
    if not diff_output:
        return {
            "conflict_sections": 0,
            "conflict_lines": 0,
            "avg_lines_per_conflict": 0,
        }

    if not isinstance(diff_output, str):
        diff_output = str(diff_output)

    conflict_sections = 0
    conflict_lines = 0
    in_conflict = False

    for line in diff_output.splitlines():
        line = line.strip()
        if "<<<<<<< " in line:
            conflict_sections += 1
            in_conflict = True
        elif "=======" in line:
            continue
        elif ">>>>>>>" in line:
            in_conflict = False
        elif in_conflict:
            conflict_lines += 1

    avg_lines = conflict_lines / conflict_sections if conflict_sections > 0 else 0

    details = {
        "conflict_sections": conflict_sections,
        "conflict_lines": conflict_lines,
        "avg_lines_per_conflict": round(avg_lines, 2),
    }

    logger.log_conflict_details(details)
    return details


def _calculate_conflict_score(conflict_details: dict[str, Any]) -> int:
    """Calculate conflict score based on details.
    
    Scoring:
    - Each conflict section: 20 points
    - Each conflicting line: 2 points
    """
    score = (conflict_details["conflict_sections"] * 20) + (conflict_details["conflict_lines"] * 2)
    return score
