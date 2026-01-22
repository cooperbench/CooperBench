"""
Git operations and workspace management for CooperBench.

This module provides utilities for running git commands, setting up agent workspaces
using git worktrees, and managing the base repository state.
"""

import shutil
import subprocess
from pathlib import Path

from cooperbench.core.paths import get_base_repo_path, get_cache_path, get_log_path


def make_dirs(file_path: Path) -> None:
    """Create necessary directories for log and cache paths."""
    get_log_path(file_path).parent.mkdir(parents=True, exist_ok=True)
    get_cache_path(file_path).parent.mkdir(parents=True, exist_ok=True)


def setup_base_repo(task_folder: Path) -> Path:
    """Run setup scripts in the given task folder to initialize the base repository.

    Args:
        task_folder: Path to the task folder containing setup scripts

    Returns:
        Path to the base repository
    """
    base_repo_path = get_base_repo_path(task_folder)

    if base_repo_path is None:
        setup_script = task_folder / "setup.sh"
        if setup_script.exists():
            subprocess.run(
                ["bash", str(setup_script.resolve())],
                cwd=str(task_folder.resolve()),
                text=True,
                capture_output=True,
                check=True,
            )
        else:
            raise FileNotFoundError(f"Setup script not found in {task_folder}")
        
        base_repo_path = get_base_repo_path(task_folder)
        if base_repo_path is None:
            raise RuntimeError(f"No git repository found in {task_folder} after running setup script.")
    
    return base_repo_path


def setup_agent_workspace(
    base_repo_path: Path,
    base_commit: str,
    branch_name: str,
    agent_workspace_path: Path,
) -> Path:
    """Create a clean worktree for the agent to work in.
    
    Args:
        base_repo_path: Path to the base git repository
        base_commit: Commit hash to base the worktree on
        branch_name: Name for the new branch
        agent_workspace_path: Destination path for the worktree

    Returns:
        Path to the agent workspace
    """
    if agent_workspace_path.exists():
        shutil.rmtree(agent_workspace_path, ignore_errors=True)

    branch_exists = run_git_command(
        str(base_repo_path.resolve()),
        "rev-parse",
        "--verify",
        "--quiet",
        f"refs/heads/{branch_name}",
        check=False,
        capture_output=True,
    )

    run_git_command(
        str(base_repo_path.resolve()),
        "worktree",
        "remove",
        str(agent_workspace_path.resolve()),
        check=False,
        capture_output=True,
    )

    if branch_exists:
        run_git_command(
            str(base_repo_path.resolve()),
            "branch",
            "-D",
            branch_name,
            check=False,
            capture_output=True,
        )

    agent_workspace_path.mkdir(parents=True, exist_ok=True)
    run_git_command(
        str(base_repo_path.resolve()),
        "worktree",
        "add",
        str(agent_workspace_path.resolve()),
        "-b",
        branch_name,
        base_commit,
        capture_output=True,
    )
    return agent_workspace_path


def get_base_commit(base_repo_path: Path) -> str:
    """Get the current HEAD commit from the base repository."""
    result = run_git_command(base_repo_path, "rev-parse", "HEAD", capture_output=True)
    return result.strip() if isinstance(result, str) else ""


def run_git_command(
    worktree_path: Path | str,
    *args: str,
    capture_output: bool = False,
    check: bool = True,
    return_returncode: bool = False,
) -> tuple[str, int] | str:
    """Run a git command in a specific worktree.
    
    Args:
        worktree_path: Path to the git worktree
        *args: Git command arguments
        capture_output: Whether to capture and return stdout
        check: Whether to raise on non-zero exit code
        return_returncode: Whether to return (output, returncode) tuple

    Returns:
        Command output string, or tuple of (output, returncode) if return_returncode=True
    """
    cmd = ["git", "-C", str(worktree_path)] + list(args)
    try:
        if capture_output:
            result = subprocess.run(
                cmd,
                check=check,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            output_result = result.stdout or ""
        else:
            result = subprocess.run(
                cmd,
                check=check,
                stdout=None,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            output_result = ""

        if return_returncode:
            return (output_result, result.returncode)
        return output_result
    except subprocess.CalledProcessError as e:
        print(f"Error running git command {' '.join(cmd)}: {e}")
        if capture_output:
            print("stdout:", e.stdout if hasattr(e, "stdout") else "")
            print("stderr:", e.stderr if hasattr(e, "stderr") else "")
        raise
