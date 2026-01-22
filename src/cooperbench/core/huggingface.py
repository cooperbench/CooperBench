"""
HuggingFace Hub integration for CooperBench experiments.

This module handles uploading and downloading experiment files (patches,
trajectories, plans) to/from HuggingFace Hub.
"""

import contextlib
import io
import subprocess
from pathlib import Path

from huggingface_hub import HfApi

from cooperbench.core.paths import CACHE_DIR, LOGS_DIR, get_cache_dir, get_cache_path, get_log_path

REPO_NAME = "CodeConflict/experiments"
API = HfApi()


def download_file_from_hf(file_path: Path, location: str) -> Path:
    """Download or locate a file based on the specified location.
    
    Args:
        file_path: Relative path to the file
        location: One of 'logs', 'cache', or 'hf'
            - 'logs': Returns path from logs directory
            - 'cache': Returns path from cache directory if exists
            - 'hf': Downloads from HuggingFace to cache directory

    Returns:
        Path to the file
    """
    if location == "logs":
        log_patch_path = get_log_path(file_path)
        assert log_patch_path.exists(), f"Logs path does not exist: {log_patch_path}"
        return log_patch_path
    
    cache_file_path = get_cache_path(file_path)
    if location == "cache":
        if cache_file_path.exists():
            return cache_file_path
    
    API.hf_hub_download(
        repo_id=REPO_NAME,
        filename=str(file_path),
        local_dir=get_cache_dir(),
        repo_type="dataset",
    )
    print(f"File cached to: {cache_file_path}")
    return cache_file_path


def upload_file_to_hf(
    file_path: Path,
    create_pr: bool = False,
) -> bool:
    """Upload a file to HuggingFace Hub.

    Args:
        file_path: Relative path for the file in the repository
        create_pr: Whether to create a PR instead of direct commit

    Returns:
        True if successful, False otherwise
    """
    log_file_path = get_log_path(file_path)

    if log_file_path.exists():
        try:
            relative_path = log_file_path.relative_to(LOGS_DIR)
            result = subprocess.run(
                ["git", "-C", str(LOGS_DIR), "status", "--porcelain", str(relative_path)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if not result.stdout.strip():
                return True
        except Exception:
            pass

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            API.upload_file(
                path_or_fileobj=log_file_path,
                path_in_repo=str(file_path),
                repo_id=REPO_NAME,
                repo_type="dataset",
                create_pr=create_pr,
            )
        return True
    except Exception as e:
        print(f"ERROR: Failed to upload file to Hugging Face - {e}")
        return False
