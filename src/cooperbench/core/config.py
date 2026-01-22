"""
Global configuration constants for CooperBench.

This module defines configuration values used across the benchmark including
directory paths, API settings, and default values.
"""

from pathlib import Path

# Directory paths
LOGS_DIR = Path("logs")
CACHE_DIR = Path.home() / ".cooperbench_cache"

# API Configuration
GITHUB_API_BASE = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"

# Default values
DEFAULT_PER_PAGE = 100
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 90

# Task acceptance criteria
MAX_CODE_FILES = 3
MAX_CODE_LINES_CHANGED = 200


def ensure_dirs() -> None:
    """Create necessary directories if they do not exist."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
