"""Shared utilities for CooperBench.

Provides:
    console: Rich console for pretty output
    get_image_name: Generate Docker image names for tasks
    clean_model_name: Clean model name for experiment naming
    ResourceTracker: Track resources (sandboxes) for cleanup on exit
    setup_cleanup_handlers: Register SIGINT/SIGTERM handlers
"""

import atexit
import re
import signal
import sys
import threading
from collections.abc import Callable
from typing import Generic, TypeVar

from rich.console import Console

console = Console()

REGISTRY = "akhatua"
IMAGE_PREFIX = "cooperbench"


def get_image_name(repo_name: str, task_id: int) -> str:
    """Generate Docker Hub image name for a task."""
    repo_clean = repo_name.replace("_task", "").replace("_", "-")
    return f"{REGISTRY}/{IMAGE_PREFIX}-{repo_clean}:task{task_id}"


def clean_model_name(model: str) -> str:
    """Clean model name for use in experiment name.

    Examples:
        vertex_ai/gemini-3-flash-preview -> gemini-3-flash
        gpt-5.2 -> gpt-5-2
        moonshotai/Kimi-K2.5 -> kimi-k2-5
    """
    # Remove provider prefix (e.g., "gemini/", "openai/")
    if "/" in model:
        model = model.split("/")[-1]
    # Remove common suffixes
    model = re.sub(r"-(preview|latest|turbo)$", "", model)
    # Replace non-alphanumeric with dash
    model = re.sub(r"[^a-zA-Z0-9]+", "-", model)
    return model.strip("-").lower()


T = TypeVar("T")


class ResourceTracker(Generic[T]):
    """Thread-safe tracker for resources that need cleanup on exit."""

    def __init__(self, cleanup_fn: Callable[[T], None], name: str = "resource"):
        self._resources: list[T] = []
        self._lock = threading.Lock()
        self._cleanup_fn = cleanup_fn
        self._name = name

    def register(self, resource: T) -> None:
        """Register a resource for cleanup."""
        with self._lock:
            self._resources.append(resource)

    def unregister(self, resource: T) -> None:
        """Unregister a resource (already cleaned up)."""
        with self._lock:
            if resource in self._resources:
                self._resources.remove(resource)

    def cleanup_all(self) -> None:
        """Clean up all registered resources."""
        with self._lock:
            resources = list(self._resources)
        if resources:
            console.print(f"\n[yellow]cleaning up {len(resources)} {self._name}(s)...[/yellow]")
            for r in resources:
                try:
                    self._cleanup_fn(r)
                except Exception:
                    pass
            console.print("[green]done[/green]")


def setup_cleanup_handlers(tracker: ResourceTracker) -> None:
    """Setup SIGINT/SIGTERM handlers for graceful cleanup."""

    def handler(signum, frame):
        console.print("\n[yellow]interrupted - cleaning up...[/yellow]")
        tracker.cleanup_all()
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    atexit.register(tracker.cleanup_all)


def get_run_totals(run_name: str, setting: str) -> dict:
    """Get time metrics and cost from all result.json files.

    Returns both:
    - wall_time: elapsed time from earliest start to latest end
    - run_time: sum of individual task durations (total agent compute time)

    Args:
        run_name: Name of the experiment run
        setting: "solo" or "coop"

    Returns:
        {"wall_time": float, "run_time": float, "total_cost": float, "task_count": int}
    """
    import json
    from datetime import datetime
    from pathlib import Path

    log_dir = Path("logs") / run_name / setting
    if not log_dir.exists():
        return {"wall_time": 0.0, "run_time": 0.0, "total_cost": 0.0, "task_count": 0}

    total_cost = 0.0
    run_time = 0.0
    task_count = 0
    earliest_start = None
    latest_end = None

    for result_file in log_dir.rglob("result.json"):
        try:
            with open(result_file) as f:
                result = json.load(f)
            total_cost += result.get("total_cost", 0)
            run_time += result.get("duration_seconds", 0)
            task_count += 1

            # Track earliest start and latest end for wall clock time
            started_at = result.get("started_at")
            ended_at = result.get("ended_at")
            if started_at:
                start_dt = datetime.fromisoformat(started_at)
                if earliest_start is None or start_dt < earliest_start:
                    earliest_start = start_dt
            if ended_at:
                end_dt = datetime.fromisoformat(ended_at)
                if latest_end is None or end_dt > latest_end:
                    latest_end = end_dt
        except Exception:
            pass

    # Calculate wall clock time
    wall_time = 0.0
    if earliest_start and latest_end:
        wall_time = (latest_end - earliest_start).total_seconds()

    return {"wall_time": wall_time, "run_time": run_time, "total_cost": total_cost, "task_count": task_count}


__all__ = [
    "console",
    "REGISTRY",
    "IMAGE_PREFIX",
    "get_image_name",
    "clean_model_name",
    "ResourceTracker",
    "setup_cleanup_handlers",
    "get_run_totals",
]
