"""Replacement for ``openhands/tools/task_tracker/__init__.py``.

Injected into the Modal sandbox at image-build time when the openhands
adapter is running team mode.  Identical to the upstream ``__init__``
plus a single ``from . import coop_definition`` side-effect import,
which re-registers ``TaskTrackerTool`` under the Redis-backed
``CoopTaskTracker`` resolver.

Lives here as a normal Python module (rather than as a string literal
in the adapter) so ruff/black/mypy keep it honest.  The adapter
``add_local_file``s this file to the sandbox path
``$OH_DIR/__init__.py``.
"""

from .definition import (
    TaskTrackerAction,
    TaskTrackerExecutor,
    TaskTrackerObservation,
    TaskTrackerStatusType,
    TaskTrackerTool,
)
from . import coop_definition  # noqa: F401, E402 — overrides TaskTrackerTool registration


__all__ = [
    "TaskTrackerAction",
    "TaskTrackerExecutor",
    "TaskTrackerObservation",
    "TaskTrackerStatusType",
    "TaskTrackerTool",
]
