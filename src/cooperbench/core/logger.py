"""
Structured logging utilities for CooperBench experiments.

Provides a consistent logging interface with support for file and console output,
context-aware log messages, and specialized logging for git operations and patches.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class BenchLogger:
    """Structured logger for CooperBench with execution context tracking."""

    def __init__(
        self,
        name: str = "cooperbench",
        log_dir: Path | None = None,
        level: int = logging.INFO,
    ) -> None:
        """Initialize the logger with file and console outputs.

        Args:
            name: Logger name
            log_dir: Directory to save log files (optional)
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if self.logger.handlers:
            self.logger.handlers.clear()

        detailed_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_formatter = logging.Formatter("%(levelname)s | %(message)s")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"cooperbench_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)

            self.logger.info(f"Logging to file: {log_file}")

    def _format_context(self, kwargs: dict[str, Any]) -> str:
        """Format context kwargs into a string."""
        if kwargs:
            return f" | Context: {kwargs}"
        return ""

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with optional context."""
        self.logger.info(message + self._format_context(kwargs))

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with optional context."""
        self.logger.debug(message + self._format_context(kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with optional context."""
        self.logger.warning(message + self._format_context(kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with optional context."""
        self.logger.error(message + self._format_context(kwargs))

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message with optional context."""
        self.logger.critical(message + self._format_context(kwargs))

    def log_execution_start(self, agent_type: str, agent_workspace_path: Path, task: str) -> None:
        """Log the start of an execution."""
        self.info(
            "Starting execution",
            agent_type=agent_type,
            agent_workspace_path=str(agent_workspace_path),
            task=task[:100],
        )

    def log_execution_end(self, agent_type: str, success: bool, duration: float | None = None) -> None:
        """Log the end of an execution."""
        status = "SUCCESS" if success else "FAILED"
        context: dict[str, Any] = {"agent_type": agent_type, "status": status}
        if duration:
            context["duration_seconds"] = round(duration, 2)
        self.info(f"Execution completed: {status}", **context)

    def log_patch_application(self, patch_file: str, success: bool) -> None:
        """Log patch application results."""
        if not success:
            self.error(f"Patch application failed: {patch_file}")

    def log_git_operation(
        self,
        operation: str,
        args: list[str],
        success: bool,
        output: str | None = None,
    ) -> None:
        """Log git operations."""
        if not success:
            context: dict[str, Any] = {
                "operation": f"git {operation}",
                "args": " ".join(args),
            }
            if output:
                context["output"] = output[:200]
            self.error("Git operation failed", **context)

    def log_conflict_details(self, details: dict[str, Any]) -> None:
        """Log merge conflict details."""
        self.debug(
            "Conflict analysis",
            sections=details.get("conflict_sections", 0),
            lines=details.get("conflict_lines", 0),
        )


def get_logger(name: str = "cooperbench", log_dir: Path | None = None) -> BenchLogger:
    """Get or create a logger instance."""
    return BenchLogger(name, log_dir)
