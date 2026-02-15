"""Modal Sandbox environment for cloud execution (v2 interface)."""

import logging
import platform
import threading
import time
from typing import Any

import modal
from pydantic import BaseModel

# Retryable error patterns
_RETRYABLE_PATTERNS = [
    "Image build",
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "INTERNAL",
    "temporarily unavailable",
    "rate limit",
    "ClientClosed",
    "NOT_FOUND",
    "Sandbox not found",
    "already shut down",
    "Container ID",
    "finished",
]

# Global thread-safe image cache to prevent duplicate builds
_image_cache: dict[str, modal.Image] = {}
_image_locks: dict[str, threading.Lock] = {}
_cache_lock = threading.Lock()

# Global Modal app (shared across all environments)
_global_app: modal.App | None = None
_app_lock = threading.Lock()


def _get_global_app() -> modal.App:
    """Get or create the global Modal app (thread-safe)."""
    global _global_app
    with _app_lock:
        if _global_app is None:
            _global_app = modal.App.lookup("cooperbench", create_if_missing=True)
        return _global_app


def _reset_global_app() -> None:
    """Reset the global app (e.g., after ClientClosed errors)."""
    global _global_app
    with _app_lock:
        _global_app = None


def _get_or_build_image(image_name: str) -> modal.Image:
    """Get cached image or build it (thread-safe, only one build per image)."""
    if image_name in _image_cache:
        return _image_cache[image_name]

    with _cache_lock:
        if image_name not in _image_locks:
            _image_locks[image_name] = threading.Lock()
        lock = _image_locks[image_name]

    with lock:
        if image_name in _image_cache:
            return _image_cache[image_name]

        image = modal.Image.from_registry(image_name).entrypoint([])
        _image_cache[image_name] = image
        return image


def _invalidate_image(image_name: str) -> None:
    """Remove an image from cache (e.g., after build failure)."""
    with _cache_lock:
        _image_cache.pop(image_name, None)


class ModalEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    timeout: int = 3600
    env: dict[str, str] = {}
    max_retries: int = 5
    retry_delay: float = 5.0


class ModalEnvironment:
    sb: modal.Sandbox | None

    def __init__(
        self,
        *,
        config_class: type = ModalEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        self.logger = logger or logging.getLogger("cooperbench.agents.mini_swe_agent_v2.modal")
        self.config = config_class(**kwargs)
        self.sb = None
        self._start_sandbox_with_retry()

    def _reset_client(self):
        """Reset Modal client state for fresh connection."""
        _reset_global_app()
        _invalidate_image(self.config.image)

    def _build_image(self):
        """Build the Modal image (globally cached, thread-safe)."""
        return _get_or_build_image(self.config.image)

    def _is_retryable(self, error: Exception) -> bool:
        """Check if error is retryable."""
        error_str = str(error) + str(type(error).__name__)
        return any(p in error_str for p in _RETRYABLE_PATTERNS)

    def _start_sandbox(self):
        """Create and start the Modal Sandbox (single attempt)."""
        self.logger.debug(f"Creating Modal Sandbox with image: {self.config.image}")
        image = self._build_image()
        self.sb = modal.Sandbox.create(
            image=image,
            timeout=self.config.timeout,
            workdir=self.config.cwd,
            app=_get_global_app(),
        )
        self.logger.debug(f"Sandbox created: {self.sb.object_id}")

    def _start_sandbox_with_retry(self):
        """Create sandbox with retry logic for transient failures."""
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                self._start_sandbox()
                return
            except Exception as e:
                last_error = e
                error_str = str(e) + str(type(e).__name__)

                if attempt < self.config.max_retries - 1 and self._is_retryable(e):
                    delay = self.config.retry_delay * (2**attempt)
                    self.logger.warning(
                        f"Sandbox creation failed (attempt {attempt + 1}/{self.config.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    if "ClientClosed" in error_str or "Image build" in error_str:
                        self._reset_client()
                else:
                    break

        raise last_error

    def _is_sandbox_dead(self, error: Exception) -> bool:
        """Check if the error indicates the sandbox has terminated."""
        error_str = str(error) + str(type(error).__name__)
        return any(
            x in error_str
            for x in [
                "NOT_FOUND",
                "Sandbox not found",
                "already shut down",
                "Container ID",
                "finished",
                "ClientClosed",
            ]
        )

    def _reconnect_sandbox(self):
        """Attempt to create a new sandbox after the old one died."""
        self.logger.warning("Sandbox terminated unexpectedly, creating new sandbox...")
        old_sb = self.sb
        self.sb = None
        if old_sb:
            try:
                old_sb.terminate()
            except Exception:
                pass
        self._reset_client()
        self._start_sandbox_with_retry()

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump() | {
            "system": "Linux",
            "release": "modal",
            "version": "",
            "machine": platform.machine(),
        }

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Modal Sandbox with retry on sandbox death.

        v2 interface: action is a dict with {"command": "..."}.
        """
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                if self.sb is None:
                    raise RuntimeError("Sandbox not initialized")
                proc = self.sb.exec("bash", "-lc", f"cd {cwd} && {command}")
                stdout = proc.stdout.read()
                stderr = proc.stderr.read()
                proc.wait()
                output = stdout + stderr if stderr else stdout
                result = {"output": output, "returncode": proc.returncode, "exception_info": ""}
                self._check_finished(result)
                return result
            except Exception as e:
                # Re-raise Submitted exceptions (task completion)
                from cooperbench.agents.mini_swe_agent_v2.exceptions import Submitted

                if isinstance(e, Submitted):
                    raise
                last_error = e
                if self._is_sandbox_dead(e) and attempt < self.config.max_retries - 1:
                    self.logger.warning(
                        f"Sandbox died during execution (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                    )
                    self._reconnect_sandbox()
                else:
                    raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("No retries attempted")

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        from cooperbench.agents.mini_swe_agent_v2.exceptions import Submitted

        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        """Terminate the Modal Sandbox."""
        if hasattr(self, "sb") and self.sb:
            try:
                self.sb.terminate()
            except Exception:
                pass

    def __del__(self):
        self.cleanup()
