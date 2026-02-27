"""Modal backend for evaluation."""

from typing import Any

import modal

from cooperbench.eval.backends.base import ExecResult, Sandbox


class ModalExecResult:
    """Wrapper for Modal exec result."""

    def __init__(self, result: Any) -> None:
        self._result = result
        self._stdout: str | None = None
        self._stderr: str | None = None

    @property
    def returncode(self) -> int:
        return self._result.returncode

    def stdout_read(self) -> str:
        if self._stdout is None:
            self._stdout = self._result.stdout.read()
        return self._stdout

    def stderr_read(self) -> str:
        if self._stderr is None:
            self._stderr = self._result.stderr.read()
        return self._stderr


class ModalSandbox:
    """Modal sandbox wrapper."""

    def __init__(self, sb: modal.Sandbox):
        self._sb = sb

    def exec(self, *args: str) -> ExecResult:
        """Execute a command in the sandbox."""
        result = self._sb.exec(*args)
        result.wait()
        return ModalExecResult(result)

    def terminate(self) -> None:
        """Terminate the sandbox."""
        self._sb.terminate()


class ModalBackend:
    """Modal backend for creating evaluation sandboxes."""

    def __init__(self, app_name: str = "cooperbench-eval"):
        self._app_name = app_name
        self._app: modal.App | None = None

    def _get_app(self) -> modal.App:
        """Get or create the Modal app."""
        if self._app is None:
            self._app = modal.App.lookup(self._app_name, create_if_missing=True)
        return self._app

    def create_sandbox(
        self,
        image: str,
        timeout: int = 600,
        workdir: str = "/workspace",
    ) -> Sandbox:
        """Create a Modal sandbox for evaluation."""
        # Eval runs need a stable long-running process for exec calls.
        # Task images may have entrypoints that exit immediately, so clear it
        # and run `sleep infinity` explicitly (mirrors Docker backend behavior).
        modal_image = modal.Image.from_registry(image).entrypoint([])
        sb = modal.Sandbox.create(
            "sleep",
            "infinity",
            image=modal_image,
            timeout=timeout,
            workdir=workdir,
            app=self._get_app(),
        )

        # Create patches directory
        result = sb.exec("mkdir", "-p", "/patches")
        result.wait()

        return ModalSandbox(sb)
