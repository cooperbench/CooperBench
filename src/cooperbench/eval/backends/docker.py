"""Docker backend for evaluation."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import docker
from docker.models.containers import Container

from cooperbench.eval.backends.base import ExecResult, Sandbox


class DockerExecResult:
    """Wrapper for Docker exec result."""

    def __init__(self, exit_code: int, output: bytes) -> None:
        self._returncode = exit_code
        self._output = output.decode("utf-8", errors="replace")

    @property
    def returncode(self) -> int:
        return self._returncode

    def stdout_read(self) -> str:
        return self._output

    def stderr_read(self) -> str:
        # Docker exec with stream=False combines stdout/stderr
        return ""


class DockerSandbox:
    """Docker container sandbox wrapper."""

    def __init__(self, container: Container, workdir: str, timeout: int = 600):
        self._container = container
        self._workdir = workdir
        self._timeout = timeout

    def exec(self, *args: str) -> ExecResult:
        """Execute a command in the container with timeout support."""

        def _run_exec():
            return self._container.exec_run(
                cmd=list(args),
                workdir=self._workdir,
                demux=False,
            )

        # Use ThreadPoolExecutor to enforce timeout on exec_run
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_exec)
            try:
                exit_code, output = future.result(timeout=self._timeout)
            except FuturesTimeoutError:
                return DockerExecResult(-1, f"Command timed out after {self._timeout} seconds".encode())

        return DockerExecResult(exit_code, output or b"")

    def terminate(self) -> None:
        """Stop and remove the container."""
        try:
            self._container.stop(timeout=5)
        except docker.errors.APIError:
            pass
        try:
            self._container.remove(force=True)
        except docker.errors.APIError:
            pass


class DockerBackend:
    """Docker backend for creating evaluation sandboxes."""

    def __init__(self) -> None:
        self._client: docker.DockerClient | None = None

    def _get_client(self) -> docker.DockerClient:
        """Get or create the Docker client."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def create_sandbox(
        self,
        image: str,
        timeout: int = 600,
        workdir: str = "/workspace",
    ) -> Sandbox:
        """Create a Docker container sandbox for evaluation."""
        client = self._get_client()

        # Run container in detached mode with a long-running command
        container = client.containers.run(
            image=image,
            command="sleep infinity",
            detach=True,
            working_dir=workdir,
            remove=False,
            # Set timeout via stop_signal behavior (container can be stopped)
            stop_signal="SIGTERM",
        )

        sandbox = DockerSandbox(container, workdir, timeout)

        # Create patches directory
        sandbox.exec("mkdir", "-p", "/patches")

        return sandbox
