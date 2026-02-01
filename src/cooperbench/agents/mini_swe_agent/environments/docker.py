"""Docker container environment for local execution."""

import logging
import platform
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

import docker
from docker.models.containers import Container
from pydantic import BaseModel


class DockerEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    timeout: int = 3600
    env: dict[str, str] = {}
    max_retries: int = 3
    retry_delay: float = 2.0


class DockerEnvironment:
    """Docker container environment for running agent commands locally."""

    container: Container | None

    def __init__(
        self,
        *,
        config_class: type = DockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        self.logger = logger or logging.getLogger("cooperbench.agents.mini_swe_agent.docker")
        self.config = config_class(**kwargs)
        self.container = None
        self._client: docker.DockerClient | None = None
        self._start_container()

    def _get_client(self) -> docker.DockerClient:
        """Get or create Docker client."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def _start_container(self):
        """Create and start the Docker container."""
        self.logger.debug(f"Creating Docker container with image: {self.config.image}")
        client = self._get_client()

        # Build environment variables
        env_vars = dict(self.config.env)

        self.container = client.containers.run(
            image=self.config.image,
            command="sleep infinity",
            detach=True,
            working_dir=self.config.cwd,
            environment=env_vars,
            remove=False,
            stdin_open=True,
            tty=True,
        )
        self.logger.debug(f"Container created: {self.container.id[:12]}")

    def get_template_vars(self) -> dict[str, Any]:
        """Return template variables for the environment."""
        return self.config.model_dump() | {
            "system": "Linux",
            "release": "docker",
            "version": "",
            "machine": platform.machine(),
        }

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container with timeout support."""
        cwd = cwd or self.config.cwd
        exec_timeout = timeout or self.config.timeout

        if self.container is None:
            raise RuntimeError("Container not initialized")

        # Refresh container state
        self.container.reload()

        # Build the command with cd
        full_command = f"cd {cwd} && {command}"

        def _run_exec():
            return self.container.exec_run(
                cmd=["bash", "-lc", full_command],
                workdir=cwd,
                demux=False,
                environment=self.config.env,
            )

        try:
            # Use ThreadPoolExecutor to enforce timeout on exec_run
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_exec)
                try:
                    exit_code, output = future.result(timeout=exec_timeout)
                except FuturesTimeoutError:
                    self.logger.warning(f"Command timed out after {exec_timeout}s: {command[:100]}")
                    return {"output": f"Command timed out after {exec_timeout} seconds", "returncode": -1}

            output_str = output.decode("utf-8", errors="replace") if output else ""
            return {"output": output_str, "returncode": exit_code}

        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            raise

    def cleanup(self):
        """Stop and remove the Docker container."""
        if hasattr(self, "container") and self.container:
            try:
                self.container.stop(timeout=5)
            except docker.errors.APIError:
                pass
            try:
                self.container.remove(force=True)
            except docker.errors.APIError:
                pass
            self.container = None

    def __del__(self):
        self.cleanup()
