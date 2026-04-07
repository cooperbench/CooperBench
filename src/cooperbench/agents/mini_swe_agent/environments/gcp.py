"""GCP VM environment for cloud execution.

Creates a GCP Compute Engine VM with Container-Optimized OS,
runs the specified Docker image, and executes commands via SSH.
"""

from __future__ import annotations

import logging
import platform
import shlex
import subprocess
import time
import uuid
from typing import Any

from pydantic import BaseModel


class GCPEnvironmentConfig(BaseModel):
    """Configuration for GCP VM environment."""

    image: str  # Docker image to run
    project_id: str | None = None  # Defaults to GOOGLE_CLOUD_PROJECT or gcloud config
    zone: str = "us-central1-a"
    machine_type: str = "e2-medium"  # 2 vCPU, 4GB RAM
    cwd: str = "/"
    timeout: int = 3600
    env: dict[str, str] = {}
    max_retries: int = 3
    retry_delay: float = 5.0
    network: str | None = None  # VPC network for git server connectivity
    vm_image_family: str | None = None  # Custom image family (e.g., "cooperbench-eval")
    vm_image_project: str | None = None  # Project for custom image (defaults to project_id)


class GCPEnvironment:
    """GCP VM environment for running agent commands.

    Creates a Container-Optimized OS VM, starts a Docker container,
    and executes commands via SSH + docker exec.

    Example:
        env = GCPEnvironment(
            image="python:3.11",
            project_id="my-project",
        )
        result = env.execute("python --version")
        print(result["output"])  # Python 3.11.x
        env.cleanup()
    """

    def __init__(
        self,
        *,
        config_class: type = GCPEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        self.logger = logger or logging.getLogger("cooperbench.agents.mini_swe_agent.gcp")
        self.config = config_class(**kwargs)

        # Resolve project ID
        self._project_id = self.config.project_id or self._get_default_project()
        if not self._project_id:
            raise ValueError(
                "project_id required. Set via config, GOOGLE_CLOUD_PROJECT env var, "
                "or gcloud config set project <project-id>"
            )

        # Generate unique VM name
        self._vm_name = f"cooperbench-agent-{uuid.uuid4().hex[:12]}"
        self._zone = self.config.zone

        # State
        self._vm_created = False
        self._container_started = False
        self._compute_client = None

        # Create VM and start container
        self._create_vm()
        self._wait_for_ssh()
        self._start_container()

    def _get_default_project(self) -> str | None:
        """Get default GCP project from environment or gcloud config."""
        import os

        # Try environment variable first
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
        if project:
            return project

        # Try cooperbench config
        try:
            from cooperbench.config import ConfigManager

            config = ConfigManager()
            project = config.get("gcp_project_id")
            if project:
                return project
        except Exception:
            pass

        # Try gcloud config
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

        return None

    def _get_compute_client(self):
        """Get or create Compute Engine client."""
        if self._compute_client is None:
            from google.cloud import compute_v1

            self._compute_client = compute_v1.InstancesClient()
        return self._compute_client

    def _create_vm(self):
        """Create the GCP VM with Container-Optimized OS."""
        from google.cloud import compute_v1

        self.logger.debug(f"Creating GCP VM {self._vm_name} in {self._zone}")

        client = self._get_compute_client()

        # Build instance configuration
        instance = compute_v1.Instance()
        instance.name = self._vm_name
        instance.machine_type = f"zones/{self._zone}/machineTypes/{self.config.machine_type}"

        # Boot disk with Container-Optimized OS or custom image
        disk = compute_v1.AttachedDisk()
        disk.auto_delete = True
        disk.boot = True
        disk.type_ = "PERSISTENT"

        disk_init = compute_v1.AttachedDiskInitializeParams()
        disk_init.disk_size_gb = 50  # 50GB should be enough for most images

        if self.config.vm_image_family:
            # Use custom image family (e.g., pre-built with cached Docker images)
            image_project = self.config.vm_image_project or self._project_id
            disk_init.source_image = f"projects/{image_project}/global/images/family/{self.config.vm_image_family}"
            self.logger.debug(f"Using custom image family: {self.config.vm_image_family}")
        else:
            # Use Container-Optimized OS (default)
            disk_init.source_image = "projects/cos-cloud/global/images/family/cos-stable"
            self.logger.debug("Using Container-Optimized OS (cos-stable)")

        disk.initialize_params = disk_init
        instance.disks = [disk]

        # Network interface
        network_interface = compute_v1.NetworkInterface()
        if self.config.network:
            network_interface.network = f"projects/{self._project_id}/global/networks/{self.config.network}"
        else:
            network_interface.network = f"projects/{self._project_id}/global/networks/default"

        # External IP for SSH access
        access_config = compute_v1.AccessConfig()
        access_config.name = "External NAT"
        access_config.type_ = "ONE_TO_ONE_NAT"
        network_interface.access_configs = [access_config]
        instance.network_interfaces = [network_interface]

        # Service account with default scopes
        service_account = compute_v1.ServiceAccount()
        service_account.email = "default"
        service_account.scopes = [
            "https://www.googleapis.com/auth/devstorage.read_only",
            "https://www.googleapis.com/auth/logging.write",
        ]
        instance.service_accounts = [service_account]

        # Metadata - startup script is not needed since we'll run docker commands via SSH

        # Create the instance
        request = compute_v1.InsertInstanceRequest()
        request.project = self._project_id
        request.zone = self._zone
        request.instance_resource = instance

        operation = client.insert(request=request)

        # Mark VM as created immediately after insert succeeds
        # This ensures cleanup() will delete the VM even if _wait_for_operation times out
        self._vm_created = True

        # Wait for operation to complete
        self._wait_for_operation(operation.name)
        self.logger.debug(f"VM {self._vm_name} created successfully")

    def _wait_for_operation(self, operation_name: str, timeout: int = 300):
        """Wait for a zone operation to complete."""
        from google.cloud import compute_v1

        client = compute_v1.ZoneOperationsClient()
        start_time = time.time()

        while time.time() - start_time < timeout:
            operation = client.get(
                project=self._project_id,
                zone=self._zone,
                operation=operation_name,
            )

            if operation.status == compute_v1.Operation.Status.DONE:
                if operation.error:
                    errors = [e.message for e in operation.error.errors]
                    raise RuntimeError(f"VM operation failed: {errors}")
                return

            time.sleep(2)

        raise TimeoutError(f"Operation {operation_name} timed out after {timeout}s")

    def _wait_for_ssh(self, timeout: int = 180):
        """Wait for SSH to become available on the VM."""
        self.logger.debug(f"Waiting for SSH on {self._vm_name}...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Try a simple SSH command
                result = subprocess.run(
                    [
                        "gcloud",
                        "compute",
                        "ssh",
                        self._vm_name,
                        f"--zone={self._zone}",
                        f"--project={self._project_id}",
                        "--command=echo ready",
                        "--quiet",
                        "--strict-host-key-checking=no",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0 and "ready" in result.stdout:
                    self.logger.debug("SSH is ready")
                    return
            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                self.logger.debug(f"SSH not ready yet: {e}")

            time.sleep(5)

        raise TimeoutError(f"SSH not available on {self._vm_name} after {timeout}s")

    def _start_container(self):
        """Start the Docker container on the VM."""
        self.logger.debug(f"Starting container with image: {self.config.image}")

        # Pull the image (may already be cached if using custom VM image)
        pull_result = self._ssh_exec(f"docker pull {self.config.image}")
        if pull_result["returncode"] != 0:
            raise RuntimeError(f"Failed to pull image {self.config.image}: {pull_result['output']}")

        # Build environment variable arguments
        env_args = ""
        for key, value in self.config.env.items():
            # Escape single quotes in values
            escaped_value = value.replace("'", "'\\''")
            env_args += f" -e '{key}={escaped_value}'"

        # Start container with sleep infinity
        # --entrypoint "" clears any entrypoint in the image (matches Modal's .entrypoint([]))
        # -w sets working directory (quoted to handle paths with spaces/special chars)
        quoted_cwd = shlex.quote(self.config.cwd)
        start_cmd = (
            f"docker run -d --name agent --entrypoint '' -w {quoted_cwd}{env_args} {self.config.image} sleep infinity"
        )
        start_result = self._ssh_exec(start_cmd)

        if start_result["returncode"] != 0:
            raise RuntimeError(f"Failed to start container: {start_result['output']}")

        # Wait for container to be running
        for _ in range(30):  # 30 seconds max
            check_result = self._ssh_exec("docker inspect -f '{{.State.Running}}' agent")
            if check_result["returncode"] == 0 and "true" in check_result["output"].lower():
                break
            time.sleep(1)
        else:
            raise RuntimeError("Container failed to start within 30 seconds")

        # Create working directory if it doesn't exist
        if self.config.cwd != "/":
            mkdir_result = self._ssh_exec(f"docker exec agent mkdir -p {quoted_cwd}")
            if mkdir_result["returncode"] != 0:
                self.logger.warning(f"Failed to create working directory: {mkdir_result['output']}")

        self._container_started = True
        self.logger.debug("Container started successfully")

    def _ssh_exec(self, command: str, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command on the VM via SSH."""
        exec_timeout = timeout or self.config.timeout

        try:
            result = subprocess.run(
                [
                    "gcloud",
                    "compute",
                    "ssh",
                    self._vm_name,
                    f"--zone={self._zone}",
                    f"--project={self._project_id}",
                    f"--command={command}",
                    "--quiet",
                    "--strict-host-key-checking=no",
                ],
                capture_output=True,
                text=True,
                timeout=exec_timeout,
            )
            output = result.stdout + result.stderr if result.stderr else result.stdout
            return {"output": output, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"output": f"Command timed out after {exec_timeout}s", "returncode": -1}
        except Exception as e:
            return {"output": str(e), "returncode": -1}

    def get_template_vars(self) -> dict[str, Any]:
        """Return template variables for the environment."""
        return self.config.model_dump() | {
            "system": "Linux",
            "release": "gcp",
            "version": "",
            "machine": platform.machine(),
            "vm_name": self._vm_name,
            "project_id": self._project_id,
            "zone": self._zone,
        }

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container on the VM.

        Args:
            command: Shell command to execute
            cwd: Working directory (defaults to config.cwd)
            timeout: Command timeout in seconds (defaults to config.timeout)

        Returns:
            Dict with 'output' (str) and 'returncode' (int)
        """
        cwd = cwd or self.config.cwd
        exec_timeout = timeout or self.config.timeout

        if not self._container_started:
            raise RuntimeError("Container not started")

        # Escape the command for shell
        # We need to escape it twice: once for the local shell, once for the remote shell
        escaped_command = command.replace("'", "'\\''")
        # Use docker exec -w to set working directory (handles paths with spaces/special chars)
        quoted_cwd = shlex.quote(cwd)
        docker_cmd = f"docker exec -w {quoted_cwd} agent bash -lc '{escaped_command}'"

        return self._ssh_exec(docker_cmd, timeout=exec_timeout)

    def cleanup(self) -> None:
        """Delete the VM and clean up resources."""
        if not self._vm_created:
            return

        self.logger.debug(f"Deleting VM {self._vm_name}")

        try:
            from google.cloud import compute_v1

            client = self._get_compute_client()
            request = compute_v1.DeleteInstanceRequest()
            request.project = self._project_id
            request.zone = self._zone
            request.instance = self._vm_name

            operation = client.delete(request=request)

            # Wait for deletion (with shorter timeout)
            try:
                self._wait_for_operation(operation.name, timeout=120)
            except TimeoutError:
                # VM deletion initiated, will complete asynchronously
                pass

            self._vm_created = False
            self._container_started = False
            self.logger.debug(f"VM {self._vm_name} deleted")
        except Exception as e:
            self.logger.warning(f"Failed to delete VM {self._vm_name}: {e}")

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        try:
            self.cleanup()
        except Exception:
            pass
