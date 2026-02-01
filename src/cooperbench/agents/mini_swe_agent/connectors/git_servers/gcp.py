"""GCP VM-based git server for code collaboration."""

from __future__ import annotations

import logging
import subprocess
import time
import uuid


class GCPGitServer:
    """Shared git server on GCP VM for code collaboration.

    Creates a GCP VM running git-daemon that agents can push/pull to.
    Supports VPC networking for agents on the same network.

    Example:
        server = GCPGitServer.create(
            run_id="my-run",
            project_id="my-project",
            network="cooperbench-vpc",
        )
        print(server.url)  # git://10.128.0.5:9418/repo.git
        print(server.network_name)  # cooperbench-vpc
        # ... agents push/pull ...
        server.cleanup()
    """

    def __init__(
        self,
        *,
        vm_name: str,
        project_id: str,
        zone: str,
        network: str | None,
        git_url: str,
    ):
        """Initialize with existing VM info.

        Use GCPGitServer.create() to create a new server.
        """
        self._vm_name = vm_name
        self._project_id = project_id
        self._zone = zone
        self._network = network
        self._git_url = git_url
        self._vm_created = False
        self._firewall_created = False
        self._compute_client = None
        self._logger = logging.getLogger("cooperbench.agents.mini_swe_agent.git_server.gcp")

    @classmethod
    def create(
        cls,
        run_id: str,
        *,
        project_id: str | None = None,
        zone: str = "us-central1-a",
        machine_type: str = "e2-micro",
        network: str | None = None,
        timeout: int = 3600,
    ) -> GCPGitServer:
        """Create and start a git server VM.

        Args:
            run_id: Unique run identifier
            project_id: GCP project ID (defaults to env/gcloud config)
            zone: GCP zone for the VM
            machine_type: VM machine type (e2-micro is smallest/cheapest)
            network: VPC network name for agent connectivity (None for external IP)
            timeout: VM timeout in seconds (not enforced, for reference)

        Returns:
            GCPGitServer instance ready to accept connections
        """
        logger = logging.getLogger("cooperbench.agents.mini_swe_agent.git_server.gcp")
        logger.debug(f"Creating GCP git server for run {run_id}")

        # Resolve project ID
        resolved_project_id = project_id or cls._get_default_project()
        if not resolved_project_id:
            raise ValueError(
                "project_id required. Set via parameter, GOOGLE_CLOUD_PROJECT env var, "
                "or gcloud config set project <project-id>"
            )

        # Generate unique VM name (sanitize run_id for GCP naming rules)
        sanitized_run_id = run_id.replace("_", "-").lower()[:20]
        vm_name = f"cooperbench-git-{sanitized_run_id}-{uuid.uuid4().hex[:8]}"

        instance = cls(
            vm_name=vm_name,
            project_id=resolved_project_id,
            zone=zone,
            network=network,
            git_url="",  # Will be set after VM is created
        )

        # Create VM and get IP
        instance._create_git_server_vm(machine_type)
        instance._wait_for_ssh()

        # Create firewall rule to allow git protocol traffic
        instance._create_firewall_rule()

        # Get the git URL (internal IP if VPC, external IP otherwise)
        ip_address = instance._get_vm_ip(use_internal=network is not None)
        instance._git_url = f"git://{ip_address}:9418/repo.git"

        logger.debug(f"Git server ready at {instance._git_url}")
        return instance

    @staticmethod
    def _get_default_project() -> str | None:
        """Get default GCP project from environment or gcloud config."""
        import os

        # Try environment variable first
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
        if project:
            return project

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

    def _create_git_server_vm(self, machine_type: str):
        """Create the GCP VM with git-daemon."""
        from google.cloud import compute_v1

        self._logger.debug(f"Creating GCP VM {self._vm_name} in {self._zone}")

        client = self._get_compute_client()

        # Startup script to install git and start git-daemon
        startup_script = """#!/bin/bash
set -e

# Install git
apt-get update && apt-get install -y git

# Create bare repo
mkdir -p /git/repo.git
cd /git/repo.git
git init --bare
git config receive.denyCurrentBranch ignore
touch git-daemon-export-ok

# Start git daemon
# --enable=receive-pack allows pushing
# --export-all exports all repos
# --base-path=/git means URL /repo.git maps to /git/repo.git
# --reuseaddr allows quick restarts
git daemon \
    --reuseaddr \
    --export-all \
    --enable=receive-pack \
    --base-path=/git \
    --listen=0.0.0.0 \
    /git &

echo "Git daemon started"
"""

        # Build instance configuration
        instance = compute_v1.Instance()
        instance.name = self._vm_name
        instance.machine_type = f"zones/{self._zone}/machineTypes/{machine_type}"

        # Boot disk with Debian
        disk = compute_v1.AttachedDisk()
        disk.auto_delete = True
        disk.boot = True
        disk.type_ = "PERSISTENT"

        disk_init = compute_v1.AttachedDiskInitializeParams()
        disk_init.disk_size_gb = 10  # Small disk for git server
        disk_init.source_image = "projects/debian-cloud/global/images/family/debian-11"
        disk.initialize_params = disk_init
        instance.disks = [disk]

        # Network interface
        network_interface = compute_v1.NetworkInterface()
        if self._network:
            network_interface.network = f"projects/{self._project_id}/global/networks/{self._network}"
        else:
            network_interface.network = f"projects/{self._project_id}/global/networks/default"

        # External IP for SSH access (and git access if no VPC)
        access_config = compute_v1.AccessConfig()
        access_config.name = "External NAT"
        access_config.type_ = "ONE_TO_ONE_NAT"
        network_interface.access_configs = [access_config]
        instance.network_interfaces = [network_interface]

        # Service account with minimal scopes
        service_account = compute_v1.ServiceAccount()
        service_account.email = "default"
        service_account.scopes = [
            "https://www.googleapis.com/auth/logging.write",
        ]
        instance.service_accounts = [service_account]

        # Metadata with startup script
        metadata = compute_v1.Metadata()
        metadata.items = [
            compute_v1.Items(key="startup-script", value=startup_script),
        ]
        instance.metadata = metadata

        # Tags for firewall rules
        tags = compute_v1.Tags()
        tags.items = [f"cooperbench-git-{self._vm_name}"]
        instance.tags = tags

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
        self._logger.debug(f"VM {self._vm_name} created successfully")

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

    def _wait_for_global_operation(self, operation_name: str, timeout: int = 300):
        """Wait for a global operation to complete."""
        from google.cloud import compute_v1

        client = compute_v1.GlobalOperationsClient()
        start_time = time.time()

        while time.time() - start_time < timeout:
            operation = client.get(
                project=self._project_id,
                operation=operation_name,
            )

            if operation.status == compute_v1.Operation.Status.DONE:
                if operation.error:
                    errors = [e.message for e in operation.error.errors]
                    raise RuntimeError(f"Global operation failed: {errors}")
                return

            time.sleep(2)

        raise TimeoutError(f"Operation {operation_name} timed out after {timeout}s")

    def _wait_for_ssh(self, timeout: int = 180):
        """Wait for SSH to become available on the VM."""
        self._logger.debug(f"Waiting for SSH on {self._vm_name}...")
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
                    self._logger.debug("SSH is ready")
                    # Wait a bit more for startup script to complete
                    self._wait_for_git_daemon()
                    return
            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                self._logger.debug(f"SSH not ready yet: {e}")

            time.sleep(5)

        raise TimeoutError(f"SSH not available on {self._vm_name} after {timeout}s")

    def _wait_for_git_daemon(self, timeout: int = 120):
        """Wait for git-daemon to start on the VM."""
        self._logger.debug("Waiting for git-daemon to start...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                result = subprocess.run(
                    [
                        "gcloud",
                        "compute",
                        "ssh",
                        self._vm_name,
                        f"--zone={self._zone}",
                        f"--project={self._project_id}",
                        "--command=pgrep -f git-daemon",
                        "--quiet",
                        "--strict-host-key-checking=no",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    self._logger.debug("git-daemon is running")
                    return
            except Exception as e:
                self._logger.debug(f"git-daemon not ready yet: {e}")

            time.sleep(5)

        raise TimeoutError(f"git-daemon not started on {self._vm_name} after {timeout}s")

    def _create_firewall_rule(self):
        """Create firewall rule to allow git protocol traffic."""
        from google.cloud import compute_v1

        firewall_name = f"cooperbench-git-{self._vm_name}"
        self._logger.debug(f"Creating firewall rule {firewall_name}")

        client = compute_v1.FirewallsClient()

        firewall = compute_v1.Firewall()
        firewall.name = firewall_name

        # Use specified VPC or default network
        network_name = self._network or "default"
        firewall.network = f"projects/{self._project_id}/global/networks/{network_name}"

        # Allow TCP port 9418 (git daemon)
        allowed = compute_v1.Allowed()
        allowed.I_p_protocol = "tcp"
        allowed.ports = ["9418"]
        firewall.allowed = [allowed]

        # Source ranges: internal only for VPC, anywhere for external mode
        if self._network:
            firewall.source_ranges = ["10.0.0.0/8"]
        else:
            firewall.source_ranges = ["0.0.0.0/0"]

        # Target only this VM
        firewall.target_tags = [f"cooperbench-git-{self._vm_name}"]

        request = compute_v1.InsertFirewallRequest()
        request.project = self._project_id
        request.firewall_resource = firewall

        operation = client.insert(request=request)

        # Mark firewall as created immediately
        self._firewall_created = True

        # Wait for operation
        self._wait_for_global_operation(operation.name)
        self._logger.debug(f"Firewall rule {firewall_name} created")

    def _get_vm_ip(self, use_internal: bool = False) -> str:
        """Get the IP address of the VM.

        Args:
            use_internal: If True, return internal IP (for VPC). Otherwise external.

        Returns:
            IP address string
        """
        from google.cloud import compute_v1

        client = self._get_compute_client()

        request = compute_v1.GetInstanceRequest()
        request.project = self._project_id
        request.zone = self._zone
        request.instance = self._vm_name

        instance = client.get(request=request)

        if not instance.network_interfaces:
            raise RuntimeError(f"VM {self._vm_name} has no network interfaces")

        network_interface = instance.network_interfaces[0]

        if use_internal:
            # Return internal IP
            return network_interface.network_i_p
        else:
            # Return external IP
            if not network_interface.access_configs:
                raise RuntimeError(f"VM {self._vm_name} has no external IP")
            return network_interface.access_configs[0].nat_i_p

    @property
    def url(self) -> str:
        """Git URL for agents to use as remote.

        Returns:
            Git URL for the repository (git://IP:9418/repo.git)
        """
        return self._git_url

    @property
    def network_name(self) -> str | None:
        """VPC network name for agents to join.

        Returns:
            Network name or None if using external IP
        """
        return self._network

    def cleanup(self) -> None:
        """Delete the VM and firewall rule."""
        # Delete firewall rule first
        if self._firewall_created:
            self._delete_firewall_rule()

        # Delete VM
        if self._vm_created:
            self._delete_vm()

    def _delete_firewall_rule(self):
        """Delete the firewall rule."""
        from google.cloud import compute_v1

        firewall_name = f"cooperbench-git-{self._vm_name}"
        self._logger.debug(f"Deleting firewall rule {firewall_name}")

        try:
            client = compute_v1.FirewallsClient()

            request = compute_v1.DeleteFirewallRequest()
            request.project = self._project_id
            request.firewall = firewall_name

            operation = client.delete(request=request)

            try:
                self._wait_for_global_operation(operation.name, timeout=120)
            except TimeoutError:
                self._logger.warning(f"Firewall deletion timed out: {firewall_name}")

            self._firewall_created = False
            self._logger.debug(f"Firewall rule {firewall_name} deleted")
        except Exception as e:
            self._logger.warning(f"Failed to delete firewall rule {firewall_name}: {e}")

    def _delete_vm(self):
        """Delete the VM."""
        self._logger.debug(f"Deleting VM {self._vm_name}")

        try:
            from google.cloud import compute_v1

            client = self._get_compute_client()

            request = compute_v1.DeleteInstanceRequest()
            request.project = self._project_id
            request.zone = self._zone
            request.instance = self._vm_name

            operation = client.delete(request=request)

            try:
                self._wait_for_operation(operation.name, timeout=120)
            except TimeoutError:
                self._logger.warning(f"VM deletion timed out: {self._vm_name}")

            self._vm_created = False
            self._logger.debug(f"VM {self._vm_name} deleted")
        except Exception as e:
            self._logger.warning(f"Failed to delete VM {self._vm_name}: {e}")

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        try:
            self.cleanup()
        except Exception:
            pass
