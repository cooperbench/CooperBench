"""Docker-based git server for code collaboration."""

from __future__ import annotations

import logging
import time

import docker


class DockerGitServer:
    """Shared git server container for code collaboration using Docker.

    Creates a Docker container running git-daemon that agents can push/pull to.
    """

    def __init__(self, container, hostname: str, port: int, network_name: str):
        """Initialize with an existing container.

        Use DockerGitServer.create() to create a new server.
        """
        self._container = container
        self._hostname = hostname
        self._port = port
        self._network_name = network_name
        self._logger = logging.getLogger("cooperbench.agents.mini_swe_agent_v2.git_server.docker")

    @classmethod
    def create(
        cls,
        run_id: str,
        timeout: int = 3600,
    ) -> DockerGitServer:
        """Create and start a git server container.

        Args:
            run_id: Unique run identifier (for container naming)
            timeout: Container timeout in seconds (not enforced, for compatibility)

        Returns:
            DockerGitServer instance ready to accept connections
        """
        logger = logging.getLogger("cooperbench.agents.mini_swe_agent_v2.git_server.docker")
        logger.debug(f"Creating docker git server for run {run_id}")

        client = docker.from_env()

        # Use a simple Debian-based image with git
        image = "debian:bookworm-slim"

        # Pull image if not present
        try:
            client.images.get(image)
        except docker.errors.ImageNotFound:
            logger.debug(f"Pulling image {image}")
            client.images.pull(image)

        # Create or get shared network for git server and agents
        network_name = f"cooperbench-git-{run_id}"
        try:
            client.networks.get(network_name)
        except docker.errors.NotFound:
            client.networks.create(network_name, driver="bridge")

        # Container name based on run_id
        container_name = f"cooperbench-git-{run_id}"

        # Remove existing container if it exists
        try:
            old_container = client.containers.get(container_name)
            old_container.remove(force=True)
        except docker.errors.NotFound:
            pass

        # Create and start container with initialization script
        # The script initializes the repo, then starts git daemon in foreground to keep container alive
        init_script = """#!/bin/bash
set -e
apt-get update -qq
apt-get install -y -qq git > /dev/null 2>&1
mkdir -p /git/repo.git
cd /git/repo.git
git init --bare
git config receive.denyCurrentBranch ignore
touch git-daemon-export-ok
exec git daemon --reuseaddr --export-all --enable=receive-pack --base-path=/git --listen=0.0.0.0 /git
"""

        container = client.containers.run(
            image=image,
            command=["bash", "-c", init_script],
            name=container_name,
            detach=True,
            network=network_name,
            ports={"9418/tcp": None},  # Auto-assign port for host access
            remove=False,
        )

        # Wait for container to start and git daemon to initialize
        time.sleep(3)

        # Verify container is running
        container.reload()
        if container.status != "running":
            logs = container.logs().decode("utf-8", errors="replace")
            container.remove(force=True)
            raise RuntimeError(f"Git server container failed to start. Logs: {logs}")

        # Reload container to get port mapping
        container.reload()

        # Get the host port
        port_bindings = container.attrs.get("NetworkSettings", {}).get("Ports", {})
        if "9418/tcp" not in port_bindings or not port_bindings["9418/tcp"]:
            container.stop()
            container.remove(force=True)
            raise RuntimeError("Failed to get port mapping for git daemon")

        # Get container's IP on the network for inter-container communication
        container.reload()
        network_settings = container.attrs.get("NetworkSettings", {})
        networks = network_settings.get("Networks", {})
        if network_name in networks:
            container_ip = networks[network_name].get("IPAddress")
            if container_ip:
                hostname = container_ip
            else:
                # Fallback to container name (DNS resolution on same network)
                hostname = container_name
        else:
            # Fallback to container name
            hostname = container_name

        logger.debug(f"Git server ready at git://{hostname}:9418 (network: {network_name})")

        return cls(container=container, hostname=hostname, port=9418, network_name=network_name)

    @property
    def url(self) -> str:
        """Git URL for agents to use as remote.

        Returns:
            Git URL for the repository (git://hostname:port/repo.git)
        """
        return f"git://{self._hostname}:{self._port}/repo.git"

    @property
    def network_name(self) -> str:
        """Docker network name for agent containers to join."""
        return self._network_name

    def cleanup(self) -> None:
        """Stop and remove the git server container and network."""
        if self._container:
            try:
                self._container.stop(timeout=5)
            except Exception:
                pass
            try:
                self._container.remove(force=True)
            except Exception:
                pass
            self._container = None

        # Clean up network
        if hasattr(self, "_network_name") and self._network_name:
            try:
                client = docker.from_env()
                try:
                    network = client.networks.get(self._network_name)
                    network.remove()
                except docker.errors.NotFound:
                    pass
            except Exception:
                pass
