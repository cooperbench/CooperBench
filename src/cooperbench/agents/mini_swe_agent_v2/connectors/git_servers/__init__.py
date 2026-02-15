"""Git servers for inter-agent code collaboration.

Provides pluggable backends for hosting shared git repositories:
- Modal: Cloud-based sandboxes (default)
- Docker: Local containers
- GCP: Google Cloud Platform VMs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cooperbench.agents.mini_swe_agent_v2.connectors.git_servers.base import GitServer
from cooperbench.agents.mini_swe_agent_v2.connectors.git_servers.docker import DockerGitServer
from cooperbench.agents.mini_swe_agent_v2.connectors.git_servers.gcp import GCPGitServer
from cooperbench.agents.mini_swe_agent_v2.connectors.git_servers.modal import ModalGitServer

if TYPE_CHECKING:
    import modal

__all__ = ["GitServer", "ModalGitServer", "DockerGitServer", "GCPGitServer", "create_git_server"]


def create_git_server(
    backend: str,
    run_id: str,
    *,
    app: modal.App | None = None,
    timeout: int = 3600,
    # GCP-specific options
    project_id: str | None = None,
    zone: str = "us-central1-a",
    machine_type: str = "e2-micro",
    network: str | None = None,
) -> ModalGitServer | DockerGitServer | GCPGitServer:
    """Create a git server for the specified backend.

    Args:
        backend: Backend name ("modal", "docker", or "gcp")
        run_id: Unique run identifier
        app: Modal app (required for modal backend)
        timeout: Server timeout in seconds
        project_id: GCP project ID (gcp backend only)
        zone: GCP zone (gcp backend only, default: us-central1-a)
        machine_type: GCP machine type (gcp backend only, default: e2-micro)
        network: VPC network name (gcp backend only, for agent connectivity)

    Returns:
        Git server instance ready to accept connections

    Example:
        # Docker backend
        server = create_git_server("docker", run_id="my-run")

        # Modal backend
        app = modal.App.lookup("cooperbench", create_if_missing=True)
        server = create_git_server("modal", run_id="my-run", app=app)

        # GCP backend
        server = create_git_server(
            "gcp",
            run_id="my-run",
            project_id="my-project",
            network="cooperbench-vpc",
        )

        # Use the server
        print(server.url)
        # ... agents push/pull ...
        server.cleanup()
    """
    if backend == "docker":
        return DockerGitServer.create(run_id=run_id, timeout=timeout)
    elif backend == "modal":
        if app is None:
            raise ValueError("Modal backend requires 'app' parameter")
        return ModalGitServer.create(app=app, run_id=run_id, timeout=timeout)
    elif backend == "gcp":
        return GCPGitServer.create(
            run_id=run_id,
            project_id=project_id,
            zone=zone,
            machine_type=machine_type,
            network=network,
            timeout=timeout,
        )
    else:
        available = "docker, modal, gcp"
        raise ValueError(f"Unknown git server backend: '{backend}'. Available: {available}")
