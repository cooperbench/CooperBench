"""Docker-based git server for code collaboration.

Architecture (mirrors how Redis is used elsewhere in CooperBench): a single
long-lived ``cooperbench-git`` container runs ``git daemon`` and serves
multiple isolated per-run repositories under ``/git/<run_id>/repo.git``.  All
agent containers join the same ``cooperbench`` bridge network so they can
resolve the server by container name.

The infra (image, network, container) is auto-created on first use and reused
thereafter.  No CLI setup step.
"""

from __future__ import annotations

import io
import logging
import time

import docker

# Singleton infra (one set per host, reused across all coop runs)
_IMAGE_TAG = "cooperbench-git-server:local"
_CONTAINER_NAME = "cooperbench-git"
_NETWORK_NAME = "cooperbench"
_VOLUME_NAME = "cooperbench-git-data"
_PORT = 9418

_DOCKERFILE = b"""FROM debian:bookworm-slim
RUN apt-get update -qq \\
 && apt-get install -y -qq git \\
 && rm -rf /var/lib/apt/lists/*
RUN mkdir /git
ENTRYPOINT ["git", "daemon", \\
            "--reuseaddr", \\
            "--export-all", \\
            "--enable=receive-pack", \\
            "--base-path=/git", \\
            "--listen=0.0.0.0", \\
            "/git"]
"""


def _wait_for_port(container, port: int, timeout: int = 30) -> None:
    """Block until the daemon binds the given port inside the container.

    Uses bash's built-in ``/dev/tcp`` (no extra packages required) so it works
    against the stripped-down debian-slim base.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        probe = container.exec_run(["bash", "-c", f"exec 3<>/dev/tcp/127.0.0.1/{port}"])
        if probe.exit_code == 0:
            return
        time.sleep(0.5)
    raise RuntimeError(f"git daemon did not bind :{port} within {timeout}s")


def _ensure_shared_infra(client: docker.DockerClient, logger: logging.Logger):
    """Idempotently bring up image, network, and shared git server container."""
    # Image
    try:
        client.images.get(_IMAGE_TAG)
    except docker.errors.ImageNotFound:
        logger.info(f"Building {_IMAGE_TAG} (one-time, ~30s)...")
        client.images.build(fileobj=io.BytesIO(_DOCKERFILE), tag=_IMAGE_TAG, rm=True)
        logger.info(f"Built {_IMAGE_TAG}")

    # Network
    try:
        client.networks.get(_NETWORK_NAME)
    except docker.errors.NotFound:
        logger.debug(f"Creating shared network {_NETWORK_NAME}")
        try:
            client.networks.create(_NETWORK_NAME, driver="bridge")
        except docker.errors.APIError:
            # race: another concurrent run created it; re-fetch
            client.networks.get(_NETWORK_NAME)

    # Volume (so /git survives container restarts; also lets us inspect history)
    try:
        client.volumes.get(_VOLUME_NAME)
    except docker.errors.NotFound:
        client.volumes.create(name=_VOLUME_NAME)

    # Container
    try:
        container = client.containers.get(_CONTAINER_NAME)
    except docker.errors.NotFound:
        logger.info(f"Starting shared git server container {_CONTAINER_NAME}")
        try:
            container = client.containers.run(
                _IMAGE_TAG,
                name=_CONTAINER_NAME,
                detach=True,
                network=_NETWORK_NAME,
                volumes={_VOLUME_NAME: {"bind": "/git", "mode": "rw"}},
                restart_policy={"Name": "unless-stopped"},
            )
        except docker.errors.APIError:
            # race: another concurrent run created it; re-fetch
            container = client.containers.get(_CONTAINER_NAME)

    container.reload()
    if container.status != "running":
        container.start()
        container.reload()

    _wait_for_port(container, _PORT, timeout=30)
    return container


class DockerGitServer:
    """Per-run handle on the shared git server.

    The shared container is a singleton; what's per-run is just a directory
    under ``/git/<run_id>/repo.git`` that this class creates and tears down.
    """

    def __init__(self, *, run_id: str, hostname: str, port: int, network_name: str):
        self._run_id = run_id
        self._hostname = hostname
        self._port = port
        self._network_name = network_name
        self._logger = logging.getLogger("cooperbench.agents.mini_swe_agent_v2.git_server.docker")

    @classmethod
    def create(cls, run_id: str, timeout: int = 3600) -> DockerGitServer:
        """Ensure shared infra is up, then init a per-run bare repo on it.

        Args:
            run_id: Unique run identifier — becomes the path prefix under /git
            timeout: Unused; kept for protocol compatibility with other backends

        Returns:
            DockerGitServer pointing at git://<container>:9418/<run_id>/repo.git
        """
        del timeout
        logger = logging.getLogger("cooperbench.agents.mini_swe_agent_v2.git_server.docker")
        client = docker.from_env()

        container = _ensure_shared_infra(client, logger)

        # Per-run repo init — fast (~50ms) inside the already-running container
        init_cmd = (
            f"set -e && "
            f"mkdir -p /git/{run_id}/repo.git && "
            f"cd /git/{run_id}/repo.git && "
            f"git init --bare && "
            f"git config receive.denyCurrentBranch ignore && "
            f"touch git-daemon-export-ok"
        )
        result = container.exec_run(["bash", "-c", init_cmd])
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to init repo /git/{run_id}/repo.git: {result.output.decode('utf-8', errors='replace')}"
            )

        logger.debug(f"Per-run repo ready at git://{_CONTAINER_NAME}:{_PORT}/{run_id}/repo.git")

        return cls(
            run_id=run_id,
            hostname=_CONTAINER_NAME,
            port=_PORT,
            network_name=_NETWORK_NAME,
        )

    @property
    def url(self) -> str:
        """Git URL for agents to use as remote."""
        return f"git://{self._hostname}:{self._port}/{self._run_id}/repo.git"

    @property
    def network_name(self) -> str:
        """Docker network name for agent containers to join."""
        return self._network_name

    def cleanup(self) -> None:
        """Remove this run's repo dir.  Leave the shared container/network/image alone."""
        try:
            client = docker.from_env()
            container = client.containers.get(_CONTAINER_NAME)
            container.exec_run(["rm", "-rf", f"/git/{self._run_id}"])
        except Exception:
            # Best-effort: we don't want cleanup failure to mask the real run result.
            pass
