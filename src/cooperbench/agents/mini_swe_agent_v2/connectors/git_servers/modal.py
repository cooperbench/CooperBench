"""Modal-based git server for code collaboration."""

from __future__ import annotations

import logging
import time

import modal


class ModalGitServer:
    """Shared git server sandbox for code collaboration using Modal.

    Creates a Modal sandbox running git-daemon that agents can push/pull to.
    """

    def __init__(self, sandbox: modal.Sandbox, hostname: str):
        """Initialize with an existing sandbox.

        Use ModalGitServer.create() to create a new server.
        """
        self._sandbox = sandbox
        self._hostname = hostname
        self._logger = logging.getLogger("cooperbench.agents.mini_swe_agent_v2.git_server.modal")

    @classmethod
    def create(
        cls,
        app: modal.App,
        run_id: str,
        timeout: int = 3600,
    ) -> ModalGitServer:
        """Create and start a git server sandbox.

        Args:
            app: Modal app to create sandbox in
            run_id: Unique run identifier (for logging)
            timeout: Sandbox timeout in seconds

        Returns:
            ModalGitServer instance ready to accept connections
        """
        logger = logging.getLogger("cooperbench.agents.mini_swe_agent_v2.git_server.modal")
        logger.debug(f"Creating git server for run {run_id}")

        # Image with git
        image = modal.Image.debian_slim().run_commands(
            "apt-get update && apt-get install -y git",
        )

        # Create sandbox with port 9418 exposed for git daemon (unencrypted TCP)
        sandbox = modal.Sandbox.create(
            image=image,
            app=app,
            timeout=timeout,
            unencrypted_ports=[9418],  # Expose git daemon port via TCP tunnel
        )

        # Initialize bare repo in /git/repo.git
        proc = sandbox.exec(
            "bash",
            "-c",
            """
            set -e
            mkdir -p /git/repo.git
            cd /git/repo.git
            git init --bare
            git config receive.denyCurrentBranch ignore
            touch git-daemon-export-ok
        """,
        )
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to init git repo: {proc.stderr.read()}")

        # Start git daemon in background
        # --enable=receive-pack allows pushing
        # --export-all exports all repos
        # --base-path=/git means URL /repo.git maps to /git/repo.git
        # --listen=0.0.0.0 to accept connections from tunnel
        proc = sandbox.exec(
            "bash",
            "-c",
            """
            git daemon \
                --reuseaddr \
                --export-all \
                --enable=receive-pack \
                --base-path=/git \
                --listen=0.0.0.0 \
                /git &

            # Wait for daemon to start
            sleep 1
            echo "Git daemon started"
        """,
        )
        proc.stdout.read()
        proc.wait()

        # Give daemon time to fully initialize
        time.sleep(1)

        # Get the tunnel URL for port 9418
        tunnels = sandbox.tunnels()

        if tunnels and 9418 in tunnels:
            tunnel = tunnels[9418]
            # Use the unencrypted endpoint for git protocol
            # Tunnel has: host, port (encrypted), unencrypted_host, unencrypted_port
            hostname = f"{tunnel.unencrypted_host}:{tunnel.unencrypted_port}"
            logger.debug(f"Using unencrypted tunnel: {hostname}")
        else:
            raise RuntimeError(f"Failed to get tunnel for port 9418. Available tunnels: {tunnels}")

        logger.debug(f"Git server ready at git://{hostname}")

        return cls(sandbox=sandbox, hostname=hostname)

    @property
    def url(self) -> str:
        """Git URL for agents to use as remote.

        Returns:
            Git URL for the repository (git://hostname/repo.git)
        """
        return f"git://{self._hostname}/repo.git"

    def cleanup(self) -> None:
        """Terminate the git server sandbox."""
        if self._sandbox:
            try:
                self._sandbox.terminate()
            except Exception:
                pass


# Backwards compatibility alias
GitServer = ModalGitServer
