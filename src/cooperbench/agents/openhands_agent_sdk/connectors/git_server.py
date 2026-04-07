"""Modal-based Git server for inter-agent code collaboration.

Creates a Modal sandbox running git-daemon that agents can push/pull to.
Uses the same pattern as ModalRedisServer.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    Git Server Sandbox                    │
    │        git daemon --enable=receive-pack (bare repo)      │
    └─────────────────────────────────────────────────────────┘
                               ▲
              ┌────────────────┼────────────────┐
              │                │                │
         git push         git fetch        git push
         git pull                          git pull
              │                │                │
    ┌─────────▼────┐                 ┌─────────▼────┐
    │   Agent A    │                 │   Agent B    │
    │   sandbox    │                 │   sandbox    │
    └──────────────┘                 └──────────────┘
"""

from __future__ import annotations

import time

import modal


class ModalGitServer:
    """Git server running in a Modal sandbox.
    
    Provides a bare git repository that agents can push to and pull from.
    Uses git-daemon with receive-pack enabled for push access.
    
    Example:
        server = ModalGitServer.create(
            app=modal_app,
            run_id="abc123",
            agents=["agent1", "agent2"],
        )
        print(f"Git URL: {server.url}")
        # Agents can now: git remote add team {server.url}
        
        server.cleanup()
    """
    
    def __init__(self, sandbox: modal.Sandbox, git_url: str, agents: list[str]):
        """Initialize with an existing sandbox.
        
        Use ModalGitServer.create() to create a new server.
        """
        self._sandbox = sandbox
        self._git_url = git_url
        self._agents = agents
    
    @classmethod
    def create(
        cls,
        app: modal.App,
        run_id: str,
        agents: list[str],
        timeout: int = 3600,
    ) -> ModalGitServer:
        """Create and start a git server sandbox.
        
        Args:
            app: Modal app to create sandbox in
            run_id: Unique run identifier (for logging)
            agents: List of agent IDs for this collaboration
            timeout: Sandbox timeout in seconds
            
        Returns:
            ModalGitServer instance ready to accept connections
        """
        
        # Image with git
        image = modal.Image.debian_slim().run_commands(
            "apt-get update && apt-get install -y git",
        )
        
        # Create sandbox with git daemon port exposed (unencrypted for TCP)
        sandbox = modal.Sandbox.create(
            image=image,
            app=app,
            timeout=timeout,
            unencrypted_ports=[9418],  # Git daemon TCP port
        )
        
        # Initialize bare repo and start git daemon
        proc = sandbox.exec(
            "bash", "-c",
            """
            set -e
            
            # Create bare repository
            mkdir -p /git/repo.git
            cd /git/repo.git
            git init --bare
            git config receive.denyCurrentBranch ignore
            touch git-daemon-export-ok
            
            # Start git daemon in background
            # --enable=receive-pack allows pushing
            # --export-all exports all repos  
            # --base-path=/git means URL /repo.git maps to /git/repo.git
            # --listen=0.0.0.0 to accept connections from tunnel
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
            """
        )
        proc.wait()
        
        if proc.returncode != 0:
            stderr = proc.stderr.read()
            raise RuntimeError(f"Failed to start git daemon: {stderr}")
        
        # Give daemon time to fully initialize
        time.sleep(1)
        
        # Get tunnel URL for port 9418
        tunnels = sandbox.tunnels()
        
        if tunnels and 9418 in tunnels:
            tunnel = tunnels[9418]
            # Use unencrypted endpoint for git protocol
            hostname = f"{tunnel.unencrypted_host}:{tunnel.unencrypted_port}"
            git_url = f"git://{hostname}/repo.git"
        else:
            raise RuntimeError(f"Failed to get tunnel for port 9418. Available: {tunnels}")
        
        return cls(sandbox=sandbox, git_url=git_url, agents=agents)
    
    @property
    def url(self) -> str:
        """Git URL for agents to use as remote."""
        return self._git_url
    
    @property
    def agents(self) -> list[str]:
        """List of agent IDs in this collaboration."""
        return self._agents
    
    def cleanup(self) -> None:
        """Terminate the git server sandbox."""
        if self._sandbox:
            try:
                self._sandbox.terminate()
            except Exception:
                pass  # Ignore cleanup errors


def create_git_server(
    app: modal.App,
    run_id: str,
    agents: list[str],
    timeout: int = 3600,
) -> ModalGitServer:
    """Create a git server (convenience function).
    
    Args:
        app: Modal app to create sandbox in
        run_id: Unique run identifier
        agents: List of agent IDs
        timeout: Sandbox timeout in seconds
        
    Returns:
        ModalGitServer instance
    """
    return ModalGitServer.create(
        app=app,
        run_id=run_id,
        agents=agents,
        timeout=timeout,
    )
