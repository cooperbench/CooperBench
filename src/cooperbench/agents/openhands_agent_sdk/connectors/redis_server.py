"""Modal-based Redis server for inter-agent messaging.

Creates a Modal sandbox running Redis that agents can connect to
for messaging and coordination.
"""

from __future__ import annotations

import time

import modal


class ModalRedisServer:
    """Redis server running in a Modal sandbox.
    
    Provides a Redis instance that agents can connect to for messaging.
    Uses the same pattern as ModalGitServer.
    
    Example:
        server = ModalRedisServer.create(
            app=modal_app,
            run_id="abc123",
            agents=["agent1", "agent2"],
        )
        print(f"Redis URL: {server.url}")
        # Agents can now connect: redis.from_url(server.url)
        
        server.cleanup()
    """
    
    def __init__(self, sandbox: modal.Sandbox, redis_url: str, agents: list[str]):
        """Initialize with an existing sandbox.
        
        Use ModalRedisServer.create() to create a new server.
        """
        self._sandbox = sandbox
        self._redis_url = redis_url
        self._agents = agents
    
    @classmethod
    def create(
        cls,
        app: modal.App,
        run_id: str,
        agents: list[str],
        timeout: int = 3600,
    ) -> ModalRedisServer:
        """Create and start a Redis server sandbox.
        
        Args:
            app: Modal app to create sandbox in
            run_id: Unique run identifier (for logging)
            agents: List of agent IDs for this collaboration
            timeout: Sandbox timeout in seconds
            
        Returns:
            ModalRedisServer instance ready to accept connections
        """
        
        # Image with Redis
        image = modal.Image.debian_slim().run_commands(
            "apt-get update && apt-get install -y redis-server",
        )
        
        # Create sandbox with Redis port exposed (unencrypted for TCP)
        sandbox = modal.Sandbox.create(
            image=image,
            app=app,
            timeout=timeout,
            unencrypted_ports=[6379],  # Redis TCP port
        )
        
        
        # Start Redis server
        # --bind 0.0.0.0 to accept connections from tunnel
        # --protected-mode no since we're in isolated sandbox
        proc = sandbox.exec(
            "bash", "-c",
            """
            redis-server \
                --bind 0.0.0.0 \
                --protected-mode no \
                --daemonize yes \
                --logfile /var/log/redis.log
            
            # Wait for Redis to start
            sleep 1
            redis-cli ping
            """
        )
        proc.wait()
        
        if proc.returncode != 0:
            stderr = proc.stderr.read()
            raise RuntimeError(f"Failed to start Redis: {stderr}")
        
        # Give Redis a moment to fully initialize
        time.sleep(1)
        
        # Get tunnel URL for port 6379
        tunnels = sandbox.tunnels()
        
        if tunnels and 6379 in tunnels:
            tunnel = tunnels[6379]
            # Use unencrypted endpoint for Redis protocol
            redis_url = f"redis://{tunnel.unencrypted_host}:{tunnel.unencrypted_port}"
        else:
            raise RuntimeError(f"Failed to get tunnel for port 6379. Available: {tunnels}")
        
        # Verify Redis is accessible
        cls._wait_for_redis(redis_url)
        
        return cls(sandbox=sandbox, redis_url=redis_url, agents=agents)
    
    @staticmethod
    def _wait_for_redis(redis_url: str, timeout: int = 30) -> None:
        """Wait for Redis to be accessible."""
        import redis
        
        start = time.time()
        last_error = None
        
        while time.time() - start < timeout:
            try:
                client = redis.from_url(redis_url, socket_timeout=5)
                if client.ping():
                    client.close()
                    return
            except Exception as e:
                last_error = e
            time.sleep(1)
        
        raise TimeoutError(
            f"Redis did not become ready within {timeout}s. Last error: {last_error}"
        )
    
    @property
    def url(self) -> str:
        """Redis URL for agents to connect to."""
        return self._redis_url
    
    @property
    def agents(self) -> list[str]:
        """List of agent IDs in this collaboration."""
        return self._agents
    
    def cleanup(self) -> None:
        """Terminate the Redis server sandbox."""
        if self._sandbox:
            try:
                self._sandbox.terminate()
            except Exception:
                pass  # Ignore cleanup errors


def create_redis_server(
    app: modal.App,
    run_id: str,
    agents: list[str],
    timeout: int = 3600,
) -> ModalRedisServer:
    """Create a Redis server (convenience function).
    
    Args:
        app: Modal app to create sandbox in
        run_id: Unique run identifier
        agents: List of agent IDs
        timeout: Sandbox timeout in seconds
        
    Returns:
        ModalRedisServer instance
    """
    return ModalRedisServer.create(
        app=app,
        run_id=run_id,
        agents=agents,
        timeout=timeout,
    )
