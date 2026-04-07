"""Git-based code sharing between agents.

Enables agents in separate containers to share code via git push/pull.
Uses a shared git server sandbox that agents connect to as a remote.

Architecture:
    +---------------------------------------------------------+
    |                    Git Server Sandbox                    |
    |        git daemon --enable=receive-pack (bare repo)      |
    +---------------------------------------------------------+
                               ^
              +----------------+----------------+
              |                |                |
         git push         git fetch        git push
         git pull                          git pull
              |                |                |
    +---------v----+                 +---------v----+
    |   Agent A    |                 |   Agent B    |
    |   sandbox    |                 |   sandbox    |
    +--------------+                 +--------------+

Example:
    # Create shared git server (once per task)
    from cooperbench.agents.mini_swe_agent_v2.connectors.git_servers import get_git_server

    GitServerClass = get_git_server("docker")  # or "modal"
    git_server = GitServerClass.create(run_id="abc123")

    # Create connector for each agent
    git = GitConnector(
        agent_id="agent1",
        agents=["agent1", "agent2"],
        server_url=git_server.url
    )

    # Configure agent's sandbox
    git.setup(env)

    # Agent can now use git normally:
    #   git push team agent1
    #   git fetch team
    #   git merge team/agent2
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cooperbench.agents.mini_swe_agent_v2.environments.docker import DockerEnvironment


class GitConnector:
    """Configures an agent's sandbox for git collaboration.

    After setup(), the agent can use standard git commands:
    - git push team <branch>  - share changes
    - git fetch team          - get other agents' branches
    - git merge team/<agent>  - merge another agent's work
    - git branch -r           - list remote branches
    """

    # Remote name used in agent's git config
    REMOTE_NAME = "team"

    def __init__(
        self,
        agent_id: str,
        agents: list[str],
        server_url: str,
    ):
        """Initialize git connector.

        Args:
            agent_id: This agent's unique identifier (e.g., "agent1")
            agents: List of all agent IDs in the collaboration
            server_url: Git server URL from GitServer.url
        """
        self.agent_id = agent_id
        self.agents = agents
        self.server_url = server_url
        self._logger = logging.getLogger("cooperbench.agents.mini_swe_agent_v2.git_connector")
        self._initialized = False

    def _exec(self, env: DockerEnvironment, command: str) -> dict:
        """Execute a command in the environment (v2 uses dict-based actions)."""
        return env.execute({"command": command})

    def setup(self, env: DockerEnvironment) -> None:
        """Configure git remote in the agent's sandbox.

        This sets up the 'team' remote pointing to the shared git server,
        creates an agent-specific branch, and pushes the initial state.

        Args:
            env: The agent's Docker environment

        Raises:
            RuntimeError: If git configuration fails
        """
        self._logger.debug(f"Setting up git for {self.agent_id}")

        # Configure git user (needed for commits)
        self._exec(env, 'git config user.email "agent@cooperbench.local"')
        self._exec(env, f'git config user.name "{self.agent_id}"')

        # Add shared remote
        result = self._exec(env, f"git remote add {self.REMOTE_NAME} {self.server_url}")
        if result.get("returncode", 0) != 0:
            # Remote might already exist
            self._exec(env, f"git remote set-url {self.REMOTE_NAME} {self.server_url}")

        # Create agent's branch
        self._exec(env, f"git checkout -b {self.agent_id}")

        # Push initial state (first agent initializes the server)
        # Use --force in case branch exists from a previous run
        result = self._exec(env, f"git push -u {self.REMOTE_NAME} {self.agent_id} --force")
        if result.get("returncode", 0) != 0:
            self._logger.warning(f"Initial push failed: {result.get('output', '')}")

        # Also push main/master as base reference
        self._exec(env, f"git push {self.REMOTE_NAME} HEAD:refs/heads/main --force 2>/dev/null || true")

        self._initialized = True
        self._logger.debug(f"Git setup complete for {self.agent_id}")

    @property
    def is_initialized(self) -> bool:
        """Whether setup() has been called."""
        return self._initialized
