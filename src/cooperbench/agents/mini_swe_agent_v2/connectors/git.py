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

import base64
import logging
from pathlib import Path
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
    REMOTE_NAME = "origin"

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
        self._base_sha = ""

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

        # Solo runs have no shared server, but the submission path should not fork on that:
        # a second mechanism for solo is exactly what let solo break silently while coop was
        # being changed. A bare repo inside the agent's own sandbox gives solo the identical
        # flow -- push, open a PR, graded from the PR -- with no extra infrastructure.
        self._detach_upstream(env)

        server = self.server_url
        if not server:
            server = "/tmp/team.git"
            self._exec(env, f"git init -q --bare {server} 2>/dev/null || true")

        # Add shared remote
        result = self._exec(env, f"git remote add {self.REMOTE_NAME} {server}")
        if result.get("returncode", 0) != 0:
            # Remote might already exist
            self._exec(env, f"git remote set-url {self.REMOTE_NAME} {server}")

        # Create agent's branch
        self._exec(env, f"git checkout -b {self.agent_id}")

        # Push initial state (first agent initializes the server)
        # Use --force in case branch exists from a previous run
        result = self._exec(env, f"git push -u {self.REMOTE_NAME} {self.agent_id} --force")
        if result.get("returncode", 0) != 0:
            self._logger.warning(f"Initial push failed: {result.get('output', '')}")

        # Also push main/master as base reference
        self._exec(env, f"git push {self.REMOTE_NAME} HEAD:refs/heads/main --force 2>/dev/null || true")

        self._install_gh_shim(env)

        # Pin the base commit. Submissions are diffed against it, and `team/main` is a
        # movable ref on a daemon with no access control -- one `git push team HEAD:main`,
        # from either agent, would silently re-baseline both submissions.
        self._base_sha = self._exec(env, "git rev-parse HEAD").get("output", "").strip()

        self._initialized = True
        self._logger.debug(f"Git setup complete for {self.agent_id}")

    @property
    def is_initialized(self) -> bool:
        """Whether setup() has been called."""
        return self._initialized

    def _install_gh_shim(self, env: DockerEnvironment) -> None:
        """Install a minimal `gh` implementing PRs over the team remote.

        Agents know `gh pr create` and `gh pr diff`; they do not know any command we invent,
        so the shim keeps the real spelling. The agent id is substituted in at install time
        rather than read from the environment, because each bash call the agent makes is a
        fresh shell and an exported variable would not survive between them.
        """
        shim = (Path(__file__).parent / "gh_shim.sh").read_text()
        shim = shim.replace(
            'AGENT="${COOPERBENCH_AGENT_ID:-}"',
            f'AGENT="${{COOPERBENCH_AGENT_ID:-{self.agent_id}}}"',
        )
        encoded = base64.b64encode(shim.encode()).decode()
        result = self._exec(
            env,
            f"echo {encoded} | base64 -d > /usr/local/bin/gh && chmod +x /usr/local/bin/gh",
        )
        if result.get("returncode", 0) != 0 or not self._exec(env, "command -v gh >/dev/null && echo ok").get(
            "output", ""
        ).strip().endswith("ok"):
            # Submission goes through `gh pr create`. Without the shim the agent cannot
            # submit anything at all, and it would only surface as an empty patch hours
            # later, indistinguishable from an agent that simply failed the task.
            raise RuntimeError(f"gh shim not installed for {self.agent_id}: {result.get('output', '')}")

    def submitted_patch(self, env: DockerEnvironment) -> str:
        """The diff the agent proposed in its PR, or "" if it never opened one.

        This is the graded artifact. An agent that opened no PR submits nothing, which is the
        correct outcome rather than something to paper over -- the previous mechanism let an
        agent submit from a local file it never shared, so its colleague could not see what
        was coming.
        """
        self._exec(env, f"git fetch -q --tags {self.REMOTE_NAME} 2>/dev/null || true")
        self._exec(env, f"git fetch -q {self.REMOTE_NAME} 2>/dev/null || true")
        # The tag records that a PR was opened; the CONTENT is the branch tip, so commits
        # pushed after opening are included -- the same way a PR on a forge updates when you
        # push. Grading the tag instead would silently freeze the submission at whatever the
        # agent had written the moment it opened the PR.
        # Checked against the REMOTE: the agent creates the tag locally first, so a local
        # check would call a PR "opened" even when the push that publishes it failed.
        opened = (
            self._exec(env, f"git ls-remote --tags {self.REMOTE_NAME} refs/tags/pr/{self.agent_id}")
            .get("output", "")
            .strip()
        )
        if not opened:
            # Never opening a PR is an agent failure, not something to paper over -- it scores
            # zero, correctly. Log it so that outcome is attributable afterwards, instead of
            # being indistinguishable from an agent whose code simply failed the tests.
            self._logger.info(f"NO PR OPENED by {self.agent_id}: submitting nothing")
            return ""
        result = self._exec(
            env,
            f"git --no-pager diff {self._base_sha} {self.REMOTE_NAME}/{self.agent_id}",
        )
        return result.get("output", "") or ""

    def _detach_upstream(self, env: DockerEnvironment) -> None:
        """Cut the sandbox off from the repository it was cloned from.

        The task image runs `git clone <upstream> && git checkout <task-sha>`, so the sandbox
        holds the project's **entire history, including every commit after the task commit**.
        Those are reachable through `refs/remotes/origin/*` and tags, which means an agent can
        run `git log --all` or `git show` and, for a task derived from a real pull request,
        read the upstream implementation of the very feature it is being asked to write.

        Removing the remote alone does not help: it stops the network (which is already
        unreachable) and leaves every object in place. The refs have to go, and the objects
        they pinned have to be pruned.

        History *before* the task commit is left intact -- that is ordinary context an engineer
        would have, and `git log` should still work.
        """
        self._exec(env, "git remote remove origin 2>/dev/null || true")
        self._exec(
            env,
            "git for-each-ref --format='%(refname)' refs/remotes | xargs -r -n1 git update-ref -d 2>/dev/null || true",
        )
        self._exec(env, "git tag -l | xargs -r git tag -d >/dev/null 2>&1 || true")
        # `git clone` leaves a LOCAL branch (usually `main`) at the tip of the default branch,
        # and `git checkout <task-sha>` only detaches HEAD -- it does not move that branch. So
        # the future stays reachable through a local ref even after every remote ref and tag
        # is gone. Drop every branch except the one HEAD is on.
        # -n1 so a branch that cannot be deleted (the checked-out one, if HEAD is attached)
        # does not abort deletion of the rest. In the task image HEAD is detached at the task
        # commit, so every branch here is one the clone left behind.
        self._exec(
            env,
            "git for-each-ref --format='%(refname:short)' refs/heads | xargs -r -n1 git branch -D >/dev/null 2>&1 || true",
        )
        self._exec(
            env,
            "git reflog expire --expire=now --all >/dev/null 2>&1; git gc --prune=now --quiet >/dev/null 2>&1 || true",
        )
