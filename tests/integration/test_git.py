"""Integration tests for git collaboration connector.

These tests require Modal sandboxes and create actual git servers.
Run with: pytest tests/integration/test_git.py --run-modal
"""

import time

import pytest

from cooperbench.agents.mini_swe_agent_v2.connectors import GitConnector, ModalGitServer

# Mark all tests in this module as requiring Modal
pytestmark = pytest.mark.modal


class SimpleEnv:
    """Adapts Modal sandbox to match DockerEnvironment interface for GitConnector."""

    def __init__(self, sandbox, cwd: str = "/workspace/repo"):
        self.sb = sandbox
        self.cwd = cwd

    def execute(self, cmd: str) -> dict:
        proc = self.sb.exec("bash", "-c", f"cd {self.cwd} && {cmd}")
        stdout = proc.stdout.read()
        stderr = proc.stderr.read()
        proc.wait()
        return {"output": stdout + stderr, "returncode": proc.returncode}


class TestModalGitServer:
    """Tests for ModalGitServer (Modal backend)."""

    def test_create_server(self, modal_app):
        """Test creating a git server."""
        server = None
        try:
            server = ModalGitServer.create(app=modal_app, run_id="test-server-create")
            assert server is not None
            assert server.url is not None
            assert "git://" in server.url or server.url.endswith(".git")
        finally:
            if server:
                server.cleanup()

    def test_server_url_format(self, modal_app):
        """Test that server URL is valid."""
        server = None
        try:
            server = ModalGitServer.create(app=modal_app, run_id="test-url-format")
            # URL should be in git:// format
            url = server.url
            assert url.endswith("/repo.git") or url.endswith(".git")
        finally:
            if server:
                server.cleanup()


class TestModalGitConnector:
    """Tests for GitConnector with Modal environment."""

    @pytest.fixture
    def git_server(self, modal_app):
        """Create a shared git server for tests."""
        server = ModalGitServer.create(app=modal_app, run_id="test-connector")
        yield server
        server.cleanup()

    @pytest.fixture
    def agent_sandbox(self, modal_app):
        """Create an agent sandbox."""
        import modal

        image = modal.Image.debian_slim().run_commands(
            "apt-get update && apt-get install -y git",
        )
        sandbox = modal.Sandbox.create(image=image, app=modal_app, timeout=300)

        # Initialize a repo
        proc = sandbox.exec(
            "bash",
            "-c",
            """
            set -e
            mkdir -p /workspace/repo
            cd /workspace/repo
            git init
            echo "Initial content" > file.txt
            git add .
            git config user.email "test@test.com"
            git config user.name "Test"
            git commit -m "Initial commit"
        """,
        )
        proc.wait()

        yield sandbox
        sandbox.terminate()

    def test_setup_configures_remote(self, git_server, agent_sandbox):
        """Test that setup configures the git remote."""
        env = SimpleEnv(agent_sandbox)
        connector = GitConnector(
            agent_id="agent1",
            agents=["agent1", "agent2"],
            server_url=git_server.url,
        )

        connector.setup(env)

        # Check remote was added
        result = env.execute("git remote -v")
        assert "team" in result["output"]
        assert connector.is_initialized

    def test_push_and_fetch(self, git_server, modal_app):
        """Test that agents can push and fetch from each other."""
        import modal

        image = modal.Image.debian_slim().run_commands(
            "apt-get update && apt-get install -y git",
        )

        # Create two agent sandboxes
        sb1 = modal.Sandbox.create(image=image, app=modal_app, timeout=300)
        sb2 = modal.Sandbox.create(image=image, app=modal_app, timeout=300)

        try:
            # Initialize both repos
            for sb in [sb1, sb2]:
                proc = sb.exec(
                    "bash",
                    "-c",
                    """
                    set -e
                    mkdir -p /workspace/repo
                    cd /workspace/repo
                    git init
                    echo "Initial content" > file.txt
                    git add .
                    git config user.email "test@test.com"
                    git config user.name "Test"
                    git commit -m "Initial commit"
                """,
                )
                proc.wait()

            env1 = SimpleEnv(sb1)
            env2 = SimpleEnv(sb2)

            # Setup connectors
            git1 = GitConnector(agent_id="agent1", agents=["agent1", "agent2"], server_url=git_server.url)
            git2 = GitConnector(agent_id="agent2", agents=["agent1", "agent2"], server_url=git_server.url)

            git1.setup(env1)
            git2.setup(env2)

            # Agent1 makes changes and pushes
            env1.execute('echo "Agent1 was here" >> file.txt')
            env1.execute("git add . && git commit -m 'Agent1 changes'")
            env1.execute("git push team agent1")

            # Agent2 fetches and merges
            time.sleep(0.5)  # Give time for push to complete
            env2.execute("git fetch team")
            env2.execute("git merge team/agent1 -m 'Merge' --allow-unrelated-histories 2>&1 || true")

            # Verify agent2 has agent1's changes
            result = env2.execute("cat file.txt")
            assert "Agent1 was here" in result["output"]

        finally:
            sb1.terminate()
            sb2.terminate()


class TestModalGitCollaborationE2E:
    """End-to-end tests for git collaboration on Modal."""

    def test_multi_agent_collaboration(self, modal_app):
        """Test that multiple agents can collaborate via git."""
        import modal

        image = modal.Image.debian_slim().run_commands(
            "apt-get update && apt-get install -y git",
        )

        server = None
        sandboxes = []
        agents = ["agent1", "agent2", "agent3"]

        try:
            # Create git server
            server = ModalGitServer.create(app=modal_app, run_id="test-multi-agent")

            # Create sandboxes for each agent
            for agent_id in agents:
                sb = modal.Sandbox.create(image=image, app=modal_app, timeout=300)
                proc = sb.exec(
                    "bash",
                    "-c",
                    f"""
                    set -e
                    mkdir -p /workspace/repo
                    cd /workspace/repo
                    git init
                    echo "Base content" > file.txt
                    git add .
                    git config user.email "{agent_id}@test.com"
                    git config user.name "{agent_id}"
                    git commit -m "Initial commit"
                """,
                )
                proc.wait()
                sandboxes.append((agent_id, sb))

            # Setup all connectors
            envs = {}
            for agent_id, sb in sandboxes:
                env = SimpleEnv(sb)
                envs[agent_id] = env
                connector = GitConnector(agent_id=agent_id, agents=agents, server_url=server.url)
                connector.setup(env)

            # Each agent makes unique changes
            for agent_id, env in envs.items():
                env.execute(f'echo "{agent_id} contribution" >> file.txt')
                env.execute(f"git add . && git commit -m '{agent_id} changes'")
                env.execute(f"git push team {agent_id}")
                time.sleep(0.3)

            # Agent3 fetches and lists all branches
            result = envs["agent3"].execute("git fetch team && git branch -r")
            assert "team/agent1" in result["output"]
            assert "team/agent2" in result["output"]

        finally:
            for _, sb in sandboxes:
                sb.terminate()
            if server:
                server.cleanup()
