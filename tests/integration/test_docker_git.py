"""Integration tests for Docker git collaboration.

These tests require Docker to be running locally.
Run with: pytest tests/integration/test_docker_git.py --run-docker
"""

import time

import pytest

from cooperbench.agents.mini_swe_agent_v2.connectors import DockerGitServer, GitConnector
from cooperbench.agents.mini_swe_agent_v2.environments.docker import DockerEnvironment

# Mark all tests in this module as requiring Docker
pytestmark = pytest.mark.docker


class TestDockerGitServer:
    """Tests for DockerGitServer."""

    def test_create_server(self):
        """Test creating a docker git server."""
        server = None
        try:
            server = DockerGitServer.create(run_id="test-docker-server-create")
            assert server is not None
            assert server.url is not None
            assert "git://" in server.url
        finally:
            if server:
                server.cleanup()

    def test_server_url_format(self):
        """Test that server URL is valid."""
        server = None
        try:
            server = DockerGitServer.create(run_id="test-docker-url-format")
            url = server.url
            assert url.startswith("git://")
            assert url.endswith("/repo.git")
            assert ":9418" in url
        finally:
            if server:
                server.cleanup()

    def test_server_network_name(self):
        """Test that server provides network name for containers."""
        server = None
        try:
            server = DockerGitServer.create(run_id="test-docker-network")
            assert server.network_name is not None
            assert "cooperbench-git-" in server.network_name
        finally:
            if server:
                server.cleanup()


class TestDockerGitConnector:
    """Tests for GitConnector with Docker environment."""

    @pytest.fixture
    def docker_git_server(self):
        """Create a shared git server for tests."""
        server = DockerGitServer.create(run_id="test-docker-connector")
        yield server
        server.cleanup()

    @pytest.fixture
    def agent_docker_env(self, docker_git_server):
        """Create an agent Docker environment."""
        env = DockerEnvironment(
            image="debian:bookworm-slim",
            cwd="/workspace/repo",
            network=docker_git_server.network_name,
        )

        # Install git and initialize repo
        env.execute("apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1", cwd="/")
        env.execute("mkdir -p /workspace/repo", cwd="/")
        env.execute("git init", cwd="/workspace/repo")
        env.execute('echo "Initial content" > file.txt', cwd="/workspace/repo")
        env.execute("git add .", cwd="/workspace/repo")
        env.execute('git config user.email "test@test.com"', cwd="/workspace/repo")
        env.execute('git config user.name "Test"', cwd="/workspace/repo")
        env.execute("git commit -m 'Initial commit'", cwd="/workspace/repo")

        yield env
        env.cleanup()

    def test_setup_configures_remote(self, docker_git_server, agent_docker_env):
        """Test that setup configures the git remote."""
        connector = GitConnector(
            agent_id="agent1",
            agents=["agent1", "agent2"],
            server_url=docker_git_server.url,
        )

        connector.setup(agent_docker_env)

        # Check remote was added
        result = agent_docker_env.execute("git remote -v")
        assert "team" in result["output"]
        assert connector.is_initialized

    def test_push_and_fetch(self, docker_git_server):
        """Test that agents can push and fetch from each other."""
        env1 = None
        env2 = None

        try:
            # Create two agent environments on the same network
            env1 = DockerEnvironment(
                image="debian:bookworm-slim",
                cwd="/workspace/repo",
                network=docker_git_server.network_name,
            )
            env2 = DockerEnvironment(
                image="debian:bookworm-slim",
                cwd="/workspace/repo",
                network=docker_git_server.network_name,
            )

            # Initialize both repos
            for env in [env1, env2]:
                env.execute("apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1", cwd="/")
                env.execute("mkdir -p /workspace/repo", cwd="/")
                env.execute("git init", cwd="/workspace/repo")
                env.execute('echo "Initial content" > file.txt', cwd="/workspace/repo")
                env.execute("git add .", cwd="/workspace/repo")
                env.execute('git config user.email "test@test.com"', cwd="/workspace/repo")
                env.execute('git config user.name "Test"', cwd="/workspace/repo")
                env.execute("git commit -m 'Initial commit'", cwd="/workspace/repo")

            # Setup connectors
            git1 = GitConnector(agent_id="agent1", agents=["agent1", "agent2"], server_url=docker_git_server.url)
            git2 = GitConnector(agent_id="agent2", agents=["agent1", "agent2"], server_url=docker_git_server.url)

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
            if env1:
                env1.cleanup()
            if env2:
                env2.cleanup()


class TestDockerGitCollaborationE2E:
    """End-to-end tests for git collaboration on Docker."""

    def test_multi_agent_collaboration(self):
        """Test that multiple agents can collaborate via git on Docker."""
        server = None
        envs = {}
        agents = ["agent1", "agent2", "agent3"]

        try:
            # Create git server
            server = DockerGitServer.create(run_id="test-docker-multi-agent")

            # Create environments for each agent
            for agent_id in agents:
                env = DockerEnvironment(
                    image="debian:bookworm-slim",
                    cwd="/workspace/repo",
                    network=server.network_name,
                )
                # Initialize the environment
                env.execute("apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1", cwd="/")
                env.execute("mkdir -p /workspace/repo", cwd="/")
                env.execute("git init", cwd="/workspace/repo")
                env.execute('echo "Base content" > file.txt', cwd="/workspace/repo")
                env.execute("git add .", cwd="/workspace/repo")
                env.execute(f'git config user.email "{agent_id}@test.com"', cwd="/workspace/repo")
                env.execute(f'git config user.name "{agent_id}"', cwd="/workspace/repo")
                env.execute("git commit -m 'Initial commit'", cwd="/workspace/repo")
                envs[agent_id] = env

            # Setup all connectors
            for agent_id, env in envs.items():
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
            for env in envs.values():
                env.cleanup()
            if server:
                server.cleanup()
