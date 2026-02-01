"""Integration tests for GCP VM environment.

These tests require:
1. GCP credentials configured (gcloud auth application-default login)
2. GOOGLE_CLOUD_PROJECT environment variable set
3. Network access to GCP

Run with: pytest tests/integration/test_gcp_environment.py -v
"""

import os

import pytest

# Skip all tests if GCP is not configured
pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_CLOUD_PROJECT"),
    reason="GOOGLE_CLOUD_PROJECT not set - skipping GCP integration tests",
)


@pytest.fixture(scope="module")
def gcp_env():
    """Create a single GCP environment for all tests in this module.

    Uses module scope to avoid creating a new VM for each test (~2min startup).
    """
    from cooperbench.agents.mini_swe_agent.environments import get_environment

    GCPEnv = get_environment("gcp")
    env = GCPEnv(
        image="python:3.11-slim",
        machine_type="e2-small",  # Smaller for tests
        cwd="/workspace",
    )
    yield env
    env.cleanup()


class TestGCPEnvironment:
    """Integration tests for GCPEnvironment.

    All tests share a single VM to avoid slow startup times.
    """

    def test_basic_command(self, gcp_env):
        """Test basic command execution."""
        result = gcp_env.execute("echo hello")
        assert result["returncode"] == 0, f"Failed: {result['output']}"
        assert "hello" in result["output"]

    def test_python_execution(self, gcp_env):
        """Test Python is available and works."""
        result = gcp_env.execute("python --version")
        assert result["returncode"] == 0, f"Failed: {result['output']}"
        assert "Python 3.11" in result["output"]

    def test_python_script(self, gcp_env):
        """Test running a Python script."""
        result = gcp_env.execute('python -c "print(1 + 1)"')
        assert result["returncode"] == 0, f"Failed: {result['output']}"
        assert "2" in result["output"]

    def test_working_directory(self, gcp_env):
        """Test working directory is respected."""
        # Create a directory and file
        gcp_env.execute("mkdir -p /workspace/testdir")
        gcp_env.execute("echo 'test content' > /workspace/testdir/test.txt")

        # Verify from the working directory
        result = gcp_env.execute("cat test.txt", cwd="/workspace/testdir")
        assert result["returncode"] == 0, f"Failed: {result['output']}"
        assert "test content" in result["output"]

    def test_exit_code(self, gcp_env):
        """Test non-zero exit codes are captured."""
        result = gcp_env.execute("exit 42")
        assert result["returncode"] == 42

    def test_stderr_captured(self, gcp_env):
        """Test stderr is captured in output."""
        result = gcp_env.execute("echo error >&2")
        assert "error" in result["output"]

    def test_template_vars(self, gcp_env):
        """Test get_template_vars returns expected data."""
        vars = gcp_env.get_template_vars()
        assert vars["system"] == "Linux"
        assert vars["release"] == "gcp"
        assert "vm_name" in vars
        assert "project_id" in vars
        assert vars["image"] == "python:3.11-slim"

    def test_file_persistence(self, gcp_env):
        """Test that files persist across commands (same container)."""
        # Create a file
        gcp_env.execute("echo 'persistent data' > /workspace/persist.txt")

        # Read it back in a separate command
        result = gcp_env.execute("cat /workspace/persist.txt")
        assert result["returncode"] == 0, f"Failed: {result['output']}"
        assert "persistent data" in result["output"]


class TestGCPEnvironmentWithEnvVars:
    """Test environment variables (requires separate VM)."""

    def test_environment_variables(self):
        """Test custom environment variables."""
        from cooperbench.agents.mini_swe_agent.environments import get_environment

        GCPEnv = get_environment("gcp")
        env = GCPEnv(
            image="python:3.11-slim",
            machine_type="e2-small",
            cwd="/",
            env={"MY_VAR": "my_value"},
        )
        try:
            result = env.execute("echo $MY_VAR")
            assert result["returncode"] == 0, f"Failed: {result['output']}"
            assert "my_value" in result["output"]
        finally:
            env.cleanup()


class TestGCPEnvironmentFactory:
    """Test the environment factory function (no GCP resources needed)."""

    def test_get_gcp_environment(self):
        """Test getting GCP environment class from factory."""
        from cooperbench.agents.mini_swe_agent.environments import get_environment

        GCPEnv = get_environment("gcp")
        assert GCPEnv.__name__ == "GCPEnvironment"

    def test_invalid_environment(self):
        """Test invalid environment name raises error."""
        from cooperbench.agents.mini_swe_agent.environments import get_environment

        with pytest.raises(ValueError, match="Unknown environment"):
            get_environment("invalid")


class TestGCPEnvironmentConfig:
    """Test GCPEnvironmentConfig validation (no GCP resources needed)."""

    def test_default_config(self):
        """Test default configuration values."""
        from cooperbench.agents.mini_swe_agent.environments.gcp import GCPEnvironmentConfig

        config = GCPEnvironmentConfig(image="python:3.11")
        assert config.zone == "us-central1-a"
        assert config.machine_type == "e2-medium"
        assert config.cwd == "/"
        assert config.timeout == 3600

    def test_custom_config(self):
        """Test custom configuration values."""
        from cooperbench.agents.mini_swe_agent.environments.gcp import GCPEnvironmentConfig

        config = GCPEnvironmentConfig(
            image="python:3.11",
            zone="us-west1-b",
            machine_type="e2-standard-4",
            cwd="/app",
            timeout=7200,
            network="my-vpc",
        )
        assert config.zone == "us-west1-b"
        assert config.machine_type == "e2-standard-4"
        assert config.cwd == "/app"
        assert config.timeout == 7200
        assert config.network == "my-vpc"
