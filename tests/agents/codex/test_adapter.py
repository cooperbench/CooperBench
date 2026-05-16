"""Tests for the Codex adapter.

Stubs the Docker environment.  Verifies:
  - registry registration
  - credential resolution
  - command shape (sandbox flag, model flag, JSON output, auth dir env)
  - model fallback when the first --model fails
  - coop + git shared plumbing reused from agents._coop
"""

import json
from unittest.mock import patch as mock_patch

import pytest

from cooperbench.agents import AgentResult, get_runner, list_agents
from cooperbench.agents.codex.adapter import resolve_credentials


class _FakeEnv:
    """Same content-keyed fake env used by the Claude Code adapter tests."""

    def __init__(self, responses_seq):
        # responses_seq is a list of {key: response} dicts.  Each entry
        # represents one "codex invocation cycle" (codex exec + stream
        # read).  We keep separate cursors for the codex command and the
        # stream-read so the fallback test correctly serves the error
        # stream for attempt 1 and the success stream for attempt 2.
        self._seq = list(responses_seq)
        self._codex_idx = 0
        self._stream_idx = 0
        self.executed: list[str] = []
        self.cleaned = False

    def _bucket(self, cursor: int) -> dict:
        return self._seq[min(cursor, len(self._seq) - 1)]

    def execute(self, action, cwd: str = "", *, timeout: int | None = None):
        command = action.get("command", "")
        self.executed.append(command)

        if "codex exec" in command:
            bucket = self._bucket(self._codex_idx)
            self._codex_idx += 1
        elif "codex-stream.jsonl" in command:
            bucket = self._bucket(self._stream_idx)
            self._stream_idx += 1
        else:
            bucket = self._bucket(self._codex_idx)

        for key, value in bucket.items():
            if key in command:
                return value
        return {"output": "", "returncode": 0}

    def cleanup(self):
        self.cleaned = True


@pytest.fixture
def fake_env_factory():
    def _factory(responses):
        return _FakeEnv(responses if isinstance(responses, list) else [responses])

    return _factory


def _stream(steps: int = 1, error: bool = False, message: str = "") -> str:
    lines = [json.dumps({"type": "thread.started", "thread_id": "t1"})]
    for _ in range(steps):
        lines.append(json.dumps({"type": "turn.started"}))
        lines.append(
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cached_input_tokens": 0,
                        "reasoning_output_tokens": 0,
                    },
                }
            )
        )
    if error:
        lines.append(json.dumps({"type": "error", "message": message or "boom"}))
    return "\n".join(lines)


class TestRegistration:
    def test_codex_is_registered(self):
        assert "codex" in list_agents()

    def test_get_runner_returns_instance(self):
        runner = get_runner("codex")
        assert runner is not None
        assert hasattr(runner, "run")


class TestResolveCredentials:
    def test_openai_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        creds = resolve_credentials()
        assert creds == {"OPENAI_API_KEY": "sk-test"}

    def test_empty_when_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        creds = resolve_credentials()
        assert creds == {}


class TestAdapterRun:
    def _responses(self, stream_text: str, patch_text: str = ""):
        return {
            "cb-setup.sh": {"output": "installed\n", "returncode": 0},
            "codex exec": {"output": "", "returncode": 0},
            "codex-stream.jsonl": {"output": stream_text, "returncode": 0},
            "patch.txt": {"output": patch_text, "returncode": 0},
        }

    def test_solo_success(self, fake_env_factory, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        env = fake_env_factory(self._responses(_stream(steps=3), "diff --git a/x b/x\n+hi\n"))

        with mock_patch(
            "cooperbench.agents.codex.adapter._build_environment",
            return_value=env,
        ):
            runner = get_runner("codex")
            result = runner.run(
                task="implement X",
                image="cooperbench/example:task1",
                model_name="gpt-5.5",
            )

        assert isinstance(result, AgentResult)
        assert result.status == "Submitted"
        assert result.steps == 3
        assert result.input_tokens == 300
        assert result.output_tokens == 150
        assert result.patch.startswith("diff --git")
        assert env.cleaned

    def test_invocation_uses_required_flags(self, fake_env_factory, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        env = fake_env_factory(self._responses(_stream(), ""))

        with mock_patch(
            "cooperbench.agents.codex.adapter._build_environment",
            return_value=env,
        ):
            get_runner("codex").run(
                task="t",
                image="cooperbench/example:task1",
                model_name="gpt-5.5",
            )

        codex_cmds = [c for c in env.executed if "codex exec" in c]
        assert len(codex_cmds) == 1
        cmd = codex_cmds[0]
        assert "--sandbox danger-full-access" in cmd
        assert "--skip-git-repo-check" in cmd
        assert "--json" in cmd
        assert "--model gpt-5.5" in cmd or "--model 'gpt-5.5'" in cmd
        # Prompt comes from file, not inlined.
        assert "cb-instruction.txt" in cmd

    def test_auth_file_written_in_container(self, fake_env_factory, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        env = fake_env_factory(self._responses(_stream(), ""))

        with mock_patch(
            "cooperbench.agents.codex.adapter._build_environment",
            return_value=env,
        ):
            get_runner("codex").run(
                task="t",
                image="cooperbench/example:task1",
                model_name="gpt-5.5",
            )

        # Auth heredoc must contain the key, written to /tmp/codex-home/auth.json
        # (so CODEX_HOME=/tmp/codex-home picks it up).
        joined = "\n".join(env.executed)
        assert "OPENAI_API_KEY" in joined
        assert "sk-test" in joined
        assert "auth.json" in joined

    def test_model_fallback_on_invalid_model_error(self, fake_env_factory, monkeypatch):
        """If --model gpt-5.5 errors with a model-not-found, retry without --model."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        first = self._responses(_stream(error=True, message="model 'gpt-5.5' does not exist"), "")
        second = self._responses(_stream(steps=2), "diff --git a/x b/x\n+ok\n")
        env = fake_env_factory([first, second])

        with mock_patch(
            "cooperbench.agents.codex.adapter._build_environment",
            return_value=env,
        ):
            result = get_runner("codex").run(
                task="t",
                image="cooperbench/example:task1",
                model_name="gpt-5.5",
            )

        codex_cmds = [c for c in env.executed if "codex exec" in c]
        assert len(codex_cmds) == 2  # original + fallback
        assert "--model gpt-5.5" in codex_cmds[0] or "--model 'gpt-5.5'" in codex_cmds[0]
        # Fallback omits --model entirely.
        assert "--model" not in codex_cmds[1]
        assert result.status == "Submitted"

    def test_no_fallback_on_non_model_error(self, fake_env_factory, monkeypatch):
        """Non-model errors (rate limit, etc) should not trigger a retry."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        responses = self._responses(_stream(error=True, message="rate limited"), "")
        env = fake_env_factory(responses)

        with mock_patch(
            "cooperbench.agents.codex.adapter._build_environment",
            return_value=env,
        ):
            result = get_runner("codex").run(
                task="t",
                image="cooperbench/example:task1",
                model_name="gpt-5.5",
            )

        codex_cmds = [c for c in env.executed if "codex exec" in c]
        assert len(codex_cmds) == 1  # no retry
        assert result.status == "Error"

    def test_missing_api_key_skips_run(self, fake_env_factory, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        env = fake_env_factory(self._responses(_stream(), ""))

        with mock_patch(
            "cooperbench.agents.codex.adapter._build_environment",
            return_value=env,
        ):
            result = get_runner("codex").run(
                task="t",
                image="cooperbench/example:task1",
                model_name="gpt-5.5",
            )

        # Adapter still runs install + invokes codex, but codex will fail.
        # Status should be Error and error message should mention auth.
        assert result.status == "Error"

    def test_coop_messaging_and_git_setup_wired(self, fake_env_factory, monkeypatch):
        """Coop + git uses the same shared plumbing as Claude Code."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        env = fake_env_factory(self._responses(_stream(steps=2), ""))

        with mock_patch(
            "cooperbench.agents.codex.adapter._build_environment",
            return_value=env,
        ):
            get_runner("codex").run(
                task="t",
                image="cooperbench/example:task1",
                model_name="gpt-5.5",
                agents=["agent1", "agent2"],
                agent_id="agent1",
                comm_url="redis://localhost:6379#run:abc",
                git_server_url="git://cooperbench-git:9418/abc/repo.git",
                git_enabled=True,
                messaging_enabled=True,
                config={"git_network": "cooperbench"},
            )

        joined = "\n".join(env.executed)
        # Coop env vars are exported when invoking codex.
        codex_cmds = [c for c in env.executed if "codex exec" in c]
        assert any("COOP_REDIS_URL=" in c for c in codex_cmds)
        assert any("host.docker.internal" in c for c in codex_cmds)
        # Git setup ran before codex.
        assert "git remote add team git://cooperbench-git:9418/abc/repo.git" in joined
        assert "git checkout -b agent1" in joined
