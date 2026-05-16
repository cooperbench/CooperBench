"""Tests for the Claude Code adapter.

These tests stub out the Docker environment so they run fast and require
no daemon. We assert that the adapter:

  - is registered under ``"claude_code"``
  - exposes the ``AgentRunner`` shape
  - in ``run``, performs install + invoke + ``cat patch.txt`` calls in order
  - returns an ``AgentResult`` populated from the parsed stream-json + session
"""

import json
from unittest.mock import patch as mock_patch

import pytest

from cooperbench.agents import AgentResult, get_runner, list_agents
from cooperbench.agents._coop import (
    build_git_setup_command,
    build_instruction,
    parse_sent_messages_log,
    rewrite_comm_url_for_container,
)
from cooperbench.agents.claude_code.adapter import resolve_credentials


class _FakeEnv:
    """Captures executed commands and returns scripted outputs.

    Outputs are dispatched by the substring keys in ``responses``; the
    first key found in the command is used.  Commands that don't match
    any key (e.g. heredoc writes) return ``returncode=0`` with empty
    stdout, which is what real shells do.
    """

    def __init__(self, responses):
        self._responses = responses
        self.executed: list[str] = []
        self.cleaned = False
        self.install_failed = False

    def execute(self, action, cwd: str = "", *, timeout: int | None = None):
        command = action.get("command", "")
        self.executed.append(command)
        for key, value in self._responses.items():
            if key in command:
                return value
        return {"output": "", "returncode": 0}

    def cleanup(self):
        self.cleaned = True


@pytest.fixture
def fake_env_factory():
    def _factory(responses):
        return _FakeEnv(responses)

    return _factory


@pytest.fixture
def stream_json_success():
    return "\n".join(
        [
            json.dumps({"type": "system", "subtype": "init"}),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "total_cost_usd": 0.12,
                    "num_turns": 5,
                    "usage": {
                        "input_tokens": 1000,
                        "output_tokens": 200,
                        "cache_read_input_tokens": 50,
                        "cache_creation_input_tokens": 10,
                    },
                }
            ),
        ]
    )


@pytest.fixture
def session_jsonl_one_turn():
    return json.dumps(
        {
            "type": "user",
            "message": {"role": "user", "content": "Hello"},
            "timestamp": "2026-01-01T00:00:00Z",
        }
    )


class TestRewriteCommUrl:
    """The coop runner gives us a host-side Redis URL like
    ``redis://localhost:6379#run:abc``.  ``localhost`` is unreachable
    from inside a Docker container; rewrite it so the in-container
    helper can reach Redis on the host.
    """

    def test_localhost_replaced_with_host_docker_internal(self):
        assert (
            rewrite_comm_url_for_container("redis://localhost:6379#run:abc")
            == "redis://host.docker.internal:6379#run:abc"
        )

    def test_127001_replaced(self):
        assert rewrite_comm_url_for_container("redis://127.0.0.1:6379") == "redis://host.docker.internal:6379"

    def test_external_host_preserved(self):
        url = "redis://my-redis.example.com:6379#run:abc"
        assert rewrite_comm_url_for_container(url) == url

    def test_none_returns_none(self):
        assert rewrite_comm_url_for_container(None) is None


class TestParseSentMessagesLog:
    """The in-container messaging helper appends one JSON object per send
    to a log file. We harvest it post-run to populate
    ``AgentResult.sent_messages``."""

    def test_parses_one_send_per_line(self):
        log = "\n".join(
            [
                json.dumps({"to": "agent2", "content": "starting on auth", "timestamp": 1.0}),
                json.dumps({"to": "agent2", "content": "done", "timestamp": 2.0}),
            ]
        )
        msgs = parse_sent_messages_log(log)
        assert len(msgs) == 2
        assert msgs[0]["to"] == "agent2"
        assert msgs[0]["content"] == "starting on auth"

    def test_skips_blank_and_invalid_lines(self):
        log = "\n".join(
            [
                "",
                "not json",
                json.dumps({"to": "agent2", "content": "ok"}),
            ]
        )
        msgs = parse_sent_messages_log(log)
        assert msgs == [{"to": "agent2", "content": "ok"}]

    def test_empty_log_returns_empty_list(self):
        assert parse_sent_messages_log("") == []


class TestBuildInstruction:
    """The prompt has two flavors: solo (just the task + submission
    protocol) and coop (also documents the messaging helpers)."""

    def test_solo_omits_messaging_section(self):
        text = build_instruction("Implement feature X")
        assert "Implement feature X" in text
        assert "patch.txt" in text
        # Solo runs don't see the coop tools.
        assert "coop-send" not in text
        assert "coop-recv" not in text

    def test_coop_mentions_messaging_helpers(self):
        text = build_instruction(
            "Implement feature X",
            agents=["agent1", "agent2"],
            agent_id="agent1",
        )
        assert "Implement feature X" in text
        assert "coop-send" in text
        assert "coop-recv" in text
        # Names the partner agent so Claude knows who to message.
        assert "agent2" in text

    def test_coop_with_self_only_still_solo_shape(self):
        # If somehow only one agent is listed, treat as solo prompt.
        text = build_instruction(
            "Implement feature X",
            agents=["agent1"],
            agent_id="agent1",
        )
        assert "coop-send" not in text

    def test_git_section_only_when_git_enabled(self):
        without_git = build_instruction(
            "Implement feature X",
            agents=["agent1", "agent2"],
            agent_id="agent1",
            git_enabled=False,
        )
        with_git = build_instruction(
            "Implement feature X",
            agents=["agent1", "agent2"],
            agent_id="agent1",
            git_enabled=True,
        )
        assert "## Git collaboration" not in without_git
        assert "## Git collaboration" in with_git
        # Must teach the remote name and partner-branch shape.
        assert "team" in with_git
        assert "team/agent2" in with_git

    def test_git_section_absent_in_solo(self):
        text = build_instruction("X", git_enabled=True)  # no agents -> solo
        assert "## Git collaboration" not in text


class TestBuildGitSetupCommand:
    """Pure helper that emits the shell snippet for configuring a fresh
    container as a participant in the shared git remote."""

    def test_contains_required_steps(self):
        cmd = build_git_setup_command(
            agent_id="agent1",
            server_url="git://cooperbench-git:9418/abc/repo.git",
        )
        # git identity (so commits/pushes work)
        assert "git config user.email" in cmd
        assert "git config user.name" in cmd
        # team remote pointing at the server
        assert "git remote add team git://cooperbench-git:9418/abc/repo.git" in cmd
        # agent-named branch
        assert "git checkout -b agent1" in cmd
        # initial push so peers can fetch
        assert "git push" in cmd
        assert "team" in cmd
        assert "agent1" in cmd

    def test_idempotent_on_pre_existing_remote(self):
        """The connector handles the case where the remote already exists
        (a previous setup ran).  The command must include a recovery path."""
        cmd = build_git_setup_command(
            agent_id="agent1",
            server_url="git://cooperbench-git:9418/abc/repo.git",
        )
        assert "git remote set-url" in cmd

    def test_server_url_shell_quoted(self):
        """URLs come from external data; we mustn't allow shell injection."""
        cmd = build_git_setup_command(
            agent_id="agent1",
            server_url="git://h:9418/r;rm -rf /",
        )
        # The dangerous payload must not appear unquoted; the only safe
        # form is inside single-quotes, where ; loses its shell meaning.
        assert "rm -rf /'" in cmd or "'rm -rf /'" in cmd or "\\;rm" in cmd


class TestAdapterGitWiring:
    """When git is enabled, the adapter must:
    (a) join the shared docker network,
    (b) run git setup in the container before invoking claude,
    (c) include the git section in the prompt.
    """

    def _responses(self, stream_json, session_jsonl, patch_text):
        return {
            "cb-setup.sh": {"output": "+ installed\n", "returncode": 0},
            "claude --verbose": {"output": "", "returncode": 0},
            "claude-stream.jsonl": {"output": stream_json, "returncode": 0},
            "find /tmp/claude-cfg/projects": {"output": session_jsonl, "returncode": 0},
            "patch.txt": {"output": patch_text, "returncode": 0},
        }

    def test_git_setup_runs_when_git_enabled(self, fake_env_factory, stream_json_success, session_jsonl_one_turn):
        env = fake_env_factory(self._responses(stream_json_success, session_jsonl_one_turn, ""))
        captured: dict[str, object] = {}

        def _capture_env(image, *, network=None, extra_run_args=None, timeout=7200):
            captured["network"] = network
            captured["extra_run_args"] = extra_run_args or []
            return env

        with mock_patch(
            "cooperbench.agents.claude_code.adapter._build_environment",
            side_effect=_capture_env,
        ):
            runner = get_runner("claude_code")
            runner.run(
                task="t",
                image="cooperbench/example:task1",
                model_name="claude-sonnet-4-5",
                agents=["agent1", "agent2"],
                agent_id="agent1",
                comm_url="redis://localhost:6379#run:abc",
                git_server_url="git://cooperbench-git:9418/abc/repo.git",
                git_enabled=True,
                messaging_enabled=True,
                config={"git_network": "cooperbench"},
            )

        # Joins shared docker network.
        assert captured["network"] == "cooperbench"
        # Some command issued must configure the team remote.
        joined = "\n".join(env.executed)
        assert "git remote add team git://cooperbench-git:9418/abc/repo.git" in joined
        assert "git checkout -b agent1" in joined

    def test_git_setup_skipped_when_git_disabled(self, fake_env_factory, stream_json_success, session_jsonl_one_turn):
        env = fake_env_factory(self._responses(stream_json_success, session_jsonl_one_turn, ""))

        with mock_patch(
            "cooperbench.agents.claude_code.adapter._build_environment",
            return_value=env,
        ):
            runner = get_runner("claude_code")
            runner.run(
                task="t",
                image="cooperbench/example:task1",
                model_name="claude-sonnet-4-5",
                agents=["agent1", "agent2"],
                agent_id="agent1",
                comm_url="redis://localhost:6379#run:abc",
                git_enabled=False,
                messaging_enabled=True,
            )

        joined = "\n".join(env.executed)
        assert "git remote add team" not in joined
        assert "git checkout -b agent1" not in joined


class TestResolveCredentials:
    """The adapter accepts either an API key or an OAuth token, with the
    OAuth token also discoverable from ``~/.claude/.credentials.json`` so
    a host that's already logged in via ``claude login`` works out of the
    box (subscription-based usage, no API key required).
    """

    def test_api_key_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-test")
        creds = resolve_credentials(credentials_path=tmp_path / "missing.json")
        assert creds == {"ANTHROPIC_API_KEY": "sk-test"}

    def test_explicit_oauth_env_var(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-test")
        creds = resolve_credentials(credentials_path=tmp_path / "missing.json")
        assert creds == {"CLAUDE_CODE_OAUTH_TOKEN": "oat-test"}

    def test_falls_back_to_credentials_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        path = tmp_path / "creds.json"
        path.write_text(json.dumps({"claudeAiOauth": {"accessToken": "oat-from-file", "expiresAt": 9999999999000}}))
        creds = resolve_credentials(credentials_path=path)
        assert creds == {"CLAUDE_CODE_OAUTH_TOKEN": "oat-from-file"}

    def test_returns_empty_when_nothing_available(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        creds = resolve_credentials(credentials_path=tmp_path / "absent.json")
        assert creds == {}

    def test_corrupt_credentials_file_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        path = tmp_path / "creds.json"
        path.write_text("not json at all")
        assert resolve_credentials(credentials_path=path) == {}


class TestRegistration:
    def test_claude_code_is_registered(self):
        assert "claude_code" in list_agents()

    def test_get_runner_returns_instance(self):
        runner = get_runner("claude_code")
        assert runner is not None
        assert hasattr(runner, "run")
        assert callable(runner.run)


class TestAdapterRun:
    """End-to-end behavior of ``ClaudeCodeRunner.run`` with the env stubbed."""

    def _responses(self, stream_json, session_jsonl, patch_text, *, install_rc=0):
        return {
            "cb-setup.sh": {"output": "+ claude installed\n", "returncode": install_rc},
            "claude --verbose": {"output": "", "returncode": 0},
            "claude-stream.jsonl": {"output": stream_json, "returncode": 0},
            "find /tmp/claude-cfg/projects": {"output": session_jsonl, "returncode": 0},
            "patch.txt": {"output": patch_text, "returncode": 0},
        }

    def test_run_returns_agent_result(self, fake_env_factory, stream_json_success, session_jsonl_one_turn):
        env = fake_env_factory(
            self._responses(
                stream_json_success,
                session_jsonl_one_turn,
                "diff --git a/x b/x\n+hello\n",
            )
        )

        with mock_patch(
            "cooperbench.agents.claude_code.adapter._build_environment",
            return_value=env,
        ):
            runner = get_runner("claude_code")
            result = runner.run(
                task="implement feature X",
                image="cooperbench/example:task1",
                model_name="claude-sonnet-4-6",
            )

        assert isinstance(result, AgentResult)
        assert result.status == "Submitted"
        assert result.patch.startswith("diff --git")
        assert result.cost == pytest.approx(0.12)
        assert result.steps == 5
        assert result.input_tokens == 1000
        assert result.output_tokens == 200
        assert result.cache_read_tokens == 50
        assert result.cache_write_tokens == 10
        assert result.messages == [{"role": "user", "content": "Hello"}]
        assert env.cleaned is True

    def test_invocation_includes_required_flags(self, fake_env_factory, stream_json_success, session_jsonl_one_turn):
        env = fake_env_factory(self._responses(stream_json_success, session_jsonl_one_turn, ""))

        with mock_patch(
            "cooperbench.agents.claude_code.adapter._build_environment",
            return_value=env,
        ):
            runner = get_runner("claude_code")
            runner.run(
                task="implement feature X",
                image="cooperbench/example:task1",
                model_name="claude-sonnet-4-6",
            )

        claude_cmds = [c for c in env.executed if "claude --verbose" in c]
        assert len(claude_cmds) == 1
        claude_cmd = claude_cmds[0]
        assert "claude " in claude_cmd
        assert "--output-format=stream-json" in claude_cmd or "--output-format stream-json" in claude_cmd
        assert "--print" in claude_cmd or " -p " in claude_cmd
        assert "--permission-mode=bypassPermissions" in claude_cmd or "bypassPermissions" in claude_cmd
        assert "--verbose" in claude_cmd
        # The instruction is written to a file and read back with `cat`,
        # not interpolated into the claude command itself.  Verify the
        # instruction landed in one of the heredoc-write commands.
        assert any("implement feature X" in c for c in env.executed)
        # And that the claude command reads the instruction from that file.
        assert "cb-instruction.txt" in claude_cmd

    def test_model_name_propagated_via_env_in_invocation(
        self, fake_env_factory, stream_json_success, session_jsonl_one_turn
    ):
        env = fake_env_factory(self._responses(stream_json_success, session_jsonl_one_turn, ""))

        with mock_patch(
            "cooperbench.agents.claude_code.adapter._build_environment",
            return_value=env,
        ):
            runner = get_runner("claude_code")
            runner.run(
                task="t",
                image="cooperbench/example:task1",
                model_name="anthropic/claude-sonnet-4-6",
            )

        claude_cmds = [c for c in env.executed if "claude --verbose" in c]
        assert len(claude_cmds) == 1
        claude_cmd = claude_cmds[0]
        # The "anthropic/" prefix is stripped before passing as ANTHROPIC_MODEL.
        assert "claude-sonnet-4-6" in claude_cmd
        assert "anthropic/claude-sonnet-4-6" not in claude_cmd

    def test_error_status_when_install_fails(self, fake_env_factory):
        env = fake_env_factory({"cb-setup.sh": {"output": "npm: command not found", "returncode": 127}})

        with mock_patch(
            "cooperbench.agents.claude_code.adapter._build_environment",
            return_value=env,
        ):
            runner = get_runner("claude_code")
            result = runner.run(
                task="t",
                image="cooperbench/example:task1",
                model_name="claude-sonnet-4-6",
            )

        assert result.status == "Error"
        assert result.error is not None
        assert result.patch == ""
        assert env.cleaned is True

    def test_empty_patch_when_agent_writes_no_file(self, fake_env_factory, stream_json_success, session_jsonl_one_turn):
        env = fake_env_factory(self._responses(stream_json_success, session_jsonl_one_turn, ""))

        with mock_patch(
            "cooperbench.agents.claude_code.adapter._build_environment",
            return_value=env,
        ):
            runner = get_runner("claude_code")
            result = runner.run(
                task="t",
                image="cooperbench/example:task1",
                model_name="claude-sonnet-4-6",
            )

        assert result.status == "Submitted"
        assert result.patch == ""

    def test_limits_exceeded_propagated(self, fake_env_factory, session_jsonl_one_turn):
        stream = json.dumps(
            {
                "type": "result",
                "subtype": "error_max_turns",
                "total_cost_usd": 0.03,
                "num_turns": 50,
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "is_error": True,
            }
        )
        env = fake_env_factory(self._responses(stream, session_jsonl_one_turn, ""))

        with mock_patch(
            "cooperbench.agents.claude_code.adapter._build_environment",
            return_value=env,
        ):
            runner = get_runner("claude_code")
            result = runner.run(
                task="t",
                image="cooperbench/example:task1",
                model_name="claude-sonnet-4-6",
            )

        assert result.status == "LimitsExceeded"
