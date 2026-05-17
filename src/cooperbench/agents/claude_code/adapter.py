"""Claude Code adapter for CooperBench.

Runs the official ``@anthropic-ai/claude-code`` CLI inside the task's
Docker image and harvests the agent's diff from ``/workspace/repo/patch.txt``.

The design mirrors Harbor's adapter (install in container, invoke in
headless ``--print --output-format=stream-json`` mode, parse the final
``result`` event for cost/tokens, walk the session JSONL for messages)
but reuses CooperBench's existing ``DockerEnvironment`` from
``mini_swe_agent_v2`` so the container lifecycle, image pulling, and
``execute()`` semantics are consistent with the other adapters.

Coop + git support: see ``cooperbench.agents._coop`` for the shared
helpers (messaging, prompt blocks, git remote setup).
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from pathlib import Path
from typing import Any

from cooperbench.agents import AgentResult
from cooperbench.agents._coop import (
    build_git_setup_command,
    build_instruction,
    parse_sent_messages_log,
    rewrite_comm_url_for_container,
)
from cooperbench.agents._coop.runtime import (
    CONTAINER_COOP_MSG_PATH,
    CONTAINER_COOP_SEND_LOG,
    CONTAINER_INSTRUCTION_PATH,
    CONTAINER_REPO_PATH,
    CONTAINER_SETUP_PATH,
    ContainerEnv,
    build_environment,
    normalize_patch,
    read_file_from_container,
    write_file_in_container,
)
from cooperbench.agents._team import build_team_env, build_team_instruction, scratchpad_mount_args
from cooperbench.agents.claude_code.parsers import parse_session_jsonl, parse_stream_json
from cooperbench.agents.registry import register

logger = logging.getLogger(__name__)


_PACKAGE_DIR = Path(__file__).parent
SETUP_SCRIPT_PATH = _PACKAGE_DIR / "setup.sh"
COOP_MSG_SCRIPT_PATH = _PACKAGE_DIR.parent / "_coop" / "coop_msg.py"
COOP_INSTALL_SNIPPET_PATH = _PACKAGE_DIR.parent / "_coop" / "install_snippet.sh"
TEAM_TASK_SCRIPT_PATH = _PACKAGE_DIR.parent / "_team" / "coop_task.py"
TEAM_INSTALL_SNIPPET_PATH = _PACKAGE_DIR.parent / "_team" / "install_snippet.sh"
TEAM_MCP_SCRIPT_PATH = _PACKAGE_DIR.parent / "_team" / "mcp_server.py"
CONTAINER_TEAM_TASK_PATH = "/tmp/cb-coop-task.py"
CONTAINER_TEAM_INSTALL_PATH = "/tmp/cb-team-install.sh"
CONTAINER_TEAM_MCP_PATH = "/tmp/cb-mcp-server.py"

# Inside the container, we redirect Claude Code's per-session state under
# /tmp so we always know where to find the JSONL trajectory after the run.
CONTAINER_CLAUDE_CONFIG_DIR = "/tmp/claude-cfg"
CONTAINER_STREAM_LOG = "/tmp/claude-stream.jsonl"

DEFAULT_CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"


def resolve_credentials(*, credentials_path: Path | None = None) -> dict[str, str]:
    """Pick the credential to forward to the in-container Claude Code CLI.

    Resolution order:

    1. ``ANTHROPIC_API_KEY`` in the host environment (API-credit billing).
    2. ``CLAUDE_CODE_OAUTH_TOKEN`` in the host environment (subscription).
    3. ``claudeAiOauth.accessToken`` from ``~/.claude/.credentials.json``,
       i.e. a host that's already logged in via ``claude login``.

    Returns the chosen credential as a one-key dict ready to merge into
    the container env; an empty dict means no credential was available
    and the run will likely fail to authenticate.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return {"ANTHROPIC_API_KEY": api_key}

    oauth = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if oauth:
        return {"CLAUDE_CODE_OAUTH_TOKEN": oauth}

    path = credentials_path if credentials_path is not None else DEFAULT_CREDENTIALS_PATH
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    token = (data.get("claudeAiOauth") or {}).get("accessToken")
    if isinstance(token, str) and token.strip():
        return {"CLAUDE_CODE_OAUTH_TOKEN": token.strip()}
    return {}


def _strip_provider_prefix(model_name: str) -> str:
    """``anthropic/claude-sonnet-4-6`` -> ``claude-sonnet-4-6``.

    Claude Code's ``ANTHROPIC_MODEL`` env var wants the bare model id when
    talking to the official Anthropic API.  Other providers' prefixes are
    not supported by Claude Code itself, so stripping the leading
    ``provider/`` is the only sane default.
    """
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def _build_claude_command(
    instruction_path: str,
    model_name: str,
    stream_log_path: str,
    *,
    extra_flags: str = "",
    coop_env: dict[str, str] | None = None,
) -> str:
    """Compose the in-container shell command that invokes Claude Code.

    We read the prompt from a file (rather than inlining via ``-p``) so
    long instructions don't blow past argv limits and don't need
    shell-escaping.
    """
    model = _strip_provider_prefix(model_name)
    coop_exports = ""
    if coop_env:
        coop_exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in coop_env.items())
    # The PATH manipulation is needed when claude-code is installed under
    # ``~/.local/bin`` (curl-based install path); npm-installed binaries
    # land in /usr/bin so this is a no-op there.
    return (
        'export PATH="$HOME/.local/bin:$PATH"; '
        f"export ANTHROPIC_MODEL={shlex.quote(model)}; "
        f"export CLAUDE_CONFIG_DIR={shlex.quote(CONTAINER_CLAUDE_CONFIG_DIR)}; "
        "export IS_SANDBOX=1; "
        "export FORCE_AUTO_BACKGROUND_TASKS=1; "
        "export ENABLE_BACKGROUND_TASKS=1; "
        "export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1; "
        + coop_exports
        + f"mkdir -p {CONTAINER_CLAUDE_CONFIG_DIR}; "
        f"cd {CONTAINER_REPO_PATH} && "
        "claude --verbose --output-format=stream-json "
        "--permission-mode=bypassPermissions "
        f"{extra_flags}"
        f'--print -- "$(cat {shlex.quote(instruction_path)})" '
        f"2>&1 | tee {shlex.quote(stream_log_path)}"
    )


def _find_session_jsonl(env: ContainerEnv) -> str:
    """Concatenate every session ``*.jsonl`` produced under CLAUDE_CONFIG_DIR.

    Claude Code writes one file per session; there will normally be
    exactly one for a fresh container.
    """
    cmd = f"find {CONTAINER_CLAUDE_CONFIG_DIR}/projects -name '*.jsonl' -type f 2>/dev/null | xargs -r cat"
    result = env.execute({"command": cmd})
    if result.get("returncode") == 0:
        return result.get("output") or ""
    return ""


# Test-time shims: the existing test suite monkey-patches
# ``cooperbench.agents.claude_code.adapter._build_environment``, so keep a
# module-level alias that forwards to the shared helper.
_build_environment = build_environment


@register("claude_code")
class ClaudeCodeRunner:
    """Adapter for the official Claude Code CLI.

    Supports solo, coop (Redis messaging), and coop + git (shared
    ``team`` remote against the ``cooperbench-git`` server).
    """

    def run(
        self,
        task: str,
        image: str,
        *,
        agent_id: str = "agent",
        model_name: str = "claude-sonnet-4-6",
        agents: list[str] | None = None,
        comm_url: str | None = None,
        git_server_url: str | None = None,
        git_enabled: bool = False,
        messaging_enabled: bool = True,
        config: dict | None = None,
        agent_config: str | None = None,
        log_dir: str | None = None,
        team_role: str | None = None,
        team_id: str | None = None,
        task_list_url: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        del agent_config, kwargs  # external-agent-config not yet wired
        config = config or {}

        credentials = resolve_credentials()
        if not credentials:
            logger.warning(
                "No Claude Code credentials found (checked ANTHROPIC_API_KEY, "
                "CLAUDE_CODE_OAUTH_TOKEN, and ~/.claude/.credentials.json). "
                "The in-container CLI will fail to authenticate."
            )

        is_coop = bool(messaging_enabled and comm_url and agents and len(agents) > 1)
        use_git = bool(git_enabled and git_server_url and agents and len(agents) > 1)
        is_team = bool(team_role and team_id and task_list_url and agents and len(agents) > 1)

        if is_team:
            instruction = build_team_instruction(
                task,
                agents=agents,
                agent_id=agent_id,
                team_role=team_role,
                git_enabled=use_git,
            )
        else:
            instruction = build_instruction(
                task,
                agents=agents if is_coop else None,
                agent_id=agent_id if is_coop else None,
                git_enabled=use_git,
            )
        setup_script = SETUP_SCRIPT_PATH.read_text()
        coop_msg_source = COOP_MSG_SCRIPT_PATH.read_text() if is_coop else None
        team_task_source = TEAM_TASK_SCRIPT_PATH.read_text() if is_team else None

        coop_env: dict[str, str] = {}
        extra_run_args: list[str] = []
        if is_coop:
            container_url = rewrite_comm_url_for_container(comm_url) or ""
            coop_env = {
                "COOP_REDIS_URL": container_url,
                "COOP_AGENT_ID": agent_id,
                "COOP_AGENTS": ",".join(agents or []),
                "COOP_LOG_PATH": CONTAINER_COOP_SEND_LOG,
            }
            extra_run_args.append("--add-host=host.docker.internal:host-gateway")
        if is_team:
            team_container_url = rewrite_comm_url_for_container(task_list_url) or ""
            coop_env.update(
                build_team_env(
                    redis_url=team_container_url,
                    run_id=team_id or "",
                    agent_id=agent_id,
                    agents=agents or [],
                    team_role=team_role,
                )
            )
            team_volume = (config or {}).get("team_volume") if isinstance(config, dict) else None
            extra_run_args.extend(scratchpad_mount_args(team_volume))
            if "--add-host=host.docker.internal:host-gateway" not in extra_run_args:
                extra_run_args.append("--add-host=host.docker.internal:host-gateway")

        max_turns = config.get("max_turns")
        extra_flags = ""
        if max_turns:
            extra_flags = f"--max-turns {int(max_turns)} "

        network = config.get("git_network") if isinstance(config, dict) else None
        env = _build_environment(image, network=network, extra_run_args=extra_run_args or None)

        status = "Error"
        error_msg: str | None = None
        stream_text = ""
        session_text = ""
        patch_text = ""
        sent_log_text = ""

        try:
            # 1. Drop the coop helper + install snippet (if coop) BEFORE
            #    running setup.sh so setup can create the coop-* wrappers
            #    under /usr/local/bin.  Drop the team helper too if in
            #    team mode; the install snippets are independent.
            if coop_msg_source is not None:
                write_file_in_container(env, CONTAINER_COOP_MSG_PATH, coop_msg_source)
                write_file_in_container(env, "/tmp/cb-coop-install.sh", COOP_INSTALL_SNIPPET_PATH.read_text())
            if team_task_source is not None:
                write_file_in_container(env, CONTAINER_TEAM_TASK_PATH, team_task_source)
                write_file_in_container(env, CONTAINER_TEAM_INSTALL_PATH, TEAM_INSTALL_SNIPPET_PATH.read_text())
                # MCP long-poll server: copy the script + register it in
                # ~/.claude.json so the CLI knows about wait_for_message.
                write_file_in_container(env, CONTAINER_TEAM_MCP_PATH, TEAM_MCP_SCRIPT_PATH.read_text())
                mcp_config = {
                    "mcpServers": {
                        "cooperbench-team": {
                            "type": "stdio",
                            "command": "python3",
                            "args": [CONTAINER_TEAM_MCP_PATH],
                        }
                    }
                }
                env.execute(
                    {"command": f"mkdir -p {shlex.quote(CONTAINER_CLAUDE_CONFIG_DIR)}"},
                    timeout=30,
                )
                write_file_in_container(
                    env,
                    f"{CONTAINER_CLAUDE_CONFIG_DIR}/.claude.json",
                    json.dumps(mcp_config, indent=2),
                )

            # 2. Install claude-code in the container.
            write_file_in_container(env, CONTAINER_SETUP_PATH, setup_script)
            install = env.execute(
                {"command": f"bash {shlex.quote(CONTAINER_SETUP_PATH)}"},
                timeout=600,
            )
            if install.get("returncode") not in (0, None):
                raise RuntimeError("claude-code install failed: " + (install.get("output") or "")[:2000])

            # 3a. Optional: configure the shared git remote so peers can
            #     fetch each other's branches.
            if use_git:
                git_cmd = build_git_setup_command(
                    agent_id=agent_id,
                    server_url=git_server_url or "",
                )
                git_setup = env.execute({"command": git_cmd}, timeout=120)
                if git_setup.get("returncode") not in (0, None):
                    logger.warning(
                        "git setup returned non-zero: %s",
                        (git_setup.get("output") or "")[:500],
                    )

            # 3. Write the instruction to a file and invoke claude.
            write_file_in_container(env, CONTAINER_INSTRUCTION_PATH, instruction)
            cred_exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in credentials.items())
            invoke_cmd = cred_exports + _build_claude_command(
                CONTAINER_INSTRUCTION_PATH,
                model_name,
                CONTAINER_STREAM_LOG,
                extra_flags=extra_flags,
                coop_env=coop_env or None,
            )
            env.execute({"command": invoke_cmd}, timeout=7200)

            # 4. Collect outputs.
            stream_text = read_file_from_container(env, CONTAINER_STREAM_LOG)
            session_text = _find_session_jsonl(env)
            patch_text = normalize_patch(read_file_from_container(env, f"{CONTAINER_REPO_PATH}/patch.txt"))
            if is_coop:
                sent_log_text = read_file_from_container(env, CONTAINER_COOP_SEND_LOG)
        except Exception as e:
            error_msg = str(e)
            logger.exception("Claude Code adapter run failed")
        finally:
            try:
                env.cleanup()
            except Exception:
                logger.warning("Env cleanup failed", exc_info=True)

        summary = parse_stream_json(stream_text)
        messages = parse_session_jsonl(session_text)
        sent_messages = parse_sent_messages_log(sent_log_text)

        if error_msg is not None:
            status = "Error"
        else:
            status = summary.status

        if log_dir:
            try:
                log_root = Path(log_dir)
                log_root.mkdir(parents=True, exist_ok=True)
                (log_root / f"{agent_id}_stream.jsonl").write_text(stream_text)
                (log_root / f"{agent_id}_session.jsonl").write_text(session_text)
                if sent_log_text:
                    (log_root / f"{agent_id}_sent.jsonl").write_text(sent_log_text)
            except OSError:
                logger.warning("Failed to persist Claude Code logs", exc_info=True)

        return AgentResult(
            status=status,
            patch=patch_text,
            cost=summary.cost,
            steps=summary.steps,
            input_tokens=summary.input_tokens,
            output_tokens=summary.output_tokens,
            cache_read_tokens=summary.cache_read_tokens,
            cache_write_tokens=summary.cache_write_tokens,
            messages=messages,
            sent_messages=sent_messages,
            error=error_msg,
        )
