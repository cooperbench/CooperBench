"""Claude Code adapter for CooperBench.

Runs the official ``@anthropic-ai/claude-code`` CLI inside the task's
Docker image and harvests the agent's diff from ``/workspace/repo/patch.txt``.

The design mirrors Harbor's adapter (install in container, invoke in
headless ``--print --output-format=stream-json`` mode, parse the final
``result`` event for cost/tokens, walk the session JSONL for messages)
but reuses CooperBench's existing ``DockerEnvironment`` from
``mini_swe_agent_v2`` so the container lifecycle, image pulling, and
``execute()`` semantics are consistent with the other adapters.

Coop support: when ``agents`` has 2+ entries and ``comm_url`` is set,
the adapter copies ``coop_msg.py`` into the container and exposes
``coop-send``/``coop-recv``/``coop-broadcast``/``coop-peek``/``coop-agents``
as shell commands.  Claude Code learns about them from the coop variant
of the prompt template.  Inter-agent messages are still routed through
the host Redis instance — we rewrite ``localhost`` to
``host.docker.internal`` (and pass ``--add-host=host.docker.internal:host-gateway``)
so the container can reach the host daemon on Linux.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from pathlib import Path
from typing import Any, Protocol

from cooperbench.agents import AgentResult
from cooperbench.agents.claude_code.parsers import parse_session_jsonl, parse_stream_json
from cooperbench.agents.claude_code.prompt import build_instruction
from cooperbench.agents.registry import register

logger = logging.getLogger(__name__)


_PACKAGE_DIR = Path(__file__).parent
SETUP_SCRIPT_PATH = _PACKAGE_DIR / "setup.sh"
COOP_MSG_SCRIPT_PATH = _PACKAGE_DIR / "coop_msg.py"

# Inside the container, we redirect Claude Code's per-session state under
# /tmp so we always know where to find the JSONL trajectory after the run.
CONTAINER_CLAUDE_CONFIG_DIR = "/tmp/claude-cfg"
CONTAINER_STREAM_LOG = "/tmp/claude-stream.jsonl"
CONTAINER_SETUP_PATH = "/tmp/claude-setup.sh"
CONTAINER_INSTRUCTION_PATH = "/tmp/claude-instruction.txt"
CONTAINER_COOP_MSG_PATH = "/tmp/claude-coop-msg.py"
CONTAINER_COOP_SEND_LOG = "/tmp/claude-coop-sent.jsonl"
CONTAINER_REPO_PATH = "/workspace/repo"

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


def rewrite_comm_url_for_container(url: str | None) -> str | None:
    """Make a host-side Redis URL reachable from inside the agent container.

    ``localhost`` and ``127.0.0.1`` point at the container itself, not
    the host where the coop runner started Redis.  Substitute
    ``host.docker.internal``, which resolves to the host gateway when
    the container is started with ``--add-host=host.docker.internal:host-gateway``
    (Linux) or natively on Docker Desktop (macOS/Windows).
    """
    if not url:
        return url
    # Use string substitution rather than urlparse to preserve the
    # ``#run:<id>`` fragment that the MessagingConnector relies on.
    for needle in ("//localhost", "//127.0.0.1"):
        if needle in url:
            return url.replace(needle, "//host.docker.internal", 1)
    return url


def build_git_setup_command(*, agent_id: str, server_url: str) -> str:
    """Shell snippet that configures the in-container repo as a participant
    in the shared git remote.

    Mirrors mini_swe_agent_v2's ``GitConnector.setup`` but emitted as a
    single ``bash -lc``-friendly string so it can be exec'd through the
    same ``env.execute`` channel as everything else.  Idempotent: re-running
    is safe (set-url replaces remote if it already exists; branch checkout
    falls back to checkout if it already exists).
    """
    server = shlex.quote(server_url)
    aid = shlex.quote(agent_id)
    branch = shlex.quote(agent_id)
    return (
        f"cd {shlex.quote(CONTAINER_REPO_PATH)} && "
        f"git config user.email 'agent@cooperbench.local' && "
        f"git config user.name {aid} && "
        f"(git remote add team {server} 2>/dev/null || git remote set-url team {server}) && "
        f"(git checkout -b {branch} 2>/dev/null || git checkout {branch}) && "
        f"git push -u team {branch} --force && "
        "git push team HEAD:refs/heads/main --force 2>/dev/null || true"
    )


def parse_sent_messages_log(text: str) -> list[dict[str, Any]]:
    """Parse the in-container coop send-log into a list of message dicts."""
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


class _Env(Protocol):
    """Minimal interface we need from the environment.

    Defined as a Protocol (not imported from mini_swe_agent_v2) so unit
    tests can stub in a tiny fake without instantiating Docker.
    """

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]: ...
    def cleanup(self) -> None: ...


def _build_environment(
    image: str,
    *,
    network: str | None = None,
    extra_run_args: list[str] | None = None,
    timeout: int = 7200,
) -> _Env:
    """Spin up a long-lived Docker container for the run.

    Factored out so tests can monkey-patch this to inject a fake env.
    ``extra_run_args`` are appended to ``docker run`` (e.g. host-gateway
    mapping for coop).
    """
    from cooperbench.agents.mini_swe_agent_v2.environments.docker import DockerEnvironment

    run_args = ["--rm"]
    if extra_run_args:
        run_args.extend(extra_run_args)

    kwargs: dict[str, Any] = {
        "image": image,
        "cwd": CONTAINER_REPO_PATH,
        "timeout": timeout,
        "run_args": run_args,
    }
    if network:
        kwargs["network"] = network
    return DockerEnvironment(**kwargs)


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


def _write_file_in_container(env: _Env, path: str, content: str) -> dict[str, Any]:
    """Write a file inside the container via a heredoc.

    Using a sentinel that's unlikely to appear in either instruction text
    or shell scripts.
    """
    sentinel = "COOPERBENCH_HEREDOC_EOF_5e7b"
    cmd = f"cat > {shlex.quote(path)} <<'{sentinel}'\n{content}\n{sentinel}\n"
    return env.execute({"command": cmd})


def _read_file_from_container(env: _Env, path: str) -> str:
    result = env.execute({"command": f"cat {shlex.quote(path)} 2>/dev/null"})
    if result.get("returncode") == 0:
        return result.get("output") or ""
    return ""


def _find_session_jsonl(env: _Env) -> str:
    """Concatenate every session ``*.jsonl`` produced under CLAUDE_CONFIG_DIR.

    Claude Code writes one file per session; there will normally be
    exactly one for a fresh container.
    """
    cmd = f"find {CONTAINER_CLAUDE_CONFIG_DIR}/projects -name '*.jsonl' -type f 2>/dev/null | xargs -r cat"
    result = env.execute({"command": cmd})
    if result.get("returncode") == 0:
        return result.get("output") or ""
    return ""


@register("claude_code")
class ClaudeCodeRunner:
    """Adapter for the official Claude Code CLI.

    Supports solo and coop runs.  Git collaboration (push/pull/merge via
    a shared remote) is not yet wired — the adapter joins the shared
    docker network if ``git_network`` is set in ``config`` so a follow-up
    can layer on top.
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
        instruction = build_instruction(
            task,
            agents=agents if is_coop else None,
            agent_id=agent_id if is_coop else None,
            git_enabled=use_git,
        )
        setup_script = SETUP_SCRIPT_PATH.read_text()
        coop_msg_source = COOP_MSG_SCRIPT_PATH.read_text() if is_coop else None

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
            # 1. Drop the coop helper (if needed) BEFORE running setup.sh
            #    so setup can symlink the coop-* wrappers under /usr/local/bin.
            if coop_msg_source is not None:
                _write_file_in_container(env, CONTAINER_COOP_MSG_PATH, coop_msg_source)

            # 2. Install claude-code in the container.
            _write_file_in_container(env, CONTAINER_SETUP_PATH, setup_script)
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
            _write_file_in_container(env, CONTAINER_INSTRUCTION_PATH, instruction)
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
            stream_text = _read_file_from_container(env, CONTAINER_STREAM_LOG)
            session_text = _find_session_jsonl(env)
            patch_text = _read_file_from_container(env, f"{CONTAINER_REPO_PATH}/patch.txt").strip()
            if is_coop:
                sent_log_text = _read_file_from_container(env, CONTAINER_COOP_SEND_LOG)
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
