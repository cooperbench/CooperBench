"""OpenAI Codex CLI adapter for CooperBench.

Runs ``codex exec --json`` inside the task's Docker image.  Mirrors the
Claude Code adapter's shape: install in container, write the prompt to
a file, invoke with ``--sandbox danger-full-access``, harvest the diff
from ``/workspace/repo/patch.txt`` and the trajectory from the JSONL
stream.

Coop + git: reuses ``cooperbench.agents._coop`` (messaging helpers,
prompt blocks, git remote setup).  Same flavors as Claude Code: solo,
coop (Redis), coop + git (shared ``team`` remote).

Auth: ``OPENAI_API_KEY`` from the host environment is written into
``${CODEX_HOME}/auth.json`` inside the container (the file Codex reads
at startup).
"""

from __future__ import annotations

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
    build_environment,
    normalize_patch,
    read_file_from_container,
    write_file_in_container,
)
from cooperbench.agents._team import build_team_env, build_team_instruction, scratchpad_mount_args
from cooperbench.agents.codex.parsers import parse_messages, parse_stream_jsonl
from cooperbench.agents.registry import register

logger = logging.getLogger(__name__)


_PACKAGE_DIR = Path(__file__).parent
SETUP_SCRIPT_PATH = _PACKAGE_DIR / "setup.sh"
COOP_MSG_SCRIPT_PATH = _PACKAGE_DIR.parent / "_coop" / "coop_msg.py"
COOP_INSTALL_SNIPPET_PATH = _PACKAGE_DIR.parent / "_coop" / "install_snippet.sh"
TEAM_TASK_SCRIPT_PATH = _PACKAGE_DIR.parent / "_team" / "coop_task.py"
TEAM_INSTALL_SNIPPET_PATH = _PACKAGE_DIR.parent / "_team" / "install_snippet.sh"
CONTAINER_TEAM_TASK_PATH = "/tmp/cb-coop-task.py"
CONTAINER_TEAM_INSTALL_PATH = "/tmp/cb-team-install.sh"

CONTAINER_CODEX_HOME = "/tmp/codex-home"
CONTAINER_AUTH_PATH = f"{CONTAINER_CODEX_HOME}/auth.json"
CONTAINER_STREAM_LOG = "/tmp/codex-stream.jsonl"

# Test-time shim: tests monkey-patch this for fake-env injection.
_build_environment = build_environment


def resolve_credentials() -> dict[str, str]:
    """Pick the OpenAI credential to forward into the container.

    Only ``OPENAI_API_KEY`` is supported today.  Codex doesn't have a
    Claude-style OAuth login flow that produces a long-lived token.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        return {"OPENAI_API_KEY": api_key}
    return {}


def _strip_provider_prefix(model_name: str) -> str:
    """``openai/gpt-5.5`` -> ``gpt-5.5``.  Codex doesn't understand
    arbitrary provider prefixes, so strip a leading ``openai/`` (or any
    other ``foo/``) before passing to ``--model``."""
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def _build_codex_command(
    instruction_path: str,
    *,
    model_name: str | None,
    stream_log_path: str,
    auth_dir: str,
    coop_env: dict[str, str] | None = None,
) -> str:
    """Compose the in-container shell command that invokes ``codex exec``.

    Reads the prompt from a file so we don't have to shell-escape the
    whole instruction.  Tees stdout (the JSONL stream) so we can read it
    back post-run.
    """
    coop_exports = ""
    if coop_env:
        coop_exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in coop_env.items())

    model_flag = ""
    if model_name:
        model_flag = f"--model {shlex.quote(_strip_provider_prefix(model_name))} "

    return (
        'export PATH="$HOME/.local/bin:$PATH"; '
        f"export CODEX_HOME={shlex.quote(auth_dir)}; " + coop_exports + f"cd {shlex.quote(CONTAINER_REPO_PATH)} && "
        "codex exec "
        "--sandbox danger-full-access "
        "--skip-git-repo-check "
        f"{model_flag}"
        "--json "
        f'-- "$(cat {shlex.quote(instruction_path)})" '
        f"2>&1 | tee {shlex.quote(stream_log_path)}"
    )


def _write_auth_file(env, api_key: str) -> None:
    """Write ``${CODEX_HOME}/auth.json`` inside the container.

    We use shell heredoc rather than ``write_file_in_container`` because
    the file lives under a directory we have to create first.
    """
    content = '{"OPENAI_API_KEY": "' + api_key.replace('"', '\\"') + '"}'
    cmd = (
        f"mkdir -p {shlex.quote(CONTAINER_CODEX_HOME)} && "
        f"cat > {shlex.quote(CONTAINER_AUTH_PATH)} <<'AUTH_EOF'\n{content}\nAUTH_EOF\n"
    )
    env.execute({"command": cmd})


@register("codex")
class CodexRunner:
    """Adapter for OpenAI's Codex CLI (``codex exec``)."""

    def run(
        self,
        task: str,
        image: str,
        *,
        agent_id: str = "agent",
        model_name: str = "gpt-5.5",
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
            # Fail fast: no point spinning up a container when we know
            # codex will reject every request.
            logger.error("OPENAI_API_KEY is not set in the host environment; skipping codex run.")
            return AgentResult(
                status="Error",
                patch="",
                cost=0.0,
                steps=0,
                error="OPENAI_API_KEY not set in host environment",
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

        network = config.get("git_network") if isinstance(config, dict) else None
        env = _build_environment(image, network=network, extra_run_args=extra_run_args or None)

        status = "Error"
        error_msg: str | None = None
        stream_text = ""
        patch_text = ""
        sent_log_text = ""

        try:
            # 1. Drop coop helper + install snippet (if coop) before setup.
            #    Drop team helper too if in team mode.
            if coop_msg_source is not None:
                write_file_in_container(env, CONTAINER_COOP_MSG_PATH, coop_msg_source)
                write_file_in_container(env, "/tmp/cb-coop-install.sh", COOP_INSTALL_SNIPPET_PATH.read_text())
            if team_task_source is not None:
                write_file_in_container(env, CONTAINER_TEAM_TASK_PATH, team_task_source)
                write_file_in_container(env, CONTAINER_TEAM_INSTALL_PATH, TEAM_INSTALL_SNIPPET_PATH.read_text())

            # 2. Install codex in the container.
            write_file_in_container(env, CONTAINER_SETUP_PATH, setup_script)
            install = env.execute(
                {"command": f"bash {shlex.quote(CONTAINER_SETUP_PATH)}"},
                timeout=600,
            )
            if install.get("returncode") not in (0, None):
                raise RuntimeError("codex install failed: " + (install.get("output") or "")[:2000])

            # 2b. Write the auth file so codex can authenticate.
            if credentials.get("OPENAI_API_KEY"):
                _write_auth_file(env, credentials["OPENAI_API_KEY"])

            # 3a. Optional: git remote setup so peers can fetch each other.
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

            # 3. Write the instruction to a file and invoke codex.
            write_file_in_container(env, CONTAINER_INSTRUCTION_PATH, instruction)

            # First attempt: with --model gpt-5.5 (or whatever user passed).
            invoke_cmd = _build_codex_command(
                CONTAINER_INSTRUCTION_PATH,
                model_name=model_name,
                stream_log_path=CONTAINER_STREAM_LOG,
                auth_dir=CONTAINER_CODEX_HOME,
                coop_env=coop_env or None,
            )
            env.execute({"command": invoke_cmd}, timeout=7200)
            stream_text = read_file_from_container(env, CONTAINER_STREAM_LOG)
            summary = parse_stream_jsonl(stream_text)

            # Fallback: if the requested model isn't available, drop the
            # --model flag and let Codex pick its default.
            if summary.is_model_error:
                logger.warning(
                    "Codex rejected model '%s' (%s); retrying without --model",
                    model_name,
                    (summary.raw_result.get("message") or "")[:200],
                )
                invoke_cmd = _build_codex_command(
                    CONTAINER_INSTRUCTION_PATH,
                    model_name=None,
                    stream_log_path=CONTAINER_STREAM_LOG,
                    auth_dir=CONTAINER_CODEX_HOME,
                    coop_env=coop_env or None,
                )
                env.execute({"command": invoke_cmd}, timeout=7200)
                stream_text = read_file_from_container(env, CONTAINER_STREAM_LOG)

            # 4. Collect outputs.
            patch_text = normalize_patch(read_file_from_container(env, f"{CONTAINER_REPO_PATH}/patch.txt"))
            if is_coop:
                sent_log_text = read_file_from_container(env, CONTAINER_COOP_SEND_LOG)
        except Exception as e:
            error_msg = str(e)
            logger.exception("Codex adapter run failed")
        finally:
            try:
                env.cleanup()
            except Exception:
                logger.warning("Env cleanup failed", exc_info=True)

        summary = parse_stream_jsonl(stream_text)
        messages = parse_messages(stream_text)
        sent_messages = parse_sent_messages_log(sent_log_text)

        if error_msg is not None:
            status = "Error"
        else:
            status = summary.status
            # Treat "no creds" as an explicit error rather than swallowing it.
            if status == "Error" and not credentials:
                error_msg = "OPENAI_API_KEY missing"

        if log_dir:
            try:
                log_root = Path(log_dir)
                log_root.mkdir(parents=True, exist_ok=True)
                (log_root / f"{agent_id}_stream.jsonl").write_text(stream_text)
                if sent_log_text:
                    (log_root / f"{agent_id}_sent.jsonl").write_text(sent_log_text)
            except OSError:
                logger.warning("Failed to persist Codex logs", exc_info=True)

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
