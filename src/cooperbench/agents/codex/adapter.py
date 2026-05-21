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

Azure OpenAI: set ``AZURE_OPENAI_API_KEY`` + ``AZURE_OPENAI_ENDPOINT``
(the OpenAI-compatible v1 base, e.g.
``https://<resource>.cognitiveservices.azure.com/openai/v1``) and pass
the Azure *deployment* name via ``-m``.  When both are present they take
precedence over ``OPENAI_API_KEY``: a custom ``model_provider`` is
written into ``config.toml`` and the key is read from the env var.
Azure runs use codex's plain output (not ``--json``) — codex 0.132's
JSONL event stream mishandles Azure's HTTP/2 ``/responses`` endpoint —
so token/cost telemetry is unavailable on that path; the patch is still
harvested from ``patch.txt``.
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
from cooperbench.agents.codex.parsers import parse_messages, parse_stream_jsonl
from cooperbench.agents.registry import register
from cooperbench.team_harness import (
    COOP_TASK_SCRIPT_PATH as TEAM_TASK_SCRIPT_PATH,
)
from cooperbench.team_harness import (
    INSTALL_SNIPPET_PATH as TEAM_INSTALL_SNIPPET_PATH,
)
from cooperbench.team_harness import (
    MCP_SERVER_NAME,
    TeamHarnessConfig,
    TeamSession,
)
from cooperbench.team_harness import (
    MCP_SERVER_SCRIPT_PATH as TEAM_MCP_SCRIPT_PATH,
)

logger = logging.getLogger(__name__)


_PACKAGE_DIR = Path(__file__).parent
SETUP_SCRIPT_PATH = _PACKAGE_DIR / "setup.sh"
COOP_MSG_SCRIPT_PATH = _PACKAGE_DIR.parent / "_coop" / "coop_msg.py"
COOP_INSTALL_SNIPPET_PATH = _PACKAGE_DIR.parent / "_coop" / "install_snippet.sh"
CONTAINER_TEAM_TASK_PATH = "/tmp/cb-coop-task.py"
CONTAINER_TEAM_INSTALL_PATH = "/tmp/cb-team-install.sh"
CONTAINER_TEAM_MCP_PATH = "/tmp/cb-mcp-server.py"

CONTAINER_CODEX_HOME = "/tmp/codex-home"
CONTAINER_AUTH_PATH = f"{CONTAINER_CODEX_HOME}/auth.json"
CONTAINER_STREAM_LOG = "/tmp/codex-stream.jsonl"
# Azure path runs codex without --json (its JSONL event stream hits a
# codex/HTTP2 bug against Azure), so we capture the final assistant
# message here via --output-last-message instead of parsing the stream.
CONTAINER_LAST_MSG = "/tmp/codex-last-message.txt"

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


def resolve_azure_config() -> dict[str, str] | None:
    """Azure OpenAI provider config from the host environment.

    Returns ``{"api_key": ..., "endpoint": ...}`` when both
    ``AZURE_OPENAI_API_KEY`` and ``AZURE_OPENAI_ENDPOINT`` are set, else
    ``None``.  When set, Azure takes precedence over the plain
    ``OPENAI_API_KEY`` path: codex is pointed at the Azure deployment via
    a custom ``model_provider`` in ``config.toml`` (see
    ``_azure_config_toml``).

    ``AZURE_OPENAI_ENDPOINT`` must be the OpenAI-compatible v1 base, e.g.
    ``https://<resource>.cognitiveservices.azure.com/openai/v1`` (codex
    appends ``/responses``).  The model name passed via ``-m`` is the
    Azure *deployment* name (e.g. ``gpt-5.5-hao``).
    """
    key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if key and endpoint:
        return {"api_key": key, "endpoint": endpoint.rstrip("/")}
    return None


def _azure_config_toml(endpoint: str) -> str:
    """Render the ``config.toml`` fragment that points codex at Azure.

    codex 0.132+ only speaks the Responses wire API (``chat`` was
    removed), and Azure's ``/openai/v1`` surface supports it, so we use
    ``wire_api = "responses"``.  The key is read from the
    ``AZURE_OPENAI_API_KEY`` env var (exported into the codex command),
    not from ``auth.json``.
    """
    return (
        'model_provider = "azure"\n\n'
        "[model_providers.azure]\n"
        'name = "Azure OpenAI"\n'
        f'base_url = "{endpoint}"\n'
        'env_key = "AZURE_OPENAI_API_KEY"\n'
        'wire_api = "responses"\n'
    )


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
    json_output: bool = True,
    last_message_path: str | None = None,
) -> str:
    """Compose the in-container shell command that invokes ``codex exec``.

    Reads the prompt from a file so we don't have to shell-escape the
    whole instruction.  Tees stdout so we can read it back post-run.

    ``json_output`` controls ``--json`` (the JSONL event stream the
    parser consumes).  It is forced off for the Azure provider: codex
    0.132's ``--json`` path mishandles Azure's HTTP/2 ``/responses``
    stream and dies with "stream disconnected", while plain (human)
    output works.  In that mode pass ``last_message_path`` so codex
    writes the final assistant message via ``--output-last-message`` —
    the patch itself is still harvested from ``patch.txt``.
    """
    coop_exports = ""
    if coop_env:
        coop_exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in coop_env.items())

    model_flag = ""
    if model_name:
        model_flag = f"--model {shlex.quote(_strip_provider_prefix(model_name))} "

    json_flag = "--json " if json_output else ""
    last_msg_flag = ""
    if last_message_path:
        last_msg_flag = f"--output-last-message {shlex.quote(last_message_path)} "

    # IMPORTANT: redirect stdin from /dev/null.  Codex's `exec` mode otherwise
    # prints "Reading additional input from stdin..." and blocks indefinitely
    # if stdin is open but no EOF arrives.  Docker's `docker exec` (non-tty)
    # gets that for free, but Modal sandbox `exec` keeps stdin open, which
    # silently hangs codex for the full sandbox lifetime (~2h) producing zero
    # output.  </dev/null gives codex an immediate EOF so it falls back to
    # the positional prompt.
    return (
        'export PATH="$HOME/.local/bin:$PATH"; '
        f"export CODEX_HOME={shlex.quote(auth_dir)}; " + coop_exports + f"cd {shlex.quote(CONTAINER_REPO_PATH)} && "
        "codex exec "
        "--sandbox danger-full-access "
        "--skip-git-repo-check "
        f"{model_flag}"
        f"{json_flag}"
        f"{last_msg_flag}"
        f'-- "$(cat {shlex.quote(instruction_path)})" '
        f"</dev/null 2>&1 | tee {shlex.quote(stream_log_path)}"
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
        team_features: TeamHarnessConfig | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        del agent_config, kwargs  # external-agent-config not yet wired
        config = config or {}

        credentials = resolve_credentials()
        azure = resolve_azure_config()
        if not azure and not credentials:
            # Fail fast: no point spinning up a container when we know
            # codex will reject every request.
            logger.error(
                "No codex credentials in host environment; skipping run. "
                "Set OPENAI_API_KEY, or AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT for Azure."
            )
            return AgentResult(
                status="Error",
                patch="",
                cost=0.0,
                steps=0,
                error="no codex credentials (OPENAI_API_KEY or AZURE_OPENAI_API_KEY+AZURE_OPENAI_ENDPOINT) set",
            )

        is_coop = bool(messaging_enabled and comm_url and agents and len(agents) > 1)
        use_git = bool(git_enabled and git_server_url and agents and len(agents) > 1)
        is_team = bool(team_role and team_id and task_list_url and agents and len(agents) > 1)

        team_session: TeamSession | None = None
        if is_team:
            # Pass the host URL — TeamSession.env_for() rewrites to the
            # container-reachable form when assembling CB_TEAM_* env vars.
            team_session = TeamSession(
                run_id=team_id or "",
                redis_url=task_list_url or "",
                agents=list(agents or []),
                team_volume=str((config or {}).get("team_volume") or ""),
                config=team_features or TeamHarnessConfig(),
            )

        if team_session is not None:
            instruction = team_session.prompt_for(
                task=task,
                agent_id=agent_id,
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
        install_team_cli = bool(team_session and (team_session.config.task_list or team_session.config.protocol))
        team_task_source = TEAM_TASK_SCRIPT_PATH.read_text() if install_team_cli else None

        coop_env: dict[str, str] = {}
        extra_run_args: list[str] = []
        if azure:
            # codex reads the Azure key from this env var (env_key in the
            # provider block); exported into the codex command below.
            coop_env["AZURE_OPENAI_API_KEY"] = azure["api_key"]
        if is_coop:
            container_url = rewrite_comm_url_for_container(comm_url) or ""
            # NB: update (not reassign) — reassigning would wipe the Azure
            # key added above, breaking codex's provider auth in coop/team.
            coop_env.update(
                {
                    "COOP_REDIS_URL": container_url,
                    "COOP_AGENT_ID": agent_id,
                    "COOP_AGENTS": ",".join(agents or []),
                    "COOP_LOG_PATH": CONTAINER_COOP_SEND_LOG,
                }
            )
            extra_run_args.append("--add-host=host.docker.internal:host-gateway")
        if team_session is not None:
            coop_env.update(team_session.env_for(agent_id))
            extra_run_args.extend(team_session.scratchpad_mount_args())
            if "--add-host=host.docker.internal:host-gateway" not in extra_run_args:
                extra_run_args.append("--add-host=host.docker.internal:host-gateway")

        network = config.get("git_network") if isinstance(config, dict) else None
        backend = config.get("backend", "docker") if isinstance(config, dict) else "docker"
        env = _build_environment(
            image,
            network=network,
            extra_run_args=extra_run_args or None,
            backend=backend,
        )

        status = "Error"
        error_msg: str | None = None
        stream_text = ""
        patch_text = ""
        sent_log_text = ""
        azure_last_message = ""
        codex_returncode: int | None = None
        use_json = not azure  # Azure runs codex in plain (non-JSON) mode

        try:
            # 1. Drop coop helper + install snippet (if coop) before setup.
            #    Drop team helper too if in team mode.
            if coop_msg_source is not None:
                write_file_in_container(env, CONTAINER_COOP_MSG_PATH, coop_msg_source)
                write_file_in_container(env, "/tmp/cb-coop-install.sh", COOP_INSTALL_SNIPPET_PATH.read_text())
            if team_task_source is not None:
                write_file_in_container(env, CONTAINER_TEAM_TASK_PATH, team_task_source)
                write_file_in_container(env, CONTAINER_TEAM_INSTALL_PATH, TEAM_INSTALL_SNIPPET_PATH.read_text())
            # Compose codex's config.toml from independent fragments:
            #   - Azure provider block (when AZURE_OPENAI_* is set)
            #   - team-mode MCP long-poll server entry
            # Both share one file, so build the parts then write once.
            install_mcp = team_session is not None and team_session.config.mcp
            toml_parts: list[str] = []
            if azure:
                toml_parts.append(_azure_config_toml(azure["endpoint"]))
            if install_mcp:
                write_file_in_container(env, CONTAINER_TEAM_MCP_PATH, TEAM_MCP_SCRIPT_PATH.read_text())
                # Codex's MCP config lives in config.toml.
                toml_parts.append(
                    f'[mcp_servers.{MCP_SERVER_NAME}]\ncommand = "python3"\nargs = ["{CONTAINER_TEAM_MCP_PATH}"]\n'
                )
            if toml_parts:
                env.execute(
                    {"command": f"mkdir -p {shlex.quote(CONTAINER_CODEX_HOME)}"},
                    timeout=30,
                )
                write_file_in_container(env, f"{CONTAINER_CODEX_HOME}/config.toml", "\n".join(toml_parts))

            # 2. Install codex in the container.
            write_file_in_container(env, CONTAINER_SETUP_PATH, setup_script)
            install = env.execute(
                {"command": f"bash {shlex.quote(CONTAINER_SETUP_PATH)}"},
                timeout=600,
            )
            if install.get("returncode") not in (0, None):
                raise RuntimeError("codex install failed: " + (install.get("output") or "")[:2000])

            # 2b. Write the auth file so codex can authenticate.  Skipped
            #     for Azure — that path authenticates via the
            #     AZURE_OPENAI_API_KEY env var (provider env_key), not auth.json.
            if not azure and credentials.get("OPENAI_API_KEY"):
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

            # Azure runs without --json (codex's JSONL stream is broken
            # against Azure's HTTP/2 endpoint); capture the final message
            # via --output-last-message instead.
            last_message_path = None if use_json else CONTAINER_LAST_MSG

            # First attempt: with --model gpt-5.5 (or whatever user passed).
            invoke_cmd = _build_codex_command(
                CONTAINER_INSTRUCTION_PATH,
                model_name=model_name,
                stream_log_path=CONTAINER_STREAM_LOG,
                auth_dir=CONTAINER_CODEX_HOME,
                coop_env=coop_env or None,
                json_output=use_json,
                last_message_path=last_message_path,
            )
            invoke_result = env.execute({"command": invoke_cmd}, timeout=7200)
            codex_returncode = invoke_result.get("returncode")
            stream_text = read_file_from_container(env, CONTAINER_STREAM_LOG)

            # Model-name fallback only applies in JSON mode (it needs the
            # parsed error).  Azure passes a fixed deployment name, so no
            # fallback there.
            if use_json:
                summary = parse_stream_jsonl(stream_text)
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
            if not use_json:
                # Best-effort final assistant message (file is absent if
                # codex never produced one).
                azure_last_message = read_file_from_container(env, CONTAINER_LAST_MSG)
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

        sent_messages = parse_sent_messages_log(sent_log_text)

        if use_json:
            # OpenAI path: rich JSONL parse (status, tokens, messages).
            summary = parse_stream_jsonl(stream_text)
            messages = parse_messages(stream_text)
            cost = summary.cost
            steps = summary.steps
            input_tokens = summary.input_tokens
            output_tokens = summary.output_tokens
            cache_read_tokens = summary.cache_read_tokens
            cache_write_tokens = summary.cache_write_tokens
            if error_msg is not None:
                status = "Error"
            else:
                status = summary.status
                # Treat "no creds" as an explicit error rather than swallowing it.
                if status == "Error" and not credentials and not azure:
                    error_msg = "OPENAI_API_KEY missing"
        else:
            # Azure path: no JSONL stream.  Derive status from codex's exit
            # code, surface the final message, and leave token/cost counts
            # at 0 (codex's plain output carries no usage data).
            cost = 0.0
            steps = 0
            input_tokens = output_tokens = cache_read_tokens = cache_write_tokens = 0
            messages = [{"role": "assistant", "content": azure_last_message}] if azure_last_message.strip() else []
            if error_msg is not None:
                status = "Error"
            elif codex_returncode in (0, None):
                status = "Submitted"
            else:
                status = "Error"
                error_msg = f"codex exited with code {codex_returncode}"

        if log_dir:
            try:
                log_root = Path(log_dir)
                log_root.mkdir(parents=True, exist_ok=True)
                suffix = "jsonl" if use_json else "log"
                (log_root / f"{agent_id}_stream.{suffix}").write_text(stream_text)
                if sent_log_text:
                    (log_root / f"{agent_id}_sent.jsonl").write_text(sent_log_text)
            except OSError:
                logger.warning("Failed to persist Codex logs", exc_info=True)

        return AgentResult(
            status=status,
            patch=patch_text,
            cost=cost,
            steps=steps,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            messages=messages,
            sent_messages=sent_messages,
            error=error_msg,
        )
