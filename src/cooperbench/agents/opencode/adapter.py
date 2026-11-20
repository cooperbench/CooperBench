"""OpenCode adapter for CooperBench.

Runs ``opencode run`` inside the task's Docker image and harvests the
agent's diff from ``/workspace/repo/patch.txt``.

opencode (https://opencode.ai) is an open-source coding agent CLI.
Installation uses the official curl-based installer; the binary lands
at ``~/.opencode/bin/opencode``.

Custom endpoints (e.g. vLLM serving Qwen3.8-27B) are configured via
environment variables forwarded into the container:

  OPENAI_BASE_URL   OpenAI-compatible API base, e.g.
                    ``http://localhost:8000/v1``  — rewritten to
                    ``host.docker.internal`` automatically so it is
                    reachable from inside the container.
  OPENAI_API_KEY    Key for the OpenAI provider.  Use ``"EMPTY"`` for
                    a local vLLM endpoint that ignores auth.
  ANTHROPIC_API_KEY For ``anthropic/*`` model names.

``config`` keys recognised by this adapter (all optional):

  ``base_url``   Override for the OpenAI-compatible endpoint.
                 ``localhost`` / ``127.0.0.1`` are rewritten
                 automatically.
  ``api_key``    Override for the OpenAI API key.
  ``backend``    ``"docker"`` (default) or ``"podman"``.

Model name convention:  pass as ``openai/<name>`` for OpenAI-compatible
endpoints (including vLLM), or ``anthropic/<name>`` for the Anthropic
API.  Bare names are passed through to opencode unchanged.
"""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path
from typing import Any

from cooperbench.agents import AgentResult
from cooperbench.agents._coop import (
    build_instruction,
    rewrite_comm_url_for_container,
)
from cooperbench.agents._coop.runtime import (
    CONTAINER_INSTRUCTION_PATH,
    CONTAINER_REPO_PATH,
    CONTAINER_SETUP_PATH,
    build_environment,
    normalize_patch,
    read_file_from_container,
    write_file_in_container,
)
from cooperbench.agents.opencode.parsers import parse_run_output
from cooperbench.agents.registry import register

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).parent
SETUP_SCRIPT_PATH = _PACKAGE_DIR / "setup.sh"

CONTAINER_OPENCODE_LOG = "/tmp/opencode-run.log"
OPENCODE_BIN = "$HOME/.opencode/bin/opencode"

# Test-time shim — mirrors the convention used by other adapters.
_build_environment = build_environment


def _resolve_base_url(config: dict[str, Any]) -> str:
    """Return the OpenAI-compatible base URL, rewritten for container access."""
    url = config.get("base_url") or os.environ.get("OPENAI_BASE_URL", "")
    if not url:
        return ""
    return rewrite_comm_url_for_container(url) or url


def _resolve_api_key(model_name: str, config: dict[str, Any]) -> str:
    """Return the API key appropriate for the given model provider."""
    if config.get("api_key"):
        return config["api_key"]
    provider = model_name.split("/")[0].lower() if "/" in model_name else ""
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "")
    # openai provider (including vLLM) — "EMPTY" is accepted by vLLM.
    return os.environ.get("OPENAI_API_KEY", "EMPTY")


def _build_opencode_command(
    model_name: str,
    base_url: str,
    api_key: str,
) -> str:
    """Compose the in-container shell command that invokes ``opencode run``."""
    provider = model_name.split("/")[0].lower() if "/" in model_name else ""

    env_exports = 'export PATH="$HOME/.opencode/bin:$PATH"; '

    if provider == "anthropic" or not base_url:
        if api_key:
            env_exports += f"export ANTHROPIC_API_KEY={shlex.quote(api_key)}; "
    else:
        env_exports += f"export OPENAI_BASE_URL={shlex.quote(base_url)}; "
        env_exports += f"export OPENAI_API_KEY={shlex.quote(api_key or 'EMPTY')}; "

    # Read the instruction from a file to avoid argv-length limits and
    # shell-quoting hazards with multi-line prompts.
    prompt_expr = f'"$(cat {shlex.quote(CONTAINER_INSTRUCTION_PATH)})"'

    return (
        env_exports
        + f"cd {CONTAINER_REPO_PATH} && "
        + f"opencode run --model {shlex.quote(model_name)} {prompt_expr} "
        + f"2>&1 | tee {shlex.quote(CONTAINER_OPENCODE_LOG)}"
    )


def _ensure_patch(env: Any) -> None:
    """Write patch.txt from the git diff if the agent didn't produce one.

    opencode should follow the submission protocol in the instruction
    (``git diff > patch.txt``).  This is a safety net for cases where
    it doesn't — it stages all modified files and captures the full diff
    against HEAD.
    """
    check = env.execute(
        {"command": f"test -s {CONTAINER_REPO_PATH}/patch.txt && echo ok || echo missing"},
        timeout=15,
    )
    if (check.get("output") or "").strip() != "ok":
        env.execute(
            {
                "command": (
                    f"cd {CONTAINER_REPO_PATH} && "
                    "git add -A 2>/dev/null; "
                    "git diff --cached HEAD > patch.txt 2>/dev/null; "
                    "true"
                )
            },
            timeout=30,
        )


@register("opencode")
class OpenCodeRunner:
    """Adapter for the ``opencode run`` CLI.

    Supports solo mode only.  Coop and git-share are not wired yet —
    extend with the ``_coop`` helpers from ``cooperbench.agents._coop``
    if needed.
    """

    def run(
        self,
        task: str,
        image: str,
        *,
        agent_id: str = "agent",
        model_name: str = "openai/gpt-4o",
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
        del agent_config, kwargs
        config = config or {}

        base_url = _resolve_base_url(config)
        api_key = _resolve_api_key(model_name, config)

        extra_run_args: list[str] = []
        if base_url:
            extra_run_args.append("--add-host=host.docker.internal:host-gateway")

        backend = config.get("backend", "docker")
        network = config.get("git_network")
        env = _build_environment(
            image,
            network=network,
            extra_run_args=extra_run_args or None,
            backend=backend,
        )

        instruction = build_instruction(task)
        setup_script = SETUP_SCRIPT_PATH.read_text()
        invoke_cmd = _build_opencode_command(model_name, base_url, api_key)

        status = "Error"
        error_msg: str | None = None
        run_log = ""
        patch_text = ""

        try:
            # 1. Install opencode.
            write_file_in_container(env, CONTAINER_SETUP_PATH, setup_script)
            install = env.execute(
                {"command": f"bash {shlex.quote(CONTAINER_SETUP_PATH)}"},
                timeout=600,
            )
            if install.get("returncode") not in (0, None):
                raise RuntimeError("opencode install failed: " + (install.get("output") or "")[:2000])

            # 2. Write the instruction and run the agent.
            write_file_in_container(env, CONTAINER_INSTRUCTION_PATH, instruction)
            env.execute({"command": invoke_cmd}, timeout=7200)

            # 3. Ensure patch.txt exists (agent should have written it;
            #    fall back to extracting git diff ourselves).
            _ensure_patch(env)

            # 4. Collect outputs.
            run_log = read_file_from_container(env, CONTAINER_OPENCODE_LOG)
            patch_text = normalize_patch(read_file_from_container(env, f"{CONTAINER_REPO_PATH}/patch.txt"))
            status = "Submitted"
        except Exception as e:
            error_msg = str(e)
            logger.exception("OpenCode adapter run failed")
        finally:
            try:
                env.cleanup()
            except Exception:
                logger.warning("Env cleanup failed", exc_info=True)

        summary = parse_run_output(run_log)

        if log_dir:
            try:
                log_root = Path(log_dir)
                log_root.mkdir(parents=True, exist_ok=True)
                (log_root / f"{agent_id}_opencode.log").write_text(run_log)
            except OSError:
                logger.warning("Failed to persist opencode logs", exc_info=True)

        return AgentResult(
            status=status if error_msg is None else "Error",
            patch=patch_text,
            cost=summary.cost,
            steps=summary.steps,
            input_tokens=summary.input_tokens,
            output_tokens=summary.output_tokens,
            error=error_msg,
        )
