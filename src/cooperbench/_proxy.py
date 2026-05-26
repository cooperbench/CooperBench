"""Auto-spawn a LiteLLM proxy for the lifetime of a CooperBench run.

The ``claude_code`` adapter drives the official ``@anthropic-ai/claude-code``
CLI, which only speaks Anthropic's ``/v1/messages``.  When the upstream
model is served on an OpenAI-compatible endpoint (vLLM, llama.cpp, ...),
we need a translation proxy in between.  Users used to have to start
LiteLLM themselves in another terminal; this module bundles it so a
single ``cooperbench run --openai-base-url ...`` is enough.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

logger = logging.getLogger(__name__)

PROXY_STARTUP_TIMEOUT_SECONDS = 60
PROXY_HEALTH_PATH = "/health/liveliness"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600
DEFAULT_AUTH_TOKEN = "sk-cooperbench-managed"


def _find_free_port() -> int:
    """Bind to port 0 to let the OS pick a free port, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, deadline: float) -> None:
    """Poll the LiteLLM health endpoint until 200 OK or the deadline passes."""
    url = base_url + PROXY_HEALTH_PATH
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = exc
        time.sleep(0.5)
    raise RuntimeError(
        f"LiteLLM proxy did not become healthy at {url} within "
        f"{PROXY_STARTUP_TIMEOUT_SECONDS}s (last error: {last_err!r})"
    )


@contextlib.contextmanager
def managed_litellm(
    *,
    openai_base_url: str,
    openai_model: str,
    api_key: str = "dummy",
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    auth_token: str = DEFAULT_AUTH_TOKEN,
) -> Iterator[tuple[str, str]]:
    """Spawn a LiteLLM proxy, yield ``(base_url, auth_token)``, tear down on exit.

    ``openai_model`` is the upstream model name on the OpenAI endpoint
    (e.g. ``Qwen/Qwen3.5-9B``).  The proxy translates between Anthropic's
    ``/v1/messages`` and OpenAI's ``/v1/chat/completions`` and forwards
    to ``openai_base_url`` (e.g. ``https://...modal.run/v1``).

    The yielded ``base_url`` is ``http://localhost:<port>``.  ``auth_token``
    is the master key the proxy expects on inbound requests — currently
    just the placeholder, since the proxy is local and short-lived.
    """
    litellm_bin = shutil.which("litellm")
    if litellm_bin is None:
        raise RuntimeError(
            "litellm CLI not found on PATH.  Install with "
            "`pip install 'litellm[proxy]'` (or `pip install cooperbench[proxy]` "
            "once that extra is published)."
        )

    port = _find_free_port()
    base_url = f"http://localhost:{port}"
    # LiteLLM's ``openai/<model>`` provider prefix tells it to forward as
    # OpenAI-format to ``api_base``.  ``--drop_params`` makes it tolerant
    # of provider-specific kwargs that the upstream doesn't accept.
    # The upstream API key is passed via ``OPENAI_API_KEY`` in the child
    # env (LiteLLM CLI has no inline ``--api_key`` flag).
    cmd = [
        litellm_bin,
        "--model",
        f"openai/{openai_model}",
        "--api_base",
        openai_base_url,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--request_timeout",
        str(int(request_timeout)),
        "--drop_params",
    ]
    child_env = {**os.environ, "OPENAI_API_KEY": api_key}

    logger.info("Spawning LiteLLM proxy on %s -> %s (%s)", base_url, openai_base_url, openai_model)
    proc = subprocess.Popen(
        cmd,
        env=child_env,
        stdout=sys.stderr,
        stderr=sys.stderr,
        # New process group so a Ctrl-C on the parent doesn't double-kill
        # the proxy mid-tear-down.
        start_new_session=True,
    )

    try:
        deadline = time.monotonic() + PROXY_STARTUP_TIMEOUT_SECONDS
        _wait_for_health(base_url, deadline)
        logger.info("LiteLLM proxy healthy on %s", base_url)
        yield base_url, auth_token
    finally:
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("LiteLLM did not exit on SIGTERM; killing")
                proc.kill()
                proc.wait(timeout=5)
