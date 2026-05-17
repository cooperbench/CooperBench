"""Stdio MCP server that exposes a ``wait_for_message`` long-poll tool.

The CLI adapters (Claude Code, Codex) run an opaque LLM loop, so we
can't auto-inject inbox messages between turns the way the Python-loop
adapters do.  The least-bad alternative is to give them a tool that
*blocks server-side* until a message arrives — i.e. ``BLPOP`` on the
Redis inbox — so "watch the inbox" becomes a natural idle behavior
rather than a busy-loop on ``coop-recv`` returning empty.

We speak JSON-RPC 2.0 over stdio per the MCP spec, supporting the
minimal surface:

  - ``initialize``
  - ``tools/list``
  - ``tools/call``        (only "wait_for_message" today)

Adapters register us via their CLI's config:
``~/.claude.json`` for Claude Code and ``~/.codex/config.toml`` for
Codex.  The actual server invocation is just
``python3 -m cooperbench.agents._team.mcp_server`` with the
``CB_TEAM_REDIS_URL`` / ``CB_TEAM_AGENT_ID`` env vars set.

The server is intentionally minimal — adding more tools later is just
another branch in ``_dispatch_tool``.  We don't implement ``logging/*``,
prompts, resources, or subscriptions; the CLI doesn't need them.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Protocol

import redis

PROTOCOL_VERSION = "2025-06-18"


class _RedisLike(Protocol):
    def blpop(self, keys, timeout: float = 0) -> Any: ...
    def rpush(self, name: str, *values: Any) -> int: ...


def build_redis_client_from_env() -> _RedisLike:
    url = os.environ.get("CB_TEAM_REDIS_URL")
    if not url:
        raise RuntimeError("CB_TEAM_REDIS_URL is not set")
    if "#" in url:
        url, _ = url.split("#", 1)
    return redis.from_url(url)


class MCPLongPollServer:
    """In-process MCP server.  ``handle(request)`` returns a response.

    For the real stdio loop, use ``serve_stdio()``.  Tests should call
    ``handle()`` directly with synthesized request dicts.
    """

    def __init__(self, redis_client: _RedisLike, agent_id: str) -> None:
        self._r = redis_client
        self._me = agent_id

    # --- JSON-RPC dispatch --------------------------------------------

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        method = request.get("method", "")
        req_id = request.get("id")

        if method == "initialize":
            return self._wrap(req_id, self._initialize(request.get("params") or {}))
        if method == "tools/list":
            return self._wrap(req_id, self._tools_list())
        if method == "tools/call":
            params = request.get("params") or {}
            try:
                result = self._dispatch_tool(params)
            except _ToolError as e:
                return self._error(req_id, code=-32601, message=str(e))
            return self._wrap(req_id, result)
        if method == "notifications/initialized":
            # Client says it's ready; nothing to send back for a notification.
            # Returning an empty dict is harmless — the stdio loop won't write it.
            return {}
        return self._error(req_id, code=-32601, message=f"unknown method {method!r}")

    # --- responses ----------------------------------------------------

    def _wrap(self, req_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _error(self, req_id: Any, *, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    # --- method impls -------------------------------------------------

    def _initialize(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "cooperbench-team", "version": "0.1.0"},
        }

    def _tools_list(self) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "name": "wait_for_message",
                    "description": (
                        "Block server-side until a message arrives in your team inbox, "
                        "or until timeout_ms elapses.  Returns the message text on hit, "
                        "or a 'no message' marker on timeout.  Prefer this to busy-polling "
                        "coop-recv when you're idle between substantive actions."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "timeout_ms": {
                                "type": "integer",
                                "minimum": 100,
                                "maximum": 30000,
                                "default": 5000,
                                "description": "How long to block before giving up.",
                            }
                        },
                    },
                }
            ]
        }

    def _dispatch_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "wait_for_message":
            return self._wait_for_message(args)
        raise _ToolError(f"unknown tool {name!r}")

    def _wait_for_message(self, args: dict[str, Any]) -> dict[str, Any]:
        timeout_ms = int(args.get("timeout_ms", 5000))
        # BLPOP timeout is in seconds (float); clamp to a sane lower
        # bound so we don't accidentally pass 0 (= block forever).
        timeout_s = max(0.1, timeout_ms / 1000.0)
        inbox_key = f"{self._me}:inbox"
        result = self._r.blpop([inbox_key], timeout=timeout_s)
        if result is None:
            return {
                "content": [{"type": "text", "text": "(no message arrived within the timeout)"}],
                "isError": False,
            }
        _key, raw = result
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        return {
            "content": [{"type": "text", "text": text}],
            "isError": False,
        }


class _ToolError(Exception):
    pass


def serve_stdio() -> None:  # pragma: no cover -- exercised in e2e
    """Run the JSON-RPC stdio loop.

    Each line of stdin is one request; each response is one line of
    stdout.  Notifications produce no output.
    """
    server = MCPLongPollServer(
        redis_client=build_redis_client_from_env(),
        agent_id=os.environ.get("CB_TEAM_AGENT_ID", "agent"),
    )
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            continue
        resp = server.handle(req)
        if not resp:
            continue
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    serve_stdio()
