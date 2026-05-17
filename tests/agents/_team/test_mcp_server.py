"""Unit tests for the MCP long-poll server.

CLI adapters (Claude Code, Codex) run an opaque LLM loop, so we can't
inject inbox messages between their turns the way Python-loop adapters
can.  The next-best thing is to give them a long-poll tool that
*blocks server-side* until a message arrives, so "watch the inbox"
becomes a natural idle behavior instead of a busy-loop on
``coop-recv``.

The server speaks the MCP JSON-RPC handshake over stdio.  Only three
methods matter for our use:

  - ``initialize``        — handshake; server reports capabilities.
  - ``tools/list``        — advertises the ``wait_for_message`` tool.
  - ``tools/call``        — executes the tool (BLPOPs the inbox).

These tests drive the server in-process by feeding it ``dict`` requests
and asserting on the response dicts.  Real Claude Code / Codex
integration is covered by the end-to-end run.
"""

from __future__ import annotations

import threading
import time

import fakeredis
import pytest

from cooperbench.agents._team.mcp_server import (
    MCPLongPollServer,
    build_redis_client_from_env,
)


@pytest.fixture
def shared_redis():
    return fakeredis.FakeRedis()


@pytest.fixture
def server(shared_redis, monkeypatch):
    monkeypatch.setenv("CB_TEAM_REDIS_URL", "redis://stub")
    monkeypatch.setenv("CB_TEAM_AGENT_ID", "agent1")
    return MCPLongPollServer(redis_client=shared_redis, agent_id="agent1")


class TestInitialize:
    def test_initialize_returns_protocol_version_and_capabilities(self, server):
        resp = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            }
        )
        assert resp["id"] == 1
        assert "result" in resp
        result = resp["result"]
        assert "protocolVersion" in result
        assert "capabilities" in result
        # We expose tools.
        assert "tools" in result["capabilities"]

    def test_initialize_includes_server_info(self, server):
        resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert resp["result"]["serverInfo"]["name"]


class TestToolsList:
    def test_advertises_wait_for_message(self, server):
        resp = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = resp["result"]["tools"]
        names = [t["name"] for t in tools]
        assert "wait_for_message" in names
        wait = next(t for t in tools if t["name"] == "wait_for_message")
        # Schema declares the timeout_ms input.
        assert "inputSchema" in wait
        assert "timeout_ms" in wait["inputSchema"]["properties"]


class TestToolsCall:
    def test_returns_message_when_one_arrives(self, server, shared_redis):
        # Push a message into our inbox.
        shared_redis.rpush("agent1:inbox", '{"from": "bob", "content": "hello"}')
        resp = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "wait_for_message", "arguments": {"timeout_ms": 100}},
            }
        )
        text = resp["result"]["content"][0]["text"]
        assert "bob" in text
        assert "hello" in text

    def test_returns_empty_on_timeout(self, server):
        resp = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "wait_for_message", "arguments": {"timeout_ms": 100}},
            }
        )
        # Should not error; should indicate timeout.
        text = resp["result"]["content"][0]["text"]
        assert "no message" in text.lower() or "timeout" in text.lower()

    def test_blocks_until_message_arrives(self, server, shared_redis):
        """Server-side blocking is the whole point of this tool."""

        def _delayed_push():
            time.sleep(0.05)
            shared_redis.rpush("agent1:inbox", '{"from": "bob", "content": "late"}')

        t = threading.Thread(target=_delayed_push)
        t.start()
        start = time.time()
        resp = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "wait_for_message", "arguments": {"timeout_ms": 2000}},
            }
        )
        t.join()
        elapsed = time.time() - start
        # Should have actually blocked for some non-zero time (we
        # injected at +50ms).
        assert elapsed >= 0.04
        text = resp["result"]["content"][0]["text"]
        assert "late" in text

    def test_unknown_tool_returns_error(self, server):
        resp = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "made_up_tool", "arguments": {}},
            }
        )
        assert "error" in resp


class TestEnvFactory:
    def test_factory_reads_env(self, monkeypatch):
        monkeypatch.setenv("CB_TEAM_REDIS_URL", "redis://x:6379#run:abc")
        client = build_redis_client_from_env()
        # We can't really probe a real client; just make sure it
        # returned something with the redis interface.
        assert hasattr(client, "blpop")

    def test_factory_raises_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("CB_TEAM_REDIS_URL", raising=False)
        with pytest.raises(RuntimeError):
            build_redis_client_from_env()
