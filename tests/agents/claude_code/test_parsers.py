"""Unit tests for Claude Code stream-json and session-JSONL parsers.

These parsers must remain pure functions that take a string (or list of
JSONL lines) and return structured data, so they can be tested without
running the CLI or a container.
"""

import json

import pytest

from cooperbench.agents.claude_code.parsers import (
    StreamSummary,
    parse_session_jsonl,
    parse_stream_json,
)


class TestParseStreamJson:
    """Stream-json from ``claude --print --output-format=stream-json``.

    The Claude Code CLI emits one JSON object per line. The final line of
    a completed run is ``{"type": "result", ..., "total_cost_usd": <float>,
    "usage": {...}, "num_turns": <int>}``.  We need to extract the
    authoritative cost + token totals from that line.
    """

    def test_extracts_total_cost_and_tokens_from_result_event(self):
        stream = "\n".join(
            [
                json.dumps({"type": "system", "subtype": "init"}),
                json.dumps({"type": "assistant", "message": {"content": []}}),
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "total_cost_usd": 0.0421,
                        "num_turns": 7,
                        "usage": {
                            "input_tokens": 1200,
                            "output_tokens": 350,
                            "cache_read_input_tokens": 8800,
                            "cache_creation_input_tokens": 200,
                        },
                    }
                ),
            ]
        )
        summary = parse_stream_json(stream)
        assert isinstance(summary, StreamSummary)
        assert summary.cost == pytest.approx(0.0421)
        assert summary.steps == 7
        assert summary.input_tokens == 1200
        assert summary.output_tokens == 350
        assert summary.cache_read_tokens == 8800
        assert summary.cache_write_tokens == 200

    def test_missing_result_event_returns_zero_values(self):
        stream = json.dumps({"type": "system", "subtype": "init"})
        summary = parse_stream_json(stream)
        assert summary.cost == 0.0
        assert summary.steps == 0
        assert summary.input_tokens == 0
        assert summary.output_tokens == 0

    def test_skips_blank_and_non_json_lines(self):
        stream = "\n".join(
            [
                "",
                "not-json garbage",
                json.dumps(
                    {
                        "type": "result",
                        "total_cost_usd": 0.5,
                        "num_turns": 3,
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    }
                ),
                "",
            ]
        )
        summary = parse_stream_json(stream)
        assert summary.cost == pytest.approx(0.5)
        assert summary.steps == 3

    def test_empty_input_returns_zero_summary(self):
        summary = parse_stream_json("")
        assert summary.cost == 0.0
        assert summary.steps == 0

    def test_status_is_error_when_result_event_signals_error(self):
        stream = json.dumps(
            {
                "type": "result",
                "subtype": "error_max_turns",
                "total_cost_usd": 0.01,
                "num_turns": 30,
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "is_error": True,
            }
        )
        summary = parse_stream_json(stream)
        assert summary.status == "LimitsExceeded"
        assert summary.cost == pytest.approx(0.01)

    def test_status_is_submitted_when_result_succeeds(self):
        stream = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "total_cost_usd": 0.0,
                "num_turns": 2,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        )
        summary = parse_stream_json(stream)
        assert summary.status == "Submitted"

    def test_status_is_error_when_no_result_event(self):
        summary = parse_stream_json('{"type": "system"}')
        assert summary.status == "Error"


class TestParseSessionJsonl:
    """Per-session JSONL written under ``$CLAUDE_CONFIG_DIR/projects/...``.

    Each line is a normalized turn (user / assistant / tool_use /
    tool_result).  We convert it to OpenAI-style chat messages that
    CooperBench's downstream extractor expects (``role`` + ``content``).
    """

    def test_user_turn_becomes_user_message(self):
        line = json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": "Hello"},
                "timestamp": "2026-01-01T00:00:00Z",
            }
        )
        messages = parse_session_jsonl(line)
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_assistant_text_turn_becomes_assistant_message(self):
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Looking at the code."}],
                },
                "timestamp": "2026-01-01T00:00:01Z",
            }
        )
        messages = parse_session_jsonl(line)
        assert messages == [{"role": "assistant", "content": "Looking at the code."}]

    def test_assistant_tool_use_serialized_into_content(self):
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool_1",
                            "name": "Bash",
                            "input": {"command": "ls"},
                        },
                    ],
                },
                "timestamp": "2026-01-01T00:00:02Z",
            }
        )
        messages = parse_session_jsonl(line)
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "assistant"
        # Tool call should appear in the content string so downstream
        # ``"send_message" in content`` checks don't blow up.
        assert "Bash" in msg["content"]
        assert "ls" in msg["content"]

    def test_tool_result_becomes_user_message(self):
        line = json.dumps(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "file1\nfile2",
                        }
                    ],
                },
                "timestamp": "2026-01-01T00:00:03Z",
            }
        )
        messages = parse_session_jsonl(line)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "file1" in messages[0]["content"]

    def test_events_sorted_by_timestamp(self):
        a = json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": "second"},
                "timestamp": "2026-01-01T00:00:02Z",
            }
        )
        b = json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": "first"},
                "timestamp": "2026-01-01T00:00:01Z",
            }
        )
        messages = parse_session_jsonl("\n".join([a, b]))
        assert [m["content"] for m in messages] == ["first", "second"]

    def test_malformed_lines_are_skipped(self):
        good = json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": "ok"},
                "timestamp": "2026-01-01T00:00:00Z",
            }
        )
        text = "\n".join(["", "not json", good])
        messages = parse_session_jsonl(text)
        assert messages == [{"role": "user", "content": "ok"}]

    def test_empty_input_returns_empty_list(self):
        assert parse_session_jsonl("") == []

    def test_content_field_never_none(self):
        """Downstream code does ``"send_message" in msg["content"]`` so
        ``content`` must always be a string, never ``None``."""
        line = json.dumps(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": None},
                "timestamp": "2026-01-01T00:00:00Z",
            }
        )
        messages = parse_session_jsonl(line)
        for m in messages:
            assert isinstance(m["content"], str)
