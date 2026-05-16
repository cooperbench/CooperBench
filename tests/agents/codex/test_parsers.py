"""Unit tests for the Codex JSONL stream parser.

Codex's ``--json`` mode emits ``ThreadEvent`` objects, one JSON per line.
The parser extracts:

  - status        — "Submitted" / "Error" based on terminal event
  - steps         — number of ``turn.completed`` events
  - input_tokens, output_tokens, cache_read_tokens — summed from ``usage``
                    objects on ``turn.completed`` events

Codex does NOT emit a total cost field; the parser leaves ``cost`` at 0.0.
The messages extractor converts ``item.completed`` events with
``agent_message`` / ``reasoning`` / ``command_execution`` payloads into
OpenAI-style chat messages.
"""

import json

import pytest

from cooperbench.agents.codex.parsers import (
    StreamSummary,
    parse_messages,
    parse_stream_jsonl,
)


def _event(**kwargs) -> str:
    return json.dumps(kwargs)


class TestParseStreamJsonl:
    def test_sums_tokens_across_turns(self):
        stream = "\n".join(
            [
                _event(type="thread.started", thread_id="t1"),
                _event(type="turn.started"),
                _event(
                    type="turn.completed",
                    usage={
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cached_input_tokens": 0,
                        "reasoning_output_tokens": 20,
                    },
                ),
                _event(type="turn.started"),
                _event(
                    type="turn.completed",
                    usage={
                        "input_tokens": 300,
                        "output_tokens": 80,
                        "cached_input_tokens": 200,
                        "reasoning_output_tokens": 10,
                    },
                ),
            ]
        )
        s = parse_stream_jsonl(stream)
        assert isinstance(s, StreamSummary)
        assert s.status == "Submitted"
        assert s.steps == 2
        assert s.input_tokens == 800
        assert s.output_tokens == 180
        assert s.cache_read_tokens == 200
        assert s.cache_write_tokens == 0
        # Codex has no native cost reporting.
        assert s.cost == pytest.approx(0.0)

    def test_fatal_error_marks_status_error(self):
        stream = "\n".join(
            [
                _event(type="thread.started", thread_id="t1"),
                _event(type="error", message="model gpt-5.5 not found"),
            ]
        )
        s = parse_stream_jsonl(stream)
        assert s.status == "Error"
        assert "gpt-5.5" in s.raw_result.get("message", "")

    def test_turn_failed_marks_status_error(self):
        stream = "\n".join(
            [
                _event(type="thread.started", thread_id="t1"),
                _event(type="turn.started"),
                _event(type="turn.failed", error={"message": "rate limited"}),
            ]
        )
        s = parse_stream_jsonl(stream)
        assert s.status == "Error"

    def test_empty_stream_is_error(self):
        s = parse_stream_jsonl("")
        assert s.status == "Error"
        assert s.steps == 0

    def test_skips_malformed_lines(self):
        stream = "\n".join(
            [
                "",
                "not json",
                _event(type="thread.started", thread_id="t1"),
                _event(
                    type="turn.completed",
                    usage={
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cached_input_tokens": 0,
                        "reasoning_output_tokens": 0,
                    },
                ),
            ]
        )
        s = parse_stream_jsonl(stream)
        assert s.steps == 1
        assert s.input_tokens == 10

    def test_invalid_model_is_distinguishable(self):
        """Adapter's fallback logic keys on this — surface it as a hint."""
        stream = _event(type="error", message="invalid_request_error: model 'gpt-5.5' does not exist")
        s = parse_stream_jsonl(stream)
        assert s.status == "Error"
        assert s.is_model_error is True

    def test_non_model_error_not_flagged(self):
        stream = _event(type="error", message="rate limited: try again later")
        s = parse_stream_jsonl(stream)
        assert s.status == "Error"
        assert s.is_model_error is False


class TestParseMessages:
    def test_agent_message_becomes_assistant(self):
        stream = _event(
            type="item.completed",
            item={
                "id": "i1",
                "type": "agent_message",
                "text": "Looking at the code.",
            },
        )
        msgs = parse_messages(stream)
        assert msgs == [{"role": "assistant", "content": "Looking at the code."}]

    def test_reasoning_prefixed(self):
        stream = _event(
            type="item.completed",
            item={"id": "i1", "type": "reasoning", "text": "Considering the structure."},
        )
        msgs = parse_messages(stream)
        assert msgs[0]["role"] == "assistant"
        assert "Considering the structure" in msgs[0]["content"]
        assert msgs[0]["content"].startswith("[reasoning]")

    def test_command_execution_serialized_with_exit(self):
        stream = _event(
            type="item.completed",
            item={
                "id": "i1",
                "type": "command_execution",
                "command": "ls /workspace/repo",
                "aggregated_output": "outlines\npatch.txt\n",
                "exit_code": 0,
                "status": "completed",
            },
        )
        msgs = parse_messages(stream)
        assert msgs[0]["role"] == "assistant"
        content = msgs[0]["content"]
        assert "ls /workspace/repo" in content
        assert "outlines" in content

    def test_only_completed_items_emit_messages(self):
        """item.started/updated would duplicate the same item."""
        stream = "\n".join(
            [
                _event(
                    type="item.started",
                    item={"id": "i1", "type": "agent_message", "text": "early"},
                ),
                _event(
                    type="item.updated",
                    item={"id": "i1", "type": "agent_message", "text": "later"},
                ),
                _event(
                    type="item.completed",
                    item={"id": "i1", "type": "agent_message", "text": "final"},
                ),
            ]
        )
        msgs = parse_messages(stream)
        assert msgs == [{"role": "assistant", "content": "final"}]

    def test_content_field_always_string(self):
        stream = _event(
            type="item.completed",
            item={"id": "i1", "type": "agent_message", "text": None},
        )
        msgs = parse_messages(stream)
        for m in msgs:
            assert isinstance(m["content"], str)

    def test_empty_input_returns_empty(self):
        assert parse_messages("") == []
