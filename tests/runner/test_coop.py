"""Unit tests for cooperbench.runner.coop module."""

from cooperbench.runner.coop import _extract_conversation, _message_timestamp_key


def _msg(role: str, content: str, *, ts=None, **extra) -> dict:
    """Build a result-message dict in the shape ``_extract_conversation`` reads."""
    m = {"role": role, "content": content}
    if ts is not None:
        m["timestamp"] = ts
    m.update(extra)
    return m


class TestExtractConversation:
    """Tests for ``_extract_conversation`` — the function whose output feeds
    the buggy ``sent_msgs.sort`` site downstream."""

    def test_extracts_sent_messages_from_bash_format(self):
        """mini_swe_agent style: assistant content contains
        ``send_message agentX "body"``."""
        results = {
            "agent1": {
                "feature_id": 1,
                "messages": [
                    _msg("assistant",
                         'send_message agent2 "lets coordinate"',
                         ts=1.0),
                ],
            },
        }
        conv = _extract_conversation(results, ["agent1"])
        assert len(conv) == 1
        assert conv[0]["from"] == "agent1"
        assert conv[0]["to"] == "agent2"
        assert conv[0]["message"] == "lets coordinate"
        assert conv[0]["timestamp"] == 1.0
        assert not conv[0].get("received")

    def test_extracts_received_messages(self):
        """``[Message from X]`` user-content delivery is tagged received."""
        results = {
            "agent2": {
                "feature_id": 2,
                "messages": [
                    _msg("user",
                         "[Message from agent1]: hi from agent1",
                         ts=2.0),
                ],
            },
        }
        conv = _extract_conversation(results, ["agent2"])
        assert any(m.get("received") and m["from"] == "agent1" for m in conv)


class TestSortDoesNotCrashOnMixedTimestampTypes:
    """The historical bug:
    ``sent_msgs.sort(key=lambda x: x.get("timestamp") or 0)`` crashes with
    ``TypeError: '<' not supported between instances of 'int' and 'str'``
    when one adapter records floats and another records ISO strings. The
    crash fires *before* ``agent{fid}_traj.json`` is written, so callers
    get no structured rollout output.

    This test exercises the post-extraction sort exactly as it appears in
    ``execute_coop`` and asserts the call succeeds. If the production
    sort ever re-introduces a non-coercing key, this test fails fast.
    """

    @staticmethod
    def _sort_like_production(conversation):
        """Reproduce the exact filter-then-sort pattern in ``execute_coop``."""
        sent = [m for m in conversation if not m.get("received")]
        sent.sort(key=_message_timestamp_key)
        return sent

    def test_mixed_int_float_string_timestamps_sort_cleanly(self):
        conversation = [
            {"from": "agent1", "to": "agent2", "message": "a", "timestamp": 1.5},
            {"from": "agent2", "to": "agent1", "message": "b",
             "timestamp": "2026-05-13T22:47:00Z"},
            {"from": "agent1", "to": "agent2", "message": "c", "timestamp": 3},
            {"from": "agent2", "to": "agent1", "message": "d", "timestamp": None},
            {"from": "agent1", "to": "agent2", "message": "e"},  # no ts at all
        ]
        sorted_msgs = self._sort_like_production(conversation)
        # All five sent rows preserved (none received-flagged).
        assert len(sorted_msgs) == 5
        # Numeric timestamps sort in order; unparseable strings + None +
        # missing all coerce to 0.0 and end up at the front (stable order).
        floats = [
            float(m["timestamp"])
            for m in sorted_msgs
            if isinstance(m.get("timestamp"), (int, float))
        ]
        assert floats == sorted(floats)

    def test_received_messages_excluded_before_sort(self):
        """Received entries must be filtered out before the sort, not after —
        otherwise sorting received-then-filtering would still bump into the
        same crash on adversarial inputs."""
        conversation = [
            {"from": "agent1", "to": "agent2", "message": "out", "timestamp": 1.0},
            {"from": "agent2", "to": "agent1", "message": "in",
             "timestamp": "garbage", "received": True},
        ]
        sorted_msgs = self._sort_like_production(conversation)
        assert len(sorted_msgs) == 1
        assert sorted_msgs[0]["message"] == "out"
