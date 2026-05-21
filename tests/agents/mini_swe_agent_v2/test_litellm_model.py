"""Tests for LitellmModel — focused on compaction-related behavior."""

from cooperbench.agents.mini_swe_agent_v2.models.litellm_model import LitellmModel


class TestSerializeTranscriptForSummary:
    """The summarizer-input serializer flattens turn-wise messages so the model
    receives the conversation as data rather than as turns to continue."""

    def test_basic_roles_get_role_headers(self):
        msgs = [
            {"role": "system", "content": "sys text"},
            {"role": "user", "content": "user text"},
            {"role": "assistant", "content": "asst text"},
        ]
        out = LitellmModel._serialize_transcript_for_summary(msgs)
        assert "[system]" in out
        assert "sys text" in out
        assert "[user]" in out
        assert "user text" in out
        assert "[assistant]" in out
        assert "asst text" in out

    def test_tool_role_becomes_tool_output(self):
        msgs = [{"role": "tool", "content": "stdout: hello"}]
        out = LitellmModel._serialize_transcript_for_summary(msgs)
        assert "[tool_output]" in out
        assert "stdout: hello" in out

    def test_assistant_tool_calls_are_rendered_inline(self):
        msgs = [
            {
                "role": "assistant",
                "content": "I'll list the directory.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "ls -la"}',
                        }
                    }
                ],
            }
        ]
        out = LitellmModel._serialize_transcript_for_summary(msgs)
        assert "[assistant]" in out
        assert "I'll list the directory." in out
        # tool call rendered as inline text marker
        assert "tool_call bash" in out
        assert "ls -la" in out

    def test_missing_content_is_empty_string(self):
        msgs = [{"role": "assistant", "content": None}]
        out = LitellmModel._serialize_transcript_for_summary(msgs)
        # No crash, role header still present
        assert "[assistant]" in out

    def test_unknown_role_uses_literal_role_label(self):
        msgs = [{"role": "weird", "content": "x"}]
        out = LitellmModel._serialize_transcript_for_summary(msgs)
        assert "[weird]" in out
        assert "x" in out

    def test_output_is_str(self):
        out = LitellmModel._serialize_transcript_for_summary([])
        assert isinstance(out, str)
        assert out == ""
