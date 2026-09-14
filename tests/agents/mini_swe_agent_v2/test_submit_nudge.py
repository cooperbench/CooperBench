"""The submit nudge must not leave the terminal role="exit" sentinel in the message list.

The nudge continues the episode after the agent tried to end with unshared work. The
exit message is harness-internal; if it stays in ``self.messages``, every later API
request replays it and OpenAI-style backends (vLLM, SGLang) reject the whole call with
"Unexpected message role" — the agent then dies in a deterministic retry loop. Measured
on a Qwen3.5-9B fleet run: 263 of 287 agents in one shard exited Error this way.
"""

from __future__ import annotations

from cooperbench.agents.mini_swe_agent_v2.agents.default import DefaultAgent


class _StubModel:
    def format_message(self, role, content, extra=None):
        return {"role": role, "content": content, "extra": extra or {}}

    def get_template_vars(self):
        return {}

    def serialize(self):
        return {}


class _GitEnv:
    """Env whose git answers make the nudge fire: dirty tree, nothing pushed, no PR."""

    def __init__(self):
        self.commands = []

    def execute(self, action):
        self.commands.append(action)
        cmd = action.get("command", "")
        if "status --porcelain" in cmd:
            return {"output": " M src/thing.py\n", "returncode": 0}
        return {"output": "", "returncode": 0}

    def get_template_vars(self):
        return {}

    def serialize(self):
        return {}


class _StubComm:
    agent_id = "agent1"
    agents = ["agent1", "agent2"]


def _agent():
    return DefaultAgent(
        _StubModel(),
        _GitEnv(),
        comm=_StubComm(),
        agent_id="agent1",
        system_template="s",
        instance_template="i",
    )


def _exit_message():
    return {"role": "exit", "content": "Submitted", "extra": {"exit_status": "Submitted", "submission": ""}}


class TestNudgeRemovesExitSentinel:
    def test_exit_message_removed_and_nudge_appended(self):
        agent = _agent()
        agent.messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "i"},
            {"role": "assistant", "content": "done!"},
            _exit_message(),
        ]

        assert agent._nudge_unsubmitted() is True
        roles = [m.get("role") for m in agent.messages]
        assert "exit" not in roles, "stale exit sentinel would be replayed to the API"
        assert roles[-1] == "user"
        assert "not submitted" in agent.messages[-1]["content"]

    def test_nudge_cap_leaves_exit_in_place(self):
        agent = _agent()
        agent._submit_nudges = DefaultAgent.MAX_SUBMIT_NUDGES
        agent.messages = [{"role": "assistant", "content": "done"}, _exit_message()]

        assert agent._nudge_unsubmitted() is False
        assert agent.messages[-1]["role"] == "exit", "after the cap the episode really ends"
