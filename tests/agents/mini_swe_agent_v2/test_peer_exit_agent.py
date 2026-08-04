"""Agent-level behaviour when a coop peer finishes first.

The connector tests cover the Redis layer. These cover what the *agent actually sees*,
which is where the original bug lived: `send()` queued into a dead mailbox and
`_handle_send_message` reported `returncode: 0, "Message sent to agent2"` regardless.
An agent then recorded "coordination is ongoing" and submitted a conflicting patch.

Driven directly rather than through a live rollout: whether an agent chooses to call
send_message is up to the model, so a real run cannot be relied on to exercise the path.
"""

from __future__ import annotations

import pytest

from cooperbench.agents.mini_swe_agent_v2.agents.default import GIT_REMOTE, DefaultAgent
from cooperbench.agents.mini_swe_agent_v2.connectors import MessagingConnector


class _StubModel:
    def format_message(self, role, content, extra=None):
        return {"role": role, "content": content, "extra": extra or {}}

    def get_template_vars(self):
        return {}

    def serialize(self):
        return {}


class _StubEnv:
    def __init__(self):
        self.commands = []

    def execute(self, action):
        self.commands.append(action)
        return {"output": "", "returncode": 0}

    def get_template_vars(self):
        return {}

    def serialize(self):
        return {}


def _agent(agent_id, comm):
    return DefaultAgent(
        _StubModel(),
        _StubEnv(),
        comm=comm,
        agent_id=agent_id,
        system_template="s",
        instance_template="i",
    )


@pytest.fixture
def pair(redis_url):
    ns = f"{redis_url}#test:peerexit-agent"
    a = MessagingConnector(agent_id="agent1", agents=["agent1", "agent2"], url=ns)
    b = MessagingConnector(agent_id="agent2", agents=["agent1", "agent2"], url=ns)
    return a, b


class TestSendToDepartedPeer:
    def test_reports_failure_not_success(self, pair):
        """
        Target:   DefaultAgent._handle_send_message
        Expected: non-zero returncode, and text that says the peer finished
        Catches:  the original bug -- "Message sent to agent2" with returncode 0 to an
                  agent that had already exited, which the sender believed.
        """
        alice_comm, bob_comm = pair
        bob_comm.mark_exited(published=True)
        out = _agent("agent1", alice_comm)._handle_send_message(
            {"recipient": "agent2", "content": "what files are you editing?"}
        )
        assert out["returncode"] == 1
        assert "completed their work and exited" in out["output"]
        assert "NOT delivered" in out["output"]

    def test_points_at_the_branch_when_the_patch_was_published(self, pair):
        alice_comm, bob_comm = pair
        bob_comm.mark_exited(published=True)
        out = _agent("agent1", alice_comm)._handle_send_message({"recipient": "agent2", "content": "hello"})
        assert f"{GIT_REMOTE}/agent2" in out["output"]
        assert "git fetch" in out["output"]

    def test_does_not_point_at_the_branch_when_publication_failed(self, pair):
        """
        Expected: the agent is told the branch is NOT usable
        Catches:  re-introducing the same class of lie -- publication is best-effort, so
                  claiming the branch holds their submission when it does not would send
                  the agent to read a baseline and treat it as their colleague's work.
        """
        alice_comm, bob_comm = pair
        bob_comm.mark_exited(published=False)
        out = _agent("agent1", alice_comm)._handle_send_message({"recipient": "agent2", "content": "hello"})
        assert "could NOT be published" in out["output"]
        assert "do not rely on it" in out["output"]

    def test_live_peer_still_succeeds(self, pair):
        alice_comm, _ = pair
        out = _agent("agent1", alice_comm)._handle_send_message({"recipient": "agent2", "content": "still here?"})
        assert out["returncode"] == 0
        assert out["output"] == "Message sent to agent2"


class TestDepartureAnnouncement:
    def test_announced_once_with_recovery_path(self, pair):
        """
        Target:   DefaultAgent._announce_departed_peers
        Expected: exactly one injected user turn naming the peer and the branch
        Catches:  an agent waiting out the rest of its run on a colleague that is gone,
                  with nothing in its context saying so; and repeated announcements
                  flooding the context on every subsequent step.
        """
        alice_comm, bob_comm = pair
        alice = _agent("agent1", alice_comm)
        alice._announce_departed_peers()
        assert alice.messages == [], "nothing to announce while the peer is running"

        bob_comm.mark_exited(published=True)
        alice._announce_departed_peers()
        alice._announce_departed_peers()  # idempotent
        notices = [m for m in alice.messages if "has completed their work and exited]" in m["content"]]
        assert len(notices) == 1
        assert f"{GIT_REMOTE}/agent2" in notices[0]["content"]

    def test_solo_run_announces_nothing(self):
        agent = DefaultAgent(
            _StubModel(), _StubEnv(), comm=None, agent_id="agent1", system_template="s", instance_template="i"
        )
        agent._announce_departed_peers()
        assert agent.messages == []


class TestPublishFinalWork:
    def test_solo_run_does_not_publish(self):
        agent = DefaultAgent(
            _StubModel(), _StubEnv(), comm=None, agent_id="agent1", system_template="s", instance_template="i"
        )
        assert agent._publish_final_work() is False

    def test_publish_reports_failure_when_the_container_command_fails(self, pair):
        """
        Expected: False, so mark_exited(published=False) and peers are not misdirected
        Catches:  treating a failed publish as success -- e.g. the `test -s patch.txt`
                  guard exiting 0 when there is no patch, which would tell the peer the
                  branch holds a submission that was never pushed.
        """
        alice_comm, _ = pair
        agent = _agent("agent1", alice_comm)

        class _FailingEnv(_StubEnv):
            def execute(self, action):
                return {"output": "no patch.txt to publish", "returncode": 3}

        agent.env = _FailingEnv()
        assert agent._publish_final_work() is False

    def test_publish_reports_success_and_targets_the_agent_branch(self, pair):
        alice_comm, _ = pair
        agent = _agent("agent1", alice_comm)
        assert agent._publish_final_work() is True
        cmd = agent.env.commands[-1]["command"]
        assert "HEAD:refs/heads/agent1" in cmd
        assert "worktree add" in cmd and "patch.txt" in cmd
        assert f"{GIT_REMOTE}/main" in cmd, "must branch from the pristine base, not HEAD"
