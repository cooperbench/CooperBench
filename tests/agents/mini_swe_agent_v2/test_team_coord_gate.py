"""Unit tests for the team-mode coordination gate in DefaultAgent.

The gate sits between action extraction and ``env.execute()`` in
``DefaultAgent.execute_actions``.  It reads the live task list from
Redis (via the agent's TeamPoller) and refuses commands that would
bypass the coordination protocol:

  1. Unclaimed-own-task gate — refuse non-claim commands while the
     agent owns a task in ``status=open``.
  2. Own-task-not-done gate — refuse final-submit commands while the
     agent owns a task in ``status=in_progress``.
  3. Peer-not-done gate — refuse final-submit commands from the lead
     while any other agent's task is not yet ``status=done``.

All tests use ``fakeredis`` so there's no daemon dependency.
"""

from __future__ import annotations

import fakeredis
import pytest

from cooperbench.agents.mini_swe_agent_v2.agents.default import DefaultAgent
from cooperbench.team_harness.task_list import TaskListClient

SUBMIT_CMD = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


class _StubPoller:
    """Minimal TeamPoller stand-in: just exposes ._ensure_client and ._run_id."""

    def __init__(self, client, run_id):
        self._client = client
        self._run_id = run_id

    def _ensure_client(self):
        return self._client


def _bare_agent(agent_id, fake_redis, run_id="test"):
    """Create a DefaultAgent without invoking __init__ (avoids pulling in
    real Model/Environment classes that need config files).  We only
    exercise the _team_coord_gate method, which needs only
    ``self.agent_id`` and ``self.team_poller``."""
    agent = DefaultAgent.__new__(DefaultAgent)
    agent.agent_id = agent_id
    agent.team_poller = _StubPoller(fake_redis, run_id)
    return agent


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis()


@pytest.fixture
def task_client(fake_redis):
    return TaskListClient(redis_client=fake_redis, run_id="test")


# -----------------------------------------------------------------------------
# No-team-mode fallthrough
# -----------------------------------------------------------------------------


def test_no_poller_means_no_gate(fake_redis):
    """Solo / coop runs have no team_poller — gate is a no-op."""
    agent = DefaultAgent.__new__(DefaultAgent)
    agent.agent_id = "agent1"
    agent.team_poller = None
    assert agent._team_coord_gate("ls") is None


def test_no_tasks_means_no_gate(fake_redis):
    """Team mode is wired but the task list is empty — nothing to gate."""
    agent = _bare_agent("agent1", fake_redis)
    assert agent._team_coord_gate("ls") is None


def test_empty_cmd_means_no_gate(fake_redis, task_client):
    """Whitespace/no-op command shouldn't be force-gated."""
    task_client.create(title="t", created_by="agent1", owner="agent1")
    agent = _bare_agent("agent1", fake_redis)
    assert agent._team_coord_gate("") is None


# -----------------------------------------------------------------------------
# Rule 1: unclaimed-own-task gate
# -----------------------------------------------------------------------------


def test_gate_blocks_when_own_task_is_unclaimed(fake_redis, task_client):
    """Agent owns a pre-assigned task at status=open — must claim first."""
    tid = task_client.create(title="Implement feature 2", created_by="bench", owner="agent2")
    agent = _bare_agent("agent2", fake_redis)
    out = agent._team_coord_gate("grep -r foo .")
    assert out is not None
    assert out["returncode"] == 1
    assert "coop-task-claim" in out["output"]
    assert tid in out["output"]


def test_gate_allows_claim_command_through(fake_redis, task_client):
    """Even with unclaimed task, a claim command itself is permitted."""
    tid = task_client.create(title="Implement feature 2", created_by="bench", owner="agent2")
    agent = _bare_agent("agent2", fake_redis)
    assert agent._team_coord_gate(f"coop-task-claim {tid}") is None


def test_other_agents_unclaimed_task_does_not_gate_me(fake_redis, task_client):
    """Rule 1 fires only when *my* tasks are unclaimed — peer state irrelevant."""
    task_client.create(title="Feature 2", created_by="bench", owner="agent2")  # peer's open task
    agent = _bare_agent("agent1", fake_redis)
    # agent1 has nothing assigned; should not be gated by agent2's open task
    assert agent._team_coord_gate("ls") is None


# -----------------------------------------------------------------------------
# Rule 2: own-task-not-done gate on submit
# -----------------------------------------------------------------------------


def test_gate_blocks_submit_when_own_task_in_progress(fake_redis, task_client):
    """Agent has claimed but not marked done — submit should be refused."""
    tid = task_client.create(title="Implement feature 2", created_by="bench", owner="agent2")
    task_client.claim(tid, by="agent2")
    agent = _bare_agent("agent2", fake_redis)
    out = agent._team_coord_gate(SUBMIT_CMD)
    assert out is not None
    assert "in_progress" in out["output"]
    assert "coop-task-update" in out["output"]
    assert tid in out["output"]


def test_gate_allows_non_submit_when_task_in_progress(fake_redis, task_client):
    """While task is in_progress, regular bash work should still execute."""
    tid = task_client.create(title="Feature 2", created_by="bench", owner="agent2")
    task_client.claim(tid, by="agent2")
    agent = _bare_agent("agent2", fake_redis)
    assert agent._team_coord_gate("pytest tests/") is None


def test_gate_allows_submit_after_marking_done(fake_redis, task_client):
    """Once status=done, member submit passes (no peer-dependency for non-lead)."""
    tid = task_client.create(title="Implement feature 2", created_by="bench", owner="agent2")
    task_client.claim(tid, by="agent2")
    task_client.update(tid, by="agent2", status="done")
    agent = _bare_agent("agent2", fake_redis)
    assert agent._team_coord_gate(SUBMIT_CMD) is None


# -----------------------------------------------------------------------------
# Rule 3: peer-not-done gate (lead only)
# -----------------------------------------------------------------------------


def test_gate_blocks_lead_submit_when_peer_still_open(fake_redis, task_client):
    """Lead's task is done but member hasn't reported — lead can't submit."""
    member_tid = task_client.create(title="Implement feature 2", created_by="bench", owner="agent2")
    lead_tid = task_client.create(title="Lead-only: integrate and submit feature 1", created_by="bench", owner="agent1")
    task_client.claim(lead_tid, by="agent1")
    task_client.update(lead_tid, by="agent1", status="done")
    agent = _bare_agent("agent1", fake_redis)
    out = agent._team_coord_gate(SUBMIT_CMD)
    assert out is not None
    assert "peer task" in out["output"].lower() or "not yet done" in out["output"]
    assert member_tid in out["output"]


def test_gate_allows_lead_submit_when_peer_done(fake_redis, task_client):
    """Both done → lead can submit."""
    member_tid = task_client.create(title="Feature 2", created_by="bench", owner="agent2")
    lead_tid = task_client.create(title="Lead-only: integrate and submit feature 1", created_by="bench", owner="agent1")
    task_client.claim(member_tid, by="agent2")
    task_client.update(member_tid, by="agent2", status="done")
    task_client.claim(lead_tid, by="agent1")
    task_client.update(lead_tid, by="agent1", status="done")
    agent = _bare_agent("agent1", fake_redis)
    assert agent._team_coord_gate(SUBMIT_CMD) is None


def test_member_submit_does_not_wait_on_lead(fake_redis, task_client):
    """A member whose own task is done can submit even if lead's task is open."""
    member_tid = task_client.create(title="Feature 2", created_by="bench", owner="agent2")
    task_client.create(title="Lead-only: integrate and submit feature 1", created_by="bench", owner="agent1")
    task_client.claim(member_tid, by="agent2")
    task_client.update(member_tid, by="agent2", status="done")
    agent = _bare_agent("agent2", fake_redis)
    assert agent._team_coord_gate(SUBMIT_CMD) is None


# -----------------------------------------------------------------------------
# Failure tolerance
# -----------------------------------------------------------------------------


def test_gate_returns_none_on_broken_poller():
    """If the poller's redis client raises, gate must silently fall through."""

    class BrokenPoller:
        _run_id = "test"

        def _ensure_client(self):
            raise RuntimeError("redis down")

    agent = DefaultAgent.__new__(DefaultAgent)
    agent.agent_id = "agent1"
    agent.team_poller = BrokenPoller()
    assert agent._team_coord_gate("ls") is None


# -----------------------------------------------------------------------------
# Split helpers: required_prefix + blocking_reason
# -----------------------------------------------------------------------------


def test_required_prefix_adds_claim_for_open_task(fake_redis, task_client):
    tid = task_client.create(title="Feature 2", created_by="bench", owner="agent2")
    agent = _bare_agent("agent2", fake_redis)
    prefix = agent._team_required_prefix("pytest tests/")
    assert prefix == [f"coop-task-claim {tid}"]


def test_required_prefix_empty_when_already_claimed(fake_redis, task_client):
    tid = task_client.create(title="Feature 2", created_by="bench", owner="agent2")
    task_client.claim(tid, by="agent2")
    agent = _bare_agent("agent2", fake_redis)
    assert agent._team_required_prefix("pytest tests/") == []


def test_required_prefix_skips_claim_when_cmd_is_claim(fake_redis, task_client):
    tid = task_client.create(title="Feature 2", created_by="bench", owner="agent2")
    agent = _bare_agent("agent2", fake_redis)
    assert agent._team_required_prefix(f"coop-task-claim {tid}") == []


def test_required_prefix_adds_update_on_submit_with_in_progress(fake_redis, task_client):
    tid = task_client.create(title="Feature 2", created_by="bench", owner="agent2")
    task_client.claim(tid, by="agent2")
    agent = _bare_agent("agent2", fake_redis)
    prefix = agent._team_required_prefix(SUBMIT_CMD)
    assert len(prefix) == 1
    assert prefix[0].startswith(f"coop-task-update {tid} done -n ")
    assert "Feature 2" in prefix[0]


def test_required_prefix_skips_update_when_cmd_already_updates(fake_redis, task_client):
    tid = task_client.create(title="Feature 2", created_by="bench", owner="agent2")
    task_client.claim(tid, by="agent2")
    agent = _bare_agent("agent2", fake_redis)
    cmd = f"coop-task-update {tid} done -n 'manual' && {SUBMIT_CMD}"
    # rule still wants a claim? no — already claimed. and update is in cmd. -> empty
    assert agent._team_required_prefix(cmd) == []


def test_required_prefix_combines_claim_and_update_on_first_submit(fake_redis, task_client):
    """Edge case: agent submits without ever claiming — prefix should chain both."""
    tid = task_client.create(title="Feature 2", created_by="bench", owner="agent2")
    agent = _bare_agent("agent2", fake_redis)
    prefix = agent._team_required_prefix(SUBMIT_CMD)
    # only claim fires here: after claim runs, status would become in_progress,
    # but the prefix is computed from the snapshot we took at call time, so
    # update doesn't appear (it would on the next submit attempt).  Verify:
    assert len(prefix) == 1
    assert prefix[0] == f"coop-task-claim {tid}"


def test_blocking_reason_none_for_non_submit(fake_redis, task_client):
    task_client.create(title="Lead-only: integrate", created_by="bench", owner="agent1")
    agent = _bare_agent("agent1", fake_redis)
    assert agent._team_blocking_reason("pytest tests/") is None


def test_blocking_reason_blocks_lead_with_open_peer(fake_redis, task_client):
    member_tid = task_client.create(title="Feature 2", created_by="bench", owner="agent2")
    task_client.create(title="Lead-only: integrate feature 1", created_by="bench", owner="agent1")
    agent = _bare_agent("agent1", fake_redis)
    reason = agent._team_blocking_reason(SUBMIT_CMD)
    assert reason is not None
    assert member_tid in reason


def test_blocking_reason_lets_member_submit(fake_redis, task_client):
    """Members don't gate on peer state; only Lead-only owners do."""
    task_client.create(title="Feature 2", created_by="bench", owner="agent2")
    task_client.create(title="Lead-only: integrate", created_by="bench", owner="agent1")
    agent = _bare_agent("agent2", fake_redis)
    assert agent._team_blocking_reason(SUBMIT_CMD) is None


def test_blocking_reason_lets_lead_submit_when_peer_done(fake_redis, task_client):
    member_tid = task_client.create(title="Feature 2", created_by="bench", owner="agent2")
    task_client.claim(member_tid, by="agent2")
    task_client.update(member_tid, by="agent2", status="done")
    task_client.create(title="Lead-only: integrate", created_by="bench", owner="agent1")
    agent = _bare_agent("agent1", fake_redis)
    assert agent._team_blocking_reason(SUBMIT_CMD) is None


# -----------------------------------------------------------------------------
# Apply-prefix: server-side mutation (bypasses in-container CLI)
# -----------------------------------------------------------------------------


def test_apply_prefix_claims_open_task_server_side(fake_redis, task_client):
    """A non-claim cmd against an open-status own task → host applies the claim."""
    tid = task_client.create(title="Implement feature 2", created_by="bench", owner="agent2")
    agent = _bare_agent("agent2", fake_redis)
    out = agent._team_apply_prefix("pytest tests/")
    assert out is not None
    assert "coop-task-claim" in out and tid in out
    # The Redis state actually flipped:
    assert task_client.get(tid)["status"] == "in_progress"
    # And the audit log saw the event:
    kinds = [e.get("kind") for e in task_client.log_events()]
    assert "claim" in kinds


def test_apply_prefix_marks_done_on_submit(fake_redis, task_client):
    """Submit cmd against own in_progress task → host applies the update."""
    tid = task_client.create(title="Implement feature 2", created_by="bench", owner="agent2")
    task_client.claim(tid, by="agent2")
    agent = _bare_agent("agent2", fake_redis)
    out = agent._team_apply_prefix(SUBMIT_CMD)
    assert out is not None
    assert "coop-task-update" in out and tid in out
    assert task_client.get(tid)["status"] == "done"
    kinds = [e.get("kind") for e in task_client.log_events()]
    assert "update" in kinds


def test_apply_prefix_noop_when_no_actions_needed(fake_redis, task_client):
    """Already claimed in_progress task + non-submit cmd → nothing to apply."""
    tid = task_client.create(title="Feature 2", created_by="bench", owner="agent2")
    task_client.claim(tid, by="agent2")
    agent = _bare_agent("agent2", fake_redis)
    assert agent._team_apply_prefix("pytest tests/") is None


def test_apply_prefix_is_idempotent_after_claim(fake_redis, task_client):
    """Calling apply twice in a row should result in exactly one claim event."""
    tid = task_client.create(title="Feature 2", created_by="bench", owner="agent2")
    agent = _bare_agent("agent2", fake_redis)
    agent._team_apply_prefix("pytest tests/")
    agent._team_apply_prefix("ls /workspace")
    kinds = [e.get("kind") for e in task_client.log_events()]
    assert kinds.count("claim") == 1
    assert task_client.get(tid)["status"] == "in_progress"


def test_apply_prefix_handles_both_claim_and_update_for_unclaimed_submit(fake_redis, task_client):
    """Submit before ever claiming → first apply claims, returned string only
    mentions the claim (status snapshot was 'open' when actions were computed).
    A subsequent apply with submit cmd would then trigger the update."""
    tid = task_client.create(title="Feature 2", created_by="bench", owner="agent2")
    agent = _bare_agent("agent2", fake_redis)
    out1 = agent._team_apply_prefix(SUBMIT_CMD)
    assert out1 is not None
    assert "coop-task-claim" in out1
    assert "coop-task-update" not in out1
    # second call now sees status=in_progress, triggers update
    out2 = agent._team_apply_prefix(SUBMIT_CMD)
    assert out2 is not None
    assert "coop-task-update" in out2
    assert task_client.get(tid)["status"] == "done"
