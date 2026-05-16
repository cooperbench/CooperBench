"""Unit tests for the Redis-backed shared task list.

The task list is the load-bearing primitive of team mode: a typed
record store where any agent can ``create`` a task, members can
``claim`` (atomically — exactly one wins on a race), the owner can
``update`` status / notes, and anyone can ``list`` the current state.

Backed by Redis hashes + sets, namespaced by ``cb:<run_id>:task...``.
Every mutation also appends a structured event to ``cb:<run_id>:task-log``
so the bench can compute coordination metrics post-run.

All tests use ``fakeredis`` so they're fast and need no real daemon.
"""

from __future__ import annotations

import json

import fakeredis
import pytest

from cooperbench.agents._team.task_list import TaskListClient


@pytest.fixture
def client():
    """One fresh in-memory Redis per test, namespaced ``cb:test``."""
    fake = fakeredis.FakeRedis()
    return TaskListClient(redis_client=fake, run_id="test")


class TestCreate:
    def test_create_returns_id_and_persists(self, client):
        task_id = client.create(title="Implement feature X", created_by="agent1")
        assert task_id
        all_ids = client.list_ids()
        assert task_id in all_ids

    def test_created_task_starts_open_and_unowned(self, client):
        task_id = client.create(title="t", created_by="agent1")
        task = client.get(task_id)
        assert task["status"] == "open"
        assert task["owner"] == ""
        assert task["title"] == "t"
        assert task["created_by"] == "agent1"
        assert isinstance(task["created_at"], float)

    def test_create_can_pre_assign_owner(self, client):
        task_id = client.create(title="t", created_by="lead", owner="agent2")
        task = client.get(task_id)
        assert task["owner"] == "agent2"
        # Pre-assignment shouldn't change status — owner can still decline
        # by leaving it open or claiming explicitly.
        assert task["status"] == "open"

    def test_create_with_metadata(self, client):
        task_id = client.create(
            title="t",
            created_by="lead",
            metadata={"feature_id": 1, "files": ["a.py", "b.py"]},
        )
        task = client.get(task_id)
        assert task["metadata"] == {"feature_id": 1, "files": ["a.py", "b.py"]}


class TestClaim:
    def test_claim_unowned_succeeds(self, client):
        task_id = client.create(title="t", created_by="lead")
        assert client.claim(task_id, by="agent1") is True
        task = client.get(task_id)
        assert task["owner"] == "agent1"
        assert task["status"] == "in_progress"

    def test_claim_already_owned_fails(self, client):
        task_id = client.create(title="t", created_by="lead", owner="agent1")
        client.claim(task_id, by="agent1")
        assert client.claim(task_id, by="agent2") is False
        task = client.get(task_id)
        assert task["owner"] == "agent1"

    def test_claim_is_atomic_under_contention(self, client):
        """Simulate two agents racing for the same task — only one wins."""
        task_id = client.create(title="t", created_by="lead")
        winners = []
        for agent in ("agent1", "agent2", "agent3"):
            if client.claim(task_id, by=agent):
                winners.append(agent)
        assert len(winners) == 1

    def test_claim_unknown_task_raises(self, client):
        with pytest.raises(KeyError):
            client.claim("does-not-exist", by="agent1")

    def test_owner_can_reclaim_no_op(self, client):
        task_id = client.create(title="t", created_by="lead")
        client.claim(task_id, by="agent1")
        # Reclaiming by the same owner should succeed (idempotent), not fail.
        assert client.claim(task_id, by="agent1") is True


class TestUpdate:
    def test_owner_can_update_status(self, client):
        task_id = client.create(title="t", created_by="lead")
        client.claim(task_id, by="agent1")
        client.update(task_id, by="agent1", status="done")
        assert client.get(task_id)["status"] == "done"

    def test_non_owner_update_rejected(self, client):
        task_id = client.create(title="t", created_by="lead")
        client.claim(task_id, by="agent1")
        with pytest.raises(PermissionError):
            client.update(task_id, by="agent2", status="done")

    def test_update_can_add_note(self, client):
        task_id = client.create(title="t", created_by="lead")
        client.claim(task_id, by="agent1")
        client.update(task_id, by="agent1", status="blocked", note="needs review")
        task = client.get(task_id)
        assert task["status"] == "blocked"
        assert task["last_note"] == "needs review"

    def test_invalid_status_rejected(self, client):
        task_id = client.create(title="t", created_by="lead")
        client.claim(task_id, by="agent1")
        with pytest.raises(ValueError):
            client.update(task_id, by="agent1", status="nonsense")


class TestList:
    def test_list_returns_all_tasks_with_fields(self, client):
        ids = [client.create(title=f"t{i}", created_by="lead") for i in range(3)]
        all_tasks = client.list()
        assert sorted(t["id"] for t in all_tasks) == sorted(ids)

    def test_list_filter_mine(self, client):
        a = client.create(title="a", created_by="lead")
        b = client.create(title="b", created_by="lead")
        client.claim(a, by="agent1")
        client.claim(b, by="agent2")
        mine = client.list(owner="agent1")
        assert len(mine) == 1
        assert mine[0]["id"] == a

    def test_list_filter_open(self, client):
        a = client.create(title="a", created_by="lead")
        b = client.create(title="b", created_by="lead")
        client.claim(a, by="agent1")  # in_progress
        open_tasks = client.list(status="open")
        assert [t["id"] for t in open_tasks] == [b]

    def test_list_empty(self, client):
        assert client.list() == []


class TestLog:
    """Every mutation should append a JSON event to the audit log.

    The bench uses this post-run to compute coordination metrics:
    time-to-first-claim, claims-per-agent, etc.
    """

    def test_create_logs_event(self, client):
        task_id = client.create(title="t", created_by="lead")
        events = client.log_events()
        assert len(events) == 1
        ev = events[0]
        assert ev["kind"] == "create"
        assert ev["task_id"] == task_id
        assert ev["by"] == "lead"
        assert isinstance(ev["ts"], float)

    def test_claim_logs_event(self, client):
        task_id = client.create(title="t", created_by="lead")
        client.claim(task_id, by="agent1")
        kinds = [e["kind"] for e in client.log_events()]
        assert kinds == ["create", "claim"]

    def test_update_logs_event_with_status(self, client):
        task_id = client.create(title="t", created_by="lead")
        client.claim(task_id, by="agent1")
        client.update(task_id, by="agent1", status="done")
        events = client.log_events()
        assert events[-1] == {
            **events[-1],  # ts and other fields preserved
            "kind": "update",
            "task_id": task_id,
            "by": "agent1",
            "status": "done",
        }

    def test_failed_claim_does_not_log(self, client):
        """Atomic claims that lose the race shouldn't pollute the log."""
        task_id = client.create(title="t", created_by="lead")
        client.claim(task_id, by="agent1")
        client.claim(task_id, by="agent2")  # loses
        kinds = [e["kind"] for e in client.log_events()]
        assert kinds == ["create", "claim"]  # only the winning claim

    def test_log_events_can_be_serialized_to_jsonl(self, client):
        client.create(title="t", created_by="lead")
        for event in client.log_events():
            # Round-trip through JSON to confirm everything is serializable.
            json.dumps(event)


class TestRunIsolation:
    """Two clients with different ``run_id`` must not see each other's tasks."""

    def test_namespaces_dont_collide(self):
        fake = fakeredis.FakeRedis()
        a = TaskListClient(redis_client=fake, run_id="run-a")
        b = TaskListClient(redis_client=fake, run_id="run-b")
        a.create(title="from-a", created_by="lead")
        b.create(title="from-b", created_by="lead")
        a_tasks = a.list()
        b_tasks = b.list()
        assert len(a_tasks) == 1 and a_tasks[0]["title"] == "from-a"
        assert len(b_tasks) == 1 and b_tasks[0]["title"] == "from-b"
