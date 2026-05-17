"""Unit tests for team-mode coordination metrics.

After a run finishes, the bench reads the task-log audit trail (one
JSON event per mutation, written by ``TaskListClient``) and computes
high-signal metrics:

  - ``time_to_first_claim`` — gap between first create and first claim;
    proxy for how quickly the team self-organized
  - ``claims_per_agent`` — who actually picked up work
  - ``updates_per_agent`` — who reported progress
  - ``unowned_at_end`` — count of open/unowned tasks; failure indicator
  - ``redundant_claims`` — claim attempts that lost the race

These tests use synthetic event lists so they're independent of Redis
and of the production producer code.
"""

from cooperbench.agents._team.metrics import compute_metrics


def _ev(kind, ts, **kwargs):
    return {"kind": kind, "ts": ts, **kwargs}


class TestComputeMetrics:
    def test_happy_path(self):
        events = [
            _ev("create", ts=0.0, task_id="t1", by="lead"),
            _ev("create", ts=0.1, task_id="t2", by="lead"),
            _ev("claim", ts=1.0, task_id="t1", by="agent2"),
            _ev("claim", ts=1.2, task_id="t2", by="agent3"),
            _ev("update", ts=10.0, task_id="t1", by="agent2", status="done"),
            _ev("update", ts=12.0, task_id="t2", by="agent3", status="done"),
        ]
        m = compute_metrics(
            events,
            final_tasks=[
                {"id": "t1", "status": "done", "owner": "agent2"},
                {"id": "t2", "status": "done", "owner": "agent3"},
            ],
        )
        assert m["tasks_total"] == 2
        assert m["tasks_done"] == 2
        assert m["unowned_at_end"] == 0
        assert m["time_to_first_claim_seconds"] == 1.0
        assert m["claims_per_agent"] == {"agent2": 1, "agent3": 1}
        assert m["updates_per_agent"] == {"agent2": 1, "agent3": 1}

    def test_unowned_task_flagged(self):
        events = [_ev("create", ts=0.0, task_id="t1", by="lead")]
        m = compute_metrics(
            events,
            final_tasks=[
                {"id": "t1", "status": "open", "owner": ""},
            ],
        )
        assert m["unowned_at_end"] == 1
        assert m["time_to_first_claim_seconds"] is None
        assert m["tasks_done"] == 0

    def test_empty_log(self):
        m = compute_metrics([], final_tasks=[])
        assert m["tasks_total"] == 0
        assert m["tasks_done"] == 0
        assert m["unowned_at_end"] == 0
        assert m["claims_per_agent"] == {}
        assert m["updates_per_agent"] == {}
        assert m["time_to_first_claim_seconds"] is None

    def test_multiple_claims_count_separately(self):
        events = [
            _ev("create", ts=0.0, task_id="t1", by="lead"),
            _ev("create", ts=0.1, task_id="t2", by="lead"),
            _ev("claim", ts=1.0, task_id="t1", by="agent1"),
            _ev("claim", ts=2.0, task_id="t2", by="agent1"),
        ]
        m = compute_metrics(events, final_tasks=[])
        assert m["claims_per_agent"] == {"agent1": 2}
