"""Coordination metrics computed from the team task-log audit trail.

After a team-mode run completes, ``compute_metrics`` walks the JSON
events the ``TaskListClient`` wrote during the run (one per mutation)
and produces a compact dict the bench saves alongside the usual
per-agent results.

Currently surfaces:

  - ``tasks_total``                  count
  - ``tasks_done``                   count where final status is "done"
  - ``unowned_at_end``               count of tasks with no owner at end
  - ``time_to_first_claim_seconds``  gap between first create and first
                                     claim (None if no claim happened)
  - ``claims_per_agent``             {agent_id: int}
  - ``updates_per_agent``            {agent_id: int}

These are intentionally simple — richer metrics like "did agents
deadlock" or "messages-per-merge" can be derived from the same event
log in follow-ups without changing the storage shape.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def compute_metrics(
    events: list[dict[str, Any]],
    *,
    final_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    creates = [e for e in events if e.get("kind") == "create"]
    claims = [e for e in events if e.get("kind") == "claim"]
    updates = [e for e in events if e.get("kind") == "update"]

    first_create_ts = min((c.get("ts", 0.0) for c in creates), default=None)
    first_claim_ts = min((c.get("ts", 0.0) for c in claims), default=None)
    time_to_first_claim: float | None = None
    if first_create_ts is not None and first_claim_ts is not None:
        time_to_first_claim = round(first_claim_ts - first_create_ts, 3)

    tasks_done = sum(1 for t in final_tasks if t.get("status") == "done")
    unowned_at_end = sum(1 for t in final_tasks if not t.get("owner"))

    return {
        "tasks_total": len(final_tasks),
        "tasks_done": tasks_done,
        "unowned_at_end": unowned_at_end,
        "time_to_first_claim_seconds": time_to_first_claim,
        "claims_per_agent": dict(Counter(c.get("by", "?") for c in claims)),
        "updates_per_agent": dict(Counter(u.get("by", "?") for u in updates)),
    }
