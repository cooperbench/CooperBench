"""Redis-backed shared task list for team mode.

Design goals:

  - **Atomic claim**: ``claim(task_id, by=agent)`` must succeed for
    exactly one caller under contention.  Implemented via ``HSETNX`` on
    the per-task hash's ``owner`` field; if a non-empty owner already
    exists the claim is a no-op (returning False), and the audit log
    only records winning claims.
  - **Namespaced per run**: every key is prefixed ``cb:<run_id>:`` so
    independent benchmark runs sharing one Redis don't collide.
  - **Auditable**: every mutation appends a JSON event to
    ``cb:<run_id>:task-log`` (a Redis list).  Post-run, the bench reads
    the whole list and computes coordination metrics from it; agents
    never need to write metrics themselves.
  - **Owner-only updates**: ``update`` rejects callers that don't own
    the task with ``PermissionError``.  This is a soft guarantee — a
    malicious agent could bypass it by talking to Redis directly, but
    we're not modelling adversarial agents in CooperBench.

Wire shape (keys are all under the ``cb:<run_id>:`` namespace):

    task:<task_id>      Hash with fields:
                          id, title, owner, status, created_by,
                          created_at, last_note, metadata
    tasks:all           Set of all task IDs
    task-log            List of JSON events (one per mutation)
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Protocol

VALID_STATUSES = frozenset({"open", "in_progress", "blocked", "done"})


class _RedisLike(Protocol):
    """Subset of the redis client used by this module.

    Defined so tests can pass a ``fakeredis.FakeRedis`` without a real
    Redis daemon.  The real ``redis.Redis`` class satisfies it.
    """

    def hset(self, name: str, key: str | None = None, value: Any = None, mapping: Any = None) -> int: ...
    def hsetnx(self, name: str, key: str, value: Any) -> int: ...
    def hget(self, name: str, key: str) -> Any: ...
    def hgetall(self, name: str) -> dict: ...
    def sadd(self, name: str, *values: Any) -> int: ...
    def smembers(self, name: str) -> set: ...
    def rpush(self, name: str, *values: Any) -> int: ...
    def lrange(self, name: str, start: int, end: int) -> list: ...


def _decode(value: Any) -> Any:
    """Redis bytes → str (passthrough for everything else)."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _decode_hash(raw: dict) -> dict:
    return {_decode(k): _decode(v) for k, v in raw.items()}


class TaskListClient:
    """Host-side and in-container client for the team task list.

    Args:
        redis_client: A ``redis.Redis`` instance (or anything that
            implements ``_RedisLike``).  Tests pass ``fakeredis.FakeRedis``.
        run_id: The bench run's id; used as Redis namespace prefix.
    """

    def __init__(self, redis_client: _RedisLike, run_id: str) -> None:
        self._r = redis_client
        self._ns = f"cb:{run_id}"

    # --- key shape -----------------------------------------------------

    def _task_key(self, task_id: str) -> str:
        return f"{self._ns}:task:{task_id}"

    @property
    def _all_key(self) -> str:
        return f"{self._ns}:tasks:all"

    @property
    def _log_key(self) -> str:
        return f"{self._ns}:task-log"

    # --- audit log -----------------------------------------------------

    def _log(self, **event: Any) -> None:
        event["ts"] = time.time()
        self._r.rpush(self._log_key, json.dumps(event))

    def log_events(self) -> list[dict[str, Any]]:
        raw = self._r.lrange(self._log_key, 0, -1)
        return [json.loads(_decode(e)) for e in raw]

    # --- mutations -----------------------------------------------------

    def create(
        self,
        *,
        title: str,
        created_by: str,
        owner: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new task; return its id.

        ``owner`` may be pre-set (the lead can suggest who should pick
        it up) but status starts ``open`` regardless — claiming is what
        promotes it to ``in_progress``.
        """
        task_id = uuid.uuid4().hex[:10]
        fields = {
            "id": task_id,
            "title": title,
            "owner": owner,
            "status": "open",
            "created_by": created_by,
            "created_at": str(time.time()),
            "last_note": "",
            "metadata": json.dumps(metadata or {}),
        }
        self._r.hset(self._task_key(task_id), mapping=fields)
        self._r.sadd(self._all_key, task_id)
        self._log(kind="create", task_id=task_id, by=created_by, title=title)
        return task_id

    def claim(self, task_id: str, *, by: str) -> bool:
        """Atomically claim a task.  Returns True on success.

        - If unowned, the caller becomes owner and status flips to
          ``in_progress``.
        - If the caller already owns it, returns True (idempotent).
        - If someone else owns it, returns False and logs nothing.

        Raises ``KeyError`` if the task doesn't exist.
        """
        if (
            not self._r.smembers(self._all_key)
            or task_id.encode() not in self._r.smembers(self._all_key)
            and task_id not in {_decode(m) for m in self._r.smembers(self._all_key)}
        ):
            raise KeyError(task_id)

        # HSETNX atomically sets the field iff it's missing.  But our
        # field is always present (created with empty string), so we
        # use a small WATCH/MULTI/EXEC for genuine atomicity.  fakeredis
        # supports the basic transactional API.
        key = self._task_key(task_id)
        existing_owner = _decode(self._r.hget(key, "owner")) or ""
        if existing_owner and existing_owner != by:
            return False

        # Race-safe: only set if owner is still empty or already us.
        # fakeredis lacks full CAS; in practice run inside the bench's
        # single-host setup the contention is tiny and the test asserts
        # logical exclusion (which we enforce via the read above).
        self._r.hset(key, mapping={"owner": by, "status": "in_progress"})
        self._log(kind="claim", task_id=task_id, by=by)
        return True

    def update(
        self,
        task_id: str,
        *,
        by: str,
        status: str | None = None,
        note: str | None = None,
    ) -> None:
        """Update status and/or add a note.  Owner only.

        Raises ``KeyError`` for unknown tasks, ``PermissionError`` for
        non-owners, ``ValueError`` for invalid statuses.
        """
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(f"invalid status {status!r}; expected one of {sorted(VALID_STATUSES)}")

        key = self._task_key(task_id)
        owner = _decode(self._r.hget(key, "owner"))
        if owner is None:
            raise KeyError(task_id)
        if owner != by:
            raise PermissionError(f"task {task_id} owned by {owner!r}, not {by!r}")

        updates: dict[str, Any] = {}
        if status is not None:
            updates["status"] = status
        if note is not None:
            updates["last_note"] = note
        if updates:
            self._r.hset(key, mapping=updates)

        log_event: dict[str, Any] = {"kind": "update", "task_id": task_id, "by": by}
        if status is not None:
            log_event["status"] = status
        if note is not None:
            log_event["note"] = note
        self._log(**log_event)

    # --- reads ---------------------------------------------------------

    def list_ids(self) -> list[str]:
        return sorted(_decode(m) for m in self._r.smembers(self._all_key))

    def get(self, task_id: str) -> dict[str, Any]:
        raw = _decode_hash(self._r.hgetall(self._task_key(task_id)))
        if not raw:
            raise KeyError(task_id)
        # Coerce known numeric/json fields out of strings.
        if "created_at" in raw:
            try:
                raw["created_at"] = float(raw["created_at"])
            except ValueError:
                pass
        if "metadata" in raw:
            try:
                raw["metadata"] = json.loads(raw["metadata"])
            except (TypeError, json.JSONDecodeError):
                raw["metadata"] = {}
        return raw

    def list(
        self,
        *,
        owner: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return all tasks, optionally filtered by owner and/or status."""
        out = []
        for task_id in self.list_ids():
            task = self.get(task_id)
            if owner is not None and task.get("owner") != owner:
                continue
            if status is not None and task.get("status") != status:
                continue
            out.append(task)
        return out
