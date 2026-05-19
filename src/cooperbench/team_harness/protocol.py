"""Typed request / response protocol layered on plain messaging.

Mirrors Claude Code's ``plan_approval_request`` / ``plan_approval_response``
shape: a sender posts a typed request with an opaque ``kind`` discriminator,
the recipient responds, and the sender can await the response by id.

Redis layout (under ``cb:<run_id>:``):

    requests:open:<request_id>    Hash: from, to, kind, body, ts
    requests:by_recipient:<peer>  Set of open request_ids
    responses:<request_id>        List the responder pushes to; the
                                  requester ``BLPOP``s it for ``await_response``

The ``await_response`` API uses ``BLPOP`` so the requester actually
blocks instead of busy-polling.  ``fetch_response`` is the
non-blocking equivalent (returns None if no response yet).

Every request and response is also written to the shared task-log
(``cb:<run_id>:task-log``) so post-run analysis can include
coordination-protocol events alongside task-list events.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Protocol


class _RedisLike(Protocol):
    def hset(self, name: str, key: str | None = None, value: Any = None, mapping: Any = None) -> int: ...
    def hgetall(self, name: str) -> dict: ...
    def sadd(self, name: str, *values: Any) -> int: ...
    def srem(self, name: str, *values: Any) -> int: ...
    def smembers(self, name: str) -> set: ...
    def delete(self, *names: str) -> int: ...
    def rpush(self, name: str, *values: Any) -> int: ...
    def lrange(self, name: str, start: int, end: int) -> list: ...
    def lpop(self, name: str, count: int | None = None) -> Any: ...
    def blpop(self, keys, timeout: float = 0) -> Any: ...
    def exists(self, *names: str) -> int: ...


def _decode(value: Any) -> Any:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _decode_hash(raw: dict) -> dict:
    return {_decode(k): _decode(v) for k, v in raw.items()}


class ProtocolClient:
    """Host-side and in-container client for the typed protocol.

    Args:
        redis_client: ``redis.Redis``-like instance.
        run_id: bench run id; used as Redis namespace prefix.
        agent_id: the calling agent's id.  Same client instance is
            used by both requester and responder roles.
    """

    def __init__(self, redis_client: _RedisLike, run_id: str, agent_id: str) -> None:
        self._r = redis_client
        self._ns = f"cb:{run_id}"
        self._me = agent_id

    # --- key shape -----------------------------------------------------

    def _req_key(self, rid: str) -> str:
        return f"{self._ns}:requests:open:{rid}"

    def _inbox_key(self, recipient: str) -> str:
        return f"{self._ns}:requests:by_recipient:{recipient}"

    def _resp_key(self, rid: str) -> str:
        return f"{self._ns}:responses:{rid}"

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

    # --- requester API -------------------------------------------------

    def request(self, *, to: str, kind: str, body: str) -> str:
        rid = uuid.uuid4().hex[:10]
        fields = {
            "id": rid,
            "from": self._me,
            "to": to,
            "kind": kind,
            "body": body,
            "ts": str(time.time()),
        }
        self._r.hset(self._req_key(rid), mapping=fields)
        self._r.sadd(self._inbox_key(to), rid)
        self._log(kind="request", request_id=rid, by=self._me, to=to, request_kind=kind)
        return rid

    def fetch_response(self, request_id: str) -> dict[str, Any] | None:
        """Non-blocking read of the response for ``request_id``.

        Returns ``None`` if no response is queued.  Pops the response
        off the queue (one-shot read).
        """
        raw = self._r.lpop(self._resp_key(request_id))
        if raw is None:
            return None
        return json.loads(_decode(raw))

    def await_response(self, request_id: str, timeout: float = 30.0) -> dict[str, Any] | None:
        """Block until a response arrives or ``timeout`` seconds pass.

        Backed by ``BLPOP`` so the caller actually sleeps instead of
        busy-polling.  Returns the response dict, or ``None`` on timeout.
        """
        # BLPOP returns (key, value) on hit, None on timeout.  Redis
        # uses 0 to mean "block forever"; we forbid that by clamping.
        clamped = max(0.05, float(timeout))
        result = self._r.blpop([self._resp_key(request_id)], timeout=clamped)
        if result is None:
            return None
        _key, raw = result
        return json.loads(_decode(raw))

    # --- responder API -------------------------------------------------

    def list_pending(self) -> list[dict[str, Any]]:
        """All open requests addressed to this agent."""
        ids = sorted(_decode(m) for m in self._r.smembers(self._inbox_key(self._me)))
        out = []
        for rid in ids:
            raw = _decode_hash(self._r.hgetall(self._req_key(rid)))
            if not raw:
                continue
            try:
                raw["ts"] = float(raw["ts"])
            except (KeyError, ValueError):
                pass
            out.append(raw)
        return out

    def respond(self, *, request_id: str, body: str) -> None:
        """Send a response to an open request.

        Raises ``KeyError`` if the request_id is unknown / already
        responded to / wasn't addressed to this agent.
        """
        if not self._r.exists(self._req_key(request_id)):
            raise KeyError(request_id)
        req = _decode_hash(self._r.hgetall(self._req_key(request_id)))
        if req.get("to") != self._me:
            raise KeyError(f"request {request_id} not addressed to {self._me}")
        response = {
            "id": request_id,
            "from": self._me,
            "to": req.get("from"),
            "body": body,
            "ts": time.time(),
        }
        self._r.rpush(self._resp_key(request_id), json.dumps(response))
        self._r.delete(self._req_key(request_id))
        self._r.srem(self._inbox_key(self._me), request_id)
        self._log(kind="response", request_id=request_id, by=self._me, to=req.get("from"))
