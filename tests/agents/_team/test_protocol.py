"""Unit tests for the typed coop-request / coop-respond protocol.

Layered on top of plain ``coop-send`` messaging.  Mirrors Claude Code's
``plan_approval_request`` / ``plan_approval_response`` shape:

  - ``request(peer, kind, body) -> request_id`` queues a request in the
    peer's inbox AND records it in ``cb:<run>:requests:open:<id>`` so
    the responder has a known target to write back to.
  - ``respond(request_id, body)`` writes to
    ``cb:<run>:responses:<id>`` and deletes the open entry.
  - ``await_response(request_id, timeout)`` blocks (BLPOP-style) until
    a response shows up or the timeout fires.

The host-side ``ProtocolClient`` is the testable seam (we use
``fakeredis``); the in-container CLI just wraps it.
"""

from __future__ import annotations

import threading
import time

import fakeredis
import pytest

from cooperbench.agents._team.protocol import ProtocolClient


@pytest.fixture
def shared_redis():
    return fakeredis.FakeRedis()


@pytest.fixture
def alice(shared_redis):
    return ProtocolClient(redis_client=shared_redis, run_id="t", agent_id="alice")


@pytest.fixture
def bob(shared_redis):
    return ProtocolClient(redis_client=shared_redis, run_id="t", agent_id="bob")


class TestRequest:
    def test_request_returns_id(self, alice, bob):
        rid = alice.request(to="bob", kind="approval", body="ok to proceed?")
        assert rid
        assert isinstance(rid, str)
        # Bob can see one open request addressed to him.
        pending = bob.list_pending()
        assert len(pending) == 1
        assert pending[0]["id"] == rid
        assert pending[0]["from"] == "alice"
        assert pending[0]["kind"] == "approval"
        assert pending[0]["body"] == "ok to proceed?"

    def test_response_round_trips(self, alice, bob):
        rid = alice.request(to="bob", kind="approval", body="?")
        bob.respond(request_id=rid, body="approved")
        # Alice should be able to fetch it without blocking.
        resp = alice.fetch_response(rid)
        assert resp is not None
        assert resp["from"] == "bob"
        assert resp["body"] == "approved"

    def test_respond_clears_pending(self, alice, bob):
        rid = alice.request(to="bob", kind="approval", body="?")
        bob.respond(request_id=rid, body="ok")
        assert bob.list_pending() == []

    def test_fetch_response_missing_returns_none(self, alice):
        assert alice.fetch_response("nonexistent") is None

    def test_respond_to_unknown_request_raises(self, bob):
        with pytest.raises(KeyError):
            bob.respond(request_id="nope", body="ok")


class TestAwait:
    def test_await_response_returns_when_responded(self, alice, bob):
        rid = alice.request(to="bob", kind="approval", body="?")

        # Have a thread respond after a short delay so the blocking call
        # actually has to wait.
        def _delayed_respond():
            time.sleep(0.05)
            bob.respond(request_id=rid, body="approved")

        t = threading.Thread(target=_delayed_respond)
        t.start()

        resp = alice.await_response(rid, timeout=2.0)
        t.join()

        assert resp is not None
        assert resp["body"] == "approved"

    def test_await_response_times_out(self, alice, bob):
        rid = alice.request(to="bob", kind="approval", body="?")
        start = time.time()
        resp = alice.await_response(rid, timeout=0.3)
        elapsed = time.time() - start
        assert resp is None
        # Timed out (allow generous slack — fakeredis behaviour varies).
        assert elapsed >= 0.25


class TestAuditLog:
    """Every request and response is appended to the task-log so the
    bench can count them post-run."""

    def test_request_logged(self, alice, bob):
        alice.request(to="bob", kind="approval", body="?")
        events = alice.log_events()
        kinds = [e.get("kind") for e in events]
        assert "request" in kinds

    def test_response_logged(self, alice, bob):
        rid = alice.request(to="bob", kind="approval", body="?")
        bob.respond(request_id=rid, body="ok")
        events = alice.log_events()
        kinds = [e.get("kind") for e in events]
        assert "request" in kinds
        assert "response" in kinds
