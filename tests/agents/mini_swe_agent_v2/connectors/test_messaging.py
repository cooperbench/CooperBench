"""Tests for cooperbench.agents.mini_swe_agent_v2.connectors.messaging module."""

import pytest

from cooperbench.agents.mini_swe_agent_v2.connectors import MessagingConnector


class TestMessagingConnector:
    """Tests for MessagingConnector."""

    @pytest.fixture
    def connector(self, redis_url):
        """Create a test connector."""
        return MessagingConnector(
            agent_id="agent1",
            agents=["agent1", "agent2"],
            url=f"{redis_url}#test:messaging",
        )

    @pytest.fixture
    def connector2(self, redis_url):
        """Create a second test connector."""
        return MessagingConnector(
            agent_id="agent2",
            agents=["agent1", "agent2"],
            url=f"{redis_url}#test:messaging",
        )

    def test_send_and_receive(self, connector, connector2):
        """Test basic send/receive."""
        connector.send("agent2", "Hello from agent1")

        messages = connector2.receive()
        assert len(messages) == 1
        assert messages[0]["from"] == "agent1"
        assert messages[0]["content"] == "Hello from agent1"

    def test_receive_empties_inbox(self, connector, connector2):
        """Test that receive empties the inbox."""
        connector.send("agent2", "Message 1")
        connector.send("agent2", "Message 2")

        messages = connector2.receive()
        assert len(messages) == 2

        # Second receive should be empty
        messages = connector2.receive()
        assert len(messages) == 0

    def test_broadcast(self, redis_url):
        """Test broadcast sends to all other agents."""
        agents = ["agent1", "agent2", "agent3"]
        connectors = {
            agent_id: MessagingConnector(
                agent_id=agent_id,
                agents=agents,
                url=f"{redis_url}#test:broadcast",
            )
            for agent_id in agents
        }

        # Agent1 broadcasts
        connectors["agent1"].broadcast("Hello everyone")

        # Agent2 and agent3 should receive, agent1 should not
        assert len(connectors["agent2"].receive()) == 1
        assert len(connectors["agent3"].receive()) == 1
        assert len(connectors["agent1"].receive()) == 0

    def test_peek(self, connector, connector2):
        """Test peek returns count without consuming."""
        connector.send("agent2", "Message 1")
        connector.send("agent2", "Message 2")

        assert connector2.peek() == 2

        # Peek should not consume
        assert connector2.peek() == 2

        # Receive consumes
        connector2.receive()
        assert connector2.peek() == 0

    def test_namespace_isolation(self, redis_url):
        """Test that different namespaces are isolated."""
        # Create connector in namespace1 (intentionally unused - tests isolation)
        _conn_ns1 = MessagingConnector(
            agent_id="agent1",
            agents=["agent1", "agent2"],
            url=f"{redis_url}#namespace1",
        )
        conn_ns2_sender = MessagingConnector(
            agent_id="agent1",
            agents=["agent1", "agent2"],
            url=f"{redis_url}#namespace2",
        )
        conn_ns2_receiver = MessagingConnector(
            agent_id="agent2",
            agents=["agent1", "agent2"],
            url=f"{redis_url}#namespace2",
        )

        conn_ns2_sender.send("agent2", "Hello in namespace2")

        # Should only receive in namespace2
        assert conn_ns2_receiver.peek() == 1

        # Namespace1 agent2 shouldn't see it
        conn_ns1_receiver = MessagingConnector(
            agent_id="agent2",
            agents=["agent1", "agent2"],
            url=f"{redis_url}#namespace1",
        )
        assert conn_ns1_receiver.peek() == 0


class TestPeerExit:
    """A peer that finishes must stop looking like a live correspondent.

    Before this, ``send`` to a departed agent returned success, the message sat unread
    in Redis forever, and ``--wait`` blocked for its full timeout on a reply that could
    never arrive.  Observed in a real run: one agent sent three messages (two blocking)
    to a colleague that had already submitted, was told "Message sent" each time,
    concluded coordination was "ongoing", and shipped a conflicting patch.
    """

    @pytest.fixture
    def alice(self, redis_url):
        return MessagingConnector(agent_id="agent1", agents=["agent1", "agent2"], url=f"{redis_url}#test:exit")

    @pytest.fixture
    def bob(self, redis_url):
        return MessagingConnector(agent_id="agent2", agents=["agent1", "agent2"], url=f"{redis_url}#test:exit")

    def test_send_to_live_peer_reports_delivered(self, alice, bob):
        assert alice.send("agent2", "still here?") is True
        assert len(bob.receive()) == 1

    def test_send_to_exited_peer_reports_not_delivered(self, alice, bob):
        bob.mark_exited()
        assert alice.send("agent2", "are you there?") is False
        assert bob.peek() == 0, "message must not be queued for an agent that has left"

    def test_has_exited_tracks_state(self, alice, bob):
        assert alice.has_exited("agent2") is False
        bob.mark_exited()
        assert alice.has_exited("agent2") is True

    def test_published_flag_distinguishes_publish_success(self, alice, bob):
        bob.mark_exited(published=False)
        assert alice.has_exited("agent2") is True
        assert alice.has_published("agent2") is False, (
            "a failed publish must not be reported as work available on the remote"
        )

    def test_published_flag_set_when_publish_succeeded(self, alice, bob):
        bob.mark_exited(published=True)
        assert alice.has_published("agent2") is True

    def test_send_and_wait_returns_immediately_when_peer_gone(self, alice, bob):
        import time as _t

        bob.mark_exited()
        start = _t.monotonic()
        delivered, replies = alice.send_and_wait("agent2", "hello?", timeout=60)
        elapsed = _t.monotonic() - start
        assert delivered is False
        assert replies == []
        assert elapsed < 5, f"must not block on a departed peer (took {elapsed:.1f}s)"

    def test_send_and_wait_stops_when_peer_exits_mid_wait(self, alice, bob):
        import threading
        import time as _t

        threading.Timer(1.0, bob.mark_exited).start()
        start = _t.monotonic()
        delivered, replies = alice.send_and_wait("agent2", "still working?", timeout=60)
        elapsed = _t.monotonic() - start
        assert delivered is True, "peer was alive at send time"
        assert replies == []
        assert elapsed < 10, f"must abandon the wait once the peer exits (took {elapsed:.1f}s)"

    def test_send_and_wait_returns_reply(self, alice, bob):
        import threading

        threading.Timer(0.5, lambda: bob.send("agent1", "yes, editing core.py")).start()
        delivered, replies = alice.send_and_wait("agent2", "what are you editing?", timeout=30)
        assert delivered is True
        assert len(replies) == 1
        assert replies[0]["content"] == "yes, editing core.py"

    def test_fresh_connector_clears_stale_exit_marker(self, redis_url):
        first = MessagingConnector(agent_id="agent2", agents=["agent1", "agent2"], url=f"{redis_url}#test:stale")
        first.mark_exited()
        # a new run reuses the agent id; it must not start out looking departed
        second = MessagingConnector(agent_id="agent2", agents=["agent1", "agent2"], url=f"{redis_url}#test:stale")
        assert second.has_exited("agent2") is False
