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
