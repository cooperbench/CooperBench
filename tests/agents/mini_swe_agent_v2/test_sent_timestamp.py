"""sent_messages must keep the enqueue timestamp, including after --wait."""

import json
from unittest.mock import MagicMock

from cooperbench.agents.mini_swe_agent_v2.agents.default import DefaultAgent
from cooperbench.agents.mini_swe_agent_v2.connectors.messaging import MessagingConnector

ENQUEUED = "2026-05-13T22:47:00"


class _Comm:
    """Stand-in that stamps at enqueue and only then pretends to wait."""

    def __init__(self):
        self.last_enqueued_timestamp = None

    def send(self, recipient, content):
        self.last_enqueued_timestamp = ENQUEUED
        return True

    def send_and_wait(self, recipient, content, timeout=60):
        self.last_enqueued_timestamp = ENQUEUED
        return True, [{"from": "agent2", "content": "ack"}]

    def has_exited(self, agent_id):
        return False

    def has_published(self, agent_id):
        return False


def _agent(comm):
    return DefaultAgent(
        model=MagicMock(),
        env=MagicMock(),
        comm=comm,
        system_template="sys",
        instance_template="task",
    )


def test_plain_send_stores_enqueue_timestamp():
    agent = _agent(_Comm())
    result = agent._handle_send_message({"recipient": "agent2", "content": "ask", "wait": False})
    assert result["returncode"] == 0
    assert agent.sent_messages == [{"to": "agent2", "content": "ask", "timestamp": ENQUEUED}]


def test_blocking_send_stores_enqueue_timestamp_not_the_reply_time():
    agent = _agent(_Comm())
    result = agent._handle_send_message({"recipient": "agent2", "content": "ask", "wait": True})
    assert result["returncode"] == 0
    assert "ack" in result["output"]
    assert agent.sent_messages == [{"to": "agent2", "content": "ask", "timestamp": ENQUEUED}]


def test_undelivered_send_is_not_recorded():
    comm = _Comm()
    comm.send = lambda recipient, content: False
    agent = _agent(comm)
    result = agent._handle_send_message({"recipient": "agent2", "content": "late", "wait": False})
    assert result["returncode"] == 1
    assert agent.sent_messages == []


def test_connector_send_publishes_the_same_timestamp_it_enqueues():
    connector = MessagingConnector.__new__(MessagingConnector)
    connector.agent_id = "agent1"
    connector._prefix = ""
    connector._client = MagicMock()
    connector.has_exited = lambda recipient: False
    connector.last_enqueued_timestamp = None

    assert connector.send("agent2", "hello") is True
    payload = json.loads(connector._client.rpush.call_args.args[1])
    assert connector.last_enqueued_timestamp == payload["timestamp"]
    assert payload["timestamp"]
