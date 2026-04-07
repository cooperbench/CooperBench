"""Redis-based mailbox messaging between agents.

Provides simple send/receive messaging via Redis lists. Each agent has an inbox
that other agents can push messages to.

Example:
    connector = MessagingConnector(
        agent_id="agent1",
        agents=["agent1", "agent2"],
        url="redis://localhost:6379#run:abc123"
    )

    # Send to specific agent
    connector.send("agent2", "I found a bug in auth.py")

    # Receive pending messages
    messages = connector.receive()

    # Broadcast to all
    connector.broadcast("I'm starting on the API changes")
"""

import json
from datetime import datetime
from typing import Any

import redis


class MessagingConnector:
    """Redis-based mailbox messaging between agents."""

    def __init__(self, agent_id: str, agents: list[str], url: str = "redis://localhost:6379"):
        """Initialize messaging connector.

        Args:
            agent_id: This agent's unique identifier (e.g., "agent1")
            agents: List of all agent IDs in the collaboration
            url: Redis URL. Supports namespacing via #prefix (e.g., "redis://host:6379#run:abc")
        """
        self.agent_id = agent_id
        self.agents = agents

        # Parse optional namespace prefix from URL (format: url#prefix)
        if "#" in url:
            url, self._prefix = url.split("#", 1)
            self._prefix += ":"
        else:
            self._prefix = ""

        self._client = redis.from_url(url)
        self._inbox_key = f"{self._prefix}{agent_id}:inbox"

        # Clear stale messages from previous runs
        self._client.delete(self._inbox_key)

    def setup(self, env: Any) -> None:
        """Configure the agent's sandbox for messaging.

        Messaging doesn't require sandbox configuration (it's pure Redis),
        but this method exists for interface consistency with other connectors.

        Args:
            env: The agent's environment (unused for messaging)
        """
        pass

    def send(self, recipient: str, content: str) -> None:
        """Send a message to another agent's inbox.

        Args:
            recipient: Target agent's ID
            content: Message content
        """
        message = {
            "from": self.agent_id,
            "to": recipient,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        self._client.rpush(f"{self._prefix}{recipient}:inbox", json.dumps(message))

    def receive(self) -> list[dict]:
        """Get all pending messages from inbox (empties the inbox).

        Returns:
            List of message dicts with from, to, content, timestamp
        """
        messages = []
        while True:
            msg = self._client.lpop(self._inbox_key)
            if msg is None:
                break
            messages.append(json.loads(msg))
        return messages

    def broadcast(self, content: str) -> None:
        """Send a message to all other agents.

        Args:
            content: Message content
        """
        for agent in self.agents:
            if agent != self.agent_id:
                self.send(agent, content)

    def peek(self) -> int:
        """Check how many messages are waiting without consuming them.

        Returns:
            Number of pending messages
        """
        return self._client.llen(self._inbox_key)
