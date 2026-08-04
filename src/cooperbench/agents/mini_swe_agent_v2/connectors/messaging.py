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
import time
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
        self._client.delete(self._exited_key(agent_id))

    def _exited_key(self, agent_id: str) -> str:
        return f"{self._prefix}{agent_id}:exited"

    def mark_exited(self, published: bool = False) -> None:
        """Record that this agent has finished, so peers stop waiting on it.

        Without this a peer's ``send`` silently succeeds into an inbox nobody will ever
        read again, and ``send_and_wait`` blocks for its full timeout on a reply that
        cannot come.

        ``published`` records whether this agent's submitted patch actually reached the
        shared remote.  Peers are told to go read that branch, so they must only be told
        that when it is true — publication is best-effort and can fail.
        """
        try:
            self._client.set(self._exited_key(self.agent_id), "published" if published else "1")
        except redis.RedisError:  # never let bookkeeping take down a run
            pass

    def has_exited(self, agent_id: str) -> bool:
        """True when ``agent_id`` has finished its work and left."""
        try:
            return bool(self._client.exists(self._exited_key(agent_id)))
        except redis.RedisError:
            return False

    def has_published(self, agent_id: str) -> bool:
        """True when ``agent_id``'s submitted patch is actually on the shared remote."""
        try:
            raw = self._client.get(self._exited_key(agent_id))
        except redis.RedisError:
            return False
        if raw is None:
            return False
        if isinstance(raw, bytes):
            raw = raw.decode()
        return raw == "published"

    def setup(self, env: Any) -> None:
        """Configure the agent's sandbox for messaging.

        Messaging doesn't require sandbox configuration (it's pure Redis),
        but this method exists for interface consistency with other connectors.

        Args:
            env: The agent's environment (unused for messaging)
        """
        pass

    def send(self, recipient: str, content: str) -> bool:
        """Send a message to another agent's inbox.

        Args:
            recipient: Target agent's ID
            content: Message content

        Returns:
            ``True`` if the message was queued, ``False`` if the recipient has already
            finished and left — in which case nothing will ever read it.  Callers must
            surface that to the agent instead of reporting success.
        """
        if self.has_exited(recipient):
            return False
        message = {
            "from": self.agent_id,
            "to": recipient,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        self._client.rpush(f"{self._prefix}{recipient}:inbox", json.dumps(message))
        return True

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

    def send_and_wait(self, recipient: str, content: str, timeout: int = 60) -> tuple[bool, list[dict]]:
        """Send, then block until the recipient replies, they exit, or ``timeout``.

        Returns ``(delivered, replies)``.  ``delivered`` is False when the recipient had
        already finished before the send.  The wait also ends the moment the recipient
        exits, so an agent never burns the full timeout on a reply that cannot arrive.
        """
        if not self.send(recipient, content):
            return False, []

        deadline = time.monotonic() + timeout
        replies: list[dict] = []
        while time.monotonic() < deadline:
            got = self.receive()
            if got:
                replies.extend(got)
                break
            if self.has_exited(recipient):
                break
            time.sleep(1.0)
        return True, replies

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
