"""Messaging tools for inter-agent communication.

Both sending and receiving happen inside the Modal sandbox via Redis.

Tools:
- SendMessageTool: Push message to teammate's inbox
- ReceiveMessageTool: Pop messages from own inbox

Supports URL fragment namespacing for shared Redis:
    redis://host:port#run:abc123
Keys become: run:abc123:agent1:inbox instead of agent1:inbox
"""

import json
import os
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


# =============================================================================
# SendMessageTool
# =============================================================================


class SendMessageAction(Action):
    """Schema for sending a message to another agent."""
    
    recipient: str = Field(
        description="The agent ID to send the message to"
    )
    content: str = Field(
        description="The message content to send"
    )


class SendMessageObservation(Observation):
    """Observation after sending a message."""
    
    success: bool = Field(description="Whether the message was sent successfully")
    recipient: str = Field(description="The agent the message was sent to")
    info: str = Field(default="", description="Additional information")
    
    @property
    def text(self) -> str:
        """Format observation for display to agent."""
        if self.success:
            return f"Message sent successfully to {self.recipient}."
        return f"Failed to send message to {self.recipient}: {self.info}"
    
    @property
    def to_llm_content(self):
        """Return content for LLM to see."""
        from openhands.sdk.llm import TextContent
        return [TextContent(text=self.text)]


class SendMessageExecutor(ToolExecutor[SendMessageAction, SendMessageObservation]):
    """Executor that sends messages via Redis."""
    
    def __init__(self, redis_url: str, agent_id: str, agents: list[str]):
        import redis
        self.agent_id = agent_id
        self.agents = agents
        
        # Parse optional namespace prefix from URL (format: url#prefix)
        # This allows multiple concurrent runs to share one Redis server
        if "#" in redis_url:
            redis_url, prefix = redis_url.split("#", 1)
            self._prefix = prefix + ":"
        else:
            self._prefix = ""
        
        self.redis_url = redis_url
        # Configure Redis client with retry logic and timeouts
        self._client = redis.from_url(
            redis_url,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )
    
    def __call__(self, action: SendMessageAction, conversation=None) -> SendMessageObservation:
        if action.recipient not in self.agents:
            return SendMessageObservation(
                success=False,
                recipient=action.recipient,
                info=f"Unknown agent '{action.recipient}'. Available: {self.agents}",
            )
        
        if action.recipient == self.agent_id:
            return SendMessageObservation(
                success=False,
                recipient=action.recipient,
                info="Cannot send a message to yourself",
            )
        
        try:
            message = {
                "from": self.agent_id,
                "to": action.recipient,
                "content": action.content,
                "timestamp": datetime.now().isoformat(),
            }
            inbox_key = f"{self._prefix}{action.recipient}:inbox"
            inbox_result = self._client.rpush(inbox_key, json.dumps(message))
            
            # Also store in a log key for conversation extraction (not consumed)
            log_key = f"{self._prefix}{self.agent_id}:sent_messages"
            log_result = self._client.rpush(log_key, json.dumps(message))
            
            return SendMessageObservation(
                success=True,
                recipient=action.recipient,
                info=f"Message sent to {action.recipient}",
            )
        except Exception as e:
            return SendMessageObservation(
                success=False,
                recipient=action.recipient,
                info=f"Failed to send: {e}",
            )


SEND_MESSAGE_DESCRIPTION = """Send a message to another agent in your team.

Messages are delivered to the recipient's inbox.
"""


class SendMessageTool(ToolDefinition[SendMessageAction, SendMessageObservation]):
    """Tool for sending messages to other agents."""
    
    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        redis_url: str | None = None,
        agent_id: str | None = None,
        agents: list[str] | None = None,
    ) -> Sequence["SendMessageTool"]:
        _ = conv_state
        
        redis_url = redis_url or os.environ.get("REDIS_URL")
        agent_id = agent_id or os.environ.get("AGENT_ID", "agent")
        agents_str = os.environ.get("AGENTS", "")
        agents = agents or (agents_str.split(",") if agents_str else [])
        
        if not redis_url or not agents or len(agents) <= 1:
            return []
        
        executor = SendMessageExecutor(
            redis_url=redis_url,
            agent_id=agent_id,
            agents=agents,
        )
        
        teammates = [a for a in agents if a != agent_id]
        enhanced_description = (
            f"{SEND_MESSAGE_DESCRIPTION}\n"
            f"You are {agent_id}. Teammates: {', '.join(teammates)}"
        )
        
        return [
            cls(
                description=enhanced_description,
                action_type=SendMessageAction,
                observation_type=SendMessageObservation,
                annotations=ToolAnnotations(
                    title="send_message",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


# =============================================================================
# ReceiveMessageTool
# =============================================================================


class ReceiveMessageAction(Action):
    """Schema for receiving messages (no parameters needed)."""
    pass


class ReceiveMessageObservation(Observation):
    """Observation with received messages."""
    
    messages: list[dict] = Field(default_factory=list, description="Received messages")
    count: int = Field(default=0, description="Number of messages received")
    error: str | None = Field(default=None, description="Error message if receiving failed")
    
    @property
    def text(self) -> str:
        """Format messages for display to agent."""
        if self.error:
            return f"Error receiving messages: {self.error}"
        
        if not self.messages:
            return "No new messages."
        
        lines = []
        for msg in self.messages:
            sender = msg.get("from", "unknown")
            content = msg.get("content", "")
            lines.append(f"[Message from {sender}]: {content}")
        return "\n".join(lines)
    
    @property
    def to_llm_content(self):
        """Return content for LLM to see."""
        from openhands.sdk.llm import TextContent
        return [TextContent(text=self.text)]


class ReceiveMessageExecutor(ToolExecutor[ReceiveMessageAction, ReceiveMessageObservation]):
    """Executor that receives messages from Redis inbox."""
    
    def __init__(self, redis_url: str, agent_id: str):
        import redis
        self.agent_id = agent_id
        
        # Parse optional namespace prefix from URL (format: url#prefix)
        if "#" in redis_url:
            redis_url, prefix = redis_url.split("#", 1)
            self._prefix = prefix + ":"
        else:
            self._prefix = ""
        
        self.redis_url = redis_url
        # Configure Redis client with retry logic and timeouts
        self._client = redis.from_url(
            redis_url,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        print(f"[ReceiveMessageExecutor] Initialized with prefix='{self._prefix}' for agent={agent_id}", flush=True)
    
    def __call__(self, action: ReceiveMessageAction, conversation=None) -> ReceiveMessageObservation:
        try:
            inbox_key = f"{self._prefix}{self.agent_id}:inbox"
            messages = []
            
            # Pop all messages from inbox
            while True:
                raw = self._client.lpop(inbox_key)
                if raw is None:
                    break
                try:
                    msg = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                    messages.append(msg)
                except json.JSONDecodeError:
                    continue
            
            return ReceiveMessageObservation(
                messages=messages,
                count=len(messages),
            )
        except Exception as e:
            return ReceiveMessageObservation(
                messages=[],
                count=0,
                error=str(e),
            )


RECEIVE_MESSAGE_DESCRIPTION = """Check for messages from teammates.

Call this periodically to see if teammates have sent you messages.
Messages are removed from your inbox once received.
"""


class ReceiveMessageTool(ToolDefinition[ReceiveMessageAction, ReceiveMessageObservation]):
    """Tool for receiving messages from other agents."""
    
    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        redis_url: str | None = None,
        agent_id: str | None = None,
        agents: list[str] | None = None,
    ) -> Sequence["ReceiveMessageTool"]:
        _ = conv_state
        
        redis_url = redis_url or os.environ.get("REDIS_URL")
        agent_id = agent_id or os.environ.get("AGENT_ID", "agent")
        agents_str = os.environ.get("AGENTS", "")
        agents = agents or (agents_str.split(",") if agents_str else [])
        
        # Debug: log what we got from env
        print(f"[ReceiveMessageTool.create] redis_url={redis_url}, agent_id={agent_id}, agents={agents}", flush=True)
        
        if not redis_url or not agents or len(agents) <= 1:
            print(f"[ReceiveMessageTool.create] SKIPPING - conditions not met", flush=True)
            return []
        
        executor = ReceiveMessageExecutor(
            redis_url=redis_url,
            agent_id=agent_id,
        )
        
        return [
            cls(
                description=RECEIVE_MESSAGE_DESCRIPTION,
                action_type=ReceiveMessageAction,
                observation_type=ReceiveMessageObservation,
                annotations=ToolAnnotations(
                    title="receive_messages",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=False,  # Messages are consumed
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


# Register tools
register_tool(SendMessageTool.name, SendMessageTool)
register_tool(ReceiveMessageTool.name, ReceiveMessageTool)
