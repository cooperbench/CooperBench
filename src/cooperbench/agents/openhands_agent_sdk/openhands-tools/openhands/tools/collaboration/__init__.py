"""Collaboration tools for inter-agent communication.

Both tools run inside the Modal sandbox, communicating via Redis:
- SendMessageTool: Push message to teammate's inbox
- ReceiveMessageTool: Pop messages from own inbox
"""

from openhands.tools.collaboration.messaging import ReceiveMessageTool, SendMessageTool

__all__ = ["SendMessageTool", "ReceiveMessageTool"]
