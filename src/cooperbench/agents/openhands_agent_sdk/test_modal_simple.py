#!/usr/bin/env python3
"""Test Modal workspace with OpenHands SDK."""

import os
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add local SDK paths
SDK_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SDK_ROOT, "openhands-sdk"))
sys.path.insert(0, os.path.join(SDK_ROOT, "openhands-tools"))
sys.path.insert(0, os.path.join(SDK_ROOT, "openhands-workspace"))
sys.path.insert(0, os.path.join(SDK_ROOT, "openhands-agent-server"))

# Test import
print("Testing imports...")
try:
    from openhands.workspace.modal import ModalWorkspace
    print("✅ ModalWorkspace imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test instantiation (won't actually start sandbox without Modal token)
print("\nModalWorkspace fields:")
import inspect
for name, field in ModalWorkspace.model_fields.items():
    print(f"  {name}: {field.annotation} = {field.default}")

print("\n✅ ModalWorkspace is ready to use!")
print("\nUsage:")
print("""
from openhands.workspace.modal import ModalWorkspace
from openhands.sdk import Agent, LLM, Conversation

with ModalWorkspace(
    image="ghcr.io/openhands/agent-server:latest-python",
    cpu=2.0,
    memory=4096,
) as workspace:
    conversation = Conversation(agent=agent, workspace=workspace)
    conversation.send_message("Hello!")
    conversation.run()
""")
