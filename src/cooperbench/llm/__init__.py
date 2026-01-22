"""
LLM (Large Language Model) integration for CooperBench.

This subpackage provides an abstraction layer for interacting with various
LLM providers via litellm, enabling agents to leverage different models
for planning and execution.
"""

from cooperbench.llm.client import call_llm

__all__ = ["call_llm"]
