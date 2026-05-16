"""Shared coop primitives reused by every CLI-style agent adapter.

The Claude Code and Codex adapters both run a third-party CLI inside the
task container.  Whatever the underlying agent looks like, the
CooperBench coop mechanics are identical:

  - submission protocol: write your diff to ``/workspace/repo/patch.txt``
  - messaging:           Redis inbox with one-line shell wrappers
                         (``coop-send`` / ``coop-recv`` / ``coop-broadcast`` /
                         ``coop-peek`` / ``coop-agents``)
  - git:                 a shared ``team`` remote configured at startup,
                         with agent-named branches and a fetch/merge/push
                         workflow

This module centralizes those bits so a new adapter doesn't have to
re-derive them.  The adapter still owns the CLI install, the invocation,
and the parsing of its own output format.
"""

from cooperbench.agents._coop.prompt import build_instruction
from cooperbench.agents._coop.runtime import (
    build_git_setup_command,
    parse_sent_messages_log,
    rewrite_comm_url_for_container,
)

__all__ = [
    "build_git_setup_command",
    "build_instruction",
    "parse_sent_messages_log",
    "rewrite_comm_url_for_container",
]
