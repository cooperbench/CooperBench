"""Oracle mode — injects ground-truth solution patches into agent prompts.

This module extends CooperBench with oracle evaluation modes:

- oracle_coop: Two agents, each receives their own ground-truth patch upfront.
- oracle_solo: One agent receives both ground-truth patches upfront.
- oracle_coop_full: Two agents, each receives BOTH ground-truth patches.

The central research question: does the curse-of-coordination persist even when
agents already know the correct solution?  If oracle_coop still scores lower than
oracle_solo, coordination itself (not implementation difficulty) is the bottleneck.
"""

from cooperbench.oracle.metrics import FaithfulnessLevel, FaithfulnessResult, compute_faithfulness
from cooperbench.oracle.prompt import OracleMode, build_oracle_task
from cooperbench.oracle.runner import execute_oracle_coop, execute_oracle_solo

__all__ = [
    "OracleMode",
    "build_oracle_task",
    "execute_oracle_coop",
    "execute_oracle_solo",
    "FaithfulnessLevel",
    "FaithfulnessResult",
    "compute_faithfulness",
]
