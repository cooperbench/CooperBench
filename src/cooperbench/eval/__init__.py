"""Evaluation package - test patches and evaluate runs."""

from cooperbench.eval.evaluate import evaluate
from cooperbench.eval.runs import discover_runs
from cooperbench.eval.sandbox import evaluate_merge, run_patch_test, test_merged, test_merged_n, test_solo, test_solo_n

__all__ = [
    "evaluate",
    "discover_runs",
    "test_merged",
    "test_merged_n",
    "test_solo",
    "test_solo_n",
    "run_patch_test",
    "evaluate_merge",
]
