"""CooperBench - Multi-agent coordination benchmark.

Run agents on coding tasks and evaluate their patches.

Quick start:
    cooperbench run -n my-exp -r llama_index_task -m gpt-4o
    cooperbench eval -n my-exp

Or use as library:
    from cooperbench import run, evaluate
    run(run_name="test", repo="llama_index_task", model_name="gpt-4o")
    evaluate(run_name="test")
"""

from cooperbench.__about__ import __version__
from cooperbench.eval import discover_runs, evaluate, evaluate_merge, test_merged, test_merged_n, test_solo, test_solo_n
from cooperbench.runner import discover_tasks, run

__all__ = [
    "__version__",
    "run",
    "discover_tasks",
    "evaluate",
    "discover_runs",
    "test_merged",
    "test_merged_n",
    "test_solo",
    "test_solo_n",
    "evaluate_merge",
]
