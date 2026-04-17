"""Evaluation backends - Modal, Docker, GCP Batch, etc."""

from cooperbench.eval.backends.base import EvalBackend, ExecResult, Sandbox
from cooperbench.eval.backends.modal import ModalBackend

__all__ = [
    "EvalBackend",
    "Sandbox",
    "ExecResult",
    "ModalBackend",
    "get_backend",
    "get_batch_evaluator",
]


def get_backend(name: str = "docker") -> EvalBackend:
    """Get an evaluation backend by name.

    For interactive/adhoc evaluation (Docker, Modal).
    For large-scale GCP evaluation, use get_batch_evaluator() instead.

    Args:
        name: Backend name ("modal", "docker", "gcp", "gcp_batch")

    Returns:
        EvalBackend instance
    """
    if name == "modal":
        return ModalBackend()
    elif name == "docker":
        from cooperbench.eval.backends.docker import DockerBackend

        return DockerBackend()
    elif name in ("gcp", "gcp_batch"):
        from cooperbench.eval.backends.gcp import GCPBatchBackend

        return GCPBatchBackend()
    else:
        available = "docker, modal, gcp, gcp_batch"
        raise ValueError(f"Unknown backend: '{name}'. Available: {available}")


def get_batch_evaluator(name: str = "gcp"):
    """Get a batch evaluator for large-scale evaluation.

    Batch evaluators submit ALL tasks at once and run them in parallel,
    which is much more efficient for large-scale evaluation on GCP.

    Args:
        name: Evaluator name ("gcp")

    Returns:
        Batch evaluator instance (e.g., GCPBatchEvaluator)
    """
    if name == "gcp":
        from cooperbench.eval.backends.gcp import GCPBatchEvaluator

        return GCPBatchEvaluator()
    else:
        raise ValueError(f"Unknown batch evaluator: '{name}'. Available: gcp")
