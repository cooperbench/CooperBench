"""Evaluation backends - Modal, Docker, GCP Batch, etc."""

from cooperbench.eval.backends.base import EvalBackend, ExecResult, Sandbox
from cooperbench.eval.backends.modal import ModalBackend

__all__ = ["EvalBackend", "Sandbox", "ExecResult", "ModalBackend", "get_backend"]


def get_backend(name: str = "modal") -> EvalBackend:
    """Get an evaluation backend by name.

    Args:
        name: Backend name ("modal", "docker", "gcp_batch")

    Returns:
        EvalBackend instance
    """
    if name == "modal":
        return ModalBackend()
    elif name == "docker":
        # Lazy import to avoid requiring docker package when not used
        from cooperbench.eval.backends.docker import DockerBackend

        return DockerBackend()
    else:
        available = "docker, modal"
        raise ValueError(f"Unknown backend: '{name}'. Available: {available}")
