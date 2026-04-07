"""OpenHands Workspace - Docker and container-based workspace implementations."""

from typing import TYPE_CHECKING

from openhands.sdk.workspace import PlatformType, TargetType

from .apptainer import ApptainerWorkspace
from .cloud import OpenHandsCloudWorkspace
from .docker import DockerWorkspace
from .remote_api import APIRemoteWorkspace


if TYPE_CHECKING:
    from .docker import DockerDevWorkspace
    from .modal import ModalWorkspace

__all__ = [
    "APIRemoteWorkspace",
    "ApptainerWorkspace",
    "DockerDevWorkspace",
    "DockerWorkspace",
    "ModalWorkspace",
    "OpenHandsCloudWorkspace",
    "PlatformType",
    "TargetType",
]


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "DockerDevWorkspace":
        from .docker import DockerDevWorkspace

        return DockerDevWorkspace
    if name == "ModalWorkspace":
        from .modal import ModalWorkspace

        return ModalWorkspace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
