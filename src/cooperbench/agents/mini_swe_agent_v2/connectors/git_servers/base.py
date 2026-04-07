"""Base protocol for git servers."""

from typing import Protocol


class GitServer(Protocol):
    """Abstract git server for code collaboration.

    Git servers provide a shared repository that agents can push to and pull from.
    Different implementations can use Docker, Modal, GCP VMs, etc.
    """

    @property
    def url(self) -> str:
        """Git URL for agents to use as remote.

        Returns:
            Git URL for the repository (e.g., git://hostname:port/repo.git)
        """
        ...

    def cleanup(self) -> None:
        """Terminate and clean up the git server resources."""
        ...
