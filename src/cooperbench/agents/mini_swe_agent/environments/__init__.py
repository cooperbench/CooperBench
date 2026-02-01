"""Execution environments for mini-swe-agent."""

from cooperbench.agents.mini_swe_agent.environments.modal import ModalEnvironment

__all__ = ["ModalEnvironment", "get_environment"]


def get_environment(name: str = "modal"):
    """Get an environment by name.

    Args:
        name: Environment name ("modal" or "docker")

    Returns:
        Environment class
    """
    if name == "modal":
        return ModalEnvironment
    elif name == "docker":
        # Lazy import to avoid requiring docker package when not used
        from cooperbench.agents.mini_swe_agent.environments.docker import DockerEnvironment

        return DockerEnvironment
    else:
        raise ValueError(f"Unknown environment: '{name}'. Available: docker, modal")
