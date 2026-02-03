"""Agent framework registry.

Allows registering and retrieving agent adapters by name.

Usage:
    from cooperbench.agents.registry import register, get_runner

    @register("my_agent")
    class MyAgentRunner:
        def run(self, task, image, **kwargs) -> AgentResult:
            ...

    runner = get_runner("my_agent")
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cooperbench.agents import AgentRunner

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator to register an agent adapter.

    Args:
        name: Unique name for this agent framework

    Example:
        @register("mini_swe")
        class MiniSweAgentRunner:
            def run(self, task, image, **kwargs) -> AgentResult:
                ...
    """

    def decorator(cls):
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_runner(name: str, **kwargs: Any) -> "AgentRunner":
    """Get an agent runner instance by name.

    Args:
        name: Name of the registered agent framework
        **kwargs: Arguments to pass to the runner constructor

    Returns:
        Instance of the agent runner

    Raises:
        ValueError: If agent name is not registered
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown agent: '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_agents() -> list[str]:
    """List all registered agent framework names."""
    return sorted(_REGISTRY.keys())


# Auto-import adapters to trigger registration
def _auto_register():
    """Import all adapter modules to register them."""
    # Built-in agents
    try:
        import cooperbench.agents.mini_swe_agent.adapter  # noqa: F401
    except ImportError:
        pass
    try:
        import cooperbench.agents.swe_agent.adapter  # noqa: F401
    except ImportError:
        pass
    try:
        import cooperbench.agents.openhands_agent_sdk.adapter  # noqa: F401
    except ImportError:
        pass

    # External agents via environment variable
    import os

    external_agents = os.getenv("COOPERBENCH_EXTERNAL_AGENTS", "")
    if external_agents:
        for module_path in external_agents.split(","):
            module_path = module_path.strip()
            if module_path:
                try:
                    __import__(module_path)
                except ImportError as e:
                    import warnings

                    warnings.warn(f"Could not import external agent module '{module_path}': {e}", stacklevel=2)


_auto_register()
