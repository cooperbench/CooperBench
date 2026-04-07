"""Runtime tools package."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("openhands-tools")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for editable/unbuilt environments

# Auto-import collaboration tools to register them in the global tool registry
# This ensures they're available when the agent-server looks them up by name
try:
    from openhands.tools import collaboration  # noqa: F401
except ImportError:
    pass
