"""
mini-swe-agent v2 - A Minimal AI Agent for Software Engineering (Tool-calling version)

Source: https://github.com/SWE-agent/mini-swe-agent
Version: 2.1.0 (commit 56613dd)
License: MIT
Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez

This code is copied directly from mini-swe-agent with modifications:
- Import paths changed from 'minisweagent' to 'cooperbench.agents.mini_swe_agent_v2'
- Removed text-based (regex) action parsing (v1 compat), keeping only tool calling
- Removed interactive agent, non-Docker environments, non-litellm models
- Added inter-agent communication (messaging + git connectors) for CooperBench

Citation:
    @misc{minisweagent2025,
        title={mini-swe-agent: A Minimal AI Agent for Software Engineering},
        author={Lieret, Kilian A. and Jimenez, Carlos E.},
        year={2025},
        url={https://github.com/SWE-agent/mini-swe-agent}
    }

This file provides:
- Path settings for global config file & relative directories
- Version numbering
- Protocols for the core components of mini-swe-agent.
"""

__version__ = "2.1.0"

import os
from pathlib import Path
from typing import Any, Protocol

import dotenv
from platformdirs import user_config_dir

from cooperbench.agents.mini_swe_agent_v2.utils.log import logger

package_dir = Path(__file__).resolve().parent

global_config_dir = Path(os.getenv("MSWEA_GLOBAL_CONFIG_DIR") or user_config_dir("mini-swe-agent"))
global_config_dir.mkdir(parents=True, exist_ok=True)
global_config_file = Path(global_config_dir) / ".env"

dotenv.load_dotenv(dotenv_path=global_config_file)


# === Protocols ===
# You can ignore them unless you want static type checking.


class Model(Protocol):
    """Protocol for language models."""

    config: Any

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict: ...

    def format_message(self, **kwargs) -> dict: ...

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...


class Environment(Protocol):
    """Protocol for execution environments."""

    config: Any

    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...


class Agent(Protocol):
    """Protocol for agents."""

    config: Any

    def run(self, task: str, **kwargs) -> dict: ...

    def save(self, path: Path | None, *extra_dicts) -> dict: ...


__all__ = [
    "Agent",
    "Model",
    "Environment",
    "package_dir",
    "__version__",
    "global_config_file",
    "global_config_dir",
    "logger",
]
