"""
mini-swe-agent - A Minimal AI Agent for Software Engineering

Source: https://github.com/SWE-agent/mini-swe-agent
Version: 1.17.3
License: MIT
Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez

This code is copied directly from mini-swe-agent with minimal modifications
(only import paths changed from 'minisweagent' to 'cooperbench.agents.mini_swe').

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

__version__ = "1.17.3"

import os
from pathlib import Path
from typing import Any, Protocol

import dotenv
from platformdirs import user_config_dir

from cooperbench.agents.mini_swe.utils.log import logger

package_dir = Path(__file__).resolve().parent

global_config_dir = Path(os.getenv("MSWEA_GLOBAL_CONFIG_DIR") or user_config_dir("mini-swe-agent"))
global_config_dir.mkdir(parents=True, exist_ok=True)
global_config_file = Path(global_config_dir) / ".env"

if not os.getenv("MSWEA_SILENT_STARTUP"):
    logger.info(f"This is mini-swe-agent version {__version__}. Loading global config from '{global_config_file}'")
dotenv.load_dotenv(dotenv_path=global_config_file)


# === Protocols ===
# You can ignore them unless you want static type checking.


class Model(Protocol):
    """Protocol for language models."""

    config: Any
    cost: float
    n_calls: int

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict: ...

    def get_template_vars(self) -> dict[str, Any]: ...


class Environment(Protocol):
    """Protocol for execution environments."""

    config: Any

    def execute(self, command: str, cwd: str = "") -> dict[str, str]: ...

    def get_template_vars(self) -> dict[str, Any]: ...


class Agent(Protocol):
    """Protocol for agents."""

    model: Model
    env: Environment
    messages: list[dict[str, str]]
    config: Any

    def run(self, task: str, **kwargs) -> tuple[str, str]: ...


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
