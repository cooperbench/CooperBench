"""
CooperBench: A benchmark for multi-agent coordination in code collaboration.

This package provides the core framework for defining, running, and evaluating
experiments involving AI agents collaborating on software engineering tasks.
"""

from cooperbench.__about__ import __version__
from cooperbench.core.settings import BenchSetting
from cooperbench.core.interface import FileInterface

__all__ = [
    "__version__",
    "BenchSetting",
    "FileInterface",
]
