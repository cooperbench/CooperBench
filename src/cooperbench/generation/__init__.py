"""Task generation package - automated creation of new benchmark features."""

from cooperbench.generation.generator import generate_feature
from cooperbench.generation.prompt import build_prompt
from cooperbench.generation.splitter import split_patch
from cooperbench.generation.validator import (
    check_conflicts_in_sandbox,
    validate_generated_feature,
)

__all__ = [
    "generate_feature",
    "build_prompt",
    "split_patch",
    "check_conflicts_in_sandbox",
    "validate_generated_feature",
]
