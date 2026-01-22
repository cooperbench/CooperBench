"""
Benchmark experiment settings and configuration.

This module defines the core experiment settings used throughout CooperBench
for controlling how agents collaborate on coding tasks.

Example usage:
    from cooperbench.core.settings import BenchSetting
    
    setting = BenchSetting.COOP
    print(setting.value)  # "coop"
"""

from enum import Enum


class BenchSetting(str, Enum):
    """Benchmark experiment settings.
    
    Defines the different modes for running experiments:
    - SINGLE: Single agent working on a single feature
    - SOLO: Single agent working on two features simultaneously
    - COOP: Two agents collaborating with communication
    - COOP_ABLATION: Two agents without communication (ablation study)
    """
    
    SINGLE = "single"
    SOLO = "solo"
    COOP = "coop"
    COOP_ABLATION = "coop_ablation"
