"""
Core modules for the FLIM-FRET analysis pipeline.

This package contains the main pipeline components:
- Configuration management
- Logging system  
- Pipeline orchestration
- Stage execution framework
"""

from .config import Config
from .logger import PipelineLogger
from .pipeline import Pipeline
from .stages import StageBase

__all__ = [
    "Config",
    "PipelineLogger", 
    "Pipeline",
    "StageBase",
] 