"""
FLIM-FRET Analysis Pipeline

A comprehensive tool for automated Fluorescence Lifetime Imaging Microscopy (FLIM) 
and Förster Resonance Energy Transfer (FRET) analysis.
"""

__version__ = "2.0.0"
__author__ = "Joshua Marcus"
__email__ = "joshua.marcus@bcm.edu"

# Package information
__title__ = "FLIM-FRET Analysis Pipeline"
__description__ = "Automated FLIM-FRET analysis from raw .bin files to complete analysis"
__url__ = "https://github.com/marcusjoshm/flimfret"

# Import main components for easier access
from .python.core.pipeline import Pipeline
from .python.core.config import Config
from .python.core.logger import PipelineLogger

__all__ = [
    "Pipeline",
    "Config", 
    "PipelineLogger",
] 