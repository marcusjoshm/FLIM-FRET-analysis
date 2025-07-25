"""
Processing modules for FLIM-FRET analysis.

This package contains all the processing modules:
- Preprocessing and file conversion
- Wavelet filtering
- Segmentation algorithms
- Visualization tools
"""

# Import stage registration function
from .stage_registry import register_all_stages

__all__ = [
    "register_all_stages",
] 