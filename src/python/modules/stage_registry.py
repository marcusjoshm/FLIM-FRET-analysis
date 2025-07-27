"""
Stage Registration Module

Registers all available pipeline stages and their implementations.
"""

from ..core.stages import register_stage
from .stage_classes import (
    PreprocessingStage,
    ProcessingStage,
    PhasorVisualizationStage,
    PhasorSegmentationStage,
    LifetimeImagesStage,
    AverageLifetimeStage
)


def register_all_stages():
    """Register all available pipeline stages."""
    
    # Core processing stages (in execution order)
    register_stage('preprocessing', order=1)(PreprocessingStage)
    register_stage('processing', order=2)(ProcessingStage)
    
    # Visualization stages
    register_stage('phasor_visualization', order=3)(PhasorVisualizationStage)
    
    # Segmentation stages
    register_stage('phasor_segmentation', order=4)(PhasorSegmentationStage)
    
    # Lifetime processing
    register_stage('lifetime_images', order=10)(LifetimeImagesStage)
    register_stage('average_lifetime', order=12)(AverageLifetimeStage) 