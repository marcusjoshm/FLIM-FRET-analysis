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
    VisualizeSegmentedStage,
    LifetimeImagesStage,
    AverageLifetimeStage,
    ApplyMaskStage
)


def register_all_stages():
    """Register all available pipeline stages."""
    
    # Core processing stages (in execution order)
    register_stage('preprocessing', order=1)(PreprocessingStage)
    register_stage('processing', order=2)(ProcessingStage)
    
    # Visualization stages
    register_stage('phasor_visualization', order=3)(PhasorVisualizationStage)
    register_stage('visualize_segmented', order=11)(VisualizeSegmentedStage)
    
    # Segmentation stages
    register_stage('phasor_segmentation', order=4)(PhasorSegmentationStage)
    
    # Mask processing
    register_stage('apply_mask', order=7)(ApplyMaskStage)
    
    # Lifetime processing
    register_stage('lifetime_images', order=10)(LifetimeImagesStage)
    register_stage('average_lifetime', order=12)(AverageLifetimeStage) 