"""
Stage Registration Module

Registers all available pipeline stages and their implementations.
"""

from ..core.stages import register_stage
from .stage_classes import (
    PreprocessingStage,
    ProcessingStage,
    PhasorVisualizationStage,
    GMMSegmentationStage,
    ManualSegmentationStage,
    ManualSegmentationUnfilteredStage,
    ManualSegmentFromMaskStage,
    ManualSegmentUnfilteredFromMaskStage,
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
    register_stage('gmm_segmentation', order=4)(GMMSegmentationStage)
    register_stage('manual_segmentation', order=5)(ManualSegmentationStage)
    register_stage('manual_segmentation_unfiltered', order=6)(ManualSegmentationUnfilteredStage)
    
    # Mask processing
    register_stage('apply_mask', order=7)(ApplyMaskStage)
    register_stage('manual_segment_from_mask', order=8)(ManualSegmentFromMaskStage)
    register_stage('manual_segment_unfiltered_from_mask', order=9)(ManualSegmentUnfilteredFromMaskStage)
    
    # Lifetime processing
    register_stage('lifetime_images', order=10)(LifetimeImagesStage)
    register_stage('average_lifetime', order=12)(AverageLifetimeStage) 