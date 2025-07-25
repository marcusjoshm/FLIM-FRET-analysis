"""
All Pipeline Stage Classes for FLIM-FRET Analysis

This file consolidates all stage class definitions for easier management.
"""

from ..core.stages import StageBase
from ..core.config import Config
from ..core.logger import PipelineLogger
import os
import sys
from pathlib import Path
from typing import Dict, Any

# --- Preprocessing Stage ---
class PreprocessingStage(StageBase):
    """
    Preprocessing stage that converts .bin files to .tif and performs phasor transformation.
    """
    def __init__(self, config: Config, logger: PipelineLogger, stage_name: str):
        super().__init__(config, logger, stage_name)
        try:
            from .preprocessing import run_preprocessing
            self.run_preprocessing = run_preprocessing
            self.preprocessing_available = True
        except ImportError as e:
            self.logger.error(f"Could not import preprocessing module: {e}")
            self.run_preprocessing = None
            self.preprocessing_available = False
    def get_description(self) -> str:
        return "Convert .bin files to .tif and perform phasor transformation"
    def validate_inputs(self, **kwargs) -> bool:
        if not self.preprocessing_available:
            self.logger.error("Preprocessing module not available")
            return False
        input_dir = kwargs.get('input_dir')
        if not input_dir or not os.path.exists(input_dir):
            self.logger.error(f"Input directory does not exist: {input_dir}")
            return False
        bin_files_found = False
        for root, dirs, files in os.walk(input_dir):
            if any(f.endswith('.bin') for f in files):
                bin_files_found = True
                break
        if not bin_files_found:
            self.logger.error(f"No .bin files found in input directory: {input_dir}")
            return False
        calibration_file = kwargs.get('calibration_file')
        if calibration_file and not os.path.exists(calibration_file):
            self.logger.warning(f"Calibration file not found: {calibration_file}")
        return True
    def run(self, **kwargs) -> bool:
        input_dir = kwargs['input_dir']
        directories = kwargs['directories']
        calibration_file = kwargs['calibration_file']
        interactive = kwargs.get('interactive', False)
        data_dir = self._find_data_directory(input_dir)
        self.logger.info(f"Processing BIN files from: {data_dir}")
        self.logger.info(f"Output directory: {directories['output']}")
        self.logger.info(f"Preprocessed directory: {directories['preprocessed']}")
        try:
            if not kwargs.get('process_all', False):
                print("\n=== Preprocessing File Selection ===")
                print("Choose how to select files for preprocessing:")
                print("  [1] Process all .bin files (default)")
                print("  [2] Select specific .bin files interactively")
                while True:
                    choice = input("Select option (1 or 2, default: 1): ").strip()
                    if choice == "" or choice == "1":
                        interactive_file_selection = False
                        print(" Processing all .bin files")
                        break
                    elif choice == "2":
                        interactive_file_selection = True
                        print(" Interactive file selection enabled")
                        break
                    else:
                        print("Please enter 1 or 2.")
                if interactive_file_selection:
                    success = self._run_interactive_preprocessing(
                        data_dir, directories, calibration_file, input_dir
                    )
                else:
                    success = self._run_batch_preprocessing(
                        data_dir, directories, calibration_file, input_dir
                    )
            else:
                success = self._run_batch_preprocessing(
                    data_dir, directories, calibration_file, input_dir
                )
            if success:
                self.logger.info("Preprocessing completed successfully")
            else:
                self.logger.error("Preprocessing failed")
            return success
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {str(e)}", e)
            return False
    def _find_data_directory(self, input_dir: str) -> str:
        if any(f.endswith('.bin') for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))):
            return input_dir
        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            if os.path.isdir(item_path):
                bin_files = [f for f in os.listdir(item_path) if f.endswith('.bin')]
                if bin_files:
                    self.logger.info(f"Found BIN files in subdirectory: {item_path}")
                    return item_path
        return input_dir
    def _run_interactive_preprocessing(self, data_dir: str, directories: Dict[str, Path], calibration_file: str, raw_data_root: str) -> bool:
        self.logger.info("Running preprocessing with interactive file selection")
        success = self.run_preprocessing(
            self.config.to_dict(),
            data_dir,
            str(directories['output']),
            str(directories['preprocessed']),
            calibration_file,
            raw_data_root,
            interactive_file_selection=True
        )
        return success
    def _run_batch_preprocessing(self, data_dir: str, directories: Dict[str, Path], calibration_file: str, raw_data_root: str) -> bool:
        self.logger.info("Running preprocessing in batch mode")
        success = self.run_preprocessing(
            self.config.to_dict(),
            data_dir,
            str(directories['output']),
            str(directories['preprocessed']),
            calibration_file,
            raw_data_root
        )
        return success



# --- Processing Stage ---
class ProcessingStage(StageBase):
    """
    Processing stage that applies complex wavelet transforms, calculates lifetimes, and generates NPZ files.
    """
    def __init__(self, config: Config, logger: PipelineLogger, stage_name: str):
        super().__init__(config, logger, stage_name)
        try:
            from .wavelet_filter import main as run_processing
            self.run_processing = run_processing
            self.processing_available = True
        except ImportError as e:
            self.logger.error(f"Could not import processing module: {e}")
            self.run_processing = None
            self.processing_available = False
    def get_description(self) -> str:
        return "Process TIFF files: apply wavelet filtering, calculate lifetimes, and generate NPZ files"
    def validate_inputs(self, **kwargs) -> bool:
        if not self.processing_available:
            self.logger.error("Processing module not available")
            return False
        directories = kwargs.get('directories', {})
        output_dir = directories.get('output')
        if not output_dir or not output_dir.exists():
            self.logger.error(f"Output directory does not exist: {output_dir}")
            return False
        tiff_files_found = False
        for root, dirs, files in os.walk(output_dir):
            if any(f.lower().endswith(('.tif', '.tiff')) for f in files):
                tiff_files_found = True
                break
        if not tiff_files_found:
            self.logger.error(f"No TIFF files found in output directory: {output_dir}")
            return False
        return True
    def setup(self, **kwargs) -> bool:
        if not self.config.get("microscope_params.frequency"):
            self.config.set("microscope_params.frequency", 78.0)
            self.logger.info("Set default frequency: 78.0 MHz")
        if not self.config.get("microscope_params.harmonic"):
            self.config.set("microscope_params.harmonic", 1)
            self.logger.info("Set default harmonic: 1")
        return True
    def run(self, **kwargs) -> bool:
        directories = kwargs['directories']
        preprocessed_dir = directories['preprocessed']
        npz_dir = directories['npz_datasets']
        self.logger.info(f"Input directory: {preprocessed_dir}")
        self.logger.info(f"Output directory: {npz_dir}")
        frequency = self.config.get("microscope_params.frequency")
        harmonic = self.config.get("microscope_params.harmonic")
        filter_level = self.config.get("wavelet_params.filter_level", 9)
        ref_g = self.config.get("wavelet_params.reference_g")
        ref_s = self.config.get("wavelet_params.reference_s")
        self.logger.info(f"Filter level: {filter_level}")
        self.logger.info(f"Reference fluorophore: G={ref_g}, S={ref_s}")
        self.logger.info(f"Frequency: {frequency} MHz, Harmonic: {harmonic}")
        try:
            success = self.run_processing(
                self.config.to_dict(),
                str(preprocessed_dir),
                str(npz_dir)
            )
            if success:
                self.logger.info("Processing completed successfully")
                npz_files = list(npz_dir.glob("*.npz"))
                self.logger.info(f"Generated {len(npz_files)} NPZ files")
            else:
                self.logger.error("Processing failed")
            return success
        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}", e)
            return False

# --- Lifetime Processing Stages ---
class LifetimeImagesStage(StageBase):
    """Lifetime images generation stage."""
    def get_description(self) -> str:
        return "Generate lifetime images from NPZ files"
    def validate_inputs(self, **kwargs) -> bool:
        return True
    def run(self, **kwargs) -> bool:
        self.logger.info("Lifetime images stage - placeholder implementation")
        return True

class AverageLifetimeStage(StageBase):
    """Average lifetime calculation stage."""
    def get_description(self) -> str:
        return "Calculate average lifetime from segmented data"
    def validate_inputs(self, **kwargs) -> bool:
        return True
    def run(self, **kwargs) -> bool:
        self.logger.info("Average lifetime stage - placeholder implementation")
        return True

# --- Mask Processing Stage ---
class ApplyMaskStage(StageBase):
    """Apply mask stage."""
    def get_description(self) -> str:
        return "Apply binary masks to NPZ data and create masked NPZ files"
    def validate_inputs(self, **kwargs) -> bool:
        return True
    def run(self, **kwargs) -> bool:
        self.logger.info("Apply mask stage - placeholder implementation")
        return True

# --- Segmentation Stages ---
class GMMSegmentationStage(StageBase):
    """GMM segmentation stage."""
    def get_description(self) -> str:
        return "GMM segmentation with interactive parameter selection"
    def validate_inputs(self, **kwargs) -> bool:
        return True
    def run(self, **kwargs) -> bool:
        self.logger.info("GMM segmentation stage - placeholder implementation")
        return True

class ManualSegmentationStage(StageBase):
    """Manual segmentation stage."""
    def get_description(self) -> str:
        return "Interactive manual ellipse-based segmentation"
    def validate_inputs(self, **kwargs) -> bool:
        return True
    def run(self, **kwargs) -> bool:
        self.logger.info("Manual segmentation stage - placeholder implementation")
        return True

class ManualSegmentationUnfilteredStage(StageBase):
    """Manual segmentation stage using unfiltered data."""
    def get_description(self) -> str:
        return "Manual segmentation using unfiltered data (GU, SU)"
    def validate_inputs(self, **kwargs) -> bool:
        return True
    def run(self, **kwargs) -> bool:
        self.logger.info("Manual segmentation unfiltered stage - placeholder implementation")
        return True

class ManualSegmentFromMaskStage(StageBase):
    """Manual segmentation from mask stage."""
    def get_description(self) -> str:
        return "Manual segmentation from masked NPZ files"
    def validate_inputs(self, **kwargs) -> bool:
        return True
    def run(self, **kwargs) -> bool:
        self.logger.info("Manual segment from mask stage - placeholder implementation")
        return True

class ManualSegmentUnfilteredFromMaskStage(StageBase):
    """Manual segmentation unfiltered from mask stage."""
    def get_description(self) -> str:
        return "Manual segmentation from masked NPZ files using unfiltered data"
    def validate_inputs(self, **kwargs) -> bool:
        return True
    def run(self, **kwargs) -> bool:
        self.logger.info("Manual segment unfiltered from mask stage - placeholder implementation")
        return True

# --- Visualization Stages ---
class VisualizeSegmentedStage(StageBase):
    """Visualize segmented data stage."""
    def get_description(self) -> str:
        return "Visualize segmented data from masked NPZ files"
    def validate_inputs(self, **kwargs) -> bool:
        return True
    def run(self, **kwargs) -> bool:
        self.logger.info("Visualize segmented stage - placeholder implementation")
        return True

# --- Phasor Visualization Stage ---
class PhasorVisualizationStage(StageBase):
    """Phasor visualization stage for interactive phasor plot generation."""
    
    def __init__(self, config: Config, logger: PipelineLogger, stage_name: str):
        super().__init__(config, logger, stage_name)
        try:
            from .phasor_visualization import run_phasor_visualization
            self.run_phasor_visualization = run_phasor_visualization
            self.phasor_visualization_available = True
        except ImportError as e:
            self.logger.error(f"Could not import phasor_visualization module: {e}")
            self.run_phasor_visualization = None
            self.phasor_visualization_available = False

    def get_description(self) -> str:
        return "Interactive phasor visualization and plot generation"

    def validate_inputs(self, **kwargs) -> bool:
        if not self.phasor_visualization_available:
            self.logger.error("Phasor visualization module not available")
            return False
        
        directories = kwargs.get('directories', {})
        output_base_dir = directories.get('output')
        if not output_base_dir:
            self.logger.error("Output base directory not specified")
            return False
            
        npz_dir = os.path.join(str(output_base_dir), "npz_datasets")
        if not os.path.isdir(npz_dir):
            self.logger.error(f"NPZ directory does not exist: {npz_dir}")
            return False
            
        if not any(f.endswith('.npz') for f in os.listdir(npz_dir)):
            self.logger.error(f"No NPZ files found in directory: {npz_dir}")
            return False
            
        return True

    def run(self, **kwargs) -> bool:
        if not self.phasor_visualization_available:
            self.logger.error("Phasor visualization module not available")
            return False
            
        directories = kwargs.get('directories', {})
        output_base_dir = str(directories.get('output'))
        
        self.logger.info("Starting interactive phasor visualization...")
        try:
            return self.run_phasor_visualization(output_base_dir)
        except Exception as e:
            self.logger.error(f"Error during phasor visualization: {str(e)}", e)
            return False