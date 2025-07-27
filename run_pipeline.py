#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to orchestrate the FLIM-FRET analysis pipeline.

Allows selective execution of different pipeline stages:
1. Preprocessing (ImageJ Macros + FLUTE TIFF processing)
2. Wavelet Filtering & NPZ Generation
3. GMM Segmentation, Plotting, and Lifetime Saving
"""

import argparse
import time
import os
import sys
import json
import subprocess
import traceback
import logging
import datetime
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Tuple

# Store import errors for logging after logger initialization
import_errors = []

def store_import_error(module_name, error_msg, suggestion=""):
    """Store import error for logging after logger initialization"""
    import_errors.append({
        'module': module_name,
        'error': error_msg,
        'suggestion': suggestion
    })

# Debug information will be logged after logger initialization

# --- Error Tracking and Logging System ---
class PipelineLogger:
    """
    Comprehensive logging system for the FLIM-FRET analysis pipeline.
    Tracks errors, performance metrics, and provides detailed reporting.
    """
    
    def __init__(self, output_base_dir: str):
        self.output_base_dir = output_base_dir
        self.log_dir = os.path.join(output_base_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        # Error tracking
        self.errors = []
        self.warnings = []
        self.stage_performance = {}
        self.file_processing_stats = {}
        
        # Performance tracking
        self.start_time = time.time()
        self.stage_timings = {}
        
    def setup_logging(self):
        """Setup comprehensive logging with multiple handlers."""
        # Create timestamp for log files
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main pipeline log
        self.pipeline_log_path = os.path.join(self.log_dir, f'pipeline_{timestamp}.log')
        
        # Error log (errors only)
        self.error_log_path = os.path.join(self.log_dir, f'errors_{timestamp}.log')
        
        # Performance log
        self.performance_log_path = os.path.join(self.log_dir, f'performance_{timestamp}.log')
        
        # Setup main logger
        self.logger = logging.getLogger('FLIM_FRET_Pipeline')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler (INFO level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (DEBUG level)
        file_handler = logging.FileHandler(self.pipeline_log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error handler (ERROR level only)
        error_handler = logging.FileHandler(self.error_log_path)
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n')
        error_handler.setFormatter(error_formatter)
        self.logger.addHandler(error_handler)
        
        # Performance handler
        self.performance_logger = logging.getLogger('FLIM_FRET_Performance')
        self.performance_logger.setLevel(logging.INFO)
        self.performance_logger.handlers = []
        
        perf_handler = logging.FileHandler(self.performance_log_path)
        perf_handler.setLevel(logging.INFO)
        perf_formatter = logging.Formatter('%(asctime)s - %(message)s')
        perf_handler.setFormatter(perf_formatter)
        self.performance_logger.addHandler(perf_handler)
        
    def log_stage_start(self, stage_name: str, stage_description: str = ""):
        """Log the start of a pipeline stage."""
        self.stage_timings[stage_name] = {'start': time.time()}
        self.logger.info(f"=== STAGE START: {stage_name} ===")
        if stage_description:
            self.logger.info(f"Description: {stage_description}")
        self.performance_logger.info(f"STAGE_START: {stage_name}")
        
    def log_stage_end(self, stage_name: str, success: bool, additional_info: str = ""):
        """Log the end of a pipeline stage with performance metrics."""
        if stage_name in self.stage_timings:
            end_time = time.time()
            duration = end_time - self.stage_timings[stage_name]['start']
            self.stage_timings[stage_name]['end'] = end_time
            self.stage_timings[stage_name]['duration'] = duration
            self.stage_timings[stage_name]['success'] = success
            
            status = "SUCCESS" if success else "FAILED"
            self.logger.info(f"=== STAGE END: {stage_name} - {status} ({duration:.2f}s) ===")
            if additional_info:
                self.logger.info(f"Additional info: {additional_info}")
            
            self.performance_logger.info(f"STAGE_END: {stage_name} - {status} - {duration:.2f}s")
            
    def log_error(self, error: Exception, context: str = "", stage: str = "", file_path: str = ""):
        """Log an error with full context and traceback."""
        error_info = {
            'timestamp': datetime.datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'stage': stage,
            'file_path': file_path,
            'traceback': traceback.format_exc()
        }
        
        self.errors.append(error_info)
        
        # Log to error log
        self.logger.error(f"ERROR in {stage}: {context}")
        self.logger.error(f"Error type: {error_info['error_type']}")
        self.logger.error(f"Error message: {error_info['error_message']}")
        if file_path:
            self.logger.error(f"File: {file_path}")
        self.logger.error(f"Traceback:\n{error_info['traceback']}")
        
    def log_warning(self, message: str, context: str = "", stage: str = ""):
        """Log a warning."""
        warning_info = {
            'timestamp': datetime.datetime.now().isoformat(),
            'message': message,
            'context': context,
            'stage': stage
        }
        
        self.warnings.append(warning_info)
        self.logger.warning(f"WARNING in {stage}: {context} - {message}")
        
    def log_file_processing(self, file_path: str, success: bool, processing_time: float, stage: str = ""):
        """Log file processing statistics."""
        if stage not in self.file_processing_stats:
            self.file_processing_stats[stage] = {'success': 0, 'failed': 0, 'total_time': 0}
        
        if success:
            self.file_processing_stats[stage]['success'] += 1
        else:
            self.file_processing_stats[stage]['failed'] += 1
            
        self.file_processing_stats[stage]['total_time'] += processing_time
        
    @contextmanager
    def error_context(self, context: str, stage: str = "", file_path: str = ""):
        """Context manager for error handling with automatic logging."""
        try:
            yield
        except Exception as e:
            self.log_error(e, context, stage, file_path)
            raise
    
    def generate_error_report(self) -> str:
        """Generate a comprehensive error report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FLIM-FRET ANALYSIS PIPELINE - ERROR REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total runtime: {time.time() - self.start_time:.2f} seconds")
        report_lines.append("")
        
        # Summary
        report_lines.append("SUMMARY:")
        report_lines.append(f"  Total errors: {len(self.errors)}")
        report_lines.append(f"  Total warnings: {len(self.warnings)}")
        report_lines.append("")
        
        # Stage performance
        report_lines.append("STAGE PERFORMANCE:")
        for stage_name, timing in self.stage_timings.items():
            status = "SUCCESS" if timing.get('success', False) else "FAILED"
            duration = timing.get('duration', 0)
            report_lines.append(f"  {stage_name}: {status} ({duration:.2f}s)")
        report_lines.append("")
        
        # File processing statistics
        if self.file_processing_stats:
            report_lines.append("FILE PROCESSING STATISTICS:")
            for stage, stats in self.file_processing_stats.items():
                total = stats['success'] + stats['failed']
                success_rate = (stats['success'] / total * 100) if total > 0 else 0
                avg_time = stats['total_time'] / total if total > 0 else 0
                report_lines.append(f"  {stage}: {stats['success']}/{total} files ({success_rate:.1f}% success, avg {avg_time:.2f}s)")
            report_lines.append("")
        
        # Detailed errors
        if self.errors:
            report_lines.append("DETAILED ERRORS:")
            for i, error in enumerate(self.errors, 1):
                report_lines.append(f"  Error {i}:")
                report_lines.append(f"    Time: {error['timestamp']}")
                report_lines.append(f"    Stage: {error['stage']}")
                report_lines.append(f"    Context: {error['context']}")
                report_lines.append(f"    Type: {error['error_type']}")
                report_lines.append(f"    Message: {error['error_message']}")
                if error['file_path']:
                    report_lines.append(f"    File: {error['file_path']}")
                report_lines.append("")
        
        # Warnings
        if self.warnings:
            report_lines.append("WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                report_lines.append(f"  Warning {i}:")
                report_lines.append(f"    Time: {warning['timestamp']}")
                report_lines.append(f"    Stage: {warning['stage']}")
                report_lines.append(f"    Context: {warning['context']}")
                report_lines.append(f"    Message: {warning['message']}")
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_error_report(self):
        """Save the error report to a file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.log_dir, f'error_report_{timestamp}.txt')
        
        with open(report_path, 'w') as f:
            f.write(self.generate_error_report())
        
        self.logger.info(f"Error report saved to: {report_path}")
        return report_path

# --- Import necessary functions from other scripts ---

# It's generally better practice to organize these into modules,
# but for simplicity, we'll import directly if they are in the same directory.

try:
    # Stage 1: Preprocessing
    from src.python.modules.TCSPC_preprocessing_AUTOcal_v2_0 import run_preprocessing
except ImportError as e:
    # Use current filename in error message
    print(f"Error: Could not import run_preprocessing from TCSPC_preprocessing_AUTOcal_v2_0.py: {e}") 
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_preprocessing = None # Placeholder
    


try:
    # Stage 2: Wavelet Filtering & NPZ Generation
    from src.python.modules.ComplexWaveletFilter_v2_0 import main as run_wavelet_filtering
    # Log this after logger initialization
    wavelet_filter_available = True
except ImportError as e:
    # Store error for logging after logger initialization
    store_import_error("ComplexWaveletFilter_v2_0", 
                      f"Could not import main (as run_wavelet_filtering): {e}",
                      "Ensure the ComplexWaveletFilter_v2_0.py module is available.")
    run_wavelet_filtering = None # Placeholder
    wavelet_filter_available = False
    
try:
    # Stage 3: Phasor Visualization
    from src.python.modules.phasor_visualization import run_phasor_visualization
except ImportError as e:
    store_import_error("phasor_visualization", 
                      f"Could not import run_phasor_visualization: {e}",
                      "Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_phasor_visualization = None # Placeholder
    
try:
    # Intensity Image Generation for Wavelet Filtering
    from src.python.modules.generate_intensity_images import process_raw_flim_files as generate_intensity_images
except ImportError as e:
    store_import_error("generate_intensity_images", 
                      f"Could not import process_raw_flim_files (as generate_intensity_images): {e}",
                      "Ensure the script is in the same directory or accessible via PYTHONPATH.")
    generate_intensity_images = None # Placeholder
    
# Additional imports needed for file operations
import shutil

try:
    # Stage 3: GMM Segmentation, Plotting, Lifetime Saving
    from src.python.modules.GMMSegmentation import main as run_gmm_segmentation
except ImportError as e:
    store_import_error("GMMSegmentation", 
                      f"Could not import main (as run_gmm_segmentation): {e}",
                      "Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_gmm_segmentation = None # Placeholder
    
try:
    # Stage 4: Phasor Transformation
    from src.python.modules.phasor_transform import process_flim_file as run_phasor_transform
except ImportError as e:
    store_import_error("phasor_transform", 
                      f"Could not import process_flim_file (as run_phasor_transform): {e}",
                      "Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_phasor_transform = None # Placeholder

try:
    # Stage 4B: Manual Segmentation
    from src.python.modules.ManualSegmentation import main as run_manual_segmentation
except ImportError as e:
    store_import_error("ManualSegmentation", 
                      f"Could not import main (as run_manual_segmentation): {e}",
                      "Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_manual_segmentation = None # Placeholder

try:
    # Stage 4B2: Manual Segmentation Unfiltered
    from src.python.modules.ManualSegmentationUnfiltered import main as run_manual_segmentation_unfiltered
except ImportError as e:
    store_import_error("ManualSegmentationUnfiltered", 
                      f"Could not import main (as run_manual_segmentation_unfiltered): {e}",
                      "Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_manual_segmentation_unfiltered = None # Placeholder

try:
    # Stage 4C: Lifetime Image Generation
    from src.python.modules.generate_lifetime_images import main as run_lifetime_generation
except ImportError as e:
    store_import_error("generate_lifetime_images", 
                      f"Could not import main (as run_lifetime_generation): {e}",
                      "Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_lifetime_generation = None # Placeholder

try:
    # Stage 4D: Average Lifetime Calculation
    from src.python.modules.calculate_average_lifetime import main as run_average_lifetime
except ImportError as e:
    store_import_error("calculate_average_lifetime", 
                      f"Could not import main (as run_average_lifetime): {e}",
                      "Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_average_lifetime = None # Placeholder

try:
    # Stage 5A: Apply Mask
    from src.python.modules.apply_mask import main as run_apply_mask
except ImportError as e:
    store_import_error("apply_mask", 
                      f"Could not import main (as run_apply_mask): {e}",
                      "Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_apply_mask = None # Placeholder

try:
    # Stage 5B: Visualize Segmented Data
    from src.python.modules.visualize_segmented_data import main as run_visualize_segmented
except ImportError as e:
    store_import_error("visualize_segmented_data", 
                      f"Could not import main (as run_visualize_segmented): {e}",
                      "Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_visualize_segmented = None # Placeholder

try:
    # Stage 5C: Manual Segment From Mask
    from src.python.modules.ManualSegmentFromMask import main as run_manual_segment_from_mask
except ImportError as e:
    store_import_error("ManualSegmentFromMask", 
                      f"Could not import main (as run_manual_segment_from_mask): {e}",
                      "Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_manual_segment_from_mask = None # Placeholder

try:
    # Stage 5D: Manual Segment Unfiltered From Mask
    from src.python.modules.ManualSegmentUnfilteredFromMask import main as run_manual_segment_unfiltered_from_mask
except ImportError as e:
    store_import_error("ManualSegmentUnfilteredFromMask", 
                      f"Could not import main (as run_manual_segment_unfiltered_from_mask): {e}",
                      "Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_manual_segment_unfiltered_from_mask = None # Placeholder
    
# Legacy FLUTE integration test function removed
# FLUTE functionality has been replaced by custom phasor_transform module
# to avoid GUI dependencies and improve maintainability

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="FLIM-FRET Analysis Pipeline")
    
    # Required arguments
    parser.add_argument("--input", required=True, help="Input directory containing raw FLIM-FRET .bin files")
    parser.add_argument("--output", required=True, help="Base output directory for all pipeline stages")
    
    # Pipeline stage control
    parser.add_argument("--all", action="store_true", help="Run all pipeline stages")
    
    # Workflow groupings
    parser.add_argument("--preprocessing", action="store_true", help="Run Stages 1-2A: convert files, phasor transformation, and organize files")
    parser.add_argument("--processing", action="store_true", help="Run Stages 1-2B: preprocessing + wavelet filtering and lifetime calculation")
    
    # Individual stages (for advanced users)
    parser.add_argument("--preprocess", action="store_true", help="[DEPRECATED] Use --preprocessing instead")
    parser.add_argument("--visualize", action="store_true", help="Run Stage 3: Interactive phasor visualization and plot generation")
    parser.add_argument("--segment", action="store_true", help="Run GMM segmentation stage")
    parser.add_argument("--manual-segment", action="store_true", help="Run manual segmentation stage")
    parser.add_argument("--manual-segment-unfiltered", action="store_true", help="Run manual segmentation stage using unfiltered data (GU, SU)")
    parser.add_argument("--lifetime-images", action="store_true", help="Run lifetime image generation from NPZ files")
    parser.add_argument("--average-lifetime", action="store_true", help="Calculate average lifetime from segmented data")

    parser.add_argument("--apply-mask", action="store_true", help="Apply binary masks to NPZ data and create masked NPZ files")
    parser.add_argument("--visualize-segmented", action="store_true", help="Visualize segmented data from masked NPZ files")
    parser.add_argument("--manual-segment-from-mask", action="store_true", help="Manual segmentation from masked NPZ files (G*mask, S*mask)")
    parser.add_argument("--manual-segment-unfiltered-from-mask", action="store_true", help="Manual segmentation from masked NPZ files using unfiltered data (GU*mask, SU*mask)")
    
    # Testing mode removed - FLUTE integration has been replaced by custom phasor_transform module
    
    # Interactive mode
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode for user input (affects GMM segmentation)")
    

    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input):
        parser.error(f"Input directory '{args.input}' does not exist or is not a directory")
    
    # If no specific stages are selected, ask the user what to do
    if not (args.all or args.preprocessing or args.processing or
            args.preprocess or args.visualize or args.segment or args.manual_segment or args.manual_segment_unfiltered or args.lifetime_images or args.average_lifetime or args.apply_mask or args.visualize_segmented or args.manual_segment_from_mask or args.manual_segment_unfiltered_from_mask):
        # Not running any specific stage
        print("\n")
        print("=" * 30)
        print("      FLIM-FRET Analysis")
        print("=" * 30)
        print("MENU:")
        print("1. Preprocessing (.bin to .tif conversion + phasor transformation)")
        print("2. Processing (preprocessing + wavelet filtering and lifetime calculation)")
        print("3. Interactive phasor visualization and plot generation")
        print("4. Lifetime Images (generate lifetime images from NPZ files)")
        print("5. Apply Mask (apply binary masks to NPZ data)")
        print("6. Visualize (interactive phasor plots)")
        print("7. Visualize Segmented (visualize segmented data from masked NPZ files)")
        print("8. Segment (GMM segmentation with interactive parameter selection)")
        print("9. Manual Segment (interactive manual ellipse-based segmentation)")
        print("10. Manual Segment From Mask (manual segmentation from masked NPZ files)")
        print("11. Manual Segment Unfiltered (manual segmentation using unfiltered data)")
        print("12. Manual Segment Unfiltered From Mask (manual segmentation from masked NPZ files using unfiltered data)")
        print("13. Average Lifetime (calculate average lifetime from segmented data)")
        print("14. _____All stages")
        print("15. Exit")
        
        choice = input("Select an option (1-15): ")
        
        if choice == "1":
            args.preprocessing = True
        elif choice == "2":
            args.processing = True
        elif choice == "3":
            args.visualize = True
        elif choice == "4":
            args.lifetime_images = True
        elif choice == "5":
            args.apply_mask = True
        elif choice == "6":
            args.visualize = True
        elif choice == "7":
            args.visualize_segmented = True
        elif choice == "8":
            args.segment = True
            args.interactive = True  # Automatically enable interactive mode for GMM segmentation
        elif choice == "9":
            args.manual_segment = True
        elif choice == "10":
            args.manual_segment_from_mask = True
        elif choice == "11":
            args.manual_segment_unfiltered = True
        elif choice == "12":
            args.manual_segment_unfiltered_from_mask = True
        elif choice == "13":
            args.average_lifetime = True
        elif choice == "14":
            args.all = True
        elif choice == "15":
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)
            
    return args

# --- Helper to load config (less prone to failure if keys change) ---
def load_pipeline_config(config_path="config/config.json"):
     try:
         with open(config_path, "r") as f:
             return json.load(f)
     except (FileNotFoundError, json.JSONDecodeError) as e:
         print(f"Error loading pipeline config ({config_path}): {e}", file=sys.stderr)
         return None # Allow pipeline to attempt running with defaults if possible

# --- Main Pipeline Execution ---
def main():
    args = parse_arguments()
    config = load_pipeline_config() # Load general config (app paths, params)
    
    if config is None:
         print("Failed to load config.json. Cannot proceed.", file=sys.stderr)
         sys.exit(1)
    
    # Initialize the pipeline logger
    logger = PipelineLogger(args.output)
    logger.logger.info("FLIM-FRET Analysis Pipeline Starting")
    logger.logger.info(f"Input directory: {args.input}")
    logger.logger.info(f"Output base directory: {args.output}")
    
    # Log debug information
    logger.logger.debug("--- sys.path before imports in run_pipeline.py ---")
    logger.logger.debug(str(sys.path))
    logger.logger.debug("--------------------------------------------------")
    
    # Log module availability
    if 'wavelet_filter_available' in globals() and wavelet_filter_available:
        logger.logger.info("Using Complex Wavelet Filter v2.0 implementation")
    
    # Log any import errors
    if import_errors:
        logger.logger.warning(f"Found {len(import_errors)} import errors:")
        for error in import_errors:
            logger.logger.error(f"Module {error['module']}: {error['error']}")
            if error['suggestion']:
                logger.logger.error(f"  Suggestion: {error['suggestion']}")
         

    
    # Define specific output subdirectories based on the output argument
    output_dir = os.path.join(args.output, 'output')
    preprocessed_dir = os.path.join(args.output, 'preprocessed')
    npz_dir = os.path.join(args.output, 'npz_datasets')
    segmented_dir = os.path.join(args.output, 'segmented')
    segmented_npz_dir = os.path.join(args.output, 'segmented_npz_datasets')
    plots_dir = os.path.join(args.output, 'plots')
    lifetime_dir = os.path.join(args.output, 'lifetime_images')
    phasor_dir = os.path.join(args.output, 'phasor_output')
    external_mask_npz_dir = os.path.join(args.output, 'external_mask_npz_datasets')
    
    # Create base output directory if it doesn't exist
    try:
         os.makedirs(args.output, exist_ok=True)
         
         # Create all required output subdirectories
         os.makedirs(output_dir, exist_ok=True)
         os.makedirs(preprocessed_dir, exist_ok=True)
         os.makedirs(npz_dir, exist_ok=True)
         os.makedirs(segmented_dir, exist_ok=True)
         os.makedirs(segmented_npz_dir, exist_ok=True)
         os.makedirs(plots_dir, exist_ok=True)
         os.makedirs(lifetime_dir, exist_ok=True)
         os.makedirs(phasor_dir, exist_ok=True)
         os.makedirs(external_mask_npz_dir, exist_ok=True)
         
         logger.logger.info("Created output directories:")
         logger.logger.info(f" - {output_dir}")
         logger.logger.info(f" - {preprocessed_dir}")
         logger.logger.info(f" - {npz_dir}")
         logger.logger.info(f" - {segmented_dir}")
         logger.logger.info(f" - {segmented_npz_dir}")
         logger.logger.info(f" - {plots_dir}")
         logger.logger.info(f" - {lifetime_dir}")
         logger.logger.info(f" - {phasor_dir}")
         logger.logger.info(f" - {external_mask_npz_dir}")
    except OSError as e:
         logger.log_error(e, "Creating output directories", "Setup")
         logger.logger.error(f"Error creating output directories: {e}")
         sys.exit(1)
    
    # Look for calibration file in input directory first, fall back to project directory
    input_calibration_path = os.path.join(args.input, "calibration.csv")
    project_calibration_path = "data/calibration.csv"
    
    if os.path.exists(input_calibration_path):
        calibration_file_path = input_calibration_path
        logger.logger.info(f"Using calibration file from input directory: {calibration_file_path}")
    else:
        calibration_file_path = project_calibration_path
        logger.logger.info(f"Calibration file not found in input directory, using project directory: {calibration_file_path}")
    
    start_pipeline_time = time.time()
    logger.logger.info("===================================")
    logger.logger.info(" FLIM-FRET Analysis Pipeline Start ")
    logger.logger.info(f" Input Dir: {args.input}")
    logger.logger.info(f" Output Base: {args.output}")
    logger.logger.info(f" Calibration: {calibration_file_path}")
    logger.logger.info("===================================")

    # --- Stage 1: Preprocessing ---
    if args.preprocess or args.preprocessing or args.processing or args.all:
        logger.log_stage_start("Stage 1: Preprocessing", "Convert .bin files to .tif and perform phasor transformation")
        if run_preprocessing:
            try:
                # Find the actual data directory (the subdirectory containing BIN files)
                data_dir = args.input
                if os.path.exists(args.input):
                    # Look for subdirectories containing BIN files
                    for item in os.listdir(args.input):
                        item_path = os.path.join(args.input, item)
                        if os.path.isdir(item_path):
                            # Check if this subdirectory contains BIN files
                            bin_files = [f for f in os.listdir(item_path) if f.endswith('.bin')]
                            if bin_files:
                                data_dir = item_path
                                logger.logger.info(f"Found BIN files in subdirectory: {data_dir}")
                                break
                
                # Prompt user for file selection mode (except for --all which processes everything)
                interactive_file_selection = False
                if not args.all:
                    logger.logger.info("Prompting user for file selection mode")
                    # Temporarily restore stdout/stderr for interactive prompt
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                    
                    try:
                        print("\n=== Preprocessing File Selection ===")
                        print("Choose how to select files for preprocessing:")
                        print("  [1] Process all .bin files (default)")
                        print("  [2] Select specific .bin files interactively")
                        
                        while True:
                            choice = input("Select option (1 or 2, default: 1): ").strip()
                            if choice == "" or choice == "1":
                                interactive_file_selection = False
                                print("→ Processing all .bin files")
                                break
                            elif choice == "2":
                                interactive_file_selection = True
                                print("→ Interactive file selection enabled")
                                break
                            else:
                                print("Please enter 1 or 2.")
                        
                    finally:
                        # Restore log file redirection
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr
                
                # Run preprocessing with or without interactive file selection
                if interactive_file_selection:
                    logger.logger.info("Running preprocessing with interactive file selection")
                    # Temporarily restore stdout/stderr for interactive mode
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                    
                    try:
                        success = run_preprocessing(
                            config,
                            data_dir,        # Use the actual data directory containing BIN files
                            output_dir,           
                            preprocessed_dir,     
                            calibration_file_path, # Pass fixed calibration path
                            args.input,        # Keep raw_data_root as the full path for path mapping
                            interactive_file_selection=True  # Enable interactive file selection
                        )
                    finally:
                        # Restore log file redirection
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr
                else:
                    logger.logger.info("Running preprocessing with all files")
                    success = run_preprocessing(
                        config,
                        data_dir,        # Use the actual data directory containing BIN files
                        output_dir,           
                        preprocessed_dir,     
                        calibration_file_path, # Pass fixed calibration path
                        args.input        # Keep raw_data_root as the full path for path mapping
                    )
                
                logger.log_stage_end("Stage 1: Preprocessing", success)
            except Exception as e:
                logger.log_error(e, "Running preprocessing", "Stage 1: Preprocessing")
                logger.log_stage_end("Stage 1: Preprocessing", False, f"Error: {str(e)}")
        else:
            error_msg = "run_preprocessing function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 1: Preprocessing")
            logger.log_stage_end("Stage 1: Preprocessing", False, error_msg)
            


    # --- Stage 2: Wavelet Filtering & NPZ Generation ---
    if args.processing or args.all:
        logger.log_stage_start("Stage 2: Wavelet Filtering", "Apply complex wavelet filtering and generate NPZ files")
        if run_wavelet_filtering:
            try:
                # Add required microscope parameters if missing in config
                if "microscope_params" not in config:
                    config["microscope_params"] = {}
                if "frequency" not in config["microscope_params"]:
                    config["microscope_params"]["frequency"] = 78.0  # Default frequency (MHz)
                if "harmonic" not in config["microscope_params"]:
                    config["microscope_params"]["harmonic"] = 1     # Default harmonic
                
                # Pass required arguments - use preprocessed directory directly
                success = run_wavelet_filtering(
                    config, 
                    preprocessed_dir,  # Use the preprocessed directory directly
                    npz_dir
                )
                logger.log_stage_end("Stage 2: Wavelet Filtering", success)
            except Exception as e:
                logger.log_error(e, "Running wavelet filtering", "Stage 2: Wavelet Filtering")
                logger.log_stage_end("Stage 2: Wavelet Filtering", False, f"Error: {str(e)}")
        else:
            error_msg = "run_wavelet_filtering function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 2: Wavelet Filtering")
            logger.log_stage_end("Stage 2: Wavelet Filtering", False, error_msg)

    # --- Stage 3: Interactive Phasor Visualization ---
    if args.visualize or args.all:
        logger.log_stage_start("Stage 3: Phasor Visualization", "Interactive phasor visualization and plot generation")
        if run_phasor_visualization:
            try:
                # Temporarily restore original stdout/stderr for interactive mode
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                
                # Restore terminal I/O for interactive visualization
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                
                logger.logger.info("Terminal I/O restored for interactive mode")
                
                # Run interactive phasor visualization
                success = run_phasor_visualization(args.output)
                
                # Restore log file redirection
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                logger.log_stage_end("Stage 3: Phasor Visualization", success, "User may have aborted" if not success else "")
            except Exception as e:
                # Make sure to restore logging even if there's an error
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                logger.log_error(e, "Running phasor visualization", "Stage 3: Phasor Visualization")
                logger.log_stage_end("Stage 3: Phasor Visualization", False, f"Error: {str(e)}")
        else:
            error_msg = "run_phasor_visualization function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 3: Phasor Visualization")
            logger.log_stage_end("Stage 3: Phasor Visualization", False, error_msg)

    # --- Stage 4: GMM Segmentation, Plotting, Lifetime Saving ---
    if args.segment or args.all:
        logger.log_stage_start("Stage 4: GMM Segmentation", "GMM segmentation, plotting, and lifetime saving")
        if run_gmm_segmentation:
            try:
                # Prompt user for interactive or config-based segmentation
                print("\nGMM Segmentation Parameter Selection:")
                print("  [1] Manually select GMM parameters (interactive)")
                print("  [2] Use parameters from config/gmm_config.json")
                while True:
                    user_choice = input("Select option (1 or 2, default: 1): ").strip()
                    if user_choice == "" or user_choice == "1":
                        use_interactive = True
                        print("→ Running GMM segmentation in interactive mode.")
                        break
                    elif user_choice == "2":
                        use_interactive = False
                        print("→ Running GMM segmentation using config/gmm_config.json.")
                        break
                    else:
                        print("Please enter 1 or 2.")

                # If interactive mode, restore terminal I/O
                if use_interactive:
                    logger.logger.info("Interactive mode enabled - user will be prompted for GMM parameters")
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                    logger.logger.info("Terminal I/O restored for interactive GMM mode")

                # Pass required arguments
                if use_interactive:
                    success = run_gmm_segmentation(
                        config, 
                        npz_dir, 
                        segmented_dir, 
                        plots_dir, 
                        lifetime_dir,
                        True  # interactive mode
                    )
                else:
                    # Load config from gmm_config.json
                    gmm_config_path = os.path.join("config", "gmm_config.json")
                    if not os.path.exists(gmm_config_path):
                        logger.log_error(Exception("gmm_config.json not found"), "Config file check", "Stage 4: GMM Segmentation")
                        print("Error: config/gmm_config.json not found.")
                        success = False
                    else:
                        # Load config file using GMM module's config loader
                        from src.python.modules.GMMSegmentation import load_gmm_config
                        gmm_params = load_gmm_config(gmm_config_path)
                        if not gmm_params:
                            logger.log_error(Exception("Failed to load GMM config parameters"), "Config loading", "Stage 4: GMM Segmentation")
                            print("Error: Failed to load GMM config parameters.")
                            success = False
                        else:
                            # Create config structure expected by GMM module
                            gmm_config = {'gmm_segmentation_params': gmm_params}
                            print(f"Loaded GMM parameters: {list(gmm_params.keys())}")
                            print(f"combine_datasets value: {gmm_params.get('combine_datasets', 'NOT FOUND')}")
                            success = run_gmm_segmentation(
                                gmm_config, 
                                npz_dir, 
                                segmented_dir, 
                                plots_dir, 
                                lifetime_dir,
                                False  # not interactive
                            )

                # Restore log file redirection if interactive mode was used
                if use_interactive:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    logger.logger.info("Terminal I/O restored to logging mode")
                
                logger.log_stage_end("Stage 4: GMM Segmentation", success)
            except Exception as e:
                if 'use_interactive' in locals() and use_interactive:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                logger.log_error(e, "Running GMM segmentation", "Stage 4: GMM Segmentation")
                logger.log_stage_end("Stage 4: GMM Segmentation", False, f"Error: {str(e)}")
        else:
            error_msg = "run_gmm_segmentation function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 4: GMM Segmentation")
            logger.log_stage_end("Stage 4: GMM Segmentation", False, error_msg)
            
    # --- Stage 4B: Manual Segmentation ---
    if args.manual_segment or args.all:
        logger.log_stage_start("Stage 4B: Manual Segmentation", "Interactive manual ellipse-based segmentation")
        if run_manual_segmentation:
            try:
                # Prompt user for NPZ directory choice
                print("\nManual Segmentation - NPZ Directory Selection:")
                print("  [1] Use original NPZ files (npz_datasets)")
                print("  [2] Use external mask NPZ files (external_mask_npz_datasets)")
                
                # Check if directories exist and show file counts
                original_count = 0
                external_count = 0
                
                if os.path.exists(npz_dir):
                    original_count = len([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
                    print(f"      → {original_count} NPZ files found in npz_datasets")
                else:
                    print("      → npz_datasets directory not found")
                
                if os.path.exists(external_mask_npz_dir):
                    external_count = len([f for f in os.listdir(external_mask_npz_dir) if f.endswith('.npz')])
                    print(f"      → {external_count} NPZ files found in external_mask_npz_datasets")
                else:
                    print("      → external_mask_npz_datasets directory not found")
                
                # Get user choice
                while True:
                    user_choice = input("Select option (1 or 2, default: 1): ").strip()
                    if user_choice == "" or user_choice == "1":
                        selected_npz_dir = npz_dir
                        dir_name = "npz_datasets"
                        print(f"→ Using {dir_name} directory for manual segmentation.")
                        break
                    elif user_choice == "2":
                        selected_npz_dir = external_mask_npz_dir
                        dir_name = "external_mask_npz_datasets"
                        print(f"→ Using {dir_name} directory for manual segmentation.")
                        break
                    else:
                        print("Please enter 1 or 2.")
                
                # Log the choice
                logger.logger.info(f"User selected NPZ directory: {dir_name}")
                logger.logger.info(f"NPZ directory path: {selected_npz_dir}")
                
                # Manual segmentation is always interactive, so restore terminal I/O
                logger.logger.info("Manual segmentation is interactive - restoring terminal I/O")
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                logger.logger.info("Terminal I/O restored for manual segmentation mode")

                # Run manual segmentation with selected directory
                success = run_manual_segmentation(
                    config, 
                    selected_npz_dir, 
                    segmented_dir, 
                    plots_dir, 
                    lifetime_dir,
                    True  # interactive mode
                )

                # Restore log file redirection
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                logger.logger.info("Terminal I/O restored to logging mode")
                
                logger.log_stage_end("Stage 4B: Manual Segmentation", success)
            except Exception as e:
                # Make sure to restore logging even if there's an error
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                logger.log_error(e, "Running manual segmentation", "Stage 4B: Manual Segmentation")
                logger.log_stage_end("Stage 4B: Manual Segmentation", False, f"Error: {str(e)}")
        else:
            error_msg = "run_manual_segmentation function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 4B: Manual Segmentation")
            logger.log_stage_end("Stage 4B: Manual Segmentation", False, error_msg)
            
    # --- Stage 4B2: Manual Segmentation Unfiltered ---
    if args.manual_segment_unfiltered or args.all:
        logger.log_stage_start("Stage 4B2: Manual Segmentation Unfiltered", "Interactive manual ellipse-based segmentation using unfiltered data (GU, SU)")
        if run_manual_segmentation_unfiltered:
            try:
                # Prompt user for NPZ directory choice
                print("\nManual Segmentation Unfiltered - NPZ Directory Selection:")
                print("  [1] Use original NPZ files (npz_datasets)")
                print("  [2] Use external mask NPZ files (external_mask_npz_datasets)")
                
                # Check if directories exist and show file counts
                original_count = 0
                external_count = 0
                
                if os.path.exists(npz_dir):
                    original_count = len([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
                    print(f"      → {original_count} NPZ files found in npz_datasets")
                else:
                    print("      → npz_datasets directory not found")
                
                if os.path.exists(external_mask_npz_dir):
                    external_count = len([f for f in os.listdir(external_mask_npz_dir) if f.endswith('.npz')])
                    print(f"      → {external_count} NPZ files found in external_mask_npz_datasets")
                else:
                    print("      → external_mask_npz_datasets directory not found")
                
                # Get user choice
                while True:
                    user_choice = input("Select option (1 or 2, default: 1): ").strip()
                    if user_choice == "" or user_choice == "1":
                        selected_npz_dir = npz_dir
                        dir_name = "npz_datasets"
                        print(f"→ Using {dir_name} directory for manual segmentation unfiltered.")
                        break
                    elif user_choice == "2":
                        selected_npz_dir = external_mask_npz_dir
                        dir_name = "external_mask_npz_datasets"
                        print(f"→ Using {dir_name} directory for manual segmentation unfiltered.")
                        break
                    else:
                        print("Please enter 1 or 2.")
                
                # Log the choice
                logger.logger.info(f"User selected NPZ directory: {dir_name}")
                logger.logger.info(f"NPZ directory path: {selected_npz_dir}")
                
                # Manual segmentation unfiltered is always interactive, so restore terminal I/O
                logger.logger.info("Manual segmentation unfiltered is interactive - restoring terminal I/O")
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                logger.logger.info("Terminal I/O restored for manual segmentation unfiltered mode")

                # Run manual segmentation unfiltered with selected directory
                success = run_manual_segmentation_unfiltered(
                    config, 
                    selected_npz_dir, 
                    segmented_dir, 
                    plots_dir, 
                    lifetime_dir,
                    True  # interactive mode
                )

                # Restore log file redirection
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                logger.logger.info("Terminal I/O restored to logging mode")
                
                logger.log_stage_end("Stage 4B2: Manual Segmentation Unfiltered", success)
            except Exception as e:
                # Make sure to restore logging even if there's an error
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                logger.log_error(e, "Running manual segmentation unfiltered", "Stage 4B2: Manual Segmentation Unfiltered")
                logger.log_stage_end("Stage 4B2: Manual Segmentation Unfiltered", False, f"Error: {str(e)}")
        else:
            error_msg = "run_manual_segmentation_unfiltered function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 4B2: Manual Segmentation Unfiltered")
            logger.log_stage_end("Stage 4B2: Manual Segmentation Unfiltered", False, error_msg)
            
    # --- Stage 4C: Lifetime Image Generation ---
    if args.lifetime_images or args.all:
        logger.log_stage_start("Stage 4C: Lifetime Image Generation", "Generate lifetime images from NPZ files")
        if run_lifetime_generation:
            try:
                # Use existing lifetime_images directory
                lifetime_images_dir = os.path.join(args.output, 'lifetime_images')
                os.makedirs(lifetime_images_dir, exist_ok=True)
                
                # Run lifetime image generation (no preview plots)
                success = run_lifetime_generation(
                    config=config,
                    npz_dir=npz_dir,
                    output_dir=lifetime_images_dir,
                    create_preview=False
                )
                
                logger.log_stage_end("Stage 4C: Lifetime Image Generation", success)
            except Exception as e:
                logger.log_error(e, "Running lifetime image generation", "Stage 4C: Lifetime Image Generation")
                logger.log_stage_end("Stage 4C: Lifetime Image Generation", False, f"Error: {str(e)}")
        else:
            error_msg = "run_lifetime_generation function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 4C: Lifetime Image Generation")
            logger.log_stage_end("Stage 4C: Lifetime Image Generation", False, error_msg)
            
    # --- Stage 4D: Average Lifetime Calculation ---
    if args.average_lifetime or args.all:
        logger.log_stage_start("Stage 4D: Average Lifetime Calculation", "Calculate average lifetime from segmented data")
        if run_average_lifetime:
            try:
                # Create results directory
                results_dir = os.path.join(args.output, 'average_lifetime_results')
                os.makedirs(results_dir, exist_ok=True)
                
                # Prompt user for NPZ directory choice
                print("\nAverage Lifetime Calculation - NPZ Directory Selection:")
                print("  [1] Use segmented NPZ files (segmented_npz_datasets)")
                print("  [2] Use external mask NPZ files (external_mask_npz_datasets)")
                
                # Check if directories exist and show file counts
                segmented_count = 0
                external_count = 0
                
                if os.path.exists(segmented_npz_dir):
                    segmented_count = len([f for f in os.listdir(segmented_npz_dir) if f.endswith('.npz')])
                    print(f"      → {segmented_count} NPZ files found in segmented_npz_datasets")
                else:
                    print("      → segmented_npz_datasets directory not found")
                
                if os.path.exists(external_mask_npz_dir):
                    external_count = len([f for f in os.listdir(external_mask_npz_dir) if f.endswith('.npz')])
                    print(f"      → {external_count} NPZ files found in external_mask_npz_datasets")
                else:
                    print("      → external_mask_npz_datasets directory not found")
                
                # Get user choice
                while True:
                    user_choice = input("Select option (1 or 2, default: 1): ").strip()
                    if user_choice == "" or user_choice == "1":
                        selected_npz_dir = segmented_npz_dir
                        dir_name = "segmented_npz_datasets"
                        print(f"→ Using {dir_name} directory for average lifetime calculation.")
                        break
                    elif user_choice == "2":
                        selected_npz_dir = external_mask_npz_dir
                        dir_name = "external_mask_npz_datasets"
                        print(f"→ Using {dir_name} directory for average lifetime calculation.")
                        break
                    else:
                        print("Please enter 1 or 2.")
                
                # Log the choice
                logger.logger.info(f"User selected NPZ directory: {dir_name}")
                logger.logger.info(f"NPZ directory path: {selected_npz_dir}")
                
                # Run average lifetime calculation with selected directory
                success = run_average_lifetime(
                    config=config,
                    segmented_npz_dir=selected_npz_dir,
                    output_dir=results_dir
                )
                
                logger.log_stage_end("Stage 4D: Average Lifetime Calculation", success)
            except Exception as e:
                logger.log_error(e, "Running average lifetime calculation", "Stage 4D: Average Lifetime Calculation")
                logger.log_stage_end("Stage 4D: Average Lifetime Calculation", False, f"Error: {str(e)}")
        else:
            error_msg = "run_average_lifetime function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 4D: Average Lifetime Calculation")
            logger.log_stage_end("Stage 4D: Average Lifetime Calculation", False, error_msg)
            
    # --- Stage 5: Phasor Transformation ---
    if args.phasor or args.all:
        logger.log_stage_start("Stage 5: Phasor Transformation", "Phasor transformation of preprocessed files")
        if run_phasor_transform:
            try:
                # Create a function to run the phasor transformation on the preprocessed files
                def process_preprocessed_files(input_dir, output_dir, calibration_file):
                    success = True
                    processed_count = 0
                    error_count = 0
                    
                    # Read calibration values if available
                    calibration_values = None
                    if os.path.exists(calibration_file):
                        try:
                            import pandas as pd
                            df = pd.read_csv(calibration_file)
                            # Simple calibration dict - we assume phi_cal and m_cal columns exist
                            calibration_values = {}
                            for _, row in df.iterrows():
                                calibration_values[row['file_path']] = (row['phi_cal'], row['m_cal'])
                            logger.logger.info(f"Loaded {len(calibration_values)} calibration values from {calibration_file}")
                        except Exception as e:
                            logger.log_error(e, "Loading calibration values", "Stage 5: Phasor Transformation")
                    
                    # Process only original FLIM TIFF files in the input directory
                    # Skip files that have already been processed (like _g.tiff, _s.tiff, _intensity.tiff)
                    for root, _, files in os.walk(input_dir):
                        for filename in files:
                            # Skip derived files that are already processed
                            if filename.lower().endswith(('_g.tiff', '_s.tiff', '_intensity.tiff')):
                                continue
                                
                            # Process only original TIFF files
                            if filename.lower().endswith(('.tif', '.tiff')):
                                input_file = os.path.join(root, filename)
                                
                                # Try to find matching calibration values
                                phi_cal, m_cal = 0.0, 1.0  # Default values
                                if calibration_values:
                                    # Simple exact path match (could be enhanced)
                                    if input_file in calibration_values:
                                        phi_cal, m_cal = calibration_values[input_file]
                                    else:
                                        # Try matching by basename
                                        base_filename = os.path.basename(input_file)
                                        for cal_path in calibration_values:
                                            if os.path.basename(cal_path) == base_filename:
                                                phi_cal, m_cal = calibration_values[cal_path]
                                                break
                                
                                try:
                                    # Create a subdirectory structure in output_dir that matches input_dir
                                    rel_path = os.path.relpath(root, input_dir)
                                    output_subdir = os.path.join(output_dir, rel_path)
                                    os.makedirs(output_subdir, exist_ok=True)
                                    
                                    # Process the file
                                    logger.logger.info(f"Processing {input_file} (phi_cal={phi_cal}, m_cal={m_cal})")
                                    file_success = run_phasor_transform(
                                        input_file=input_file,
                                        output_dir=output_subdir,
                                        phi_cal=phi_cal,
                                        m_cal=m_cal,
                                        bin_width_ns=0.2208,  # Standard bin width
                                        freq_mhz=80,          # Standard laser frequency
                                        harmonic=1,            # First harmonic
                                        apply_filter=1,        # Apply median filter once
                                        threshold_min=0,       # No minimum intensity threshold
                                        threshold_max=None     # No maximum intensity threshold
                                    )
                                    
                                    if file_success:
                                        processed_count += 1
                                    else:
                                        error_count += 1
                                        success = False
                                except Exception as e:
                                    logger.log_error(e, f"Processing file {input_file}", "Stage 5: Phasor Transformation", input_file)
                                    error_count += 1
                                    success = False
                    
                    logger.logger.info(f"Phasor transformation complete: {processed_count} files processed, {error_count} errors")
                    return success
                
                # Run the phasor transformation on preprocessed files
                success = process_preprocessed_files(
                    input_dir=preprocessed_dir,
                    output_dir=phasor_dir,
                    calibration_file=calibration_file_path
                )
                
                logger.log_stage_end("Stage 5: Phasor Transformation", success, f"Completed with some errors" if not success else "")
            except Exception as e:
                logger.log_error(e, "Running phasor transformation", "Stage 5: Phasor Transformation")
                logger.log_stage_end("Stage 5: Phasor Transformation", False, f"Error: {str(e)}")
        else:
            error_msg = "run_phasor_transform function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 5: Phasor Transformation")
            logger.log_stage_end("Stage 5: Phasor Transformation", False, error_msg)
            
    # --- Stage 5A: Apply Mask ---
    if args.apply_mask or args.all:
        logger.log_stage_start("Stage 5A: Apply Mask", "Apply binary masks to NPZ data and create masked NPZ files")
        if run_apply_mask:
            try:
                # Run apply mask
                success = run_apply_mask(
                    config=config,
                    segmented_dir=segmented_dir,
                    npz_dir=npz_dir,
                    output_dir=external_mask_npz_dir
                )
                
                logger.log_stage_end("Stage 5A: Apply Mask", success)
            except Exception as e:
                logger.log_error(e, "Running apply mask", "Stage 5A: Apply Mask")
                logger.log_stage_end("Stage 5A: Apply Mask", False, f"Error: {str(e)}")
        else:
            error_msg = "run_apply_mask function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 5A: Apply Mask")
            logger.log_stage_end("Stage 5A: Apply Mask", False, error_msg)
            
    # --- Stage 5B: Visualize Segmented Data ---
    if args.visualize_segmented or args.all:
        logger.log_stage_start("Stage 5B: Visualize Segmented Data", "Visualize segmented data from masked NPZ files")
        if run_visualize_segmented:
            try:
                # Visualize segmented data is always interactive, so restore terminal I/O
                logger.logger.info("Segmented data visualization is interactive - restoring terminal I/O")
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                logger.logger.info("Terminal I/O restored for segmented visualization mode")

                # Run visualize segmented data
                success = run_visualize_segmented(
                    config=config,
                    external_mask_npz_dir=external_mask_npz_dir,
                    output_dir=args.output
                )

                # Restore log file redirection
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                logger.logger.info("Terminal I/O restored to logging mode")
                
                logger.log_stage_end("Stage 5B: Visualize Segmented Data", success)
            except Exception as e:
                # Make sure to restore logging even if there's an error
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                logger.log_error(e, "Running visualize segmented data", "Stage 5B: Visualize Segmented Data")
                logger.log_stage_end("Stage 5B: Visualize Segmented Data", False, f"Error: {str(e)}")
        else:
            error_msg = "run_visualize_segmented function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 5B: Visualize Segmented Data")
            logger.log_stage_end("Stage 5B: Visualize Segmented Data", False, error_msg)
            
    # --- Stage 5C: Manual Segment From Mask ---
    if args.manual_segment_from_mask or args.all:
        logger.log_stage_start("Stage 5C: Manual Segment From Mask", "Interactive manual segmentation from masked NPZ files")
        if run_manual_segment_from_mask:
            try:
                # Prompt user for NPZ directory choice
                print("\nManual Segment From Mask - NPZ Directory Selection:")
                print("  [1] Use external mask NPZ files (external_mask_npz_datasets)")
                print("  [2] Use segmented NPZ files (segmented_npz_datasets)")
                
                # Check if directories exist and show file counts
                external_count = 0
                segmented_count = 0
                
                if os.path.exists(external_mask_npz_dir):
                    external_count = len([f for f in os.listdir(external_mask_npz_dir) if f.endswith('.npz')])
                    print(f"      → {external_count} NPZ files found in external_mask_npz_datasets")
                else:
                    print("      → external_mask_npz_datasets directory not found")
                
                if os.path.exists(segmented_npz_dir):
                    segmented_count = len([f for f in os.listdir(segmented_npz_dir) if f.endswith('.npz')])
                    print(f"      → {segmented_count} NPZ files found in segmented_npz_datasets")
                else:
                    print("      → segmented_npz_datasets directory not found")
                
                # Get user choice
                while True:
                    user_choice = input("Select option (1 or 2, default: 1): ").strip()
                    if user_choice == "" or user_choice == "1":
                        selected_npz_dir = external_mask_npz_dir
                        dir_name = "external_mask_npz_datasets"
                        print(f"→ Using {dir_name} directory for manual segmentation from mask.")
                        break
                    elif user_choice == "2":
                        selected_npz_dir = segmented_npz_dir
                        dir_name = "segmented_npz_datasets"
                        print(f"→ Using {dir_name} directory for manual segmentation from mask.")
                        break
                    else:
                        print("Please enter 1 or 2.")
                
                # Log the choice
                logger.logger.info(f"User selected NPZ directory: {dir_name}")
                logger.logger.info(f"NPZ directory path: {selected_npz_dir}")
                
                # Manual segment from mask is always interactive, so restore terminal I/O
                logger.logger.info("Manual segment from mask is interactive - restoring terminal I/O")
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                logger.logger.info("Terminal I/O restored for manual segment from mask mode")

                # Run manual segment from mask with selected directory
                success = run_manual_segment_from_mask(
                    config, 
                    selected_npz_dir, 
                    segmented_dir, 
                    plots_dir, 
                    lifetime_dir,
                    True  # interactive mode
                )

                # Restore log file redirection
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                logger.logger.info("Terminal I/O restored to logging mode")
                
                logger.log_stage_end("Stage 5C: Manual Segment From Mask", success)
            except Exception as e:
                # Make sure to restore logging even if there's an error
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                logger.log_error(e, "Running manual segment from mask", "Stage 5C: Manual Segment From Mask")
                logger.log_stage_end("Stage 5C: Manual Segment From Mask", False, f"Error: {str(e)}")
        else:
            error_msg = "run_manual_segment_from_mask function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 5C: Manual Segment From Mask")
            logger.log_stage_end("Stage 5C: Manual Segment From Mask", False, error_msg)
            
    # --- Stage 5D: Manual Segment Unfiltered From Mask ---
    if args.manual_segment_unfiltered_from_mask or args.all:
        logger.log_stage_start("Stage 5D: Manual Segment Unfiltered From Mask", "Interactive manual segmentation from masked NPZ files using unfiltered data (GU*mask, SU*mask)")
        if run_manual_segment_unfiltered_from_mask:
            try:
                # Prompt user for NPZ directory choice
                print("\nManual Segment Unfiltered From Mask - NPZ Directory Selection:")
                print("  [1] Use external mask NPZ files (external_mask_npz_datasets)")
                print("  [2] Use segmented NPZ files (segmented_npz_datasets)")
                
                # Check if directories exist and show file counts
                external_count = 0
                segmented_count = 0
                
                if os.path.exists(external_mask_npz_dir):
                    external_count = len([f for f in os.listdir(external_mask_npz_dir) if f.endswith('.npz')])
                    print(f"      → {external_count} NPZ files found in external_mask_npz_datasets")
                else:
                    print("      → external_mask_npz_datasets directory not found")
                
                if os.path.exists(segmented_npz_dir):
                    segmented_count = len([f for f in os.listdir(segmented_npz_dir) if f.endswith('.npz')])
                    print(f"      → {segmented_count} NPZ files found in segmented_npz_datasets")
                else:
                    print("      → segmented_npz_datasets directory not found")
                
                # Get user choice
                while True:
                    user_choice = input("Select option (1 or 2, default: 1): ").strip()
                    if user_choice == "" or user_choice == "1":
                        selected_npz_dir = external_mask_npz_dir
                        dir_name = "external_mask_npz_datasets"
                        print(f"→ Using {dir_name} directory for manual segmentation unfiltered from mask.")
                        break
                    elif user_choice == "2":
                        selected_npz_dir = segmented_npz_dir
                        dir_name = "segmented_npz_datasets"
                        print(f"→ Using {dir_name} directory for manual segmentation unfiltered from mask.")
                        break
                    else:
                        print("Please enter 1 or 2.")
                
                # Log the choice
                logger.logger.info(f"User selected NPZ directory: {dir_name}")
                logger.logger.info(f"NPZ directory path: {selected_npz_dir}")
                
                # Manual segment unfiltered from mask is always interactive, so restore terminal I/O
                logger.logger.info("Manual segment unfiltered from mask is interactive - restoring terminal I/O")
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                logger.logger.info("Terminal I/O restored for manual segment unfiltered from mask mode")

                # Run manual segment unfiltered from mask with selected directory
                success = run_manual_segment_unfiltered_from_mask(
                    config, 
                    selected_npz_dir, 
                    segmented_dir, 
                    plots_dir, 
                    lifetime_dir,
                    True  # interactive mode
                )

                # Restore log file redirection
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                logger.logger.info("Terminal I/O restored to logging mode")
                
                logger.log_stage_end("Stage 5D: Manual Segment Unfiltered From Mask", success)
            except Exception as e:
                # Make sure to restore logging even if there's an error
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                logger.log_error(e, "Running manual segment unfiltered from mask", "Stage 5D: Manual Segment Unfiltered From Mask")
                logger.log_stage_end("Stage 5D: Manual Segment Unfiltered From Mask", False, f"Error: {str(e)}")
        else:
            error_msg = "run_manual_segment_unfiltered_from_mask function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 5D: Manual Segment Unfiltered From Mask")
            logger.log_stage_end("Stage 5D: Manual Segment Unfiltered From Mask", False, error_msg)
            
    end_pipeline_time = time.time()
    total_time = end_pipeline_time - start_pipeline_time
    
    logger.logger.info("=================================")
    logger.logger.info(" FLIM-FRET Analysis Pipeline End ")
    logger.logger.info(f" Total Time: {total_time:.2f} seconds")
    logger.logger.info("=================================")
    
    # Generate and save error report
    logger.save_error_report()
    
    # Log final summary
    logger.logger.info("Pipeline completed. Check error report for detailed analysis.")

if __name__ == "__main__":
    # Remove config check here, paths are checked in parse_arguments
    main() 