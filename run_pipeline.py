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

# Print sys.path right before imports
print("--- sys.path before imports in run_pipeline.py ---")
print(sys.path)
print("--------------------------------------------------")

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
    # Stage 1B: Filename simplification (optional)
    from src.python.modules.simplify_filenames import simplify_filenames
except ImportError as e:
    # Use current filename in error message
    print(f"Error: Could not import simplify_filenames from simplify_filenames.py: {e}")
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    simplify_filenames = None # Placeholder

try:
    # Stage 2: Wavelet Filtering & NPZ Generation
    from src.python.modules.ComplexWaveletFilter_v2_0 import main as run_wavelet_filtering
    print("Using Complex Wavelet Filter v2.0 implementation")
except ImportError as e:
    print(f"Error: Could not import main (as run_wavelet_filtering) from ComplexWaveletFilter_v2_0.py: {e}") 
    print("Ensure the ComplexWaveletFilter_v2_0.py module is available.")
    run_wavelet_filtering = None # Placeholder
    
try:
    # Stage 3: Phasor Visualization
    from src.python.modules.phasor_visualization import run_phasor_visualization
except ImportError as e:
    print(f"Error: Could not import run_phasor_visualization from phasor_visualization.py: {e}")
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_phasor_visualization = None # Placeholder
    
try:
    # Intensity Image Generation for Wavelet Filtering
    from src.python.modules.generate_intensity_images import process_raw_flim_files as generate_intensity_images
except ImportError as e:
    # Use current filename in error message
    print(f"Error: Could not import process_raw_flim_files (as generate_intensity_images) from generate_intensity_images.py: {e}") 
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    generate_intensity_images = None # Placeholder
    
# Additional imports needed for file operations
import shutil

try:
    # Stage 3: GMM Segmentation, Plotting, Lifetime Saving
    from src.python.modules.GMMSegmentation_v2_6 import main as run_gmm_segmentation
except ImportError as e:
    # Use current filename in error message
    print(f"Error: Could not import main (as run_gmm_segmentation) from GMMSegmentation_v2_6.py: {e}") 
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_gmm_segmentation = None # Placeholder
    
try:
    # Stage 4: Phasor Transformation
    from src.python.modules.phasor_transform import process_flim_file as run_phasor_transform
except ImportError as e:
    # Use current filename in error message
    print(f"Error: Could not import process_flim_file (as run_phasor_transform) from phasor_transform.py: {e}") 
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_phasor_transform = None # Placeholder

try:
    # Stage 4B: Manual Segmentation
    from src.python.modules.ManualSegmentation import main as run_manual_segmentation
except ImportError as e:
    # Use current filename in error message
    print(f"Error: Could not import main (as run_manual_segmentation) from ManualSegmentation.py: {e}") 
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_manual_segmentation = None # Placeholder
    
# --- New Test Function ---
def test_flute_integration(config, logger=None):
    """
    Tests the FLUTE integration to ensure that all components are working correctly.
    
    Args:
        config (dict): Configuration dictionary with paths and parameters
        
    Returns:
        bool: True if tests pass, False otherwise
    """
    print("\n===================================")
    print("       FLUTE Integration Test      ")
    print("===================================")
    
    # Test 1: FLUTE Python Environment
    print("\n--- Test 1: FLUTE Python Environment ---")
    
    flute_path = config.get("flute_path")
    flute_python_path = config.get("flute_python_path")
    
    if not flute_path or not os.path.exists(flute_path):
        error_msg = f"flute_path not found or invalid: {flute_path}"
        print(f"❌ Error: {error_msg}")
        if logger:
            logger.log_error(Exception(error_msg), "FLUTE path validation", "Test")
        return False
    
    if not flute_python_path or not os.path.exists(flute_python_path):
        error_msg = f"flute_python_path not found or invalid: {flute_python_path}"
        print(f"❌ Error: {error_msg}")
        if logger:
            logger.log_error(Exception(error_msg), "FLUTE Python path validation", "Test")
        return False
    
    print(f"✓ FLUTE path: {flute_path}")
    print(f"✓ FLUTE Python path: {flute_python_path}")
    
    # Test 2: Required Packages
    print("\n--- Test 2: Required Packages ---")
    
    # Create a test script to verify required packages
    test_script = "test_flute_packages.py"
    
    # Get FLUTE directory
    flute_dir = os.path.dirname(flute_path)
    
    with open(test_script, 'w') as f:
        f.write(f"""
import sys
import os

packages = [
    "PyQt5", 
    "numpy", 
    "cv2",  # OpenCV
    "matplotlib", 
    "scipy",
    "skimage"  # scikit-image
]

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("\\nTesting package imports:")

success = True
for package in packages:
    try:
        if package == "PyQt5":
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtCore import QObject
            print(f"  ✓ {{package}} (and QtWidgets, QtCore)")
        elif package == "cv2":
            import cv2
            print(f"  ✓ {{package}} (version: {{cv2.__version__}})")
        elif package == "numpy":
            import numpy as np
            print(f"  ✓ {{package}} (version: {{np.__version__}})")
        elif package == "matplotlib":
            import matplotlib
            print(f"  ✓ {{package}} (version: {{matplotlib.__version__}})")
        elif package == "scipy":
            import scipy
            print(f"  ✓ {{package}} (version: {{scipy.__version__}})")
        elif package == "skimage":
            import skimage
            print(f"  ✓ {{package}} (version: {{skimage.__version__}})")
        else:
            module = __import__(package)
            print(f"  ✓ {{package}}")
    except ImportError as e:
        print(f"  ❌ {{package}}: {{e}}")
        success = False

sys.exit(0 if success else 1)
""")
    
    try:
        result = subprocess.run(
            [flute_python_path, test_script],
            check=False,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"❌ Some packages are missing. Check output above.")
            # Clean up and return
            if os.path.exists(test_script):
                os.remove(test_script)
            return False
            
        print("✓ All required packages are installed correctly!")
    except Exception as e:
        print(f"❌ Error running package test: {e}")
        if os.path.exists(test_script):
            os.remove(test_script)
        return False
        
    # Clean up
    if os.path.exists(test_script):
        os.remove(test_script)
    
    # Test 3: ImageHandler Import
    print("\n--- Test 3: ImageHandler Import ---")
    
    # Create a test script to verify ImageHandler import
    test_script = "test_flute_handler.py"
    
    with open(test_script, "w") as f:
        f.write(f"""
import os
import sys

# Add FLUTE directory to Python path
flute_dir = "{flute_dir}"
if flute_dir not in sys.path:
    sys.path.append(flute_dir)

try:
    from ImageHandler import ImageHandler
    print("Successfully imported ImageHandler from:", flute_dir)

    # List available attributes
    handler_attrs = [attr for attr in dir(ImageHandler) if not attr.startswith('__')]
    print("\\nImageHandler attributes (first 10):")
    for attr in sorted(handler_attrs)[:10]:
        print(f"  - {{attr}}")
        
    print("\\nImageHandler test successful!")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Error importing ImageHandler: {{e}}")
    print("Python path:")
    for p in sys.path:
        print(f"  {{p}}")
    print("ImageHandler test failed.")
    sys.exit(1)
""")
    
    try:
        # Run the test script with the FLUTE virtual environment's Python
        result = subprocess.run(
            [flute_python_path, test_script], 
            check=False, 
            capture_output=True, 
            text=True
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"❌ Failed to import ImageHandler. Check output above.")
            # Clean up and return
            if os.path.exists(test_script):
                os.remove(test_script)
            return False
            
        print("✓ ImageHandler can be imported and used correctly!")
    except Exception as e:
        print(f"❌ Error running ImageHandler test: {e}")
        if os.path.exists(test_script):
            os.remove(test_script)
        return False
        
    # Clean up
    if os.path.exists(test_script):
        os.remove(test_script)
    
    # Test 4: Simple Test Script (Sample Run)
    print("\n--- Test 4: Sample Run ---")
    
    # Create a test script for a simple sample run
    test_script = "test_flute_sample.py"
    
    with open(test_script, "w") as f:
        f.write(f"""
import os
import sys

# Add FLUTE directory to Python path
flute_dir = "{flute_dir}"
if flute_dir not in sys.path:
    sys.path.append(flute_dir)

from ImageHandler import ImageHandler

print("Creating and checking a simple ImageHandler instance...")
try:
    # Create a dummy handler (just to test the class)
    # Note: This doesn't actually process an image
    handler = ImageHandler.__new__(ImageHandler)
    print("ImageHandler instance created successfully!")
    print("Sample run successful!")
    sys.exit(0)
except Exception as e:
    print(f"❌ Error creating ImageHandler instance: {{e}}")
    print("Sample run failed.")
    sys.exit(1)
""")
    
    try:
        # Run the test script with the FLUTE virtual environment's Python
        result = subprocess.run(
            [flute_python_path, test_script], 
            check=False, 
            capture_output=True, 
            text=True
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"❌ Sample run failed. Check output above.")
            # Clean up and return
            if os.path.exists(test_script):
                os.remove(test_script)
            return False
            
        print("✓ Sample run completed successfully!")
    except Exception as e:
        print(f"❌ Error running sample test: {e}")
        if os.path.exists(test_script):
            os.remove(test_script)
        return False
    
    # Clean up
    if os.path.exists(test_script):
        os.remove(test_script)
    
    print("\n====================================")
    print("  All FLUTE integration tests passed!")
    print("====================================")
    
    return True

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
    parser.add_argument("--LF-preprocessing", action="store_true", help="LF workflow: Run preprocessing with automatic filename simplification")
    
    # Individual stages (for advanced users)
    parser.add_argument("--preprocess", action="store_true", help="[DEPRECATED] Use --preprocessing instead")
    parser.add_argument("--filter", action="store_true", help="Run only Stage 2B: wavelet filtering and lifetime calculation")
    parser.add_argument("--visualize", action="store_true", help="Run Stage 3: Interactive phasor visualization and plot generation")
    parser.add_argument("--segment", action="store_true", help="Run GMM segmentation stage")
    parser.add_argument("--manual-segment", action="store_true", help="Run manual segmentation stage")
    parser.add_argument("--phasor", action="store_true", help="Run phasor transformation stage")
    
    # Testing mode
    parser.add_argument("--test", action="store_true", help="Run in test mode to verify the environment")
    
    # Interactive mode
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode for user input (affects GMM segmentation)")
    
    # File naming options
    parser.add_argument("--simplify-filenames", action="store_true", help="Simplify filenames in preprocessed directory (e.g., R_1_s2_g.tiff -> 2.tiff)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input directory
    if not args.test:
        if not os.path.isdir(args.input):
            parser.error(f"Input directory '{args.input}' does not exist or is not a directory")
    
    # If no specific stages are selected, and not running in test mode
    # ask the user what to do
    if not (args.all or args.preprocessing or args.processing or args.LF_preprocessing or
            args.preprocess or args.filter or args.visualize or args.segment or args.manual_segment or args.phasor or args.test):
        # Not running any specific stage and not in test mode
        print("No pipeline stages specified. Options:")
        print("1. Preprocessing (.bin to .tif conversion + phasor transformation)")
        print("2. Processing (preprocessing + wavelet filtering and lifetime calculation)")
        print("3. LF preprocessing (preprocessing with simplified filenames)")
        print("4. Filter only (wavelet filtering)")
        print("5. Visualize (interactive phasor plots)")
        print("6. Segment (GMM segmentation with interactive parameter selection)")
        print("7. Manual Segment (interactive manual ellipse-based segmentation)")
        print("8. Phasor (phasor transformation only)")
        print("9. All stages")
        print("10. Exit")
        
        choice = input("Select an option (1-10): ")
        
        if choice == "1":
            args.preprocessing = True
        elif choice == "2":
            args.processing = True
        elif choice == "3":
            args.LF_preprocessing = True
        elif choice == "4":
            args.filter = True
        elif choice == "5":
            args.visualize = True
        elif choice == "6":
            args.segment = True
            args.interactive = True  # Automatically enable interactive mode for GMM segmentation
        elif choice == "7":
            args.manual_segment = True
        elif choice == "8":
            args.phasor = True
        elif choice == "9":
            args.all = True
        elif choice == "10":
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
         
    # Handle test mode
    if args.test:
        logger.log_stage_start("FLUTE Integration Test", "Testing FLUTE environment and dependencies")
        success = test_flute_integration(config, logger)
        logger.log_stage_end("FLUTE Integration Test", success)
        logger.save_error_report()
        sys.exit(0 if success else 1)
    
    # Define specific output subdirectories based on the output argument
    output_dir = os.path.join(args.output, 'output')
    preprocessed_dir = os.path.join(args.output, 'preprocessed')
    npz_dir = os.path.join(args.output, 'npz_datasets')
    segmented_dir = os.path.join(args.output, 'segmented')
    plots_dir = os.path.join(args.output, 'plots')
    lifetime_dir = os.path.join(args.output, 'lifetime_images')
    phasor_dir = os.path.join(args.output, 'phasor_output')
    
    # Create base output directory if it doesn't exist
    try:
         os.makedirs(args.output, exist_ok=True)
         
         # Create all required output subdirectories
         os.makedirs(output_dir, exist_ok=True)
         os.makedirs(preprocessed_dir, exist_ok=True)
         os.makedirs(npz_dir, exist_ok=True)
         os.makedirs(segmented_dir, exist_ok=True)
         os.makedirs(plots_dir, exist_ok=True)
         os.makedirs(lifetime_dir, exist_ok=True)
         os.makedirs(phasor_dir, exist_ok=True)
         
         logger.logger.info("Created output directories:")
         logger.logger.info(f" - {output_dir}")
         logger.logger.info(f" - {preprocessed_dir}")
         logger.logger.info(f" - {npz_dir}")
         logger.logger.info(f" - {segmented_dir}")
         logger.logger.info(f" - {plots_dir}")
         logger.logger.info(f" - {lifetime_dir}")
         logger.logger.info(f" - {phasor_dir}")
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
    if args.preprocess or args.preprocessing or args.processing or args.LF_preprocessing or args.all:
        logger.log_stage_start("Stage 1: Preprocessing", "Convert .bin files to .tif and perform phasor transformation")
        if run_preprocessing:
            try:
                success = run_preprocessing(
                    config,
                    args.input,       
                    output_dir,           
                    preprocessed_dir,     
                    calibration_file_path, # Pass fixed calibration path
                    args.input        
                )
                logger.log_stage_end("Stage 1: Preprocessing", success)
            except Exception as e:
                logger.log_error(e, "Running preprocessing", "Stage 1: Preprocessing")
                logger.log_stage_end("Stage 1: Preprocessing", False, f"Error: {str(e)}")
        else:
            error_msg = "run_preprocessing function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 1: Preprocessing")
            logger.log_stage_end("Stage 1: Preprocessing", False, error_msg)
            
    # --- Stage 2A: Optional Filename Simplification ---
    if args.preprocess or args.preprocessing or args.processing or args.LF_preprocessing or args.all:
        if (args.simplify_filenames or args.LF_preprocessing) and simplify_filenames:
            logger.log_stage_start("Stage 2A: Filename Simplification", "Simplify filenames for LF workflow")
            try:
                simple_success, simple_errors = simplify_filenames(preprocessed_dir, dry_run=False)
                
                if simple_success > 0:
                    logger.logger.info(f"Successfully simplified {simple_success} filenames (with {simple_errors} errors)")
                    logger.log_stage_end("Stage 2A: Filename Simplification", True, f"Simplified {simple_success} files")
                else:
                    logger.log_warning("No files were successfully simplified", "Filename simplification", "Stage 2A: Filename Simplification")
                    logger.log_stage_end("Stage 2A: Filename Simplification", False, "No files simplified")
            except Exception as e:
                logger.log_error(e, "Filename simplification", "Stage 2A: Filename Simplification")
                logger.log_stage_end("Stage 2A: Filename Simplification", False, f"Error: {str(e)}")
        elif args.simplify_filenames and not simplify_filenames:
            error_msg = "simplify_filenames function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 2A: Filename Simplification")
            logger.log_stage_end("Stage 2A: Filename Simplification", False, error_msg)

    # --- Stage 2B: Wavelet Filtering & NPZ Generation ---
    if args.filter or args.processing or args.all:
        logger.log_stage_start("Stage 2B: Wavelet Filtering", "Apply complex wavelet filtering and generate NPZ files")
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
                logger.log_stage_end("Stage 2B: Wavelet Filtering", success)
            except Exception as e:
                logger.log_error(e, "Running wavelet filtering", "Stage 2B: Wavelet Filtering")
                logger.log_stage_end("Stage 2B: Wavelet Filtering", False, f"Error: {str(e)}")
        else:
            error_msg = "run_wavelet_filtering function not available"
            logger.log_error(Exception(error_msg), "Import check", "Stage 2B: Wavelet Filtering")
            logger.log_stage_end("Stage 2B: Wavelet Filtering", False, error_msg)

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
                        from src.python.modules.GMMSegmentation_v2_6 import load_gmm_config
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
                # Manual segmentation is always interactive, so restore terminal I/O
                logger.logger.info("Manual segmentation is interactive - restoring terminal I/O")
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                logger.logger.info("Terminal I/O restored for manual segmentation mode")

                # Run manual segmentation
                success = run_manual_segmentation(
                    config, 
                    npz_dir, 
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