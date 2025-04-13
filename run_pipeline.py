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

# Print sys.path right before imports
print("--- sys.path before imports in run_pipeline.py ---")
print(sys.path)
print("--------------------------------------------------")

# --- Import necessary functions from other scripts ---

# It's generally better practice to organize these into modules,
# but for simplicity, we'll import directly if they are in the same directory.

try:
    # Stage 1: Preprocessing
    from TCSPC_preprocessing_AUTOcal_v2_0 import run_preprocessing
except ImportError as e:
    # Use current filename in error message
    print(f"Error: Could not import run_preprocessing from TCSPC_preprocessing_AUTOcal_v2_0.py: {e}") 
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_preprocessing = None # Placeholder

try:
    # Stage 2: Wavelet Filtering & NPZ Generation
    from ComplexWaveletFilter_v1_6 import main as run_wavelet_filtering
except ImportError as e:
    # Use current filename in error message
    print(f"Error: Could not import main (as run_wavelet_filtering) from ComplexWaveletFilter_v1_6.py: {e}") 
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_wavelet_filtering = None # Placeholder
    
try:
    # Stage 3: GMM Segmentation, Plotting, Lifetime Saving
    from GMMSegmentation_v2_6 import main as run_gmm_segmentation
except ImportError as e:
    # Use current filename in error message
    print(f"Error: Could not import main (as run_gmm_segmentation) from GMMSegmentation_v2_6.py: {e}") 
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_gmm_segmentation = None # Placeholder
    
# --- New Test Function ---
def test_flute_integration(config):
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
        print(f"❌ Error: flute_path not found or invalid: {flute_path}")
        return False
    
    if not flute_python_path or not os.path.exists(flute_python_path):
        print(f"❌ Error: flute_python_path not found or invalid: {flute_python_path}")
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
    
    print("\n===================================")
    print("  All FLUTE integration tests passed!")
    print("===================================")
    
    return True
    
# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run FLIM-FRET analysis pipeline stages.")
    
    # Input/Output Arguments
    parser.add_argument('-i', '--input_dir', 
                        help="Root directory containing the raw input data (.bin files). Equivalent to 'raw_data_root'.")
    parser.add_argument('-o', '--output_base_dir',  
                        help="Base directory where all output subfolders (output, preprocessed, npz_datasets, etc.) will be created.")

    # Stage Selection Arguments
    parser.add_argument('--preprocess', action='store_true',
                        help="Run Stage 1: Preprocessing (ImageJ + FLUTE)")
    parser.add_argument('--filter', action='store_true',
                        help="Run Stage 2: Wavelet Filtering & NPZ Generation")
    parser.add_argument('--segment', action='store_true',
                        help="Run Stage 3: GMM Segmentation, Plotting, Lifetime Saving")
    parser.add_argument('--all', action='store_true',
                        help="Run all stages sequentially.")
    
    # New Test Argument
    parser.add_argument('--test', action='store_true',
                        help="Run FLUTE integration tests to verify setup")
    
    args = parser.parse_args()
    
    # If --test is specified, we don't need input/output dirs
    if args.test:
        return args
    
    # Check for required arguments when running pipeline stages
    if not args.input_dir:
        parser.error("--input_dir/-i is required when running pipeline stages")
    if not args.output_base_dir:
        parser.error("--output_base_dir/-o is required when running pipeline stages")
        
    # Default to running all if no specific stage is selected
    if not any([args.preprocess, args.filter, args.segment, args.all, args.test]):
        print("No specific stage selected. Defaulting to running all stages (--all).")
        args.all = True
        
    # Validate paths
    if not os.path.isdir(args.input_dir):
        parser.error(f"Input directory not found: {args.input_dir}")
    # Check for fixed calibration file in current directory
    if not os.path.exists("calibration.csv"):
        parser.error(f"Calibration file not found: calibration.csv (expected in current directory)")
        
    return args

# --- Helper to load config (less prone to failure if keys change) ---
def load_pipeline_config(config_path="config.json"):
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
         
    # Handle test mode
    if args.test:
        success = test_flute_integration(config)
        sys.exit(0 if success else 1)
    
    # Define specific output subdirectories based on the output_base_dir argument
    output_dir = os.path.join(args.output_base_dir, 'output')
    preprocessed_dir = os.path.join(args.output_base_dir, 'preprocessed')
    npz_dir = os.path.join(args.output_base_dir, 'npz_datasets')
    segmented_dir = os.path.join(args.output_base_dir, 'segmented')
    plots_dir = os.path.join(args.output_base_dir, 'plots')
    lifetime_dir = os.path.join(args.output_base_dir, 'lifetime_images')
    
    # Create base output directory if it doesn't exist
    try:
         os.makedirs(args.output_base_dir, exist_ok=True)
         
         # Create all required output subdirectories
         os.makedirs(output_dir, exist_ok=True)
         os.makedirs(preprocessed_dir, exist_ok=True)
         os.makedirs(npz_dir, exist_ok=True)
         os.makedirs(segmented_dir, exist_ok=True)
         os.makedirs(plots_dir, exist_ok=True)
         os.makedirs(lifetime_dir, exist_ok=True)
         
         print(f"Created output directories:")
         print(f" - {output_dir}")
         print(f" - {preprocessed_dir}")
         print(f" - {npz_dir}")
         print(f" - {segmented_dir}")
         print(f" - {plots_dir}")
         print(f" - {lifetime_dir}")
    except OSError as e:
         print(f"Error creating output directories: {e}", file=sys.stderr)
         sys.exit(1)
    
    # Define fixed calibration file path
    calibration_file_path = "calibration.csv"
    
    start_pipeline_time = time.time()
    print("\n===================================")
    print(" FLIM-FRET Analysis Pipeline Start ")
    print(f" Input Dir: {args.input_dir}")
    print(f" Output Base: {args.output_base_dir}")
    print(f" Calibration: {calibration_file_path}")
    print("===================================")

    # --- Stage 1: Preprocessing ---
    if args.preprocess or args.all:
        print("\n--- Running Stage 1: Preprocessing ---")
        if run_preprocessing:
            try:
                stage_start = time.time()
                success = run_preprocessing(
                    config,
                    args.input_dir,       
                    output_dir,           
                    preprocessed_dir,     
                    calibration_file_path, # Pass fixed calibration path
                    args.input_dir        
                )
                stage_end = time.time()
                if success:
                    print(f"--- Stage 1 Finished ({stage_end - stage_start:.2f} seconds) ---")
                else:
                    print(f"!!! Stage 1 Failed (check errors above) ({stage_end - stage_start:.2f} seconds) !!!")
                    # Decide if pipeline should stop on failure
                    # sys.exit(1) 
            except Exception as e:
                print(f"!!! Uncaught Error during Stage 1: Preprocessing: {e}", file=sys.stderr)
        else:
            print("!!! Cannot run Stage 1: run_preprocessing function not available.", file=sys.stderr)
            
    # --- Stage 2: Wavelet Filtering & NPZ Generation ---
    if args.filter or args.all:
        print("\n--- Running Stage 2: Wavelet Filtering & NPZ Generation ---")
        if run_wavelet_filtering:
            try:
                stage_start = time.time()
                # Pass required arguments
                success = run_wavelet_filtering(
                    config, 
                    preprocessed_dir, 
                    npz_dir
                )
                stage_end = time.time()
                if success:
                    print(f"--- Stage 2 Finished ({stage_end - stage_start:.2f} seconds) ---")
                else:
                    print(f"!!! Stage 2 Failed (check errors above) ({stage_end - stage_start:.2f} seconds) !!!")
            except Exception as e:
                print(f"!!! Uncaught Error during Stage 2: Wavelet Filtering: {e}", file=sys.stderr)
        else:
            print("!!! Cannot run Stage 2: run_wavelet_filtering function not available.", file=sys.stderr)

    # --- Stage 3: GMM Segmentation, Plotting, Lifetime Saving ---
    if args.segment or args.all:
        print("\n--- Running Stage 3: GMM Segmentation, Plotting, Lifetime Saving ---")
        if run_gmm_segmentation:
            try:
                stage_start = time.time()
                # Pass required arguments
                success = run_gmm_segmentation(
                    config, 
                    npz_dir, 
                    segmented_dir, 
                    plots_dir, 
                    lifetime_dir
                )
                stage_end = time.time()
                if success:
                    print(f"--- Stage 3 Finished ({stage_end - stage_start:.2f} seconds) ---")
                else:
                    print(f"!!! Stage 3 Failed (check errors above) ({stage_end - stage_start:.2f} seconds) !!!")
            except Exception as e:
                print(f"!!! Uncaught Error during Stage 3: GMM Segmentation: {e}", file=sys.stderr)
        else:
            print("!!! Cannot run Stage 3: run_gmm_segmentation function not available.", file=sys.stderr)
            
    end_pipeline_time = time.time()
    print("\n=================================")
    print(" FLIM-FRET Analysis Pipeline End ")
    print(f" Total Time: {end_pipeline_time - start_pipeline_time:.2f} seconds")
    print("=================================")

if __name__ == "__main__":
    # Remove config check here, paths are checked in parse_arguments
    main() 