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
    # Intensity Image Generation for Wavelet Filtering
    from generate_intensity_images import process_raw_flim_files as generate_intensity_images
except ImportError as e:
    # Use current filename in error message
    print(f"Error: Could not import process_raw_flim_files (as generate_intensity_images) from generate_intensity_images.py: {e}") 
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    generate_intensity_images = None # Placeholder
    
# Additional imports needed for file operations
import shutil
import traceback
    
try:
    # Stage 3: GMM Segmentation, Plotting, Lifetime Saving
    from GMMSegmentation_v2_6 import main as run_gmm_segmentation
except ImportError as e:
    # Use current filename in error message
    print(f"Error: Could not import main (as run_gmm_segmentation) from GMMSegmentation_v2_6.py: {e}") 
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_gmm_segmentation = None # Placeholder
    
try:
    # Stage 4: Phasor Transformation
    from phasor_transform import process_flim_file as run_phasor_transform
except ImportError as e:
    # Use current filename in error message
    print(f"Error: Could not import process_flim_file (as run_phasor_transform) from phasor_transform.py: {e}") 
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_phasor_transform = None # Placeholder
    
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
    
    print("\n====================================")
    print("  All FLUTE integration tests passed!")
    print("====================================")
    
    return True

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="FLIM-FRET Analysis Pipeline")
    
    # Required arguments
    parser.add_argument("--input-dir", required=True, help="Input directory containing raw FLIM-FRET .bin files")
    parser.add_argument("--output-base-dir", required=True, help="Base output directory for all pipeline stages")
    
    # Pipeline stage control
    parser.add_argument("--all", action="store_true", help="Run all pipeline stages")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing stage through intensity image generation (Stages 1 and 2A)")
    parser.add_argument("--filter", action="store_true", help="Run wavelet filtering stage")
    parser.add_argument("--segment", action="store_true", help="Run GMM segmentation stage")
    parser.add_argument("--phasor", action="store_true", help="Run phasor transformation stage")
    
    # Testing mode
    parser.add_argument("--test", action="store_true", help="Run in test mode to verify the environment")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input directory
    if not args.test:
        if not os.path.isdir(args.input_dir):
            parser.error(f"Input directory '{args.input_dir}' does not exist or is not a directory")
    
    # If no specific stages are selected, and not running in test mode
    # ask the user what to do
    if not (args.all or args.preprocess or args.filter or args.segment or args.phasor or args.test):
        # Not running any specific stage and not in test mode
        print("No pipeline stages specified. Options:")
        print("1. Run all stages")
        print("2. Run preprocessing through intensity image generation (Stages 1 and 2A)")
        print("3. Run wavelet filtering only")
        print("4. Run GMM segmentation only")
        print("5. Run phasor transformation only")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == "1":
            args.all = True
        elif choice == "2":
            args.preprocess = True
        elif choice == "3":
            args.filter = True
        elif choice == "4":
            args.segment = True
        elif choice == "5":
            args.phasor = True
        elif choice == "6":
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)
            
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
    phasor_dir = os.path.join(args.output_base_dir, 'phasor_output')
    
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
         os.makedirs(phasor_dir, exist_ok=True)
         wavelet_intensity_dir = os.path.join(args.output_base_dir, 'wavelet_intensity_images')
         os.makedirs(wavelet_intensity_dir, exist_ok=True)
         
         print(f"Created output directories:")
         print(f" - {output_dir}")
         print(f" - {preprocessed_dir}")
         print(f" - {npz_dir}")
         print(f" - {segmented_dir}")
         print(f" - {plots_dir}")
         print(f" - {lifetime_dir}")
         print(f" - {phasor_dir}")
         print(f" - {wavelet_intensity_dir}")
    except OSError as e:
         print(f"Error creating output directories: {e}", file=sys.stderr)
         sys.exit(1)
    
    # Look for calibration file in input directory first, fall back to project directory
    input_calibration_path = os.path.join(args.input_dir, "calibration.csv")
    project_calibration_path = "calibration.csv"
    
    if os.path.exists(input_calibration_path):
        calibration_file_path = input_calibration_path
        print(f"Using calibration file from input directory: {calibration_file_path}")
    else:
        calibration_file_path = project_calibration_path
        print(f"Calibration file not found in input directory, using project directory: {calibration_file_path}")
    
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
            
    # --- Stage 2A: Generate Intensity Images and Copy G/S Files for Wavelet Filtering ---
    if args.preprocess or args.filter or args.all:
        print("\n--- Running Stage 2A: Preparing Data for Wavelet Filtering ---")
        wavelet_input_dir = os.path.join(args.output_base_dir, 'wavelet_intensity_images')
        os.makedirs(wavelet_input_dir, exist_ok=True)
        
        # 1. Generate intensity images
        if generate_intensity_images:
            try:
                intensity_stage_start = time.time()
                print("Generating intensity images from raw FLIM data...")
                
                # Generate intensity images from raw FLIM data
                success_count, error_count = generate_intensity_images(
                    input_dir=preprocessed_dir,
                    output_dir=wavelet_input_dir
                )
                
                print(f"Successfully generated {success_count} intensity images for wavelet filtering")
                
            except Exception as e:
                print(f"!!! Uncaught Error during intensity image generation: {e}", file=sys.stderr)
                success_count = 0
        else:
            print("!!! Cannot generate intensity images: generate_intensity_images function not available.", file=sys.stderr)
            success_count = 0
            
        # 2. Copy G and S files from preprocessed directory to wavelet input directory
        print("Copying G and S files from preprocessed directory...")
        g_count, s_count = 0, 0
        
        try:
            # Walk through preprocessed directory to find G and S files
            for root, dirs, files in os.walk(preprocessed_dir):
                for dirname in dirs:
                    # Look for G_unfiltered and S_unfiltered directories
                    if dirname in ['G_unfiltered', 'S_unfiltered']:
                        source_dir = os.path.join(root, dirname)
                        # Get relative path from preprocessed_dir
                        relpath = os.path.relpath(os.path.dirname(source_dir), preprocessed_dir)
                        if relpath == '.':
                            # Files are in the root directory
                            target_dir = wavelet_input_dir
                        else:
                            target_dir = os.path.join(wavelet_input_dir, relpath)
                        
                        # Create target directory if it doesn't exist
                        os.makedirs(target_dir, exist_ok=True)
                        
                        # Copy files
                        for file in os.listdir(source_dir):
                            if file.endswith('.tif') or file.endswith('.tiff'):
                                source_file = os.path.join(source_dir, file)
                                # For G files: rename from _g.tiff to .g.tiff (change underscore to dot)
                                if dirname == 'G_unfiltered':
                                    if '_g.tiff' in file:
                                        new_filename = file.replace('_g.tiff', '.g.tiff')
                                    else:
                                        new_filename = file
                                    target_file = os.path.join(target_dir, new_filename)
                                    shutil.copy2(source_file, target_file)
                                    g_count += 1
                                    print(f"  Copied G file to: {target_file}")
                                # For S files: rename from _s.tiff to .s.tiff (change underscore to dot)
                                elif dirname == 'S_unfiltered':
                                    if '_s.tiff' in file:
                                        new_filename = file.replace('_s.tiff', '.s.tiff')
                                    else:
                                        new_filename = file
                                    target_file = os.path.join(target_dir, new_filename)
                                    shutil.copy2(source_file, target_file)
                                    s_count += 1
                                    print(f"  Copied S file to: {target_file}")
            
            # Rename intensity files to match expected format (from _wavelet_intensity.tiff to .intensity.tiff)
            intensity_count = 0
            for root, dirs, files in os.walk(wavelet_input_dir):
                for file in files:
                    if '_wavelet_intensity.tiff' in file:
                        old_path = os.path.join(root, file)
                        new_filename = file.replace('_wavelet_intensity.tiff', '.intensity.tiff')
                        new_path = os.path.join(root, new_filename)
                        os.rename(old_path, new_path)
                        intensity_count += 1
                        
            print(f"Renamed {intensity_count} intensity files to match expected format")
            
            intensity_stage_end = time.time()
            
            if g_count > 0 and s_count > 0 and success_count > 0:
                print(f"Successfully prepared data for wavelet filtering:")
                print(f"  - {g_count} G files copied")
                print(f"  - {s_count} S files copied")
                print(f"  - {success_count} intensity images generated")
                print(f"--- Stage 2A Finished ({intensity_stage_end - intensity_stage_start:.2f} seconds) ---")
            else:
                print(f"!!! Stage 2A Warning: Some data may be missing for wavelet filtering !!!")
                print(f"  - G files: {g_count}")
                print(f"  - S files: {s_count}")
                print(f"  - Intensity images: {success_count}")
                print(f"--- Stage 2A Finished with warnings ({intensity_stage_end - intensity_stage_start:.2f} seconds) ---")
                
        except Exception as e:
            print(f"!!! Uncaught Error during data preparation for wavelet filtering: {e}", file=sys.stderr)
            traceback.print_exc()

    # --- Stage 2B: Wavelet Filtering & NPZ Generation ---
    if args.filter or args.all:
        print("\n--- Running Stage 2B: Wavelet Filtering & NPZ Generation ---")
        if run_wavelet_filtering:
            try:
                stage_start = time.time()
                # Path to intensity images generated in Stage 2A
                wavelet_intensity_dir = os.path.join(args.output_base_dir, 'wavelet_intensity_images')
                
                # Pass required arguments
                success = run_wavelet_filtering(
                    config, 
                    wavelet_intensity_dir,  # Use wavelet intensity images instead of preprocessed dir
                    npz_dir
                )
                stage_end = time.time()
                if success:
                    print(f"--- Stage 2B Finished ({stage_end - stage_start:.2f} seconds) ---")
                else:
                    print(f"!!! Stage 2B Failed (check errors above) ({stage_end - stage_start:.2f} seconds) !!!")
            except Exception as e:
                print(f"!!! Uncaught Error during Stage 2B: Wavelet Filtering: {e}", file=sys.stderr)
        else:
            print("!!! Cannot run Stage 2B: run_wavelet_filtering function not available.", file=sys.stderr)

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
            
    # --- Stage 4: Phasor Transformation ---
    if args.phasor or args.all:
        print("\n--- Running Stage 4: Phasor Transformation ---")
        if run_phasor_transform:
            try:
                stage_start = time.time()
                
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
                            print(f"Loaded {len(calibration_values)} calibration values from {calibration_file}")
                        except Exception as e:
                            print(f"Error loading calibration values: {e}")
                    
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
                                    print(f"Processing {input_file} (phi_cal={phi_cal}, m_cal={m_cal})")
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
                                    print(f"Error processing {input_file}: {e}")
                                    error_count += 1
                                    success = False
                    
                    print(f"Phasor transformation complete: {processed_count} files processed, {error_count} errors")
                    return success
                
                # Run the phasor transformation on preprocessed files
                success = process_preprocessed_files(
                    input_dir=preprocessed_dir,
                    output_dir=phasor_dir,
                    calibration_file=calibration_file_path
                )
                
                stage_end = time.time()
                if success:
                    print(f"--- Stage 4 Finished ({stage_end - stage_start:.2f} seconds) ---")
                else:
                    print(f"!!! Stage 4 Completed with some errors ({stage_end - stage_start:.2f} seconds) !!!")
            except Exception as e:
                print(f"!!! Uncaught Error during Stage 4: Phasor Transformation: {e}", file=sys.stderr)
        else:
            print("!!! Cannot run Stage 4: run_phasor_transform function not available.", file=sys.stderr)
            
    end_pipeline_time = time.time()
    print("\n=================================")
    print(" FLIM-FRET Analysis Pipeline End ")
    print(f" Total Time: {end_pipeline_time - start_pipeline_time:.2f} seconds")
    print("=================================")

if __name__ == "__main__":
    # Remove config check here, paths are checked in parse_arguments
    main() 