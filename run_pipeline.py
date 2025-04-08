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

# --- Import necessary functions from other scripts ---

# It's generally better practice to organize these into modules,
# but for simplicity, we'll import directly if they are in the same directory.

try:
    # Stage 1: Preprocessing
    from TCSPC_preprocessing_AUTOcal_v2_0 import run_preprocessing
except ImportError:
    print("Error: Could not import run_preprocessing from TCSPC_preprocessing_AUTOcal_v2.0.py")
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_preprocessing = None # Placeholder

try:
    # Stage 2: Wavelet Filtering & NPZ Generation
    from ComplexWaveletFilter_v1_6 import main as run_wavelet_filtering
except ImportError:
    print("Error: Could not import main (as run_wavelet_filtering) from ComplexWaveletFilter_v1.6.py")
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_wavelet_filtering = None # Placeholder
    
try:
    # Stage 3: GMM Segmentation, Plotting, Lifetime Saving
    from GMMSegmentation_v2_6 import main as run_gmm_segmentation
except ImportError:
    print("Error: Could not import main (as run_gmm_segmentation) from GMMSegmentation_v2.6.py")
    print("Ensure the script is in the same directory or accessible via PYTHONPATH.")
    run_gmm_segmentation = None # Placeholder
    
# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run FLIM-FRET analysis pipeline stages.")
    
    # Input/Output Arguments
    parser.add_argument('-i', '--input_dir', required=True, 
                        help="Root directory containing the raw input data (.bin files). Equivalent to 'raw_data_root'.")
    parser.add_argument('-o', '--output_base_dir', required=True, 
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
    
    args = parser.parse_args()
    
    # Default to running all if no specific stage is selected
    if not any([args.preprocess, args.filter, args.segment, args.all]):
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
    except OSError as e:
         print(f"Error creating base output directory {args.output_base_dir}: {e}", file=sys.stderr)
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