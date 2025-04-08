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
    
    parser.add_argument('--preprocess', action='store_true',
                        help="Run Stage 1: Preprocessing (ImageJ + FLUTE)")
    parser.add_argument('--filter', action='store_true',
                        help="Run Stage 2: Wavelet Filtering & NPZ Generation")
    parser.add_argument('--segment', action='store_true',
                        help="Run Stage 3: GMM Segmentation, Plotting, Lifetime Saving")
    parser.add_argument('--all', action='store_true',
                        help="Run all stages sequentially.")
    
    args = parser.parse_args()
    
    # If no specific stage is selected, default to running all
    if not any([args.preprocess, args.filter, args.segment, args.all]):
        print("No specific stage selected. Defaulting to running all stages (--all).")
        args.all = True
        
    return args

# --- Main Pipeline Execution ---
def main():
    args = parse_arguments()
    
    start_pipeline_time = time.time()
    print("\n===================================")
    print(" FLIM-FRET Analysis Pipeline Start ")
    print("===================================")

    # --- Stage 1: Preprocessing ---
    if args.preprocess or args.all:
        print("\n--- Running Stage 1: Preprocessing ---")
        if run_preprocessing:
            try:
                stage_start = time.time()
                run_preprocessing()
                stage_end = time.time()
                print(f"--- Stage 1 Finished ({stage_end - stage_start:.2f} seconds) ---")
            except Exception as e:
                print(f"!!! Error during Stage 1: Preprocessing: {e}", file=sys.stderr)
                # Decide if pipeline should stop on error
                # sys.exit(1)
        else:
            print("!!! Cannot run Stage 1: run_preprocessing function not available.", file=sys.stderr)
            
    # --- Stage 2: Wavelet Filtering & NPZ Generation ---
    if args.filter or args.all:
        print("\n--- Running Stage 2: Wavelet Filtering & NPZ Generation ---")
        if run_wavelet_filtering:
            try:
                stage_start = time.time()
                run_wavelet_filtering()
                stage_end = time.time()
                print(f"--- Stage 2 Finished ({stage_end - stage_start:.2f} seconds) ---")
            except Exception as e:
                print(f"!!! Error during Stage 2: Wavelet Filtering: {e}", file=sys.stderr)
        else:
            print("!!! Cannot run Stage 2: run_wavelet_filtering function not available.", file=sys.stderr)

    # --- Stage 3: GMM Segmentation, Plotting, Lifetime Saving ---
    if args.segment or args.all:
        print("\n--- Running Stage 3: GMM Segmentation, Plotting, Lifetime Saving ---")
        if run_gmm_segmentation:
            try:
                stage_start = time.time()
                run_gmm_segmentation()
                stage_end = time.time()
                print(f"--- Stage 3 Finished ({stage_end - stage_start:.2f} seconds) ---")
            except Exception as e:
                print(f"!!! Error during Stage 3: GMM Segmentation: {e}", file=sys.stderr)
        else:
            print("!!! Cannot run Stage 3: run_gmm_segmentation function not available.", file=sys.stderr)
            
    end_pipeline_time = time.time()
    print("\n=================================")
    print(" FLIM-FRET Analysis Pipeline End ")
    print(f" Total Time: {end_pipeline_time - start_pipeline_time:.2f} seconds")
    print("=================================")

if __name__ == "__main__":
    # Check if config file exists before starting
    if not os.path.exists("config.json"):
        print("Error: config.json not found in the current directory.", file=sys.stderr)
        print("Please ensure the configuration file exists.", file=sys.stderr)
        sys.exit(1)
        
    main() 