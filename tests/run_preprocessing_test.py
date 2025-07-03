#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to run just the preprocessing stage with FLUTE.
"""

import os
import sys
import json
import time
from TCSPC_preprocessing_AUTOcal_v2_0 import run_preprocessing

def main():
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Define paths
    input_dir = "/Volumes/NX-01-A/FLIM_workflow_test_data"
    output_base_dir = "/Volumes/NX-01-A/FLIM_workflow_test_data_analysis"
    output_dir = os.path.join(output_base_dir, 'output')
    preprocessed_dir = os.path.join(output_base_dir, 'preprocessed')
    calibration_file = "calibration.csv"
    
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    print("\n===================================")
    print(" FLIM Preprocessing Test ")
    print(f" Input Dir: {input_dir}")
    print(f" Output Dir: {output_dir}")
    print(f" Preprocessed Dir: {preprocessed_dir}")
    print(f" Calibration: {calibration_file}")
    print(f" FLUTE Path: {config['flute_path']}")
    print(f" FLUTE Python: {config.get('flute_python_path', 'Not specified')}")
    print("===================================\n")
    
    # Run preprocessing
    try:
        start_time = time.time()
        
        success = run_preprocessing(
            config,
            input_dir,
            output_dir,
            preprocessed_dir,
            calibration_file,
            input_dir
        )
        
        end_time = time.time()
        
        if success:
            print(f"\nPreprocessing completed successfully in {end_time - start_time:.2f} seconds!")
        else:
            print("\nPreprocessing failed. Check the errors above.")
            
    except Exception as e:
        print(f"\nError running preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 