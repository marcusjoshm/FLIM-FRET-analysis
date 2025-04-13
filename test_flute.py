#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to run the FLUTE processing step on TIF files.
"""

import os
import json
import sys
from TCSPC_preprocessing_AUTOcal_v2_0 import process_tiffs_with_flute

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    # Define paths
    input_dir = "/Volumes/NX-01-A/FLIM_workflow_test_data"
    base_output_dir = "/Volumes/NX-01-A/FLIM_workflow_test_data_analysis"
    output_dir = os.path.join(base_output_dir, 'output')
    
    # Load config and parameters
    config = load_config()
    flute_path = config["flute_path"]
    microscope_params = config["microscope_params"]
    calibration_file = "calibration.csv"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Run FLUTE processing on the TIF files in output_dir
    print("\n=== Starting FLUTE Processing ===")
    print(f"FLUTE path: {flute_path}")
    print(f"Calibration file: {calibration_file}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Microscope parameters: {microscope_params}\n")
    
    # Process TIFFs with FLUTE
    result = process_tiffs_with_flute(
        calibration_file=calibration_file,
        base_output_dir=output_dir, 
        raw_data_root=input_dir,
        microscope_params=microscope_params,
        flute_path=flute_path
    )
    
    if result:
        print("\n=== FLUTE Processing Completed Successfully ===")
    else:
        print("\n=== FLUTE Processing Failed ===")

if __name__ == "__main__":
    main() 