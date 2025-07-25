#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to run the preprocessing part of the FLIM-FRET analysis pipeline directly.
"""

import json
import os
import sys

# Import directly from the current directory
from src.python.modules.preprocessing import run_preprocessing

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    # Define paths
    input_dir = "/Volumes/NX-01-A/FLIM_workflow_test_data"
    output_base_dir = "/Volumes/NX-01-A/FLIM_workflow_test_data_analysis"
    output_dir = os.path.join(output_base_dir, 'output')
    preprocessed_dir = os.path.join(output_base_dir, 'preprocessed')
    
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Load config
    config = load_config()
    
    # Run preprocessing
    print("Running preprocessing directly...")
    calibration_file = "calibration.csv"
    
    # Make sure the function has the right signature
    result = run_preprocessing(
        config=config,
        input_dir=input_dir,
        output_dir=output_dir,
        preprocessed_dir=preprocessed_dir,
        calibration_file=calibration_file,
        raw_data_root=input_dir
    )
    
    if result:
        print("Preprocessing completed successfully!")
    else:
        print("Preprocessing failed!")

if __name__ == "__main__":
    main() 