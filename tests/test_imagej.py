#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to run ImageJ macros.
"""
import subprocess
import json
import os
import time

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def run_imagej_macro(imagej_path, macro_file, *args):
    """Run an ImageJ macro with arguments"""
    command = [
        imagej_path,
        '-macro', macro_file, ",".join(args)
    ]
    print(f"Running command: {' '.join(command)}")
    
    try:
        # Run ImageJ with stdout captured
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"ImageJ Command Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print(f"Exit code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def main():
    config = load_config()
    
    imagej_path = config["imagej_path"]
    macro_files = config["macro_files"]
    
    # Define the paths
    input_dir = "/Volumes/NX-01-A/FLIM_workflow_test_data"
    base_output_dir = "/Volumes/NX-01-A/FLIM_workflow_test_data_analysis"
    output_dir = os.path.join(base_output_dir, 'output')
    preprocessed_dir = os.path.join(base_output_dir, 'preprocessed')
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Clean any existing output files
    print("Clearing existing output files...")
    subprocess.run(f"rm -rf {output_dir}/*", shell=True)
    subprocess.run(f"rm -rf {preprocessed_dir}/*", shell=True)
    
    # Run the single macro that processes all .bin files
    print("Running ImageJ Macro (All .bin files)...")
    success = run_imagej_macro(imagej_path, macro_files[0], input_dir, output_dir)
    if success:
        print("ImageJ macro ran successfully!")
    else:
        print("ImageJ macro failed!")
    
    # Print overall status
    print("\nTest Summary:")
    print(f"Macro: {'SUCCESS' if success else 'FAILED'}")

if __name__ == "__main__":
    main() 