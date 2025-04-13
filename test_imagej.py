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
    
    # Run the first macro that processes FITC.bin files
    print("Running ImageJ Macro 1 (FITC.bin files)...")
    success1 = run_imagej_macro(imagej_path, macro_files[0], input_dir, output_dir)
    if success1:
        print("ImageJ macro 1 ran successfully!")
    else:
        print("ImageJ macro 1 failed!")
    
    # Wait a moment before continuing
    time.sleep(1)
    
    # Run the second macro that processes all bin files
    print("Running ImageJ Macro 2 (All .bin files)...")
    success2 = run_imagej_macro(imagej_path, macro_files[1], input_dir, output_dir)
    if success2:
        print("ImageJ macro 2 ran successfully!")
    else:
        print("ImageJ macro 2 failed!")
    
    # Wait a moment before continuing
    time.sleep(1)
    
    # Run the third macro that processes G and S files
    print("Running ImageJ Macro 3 (G and S files)...")
    success3 = run_imagej_macro(imagej_path, macro_files[2], output_dir, preprocessed_dir)
    if success3:
        print("ImageJ macro 3 ran successfully!")
    else:
        print("ImageJ macro 3 failed!")
    
    # Wait a moment before continuing
    time.sleep(1)
    
    # Run the fourth macro that processes intensity files
    print("Running ImageJ Macro 4 (intensity files)...")
    success4 = run_imagej_macro(imagej_path, macro_files[3], output_dir, preprocessed_dir)
    if success4:
        print("ImageJ macro 4 ran successfully!")
    else:
        print("ImageJ macro 4 failed!")
    
    # Wait a moment before continuing
    time.sleep(1)
    
    # Run the fifth macro that renames files
    print("Running ImageJ Macro 5 (file renaming)...")
    success5 = run_imagej_macro(imagej_path, macro_files[4], preprocessed_dir)
    if success5:
        print("ImageJ macro 5 ran successfully!")
    else:
        print("ImageJ macro 5 failed!")
    
    # Print overall status
    print("\nTest Summary:")
    print(f"Macro 1: {'SUCCESS' if success1 else 'FAILED'}")
    print(f"Macro 2: {'SUCCESS' if success2 else 'FAILED'}")
    print(f"Macro 3: {'SUCCESS' if success3 else 'FAILED'}")
    print(f"Macro 4: {'SUCCESS' if success4 else 'FAILED'}")
    print(f"Macro 5: {'SUCCESS' if success5 else 'FAILED'}")

if __name__ == "__main__":
    main() 