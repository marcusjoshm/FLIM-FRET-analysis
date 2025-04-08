#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:45:02 2024

@author: leelab & joshuamarcus
"""

import subprocess
import json
import os
import pandas as pd
import sys

# Load configurations from the JSON file
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Assign paths from configuration
imagej_path = config["imagej_path"]
flute_path = config["flute_path"]
macro_files = config["macro_files"]
input_dir = config["input_dir"]
output_dir = config["output_dir"]
preprocessed_dir = config["preprocessed_dir"]

# Add FLUTE directory to sys.path to allow importing ImageHandler
flute_dir = os.path.dirname(flute_path)
if flute_dir not in sys.path:
    sys.path.append(flute_dir)

try:
    # Dynamically import ImageHandler
    from ImageHandler_noGUI import ImageHandler
except ImportError as e:
    print(f"Error importing ImageHandler from {flute_dir}: {e}")
    print("Please ensure ImageHandler_noGUI.py exists in the FLUTE directory specified in config.json.")
    sys.exit(1) # Exit if import fails

# Function to run ImageJ with a given macro and directories
def run_imagej(macro_file, *args):
    command = [
        imagej_path,
        '-macro', macro_file, ",".join(args)
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the macro: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check the path to the ImageJ executable.")
    except PermissionError as e:
        print(f"Permission error: {e}")
        print("Please check the permissions of the ImageJ executable.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- New function to process TIFs using ImageHandler ---
def process_tiffs_with_flute(calibration_file, base_output_dir, bin_width_ns, freq_mhz, harmonic):
    """
    Processes TIFF files using FLUTE's ImageHandler based on calibration data.

    Args:
        calibration_file (str): Path to the CSV file with calibration data.
                                Expected columns: 'file_path', 'phi', 'modulation'.
        base_output_dir (str): The main output directory where subdirectories
                               (matching 'file_path' in CSV) contain TIFF files.
        bin_width_ns (float): Temporal bin width in nanoseconds.
        freq_mhz (float): Laser repetition frequency in MHz.
        harmonic (int): Harmonic number.
    """
    try:
        calibration_data = pd.read_csv(calibration_file)
    except FileNotFoundError:
        print(f"Error: Calibration file not found at {calibration_file}")
        return
    except Exception as e:
        print(f"Error reading calibration file {calibration_file}: {e}")
        return

    print(f"Starting FLUTE processing using calibration from: {calibration_file}")

    for index, row in calibration_data.iterrows():
        try:
            relative_subdir = str(row['file_path'])
            phi_cal = float(row['phi'])
            m_cal = float(row['modulation'])
            subdirectory_path = os.path.join(base_output_dir, relative_subdir)

            if not os.path.isdir(subdirectory_path):
                print(f"Warning: Subdirectory not found for calibration row {index}: {subdirectory_path}. Skipping.")
                continue

            print(f"Processing directory: {subdirectory_path} with Phi={phi_cal}, Mod={m_cal}")
            tif_files_processed = 0
            for filename in os.listdir(subdirectory_path):
                if filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
                    tif_file_path = os.path.join(subdirectory_path, filename)
                    print(f"  Processing file: {filename}")
                    try:
                        # Instantiate ImageHandler
                        handler = ImageHandler(
                            filename=tif_file_path,
                            phi_cal=phi_cal,
                            m_cal=m_cal,
                            bin_width=bin_width_ns,
                            freq=freq_mhz,
                            harmonic=harmonic
                        )
                        # Save the g and s coordinate TIFFs
                        handler.save_data(file=subdirectory_path, save_type=None) # save_type seems unused
                        print(f"    Successfully processed and saved output for {filename}")
                        tif_files_processed += 1
                    except Exception as e:
                        print(f"    Error processing file {filename} with ImageHandler: {e}")
            
            if tif_files_processed == 0:
                 print(f"  Warning: No .tif files found or processed in {subdirectory_path}")
            else:
                 print(f"  Finished processing {tif_files_processed} TIFF file(s) in {subdirectory_path}")

        except KeyError as e:
            print(f"Error: Missing expected column in {calibration_file}: {e}. Skipping row {index}.")
        except ValueError as e:
            print(f"Error: Invalid numerical value (phi/modulation) in {calibration_file}, row {index}: {e}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing row {index} ({relative_subdir}): {e}")

    print("FLUTE processing complete.")

# Function to list all files with a specific extension in a given directory and its subdirectories
def list_files_with_extension(base_dir, extension):
    files = []
    for root, dirs, file_list in os.walk(base_dir):
        for file in file_list:
            if file.endswith(extension):
                files.append(os.path.join(root, file))
    return files

def run_preprocessing():
    """Runs the full TCSPC preprocessing pipeline (ImageJ + FLUTE)."""
    # Load config inside the function in case it's called externally
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    imagej_path = config["imagej_path"]
    flute_path = config["flute_path"]
    macro_files = config["macro_files"]
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    preprocessed_dir = config["preprocessed_dir"]
    
    # Check if ImageHandler was imported correctly (already done at top level)
    # Consider moving the import logic inside here if making a standalone module

    # === Main script execution ===

    # Run the first macro for calibration (assumed to generate initial data if needed)
    print("Running ImageJ Macro 1...")
    run_imagej(macro_files[0], input_dir, output_dir)
    print("ImageJ Macro 1 finished.")

    # Run the second macro to process .bin files into .tif files in subdirectories
    print("Running ImageJ Macro 2...")
    run_imagej(macro_files[1], input_dir, output_dir)
    print("ImageJ Macro 2 finished.")

    # Process the generated TIF files using FLUTE ImageHandler
    print("Starting FLUTE processing...")
    # Define microscope parameters (Consider moving these to config.json if not already done)
    # These should ideally be loaded from config if microscope_params are there
    try:
        with open("config.json", "r") as cfg_file:
            cfg = json.load(cfg_file)
            bin_width_ns = cfg["microscope_params"]["bin_width_ns"]
            freq_mhz = cfg["microscope_params"]["freq_mhz"]
            harmonic = cfg["microscope_params"]["harmonic"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        print("Warning: Could not load microscope params from config.json. Using defaults.")
        bin_width_ns = 0.097 # Default fallback
        freq_mhz = 78.0    # Default fallback
        harmonic = 1       # Default fallback
        
    process_tiffs_with_flute('calibration.csv', output_dir, bin_width_ns, freq_mhz, harmonic)

    # Run the third macro to process FLUTE output (_g.tiff, _s.tiff)
    print("Running ImageJ Macro 3...")
    run_imagej(macro_files[2], output_dir, preprocessed_dir)
    print("ImageJ Macro 3 finished.")

    # Run the fourth macro to generate the intensity images
    print("Running ImageJ Macro 4...")
    run_imagej(macro_files[3], output_dir, preprocessed_dir)
    print("ImageJ Macro 4 finished.")

    # Run the fifth macro to rename files
    print("Running ImageJ Macro 5...")
    run_imagej(macro_files[4], preprocessed_dir)
    print("ImageJ Macro 5 finished.")

    print("Preprocessing pipeline complete.")

# === Main script execution block ===
if __name__ == "__main__":
    run_preprocessing()