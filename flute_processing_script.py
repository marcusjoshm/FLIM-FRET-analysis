# Generated FLUTE processing script
# This script recreates the operations performed in the FLUTE GUI

import os
import sys
import numpy as np
from skimage import io
import tifffile
import csv
import pandas as pd
from pathlib import Path

# Add the FLUTE directory to Python path
FLUTE_PATH = "/Users/joshuamarcus/FLUTE"
if FLUTE_PATH not in sys.path:
    sys.path.append(FLUTE_PATH)

# Import FLUTE modules
import ImageHandler
import Calibration

def read_calibration_csv(csv_file):
    """
    Read calibration values from a CSV file.
    
    Expected CSV format:
    file_path,phi_cal,m_cal
    path/to/file1.bin,0.123,0.987
    path/to/file2.bin,0.456,0.876
    
    Args:
        csv_file (str): Path to the CSV file containing calibration values
        
    Returns:
        dict: Dictionary mapping base filenames to (phi_cal, m_cal) tuples
    """
    try:
        # Read CSV using pandas for better handling of different formats
        df = pd.read_csv(csv_file)
        
        # Validate required columns
        required_cols = ['file_path', 'phi_cal', 'm_cal']
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: CSV file must contain a '{col}' column")
                return {}
        
        # Build dictionary of calibration values by base filename (without extension)
        calibration_values = {}
        for _, row in df.iterrows():
            # Extract just the base filename part (without path and extension)
            full_path = row['file_path']
            filename = os.path.basename(full_path)
            base_name = os.path.splitext(filename)[0]
            
            # Store phi and modulation values keyed by base name
            calibration_values[base_name] = (float(row['phi_cal']), float(row['m_cal']))
        
        print(f"Loaded calibration values for {len(calibration_values)} base filenames from {csv_file}")
        return calibration_values
        
    except Exception as e:
        print(f"Error reading calibration CSV file: {e}")
        return {}

def process_tiff_file(input_file, output_dir, phi_cal=0.0, m_cal=1.0, bin_width=0.2208, freq=80, harmonic=1):
    """
    Process a TIFF file with the parameters specified through the FLUTE GUI.
    
    Args:
        input_file (str): Path to the input TIFF file
        output_dir (str): Directory to save output files
        phi_cal (float): Phase calibration value
        m_cal (float): Modulation calibration value
        bin_width (float): Width of each time bin in nanoseconds
        freq (float): Laser frequency in MHz
        harmonic (int): Harmonic to use for phasor calculation
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Processing {input_file}...")
    print(f"  Using calibration: phi_cal={phi_cal}, m_cal={m_cal}")
    
    # Create FLUTE_output directory inside output_dir
    flute_output_dir = os.path.join(output_dir, "FLUTE_output")
    os.makedirs(flute_output_dir, exist_ok=True)

    try:
        # Initialize ImageHandler
        image_handler = ImageHandler.ImageHandler(input_file, phi_cal, m_cal, bin_width, freq, harmonic)

        # Apply median filter (you can adjust the filter size if needed)
        filter_size = 0  # 0 means no filter, 1 means 3x3 filter, etc.
        image_handler.convolution(filter_size)
        
        # Set intensity threshold (adjust these values as needed)
        min_threshold = 0.0
        max_threshold = 1000000.0
        image_handler.update_threshold(min_threshold, max_threshold)
        
        # Set TauP angle range (adjust these values as needed)
        min_angle = 0.0
        max_angle = 90.0
        image_handler.update_angle_range(min_angle, max_angle)
        
        # Set TauM range (adjust these values as needed)
        min_modulation = 0.0
        max_modulation = 120.0
        image_handler.update_circle_range(min_modulation, max_modulation)
        
        # Save data for image
        image_handler.save_data(flute_output_dir, "all")
        
        print(f"Processing of {input_file} completed successfully.")
        print(f"Output saved to {flute_output_dir}")
        return True
    
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False

def process_folder(input_folder, output_folder, calibration_values=None, default_phi_cal=0.0, default_m_cal=1.0, 
                  bin_width=0.2208, freq=80, harmonic=1):
    """
    Process all TIFF files in the input folder and save results to the output folder.
    
    Args:
        input_folder (str): Path to folder containing TIFF files
        output_folder (str): Path to save output files (FLUTE_output will be created inside)
        calibration_values (dict): Dictionary mapping base filenames to (phi_cal, m_cal) tuples
        default_phi_cal (float): Default phase calibration value if not found in calibration_values
        default_m_cal (float): Default modulation calibration value if not found in calibration_values
        bin_width (float): Width of each time bin in nanoseconds
        freq (float): Laser frequency in MHz
        harmonic (int): Harmonic to use for phasor calculation
    """
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Processing all TIFF files in {input_folder}")
    print(f"Saving results to {output_folder}/FLUTE_output")
    
    if calibration_values:
        print(f"Using calibration values from CSV file for matched files")
        print(f"Using default calibration (phi={default_phi_cal}, mod={default_m_cal}) for unmatched files")
    else:
        print(f"Using default calibration: phi_cal={default_phi_cal}, m_cal={default_m_cal}")
    
    # Get all TIFF files in the input folder
    tiff_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                tiff_files.append(os.path.join(root, file))
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Process each TIFF file
    success_count = 0
    for i, tiff_file in enumerate(tiff_files):
        # Get filename only (without path)
        filename = os.path.basename(tiff_file)
        base_name = os.path.splitext(filename)[0]
        
        # Get calibration values for this file (trying by base name)
        if calibration_values and base_name in calibration_values:
            phi_cal, m_cal = calibration_values[base_name]
            print(f"Using calibration from CSV for {filename}: phi={phi_cal}, mod={m_cal}")
        else:
            phi_cal, m_cal = default_phi_cal, default_m_cal
            if calibration_values:
                print(f"No calibration found for {filename}, using default values")
                
        print(f"Processing file {i+1}/{len(tiff_files)}: {filename}")
        success = process_tiff_file(
            tiff_file, 
            output_folder, 
            phi_cal, 
            m_cal, 
            bin_width, 
            freq, 
            harmonic
        )
        if success:
            success_count += 1
    
    print(f"\nProcessing complete. Successfully processed {success_count}/{len(tiff_files)} files.")
    print(f"Results saved to {os.path.join(output_folder, 'FLUTE_output')}")

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 3:
        print("Usage:")
        print("  For a single file: python script.py --file <input_tiff_file> <output_directory>")
        print("  For a folder:      python script.py --folder <input_folder> <output_directory>")
        print("\nOptional parameters:")
        print("  --calibration <csv_file> CSV file with calibration values (file_path,phi_cal,m_cal)")
        print("  --phi <phi_cal>          Phase calibration value (default: 0.0)")
        print("  --mod <m_cal>            Modulation calibration value (default: 1.0)")
        print("  --bin <bin_width>        Bin width in nanoseconds (default: 0.2208)")
        print("  --freq <freq>            Laser frequency in MHz (default: 80)")
        print("  --harmonic <harmonic>    Harmonic to use (default: 1)")
        sys.exit(1)

    # Parse command line arguments
    # Default values
    phi_cal = 0.0
    m_cal = 1.0
    bin_width = 0.2208
    freq = 80
    harmonic = 1
    calibration_file = None
    calibration_values = None
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--calibration" and i+1 < len(sys.argv):
            calibration_file = sys.argv[i+1]
            calibration_values = read_calibration_csv(calibration_file)
            i += 2
        elif sys.argv[i] == "--phi" and i+1 < len(sys.argv):
            phi_cal = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--mod" and i+1 < len(sys.argv):
            m_cal = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--bin" and i+1 < len(sys.argv):
            bin_width = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--freq" and i+1 < len(sys.argv):
            freq = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--harmonic" and i+1 < len(sys.argv):
            harmonic = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--file" and i+1 < len(sys.argv) and i+2 < len(sys.argv):
            input_file = sys.argv[i+1]
            output_dir = sys.argv[i+2]
            
            # Check if we have calibration values for this file
            if calibration_values:
                filename = os.path.basename(input_file)
                base_name = os.path.splitext(filename)[0]
                if base_name in calibration_values:
                    file_phi, file_m = calibration_values[base_name]
                    print(f"Using calibration from CSV for {filename}")
                    process_tiff_file(input_file, output_dir, file_phi, file_m, bin_width, freq, harmonic)
                else:
                    print(f"No calibration found for {filename}, using default values")
                    process_tiff_file(input_file, output_dir, phi_cal, m_cal, bin_width, freq, harmonic)
            else:
                process_tiff_file(input_file, output_dir, phi_cal, m_cal, bin_width, freq, harmonic)
            i += 3
        elif sys.argv[i] == "--folder" and i+1 < len(sys.argv) and i+2 < len(sys.argv):
            input_folder = sys.argv[i+1]
            output_dir = sys.argv[i+2]
            process_folder(input_folder, output_dir, calibration_values, phi_cal, m_cal, bin_width, freq, harmonic)
            i += 3
        else:
            i += 1
