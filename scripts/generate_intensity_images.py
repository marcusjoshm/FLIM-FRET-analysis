#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Intensity Images for Wavelet Filtering

This script replicates the functionality of FLIM_processing_macro_4.ijm
by generating appropriate intensity images from raw FLIM data TIFFs.
These intensity images are required for the wavelet filtering step.
"""

import os
import glob
import numpy as np
import tifffile
from pathlib import Path

def load_tiff(file_path):
    """
    Load a TIFF file safely with proper error handling.
    
    Args:
        file_path (str): Path to the TIFF file
        
    Returns:
        numpy.ndarray or None: The image data, or None if loading failed
    """
    try:
        return tifffile.imread(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def generate_intensity_image(input_file, output_file):
    """
    Generate an intensity image by summing all time slices in a FLIM data TIFF.
    This replicates the Z-projection functionality in ImageJ.
    
    Args:
        input_file (str): Path to the input FLIM data TIFF
        output_file (str): Path to save the generated intensity image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the input TIFF
        data = load_tiff(input_file)
        if data is None:
            return False
        
        # Handle different dimensions
        if data.ndim == 3:  # 3D: (time, height, width)
            # Sum along time dimension (axis 0)
            intensity = np.sum(data, axis=0).astype(np.float32)
        elif data.ndim == 2:  # 2D: (height, width)
            # Already an intensity image, just convert to float32
            intensity = data.astype(np.float32)
        else:
            print(f"Unexpected dimensions in {input_file}: {data.shape}")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the intensity image
        tifffile.imwrite(output_file, intensity)
        print(f"Generated intensity image: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error generating intensity image for {input_file}: {e}")
        return False

def process_raw_flim_files(input_dir, output_dir):
    """
    Process all raw FLIM data TIFFs in the input directory and generate 
    intensity images for the wavelet filtering step.
    
    Args:
        input_dir (str): Directory containing raw FLIM data TIFFs
        output_dir (str): Directory to save generated intensity images
        
    Returns:
        tuple: (success_count, error_count)
    """
    success_count = 0
    error_count = 0
    raw_tiff_dir = os.path.join(os.path.dirname(input_dir), 'output')
    
    print(f"Looking for raw TIFF files in {raw_tiff_dir}")
    
    # Look for raw TIFF files in the output directory (where ImageJ writes the converted .bin files)
    if not os.path.exists(raw_tiff_dir):
        print(f"Error: Raw TIFF directory not found: {raw_tiff_dir}")
        return 0, 1
    
    # Walk through the raw TIFF directory
    for root, _, files in os.walk(raw_tiff_dir):
        for file in files:
            # Process only the original raw TIFF files from .bin conversion
            # These are the TIFF files that don't have any suffix like _g, _s, etc.
            if file.lower().endswith(('.tif', '.tiff')):
                # Skip files that are outputs from phasor transformation
                if file.lower().endswith(('_g.tiff', '_s.tiff', '_intensity.tiff', '_mask.tiff',
                                          '_taup.tiff', '_taum.tiff', '_wavelet_intensity.tiff')):
                    continue
                
                # This is an original raw FLIM TIFF file
                input_file = os.path.join(root, file)
                print(f"Found raw FLIM file: {input_file}")
                
                # Create relative path to maintain directory structure
                rel_path = os.path.relpath(root, raw_tiff_dir)
                if rel_path == '.':  # File is in the root directory
                    output_subdir = output_dir
                else:
                    output_subdir = os.path.join(output_dir, rel_path)
                    
                # Create output filename (basename + _wavelet_intensity.tiff)
                basename = os.path.splitext(file)[0]
                output_file = os.path.join(output_subdir, f"{basename}_wavelet_intensity.tiff")
                
                # Generate intensity image
                if generate_intensity_image(input_file, output_file):
                    success_count += 1
                else:
                    error_count += 1
                    
    if success_count == 0:
        print(f"No raw FLIM TIFF files found in {raw_tiff_dir}")
        print("Make sure to run the preprocessing step first to convert .bin files to .tif files")
                
    return success_count, error_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate intensity images for wavelet filtering")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw FLIM data TIFFs")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated intensity images")
    args = parser.parse_args()
    
    success_count, error_count = process_raw_flim_files(args.input_dir, args.output_dir)
    print(f"Processed {success_count + error_count} files: {success_count} successful, {error_count} failed")
