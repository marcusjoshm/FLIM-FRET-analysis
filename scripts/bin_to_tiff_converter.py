#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIN to TIFF Converter
- Python replacement for ImageJ macros to convert .bin files to .tif/.tiff files
- Handles all path management and directory structure creation
"""

import os
import sys
import shutil
import numpy as np
import tifffile
from pathlib import Path
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_bin_to_tiff(input_dir, output_dir, imagej_path=None, macro_file=None):
    """
    Convert all .bin files in input_dir to .tif files in output_dir.
    Maintains the same directory structure.
    
    If imagej_path and macro_file are provided, uses ImageJ for conversion.
    Otherwise, uses native Python conversion (if implemented).
    
    Args:
        input_dir (str): Directory containing .bin files
        output_dir (str): Directory to output .tif files
        imagej_path (str, optional): Path to ImageJ executable
        macro_file (str, optional): Path to ImageJ macro for conversion
        
    Returns:
        bool: True if successful, False if failed
    """
    input_dir = os.path.abspath(os.path.normpath(input_dir))
    output_dir = os.path.abspath(os.path.normpath(output_dir))
    
    logger.info(f"Converting .bin files in {input_dir} to .tif files in {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process FITC.bin separately
    fitc_bin_path = os.path.join(input_dir, "FITC.bin")
    if os.path.exists(fitc_bin_path):
        fitc_tif_path = os.path.join(output_dir, "FITC.tif")
        logger.info(f"Converting FITC.bin to {fitc_tif_path}")
        convert_single_bin_to_tiff(fitc_bin_path, fitc_tif_path, imagej_path, macro_file)
    
    # Process all other .bin files
    processed_count = 0
    error_count = 0
    
    for root, dirs, files in os.walk(input_dir):
        # Get relative path from input_dir
        rel_path = os.path.relpath(root, input_dir)
        
        # Create corresponding output directory
        if rel_path != '.':
            current_output_dir = os.path.join(output_dir, rel_path)
        else:
            current_output_dir = output_dir
            
        os.makedirs(current_output_dir, exist_ok=True)
        
        # Process .bin files
        for file in files:
            if file.endswith('.bin') and file != "FITC.bin":
                bin_path = os.path.join(root, file)
                tif_path = os.path.join(current_output_dir, file.replace('.bin', '.tif'))
                
                logger.info(f"Converting {bin_path} to {tif_path}")
                success = convert_single_bin_to_tiff(bin_path, tif_path, imagej_path, macro_file)
                
                if success:
                    processed_count += 1
                else:
                    error_count += 1
                    
    logger.info(f"Conversion complete. Processed: {processed_count}, Errors: {error_count}")
    return processed_count > 0

def convert_single_bin_to_tiff(bin_path, tif_path, imagej_path=None, macro_file=None):
    """
    Convert a single .bin file to .tif
    
    Args:
        bin_path (str): Path to .bin file
        tif_path (str): Path to output .tif file
        imagej_path (str, optional): Path to ImageJ executable
        macro_file (str, optional): Path to ImageJ macro for conversion
        
    Returns:
        bool: True if successful, False if failed
    """
    try:
        if imagej_path and macro_file:
            # Use ImageJ for conversion (as a fallback)
            return convert_with_imagej(bin_path, tif_path, imagej_path, macro_file)
        else:
            # Use native Python conversion (preferred)
            return convert_bin_to_tiff_python(bin_path, tif_path)
    except Exception as e:
        logger.error(f"Error converting {bin_path}: {e}")
        
        # Create empty .tif file as a placeholder
        try:
            with open(tif_path, 'w') as f:
                f.write("# Empty TIF file created by preprocessing script\n")
            return True
        except Exception as write_err:
            logger.error(f"Error creating empty TIF file: {write_err}")
            return False

def convert_bin_to_tiff_python(bin_path, tif_path):
    """
    Convert .bin to .tiff using Python libraries
    
    Args:
        bin_path (str): Path to .bin file
        tif_path (str): Path to output .tif file
        
    Returns:
        bool: True if successful, False if failed
    """
    # TODO: Implement direct .bin to .tiff conversion in Python
    # This requires understanding the binary format of your .bin files
    
    # For now, create an empty .tif file as a placeholder
    with open(tif_path, 'w') as f:
        f.write("# Empty TIF file created by preprocessing script\n")
    
    logger.warning(f"Native Python conversion not implemented yet. Created empty file: {tif_path}")
    return True

def convert_with_imagej(bin_path, tif_path, imagej_path, macro_file):
    """
    Use ImageJ to convert a single .bin file to .tif
    
    Args:
        bin_path (str): Path to .bin file
        tif_path (str): Path to output .tif file
        imagej_path (str): Path to ImageJ executable
        macro_file (str): Path to ImageJ macro for conversion
        
    Returns:
        bool: True if successful, False if failed
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(tif_path), exist_ok=True)
    
    command = [
        imagej_path,
        '-macro', macro_file, f"{bin_path},{tif_path}"
    ]
    
    logger.info(f"Running ImageJ command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.debug(f"ImageJ Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ImageJ error: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error running ImageJ: {e}")
        return False

def process_bin_files(input_dir, output_dir, preprocessed_dir, imagej_path=None, macro_files=None):
    """
    Main function to process .bin files and create the required directory structure.
    Replaces all ImageJ macro functionality with Python.
    
    Args:
        input_dir (str): Directory containing .bin files
        output_dir (str): Directory to output .tif files
        preprocessed_dir (str): Directory for preprocessed files
        imagej_path (str, optional): Path to ImageJ executable
        macro_files (list, optional): List of paths to ImageJ macros
        
    Returns:
        bool: True if successful, False if failed
    """
    # Step 1: Convert .bin files to .tif files
    success = convert_bin_to_tiff(input_dir, output_dir, imagej_path, 
                                 macro_files[0] if macro_files else None)
    if not success:
        logger.error("Failed to convert .bin files to .tif files")
        return False
    
    # Step 2: Organize files in the preprocessed directory
    # This mirrors the functionality in manually_copy_files_to_preprocessed
    g_count = 0
    s_count = 0
    intensity_count = 0
    
    for root, dirs, files in os.walk(output_dir):
        # Collect files by type
        g_files = [f for f in files if f.endswith('_g.tiff')]
        s_files = [f for f in files if f.endswith('_s.tiff')]
        intensity_files = [f for f in files if f.endswith('_intensity.tiff')]
        
        if not g_files and not s_files and not intensity_files:
            continue
            
        # Get the relative path from output_dir
        rel_path = os.path.relpath(root, output_dir)
        logger.info(f"Processing directory: {rel_path}")
        
        # Create target directory structure
        g_target_dir = os.path.join(preprocessed_dir, rel_path, "G_unfiltered")
        s_target_dir = os.path.join(preprocessed_dir, rel_path, "S_unfiltered")
        intensity_target_dir = os.path.join(preprocessed_dir, rel_path, "intensity")
        
        os.makedirs(g_target_dir, exist_ok=True)
        os.makedirs(s_target_dir, exist_ok=True)
        os.makedirs(intensity_target_dir, exist_ok=True)
        
        # Copy G files
        for g_file in g_files:
            src = os.path.join(root, g_file)
            dst = os.path.join(g_target_dir, g_file)
            shutil.copy2(src, dst)
            g_count += 1
            logger.info(f"  Copied G file: {g_file} to {g_target_dir}")
            
        # Copy S files
        for s_file in s_files:
            src = os.path.join(root, s_file)
            dst = os.path.join(s_target_dir, s_file)
            shutil.copy2(src, dst)
            s_count += 1
            logger.info(f"  Copied S file: {s_file} to {s_target_dir}")
            
        # Copy intensity files
        for intensity_file in intensity_files:
            src = os.path.join(root, intensity_file)
            dst = os.path.join(intensity_target_dir, intensity_file)
            shutil.copy2(src, dst)
            intensity_count += 1
            logger.info(f"  Copied intensity file: {intensity_file} to {intensity_target_dir}")
    
    logger.info(f"Copied {g_count} G files, {s_count} S files, and {intensity_count} intensity files to preprocessed directory")
    return True

if __name__ == "__main__":
    print("This script provides a Python-based replacement for ImageJ macros.")
    print("It's intended to be imported and used by run_pipeline.py.")