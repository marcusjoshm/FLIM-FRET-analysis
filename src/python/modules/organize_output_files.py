#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLIM-FRET Output File Organizer

This script replaces ImageJ macros 3-5 by organizing the output files 
from the phasor transformation into the preprocessed directory structure.
It handles G, S, and intensity files, maintaining the same directory structure.
"""

import os
import shutil
import glob
import argparse
from pathlib import Path
import time

def copy_file_with_subdirectory(src_file, output_base_dir, subdir_name):
    """
    Copy a file to a target directory with a specific subdirectory structure.
    
    Args:
        src_file (str): Source file path
        output_base_dir (str): Base output directory
        subdir_name (str): Name of subdirectory to place file in
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get source file info
        src_path = Path(src_file)
        if not src_path.exists():
            print(f"Source file does not exist: {src_file}")
            return False
            
        # Extract relative directory structure
        src_dir = src_path.parent
        filename = src_path.name
        
        # Create target directory path
        rel_path = os.path.relpath(src_dir, start=args.input_dir)
        if rel_path == '.':
            # File is in the root of input_dir
            target_dir = Path(output_base_dir) / subdir_name
        else:
            # File is in a subdirectory
            target_dir = Path(output_base_dir) / rel_path / subdir_name
            
        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Create target file path
        target_file = target_dir / filename
        
        # Check if target file already exists
        if target_file.exists():
            print(f"Target file already exists, skipping: {target_file}")
            return True
            
        # Copy the file
        shutil.copy2(src_file, target_file)
        print(f"Copied {src_file} -> {target_file}")
        return True
        
    except Exception as e:
        print(f"Error copying {src_file}: {e}")
        return False

def organize_output_files(input_dir, output_dir):
    """
    Organize output files (_g.tiff, _s.tiff, _intensity.tiff) into subdirectories
    in the preprocessed directory.
    
    Args:
        input_dir (str): Input directory containing output files
        output_dir (str): Output directory for organized files
        
    Returns:
        tuple: (success_count, error_count)
    """
    success_count = 0
    error_count = 0
    
    # Find all files recursively
    input_path = Path(input_dir)
    
    # Process G files
    print(f"\nProcessing G files from {input_dir}...")
    g_files = list(input_path.rglob('*_g.tiff'))
    print(f"Found {len(g_files)} G files")
    
    for g_file in g_files:
        if copy_file_with_subdirectory(g_file, output_dir, "G_unfiltered"):
            success_count += 1
        else:
            error_count += 1
    
    # Process S files
    print(f"\nProcessing S files from {input_dir}...")
    s_files = list(input_path.rglob('*_s.tiff'))
    print(f"Found {len(s_files)} S files")
    
    for s_file in s_files:
        if copy_file_with_subdirectory(s_file, output_dir, "S_unfiltered"):
            success_count += 1
        else:
            error_count += 1
    
    # Process intensity files
    print(f"\nProcessing intensity files from {input_dir}...")
    intensity_files = list(input_path.rglob('*_intensity.tiff'))
    print(f"Found {len(intensity_files)} intensity files")
    
    for intensity_file in intensity_files:
        if copy_file_with_subdirectory(intensity_file, output_dir, "Intensity"):
            success_count += 1
        else:
            error_count += 1
    
    return success_count, error_count

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Organize FLIM-FRET output files into preprocessed directory structure")
    parser.add_argument("--input-dir", required=True, help="Input directory containing output files")
    parser.add_argument("--output-dir", required=True, help="Output directory for organized files")
    args = parser.parse_args()
    
    # Normalize paths
    input_dir = os.path.normpath(args.input_dir)
    output_dir = os.path.normpath(args.output_dir)
    
    print(f"Organizing files from {input_dir} to {output_dir}")
    
    # Check if directories exist
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        exit(1)
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process files
    start_time = time.time()
    success_count, error_count = organize_output_files(input_dir, output_dir)
    end_time = time.time()
    
    # Print summary
    print(f"\nOrganization complete in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {success_count} files")
    print(f"Errors: {error_count}")
