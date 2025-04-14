#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename Simplification Script for FLIM-FRET Analysis Pipeline

This script simplifies complex filenames (e.g., R_1_s2_g.tiff) to sequential numbers (e.g., 2.tiff)
while maintaining the existing directory structure. This makes the files more compatible with
external analysis scripts used by collaborators.
"""

import os
import re
import shutil
import glob
import argparse

def extract_sample_number(filename):
    """
    Extract the sample number from a filename like R_1_s2_g.tiff -> 2
    
    Args:
        filename (str): Original filename
        
    Returns:
        int: Extracted sample number, or None if no match found
    """
    # Match pattern like R_1_s2_g.tiff where 2 is the sample number
    # Also handle cases like R_1_s10_g.tiff for numbers > 9
    pattern = r'.*_s(\d+)_.*\.tiff?'
    match = re.search(pattern, filename)
    
    if match:
        return int(match.group(1))
    return None

def simplify_filenames(input_dir, dry_run=False):
    """
    Simplify filenames in the directory structure to sequential numbers.
    
    Args:
        input_dir (str): Root directory containing regions with G_unfiltered, S_unfiltered, intensity folders
        dry_run (bool): If True, print changes without actually renaming files
        
    Returns:
        tuple: (success_count, error_count)
    """
    success_count = 0
    error_count = 0
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Simplifying filenames in: {input_dir}")
    
    # Walk through the input directory
    for root, dirs, files in os.walk(input_dir):
        # Check if we're in a G_unfiltered, S_unfiltered, or intensity directory
        current_dir = os.path.basename(root)
        parent_dir = os.path.basename(os.path.dirname(root))
        
        if current_dir in ['G_unfiltered', 'S_unfiltered', 'intensity']:
            print(f"\nProcessing directory: {root}")
            
            # Process files in this directory
            file_list = [f for f in files if f.endswith('.tiff') or f.endswith('.tif')]
            
            # Skip if no files
            if not file_list:
                continue
                
            # Sort files by sample number
            numbered_files = []
            for filename in file_list:
                sample_num = extract_sample_number(filename)
                if sample_num is not None:
                    numbered_files.append((sample_num, filename))
                    
            # Sort by sample number
            numbered_files.sort(key=lambda x: x[0])
            
            # Rename files
            for i, (sample_num, old_filename) in enumerate(numbered_files):
                new_filename = f"{sample_num}.tiff"
                old_path = os.path.join(root, old_filename)
                new_path = os.path.join(root, new_filename)
                
                # If the simplified file already exists, skip
                if os.path.exists(new_path) and old_filename != new_filename:
                    print(f"  Skipping: {old_filename} -> {new_filename} (destination already exists)")
                    error_count += 1
                    continue
                
                # Rename file
                print(f"  {'Would rename' if dry_run else 'Renaming'}: {old_filename} -> {new_filename}")
                
                if not dry_run:
                    try:
                        # Use copy instead of rename to ensure it works across devices
                        # Then remove the original
                        if old_filename != new_filename:  # Avoid unnecessary copies
                            shutil.copy2(old_path, new_path)
                            os.remove(old_path)
                        success_count += 1
                    except Exception as e:
                        print(f"  Error renaming {old_filename}: {e}")
                        error_count += 1
                else:
                    # In dry run mode, count as success
                    success_count += 1
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Simplified {success_count} files with {error_count} errors")
    return success_count, error_count

def main():
    parser = argparse.ArgumentParser(description="Simplify FLIM-FRET file names for compatibility with external tools")
    parser.add_argument("--input-dir", required=True, help="Input directory containing preprocessed files")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without actually renaming files")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist or is not a directory")
        return 1
        
    success_count, error_count = simplify_filenames(args.input_dir, args.dry_run)
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
