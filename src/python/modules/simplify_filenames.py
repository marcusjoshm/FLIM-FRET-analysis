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

def extract_file_info(filename):
    """
    Extract region and sample numbers from a filename (e.g., R_1_s2_g.tiff)
    
    Args:
        filename (str): Original filename
        
    Returns:
        tuple: (region_number, sample_number) or None if no match found
    """
    # Match pattern like R_1_s2_g.tiff where 1 is region and 2 is sample
    # Also handle cases with double-digit numbers
    pattern = r'R_*(\d+)_s(\d+)_.*\.tiff?'
    match = re.search(pattern, filename)
    
    if match:
        region_num = int(match.group(1))
        sample_num = int(match.group(2))
        return (region_num, sample_num)
    
    # Also try just extracting sample number if no region
    pattern = r'.*_s(\d+)_.*\.tiff?'
    match = re.search(pattern, filename)
    if match:
        sample_num = int(match.group(1))
        return (0, sample_num)  # Return 0 as region if no region found
        
    return None

def simplify_filenames(input_dir, dry_run=False):
    """
    Simplify filenames in the directory structure to sequential numbers.
    The function can handle both hierarchical (region folders) and flat directory structures.
    
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
                
            # Sort files by both region and sample number
            numbered_files = []
            for filename in file_list:
                file_info = extract_file_info(filename)
                if file_info is not None:
                    region_num, sample_num = file_info
                    numbered_files.append((region_num, sample_num, filename))
                    
            # Detect if we have a flat or hierarchical structure by looking at parent directories
            is_flat_structure = True
            parent_parts = os.path.normpath(root).split(os.sep)
            for i, part in enumerate(parent_parts):
                if part.startswith('R') and len(part) <= 3:
                    # Found a region directory (like R1, R2) in the path
                    is_flat_structure = False
                    break
            
            if is_flat_structure:
                # For flat structure, sort by both region and sample numbers 
                # and assign sequential numbers
                numbered_files.sort(key=lambda x: (x[0], x[1]))  # Sort by region, then sample
                print(f"  Using flat directory structure mode with sequential numbering")
                
                # Create sequential numbering
                for i, (region_num, sample_num, old_filename) in enumerate(numbered_files, 1):
                    new_filename = f"{i}.tiff"
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
            else:
                # For hierarchical structure, just use sample numbers
                numbered_files.sort(key=lambda x: x[1])  # Sort by sample number only
                print(f"  Using hierarchical directory structure mode with sample numbers")
                
                # Use sample numbers as filenames
                for i, (region_num, sample_num, old_filename) in enumerate(numbered_files):
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
    parser.add_argument("--flat", action="store_true", help="Force flat directory mode (sequential numbering)")
    parser.add_argument("--hierarchical", action="store_true", help="Force hierarchical directory mode (sample numbering)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist or is not a directory")
        return 1
        
    success_count, error_count = simplify_filenames(args.input_dir, args.dry_run)
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
