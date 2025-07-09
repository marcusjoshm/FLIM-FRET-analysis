#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply Binary Masks to NPZ Data
===============================

This module applies binary masks from the segmented directory to NPZ data files
and creates new NPZ files with the mask data saved under 'full_mask'.
The new NPZ files are saved in external_mask_npz_datasets directory.

Part of FLIM-FRET Analysis Pipeline
"""

import os
import sys
import glob
import numpy as np
from PIL import Image
import argparse
import re

def find_mask_files(segmented_dir):
    """
    Find all mask TIFF files in the segmented directory.
    
    Args:
        segmented_dir (str): Directory containing segmented mask files
        
    Returns:
        list: List of dictionaries with mask file info
    """
    if not os.path.isdir(segmented_dir):
        print(f"Error: Segmented directory '{segmented_dir}' does not exist")
        return []
    
    # Find all TIFF mask files
    mask_files = []
    
    # Look for mask files with common patterns
    patterns = [
        "*_mask.tiff",
        "*_mask.tif", 
        "*_manually_segmented_mask.tiff",
        "*_manually_segmented_mask.tif",
        "*_segmentation_*_mask.tiff",
        "*_segmentation_*_mask.tif"
    ]
    
    for pattern in patterns:
        files = glob.glob(os.path.join(segmented_dir, pattern))
        files.extend(glob.glob(os.path.join(segmented_dir, "**", pattern), recursive=True))
        
        for mask_file in files:
            # Extract base name and mask type
            base_name = os.path.basename(mask_file)
            
            # Try to extract the original NPZ filename
            original_name = None
            mask_type = "unknown"
            
            if "_manually_segmented_mask" in base_name:
                original_name = base_name.replace("_manually_segmented_mask.tiff", "").replace("_manually_segmented_mask.tif", "")
                mask_type = "manual_segmentation"
            elif "_segmentation_" in base_name and "_mask" in base_name:
                # GMM segmentation masks like "filename_ellipse_segmentation_component_1_mask.tiff"
                match = re.match(r"(.+)_([^_]+_segmentation_[^_]+)_mask\.(tiff?)", base_name)
                if match:
                    original_name = match.group(1)
                    mask_type = match.group(2)
            elif "_mask" in base_name:
                # Generic mask files
                original_name = base_name.replace("_mask.tiff", "").replace("_mask.tif", "")
                mask_type = "generic_mask"
            
            if original_name:
                mask_files.append({
                    'mask_path': mask_file,
                    'original_name': original_name,
                    'mask_type': mask_type,
                    'base_name': base_name
                })
    
    # Remove duplicates based on mask_path
    seen_paths = set()
    unique_mask_files = []
    for mask_file in mask_files:
        if mask_file['mask_path'] not in seen_paths:
            seen_paths.add(mask_file['mask_path'])
            unique_mask_files.append(mask_file)
    
    return unique_mask_files

def find_matching_npz_file(original_name, npz_dir):
    """
    Find the NPZ file that matches the original name from the mask.
    
    Args:
        original_name (str): Original filename without extensions
        npz_dir (str): Directory containing NPZ files
        
    Returns:
        str: Path to matching NPZ file, or None if not found
    """
    if not os.path.isdir(npz_dir):
        return None
    
    # Try exact match first
    exact_path = os.path.join(npz_dir, f"{original_name}.npz")
    if os.path.exists(exact_path):
        return exact_path
    
    # Search recursively for matching NPZ files
    for root, dirs, files in os.walk(npz_dir):
        for file in files:
            if file.endswith('.npz'):
                file_base = os.path.splitext(file)[0]
                if file_base == original_name:
                    return os.path.join(root, file)
                # Also try partial matching (in case of slight name differences)
                if original_name in file_base or file_base in original_name:
                    return os.path.join(root, file)
    
    return None

def load_mask_file(mask_path):
    """
    Load a mask TIFF file and convert to binary mask.
    
    Args:
        mask_path (str): Path to mask TIFF file
        
    Returns:
        numpy.ndarray: Binary mask array, or None if failed
    """
    try:
        # Load the mask image
        mask_img = Image.open(mask_path)
        mask_array = np.array(mask_img)
        
        # Convert to binary (0 and 1)
        if mask_array.dtype == bool:
            mask_binary = mask_array.astype(np.int32)
        elif mask_array.max() <= 1:
            # Already in 0-1 range
            mask_binary = mask_array.astype(np.int32)
        else:
            # Scale to binary (assume non-zero values are selected region)
            mask_binary = (mask_array > 0).astype(np.int32)
        
        return mask_binary
        
    except Exception as e:
        print(f"Error loading mask file {mask_path}: {e}")
        return None

def load_npz_file(npz_path):
    """
    Load NPZ file and return data dictionary.
    
    Args:
        npz_path (str): Path to NPZ file
        
    Returns:
        dict: NPZ data dictionary, or None if failed
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        return dict(data)
    except Exception as e:
        print(f"Error loading NPZ file {npz_path}: {e}")
        return None

def apply_mask_to_npz(npz_data, mask_array, mask_info):
    """
    Apply mask to NPZ data and create new data dictionary with full_mask.
    
    Args:
        npz_data (dict): Original NPZ data
        mask_array (numpy.ndarray): Binary mask array
        mask_info (dict): Information about the mask
        
    Returns:
        dict: New NPZ data with mask applied, or None if failed
    """
    try:
        # Copy all original data
        new_data = npz_data.copy()
        
        # Add the mask as 'full_mask'
        new_data['full_mask'] = mask_array
        
        # Add mask metadata
        mask_metadata = {
            'mask_type': mask_info['mask_type'],
            'mask_source': mask_info['mask_path'],
            'original_npz': mask_info.get('npz_path', 'unknown'),
            'pixels_selected': int(np.sum(mask_array)),
            'total_pixels': int(mask_array.size),
            'selection_percentage': float(np.sum(mask_array) / mask_array.size * 100)
        }
        
        # Add or update metadata
        if 'metadata' in new_data:
            if isinstance(new_data['metadata'], dict):
                new_data['metadata'].update(mask_metadata)
            else:
                new_data['mask_metadata'] = mask_metadata
        else:
            new_data['mask_metadata'] = mask_metadata
        
        return new_data
        
    except Exception as e:
        print(f"Error applying mask to NPZ data: {e}")
        return None

def process_mask_npz_pair(mask_info, npz_dir, output_dir):
    """
    Process a single mask-NPZ pair.
    
    Args:
        mask_info (dict): Information about the mask file
        npz_dir (str): Directory containing NPZ files
        output_dir (str): Output directory for masked NPZ files
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\nProcessing: {mask_info['base_name']}")
    
    # Find matching NPZ file
    npz_path = find_matching_npz_file(mask_info['original_name'], npz_dir)
    if npz_path is None:
        print(f"  Error: Could not find matching NPZ file for '{mask_info['original_name']}'")
        return False
    
    print(f"  Found NPZ: {os.path.basename(npz_path)}")
    mask_info['npz_path'] = npz_path
    
    # Load mask file
    mask_array = load_mask_file(mask_info['mask_path'])
    if mask_array is None:
        print(f"  Error: Could not load mask file")
        return False
    
    print(f"  Loaded mask: {mask_array.shape}, {np.sum(mask_array)} selected pixels")
    
    # Load NPZ file
    npz_data = load_npz_file(npz_path)
    if npz_data is None:
        print(f"  Error: Could not load NPZ file")
        return False
    
    print(f"  Loaded NPZ: {len(npz_data)} arrays")
    
    # Check dimensions compatibility
    expected_shape = None
    for key in ['G', 'S', 'A', 'GU', 'SU']:
        if key in npz_data:
            expected_shape = npz_data[key].shape
            break
    
    if expected_shape is None:
        print(f"  Error: No recognizable data arrays found in NPZ file")
        return False
    
    if mask_array.shape != expected_shape:
        print(f"  Error: Mask shape {mask_array.shape} doesn't match NPZ data shape {expected_shape}")
        return False
    
    # Apply mask to NPZ data
    masked_data = apply_mask_to_npz(npz_data, mask_array, mask_info)
    if masked_data is None:
        print(f"  Error: Could not apply mask to NPZ data")
        return False
    
    # Create output filename
    output_filename = f"{mask_info['original_name']}_{mask_info['mask_type']}_masked.npz"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save masked NPZ file
    try:
        np.savez_compressed(output_path, **masked_data)
        print(f"  Saved: {output_filename}")
        return True
    except Exception as e:
        print(f"  Error saving NPZ file: {e}")
        return False

def process_all_masks(segmented_dir, npz_dir, output_dir):
    """
    Process all mask files and create masked NPZ files.
    
    Args:
        segmented_dir (str): Directory containing segmented mask files
        npz_dir (str): Directory containing NPZ files
        output_dir (str): Output directory for masked NPZ files
        
    Returns:
        tuple: (success_count, error_count)
    """
    print(f"=== Apply Binary Masks to NPZ Data ===")
    print(f"Segmented masks directory: {segmented_dir}")
    print(f"NPZ data directory: {npz_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all mask files
    mask_files = find_mask_files(segmented_dir)
    if not mask_files:
        print("No mask files found in the segmented directory")
        return 0, 0
    
    print(f"\nFound {len(mask_files)} mask files:")
    for mask_info in mask_files:
        print(f"  {mask_info['base_name']} -> {mask_info['original_name']} ({mask_info['mask_type']})")
    
    # Process each mask-NPZ pair
    success_count = 0
    error_count = 0
    
    for mask_info in mask_files:
        try:
            if process_mask_npz_pair(mask_info, npz_dir, output_dir):
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"  Unexpected error processing {mask_info['base_name']}: {e}")
            error_count += 1
    
    print(f"\n=== Apply Masks Complete ===")
    print(f"Successfully processed: {success_count} files")
    print(f"Errors: {error_count} files")
    
    return success_count, error_count

def main(config=None, segmented_dir=None, npz_dir=None, output_dir=None):
    """
    Main execution function for applying masks to NPZ data.
    
    Args:
        config: Configuration dictionary (optional)
        segmented_dir: Directory containing segmented mask files
        npz_dir: Directory containing NPZ files  
        output_dir: Directory to save masked NPZ files
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if running as standalone script
    if config is None:
        parser = argparse.ArgumentParser(description="Apply binary masks to NPZ files")
        parser.add_argument("segmented_dir", help="Directory containing segmented mask files")
        parser.add_argument("npz_dir", help="Directory containing NPZ files")
        parser.add_argument("output_dir", help="Directory to save masked NPZ files")
        
        args = parser.parse_args()
        
        segmented_dir = args.segmented_dir
        npz_dir = args.npz_dir
        output_dir = args.output_dir
    
    if not segmented_dir or not npz_dir or not output_dir:
        print("Error: All directories (segmented_dir, npz_dir, output_dir) must be specified")
        return False
    
    # Process all masks
    success_count, error_count = process_all_masks(segmented_dir, npz_dir, output_dir)
    
    if success_count > 0:
        print(f"\n✅ Successfully applied masks to {success_count} NPZ files")
        if error_count > 0:
            print(f"⚠️  {error_count} files had errors")
        return True
    else:
        print(f"\n❌ No masks were successfully applied")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 