#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Average Lifetime from Segmented Data
============================================

This module calculates average lifetime values from segmented NPZ files.
It works with both manually segmented files and external mask files created by apply_mask.py.
It extracts TU (unfiltered lifetime) data, applies the segmentation mask, and
calculates the average lifetime for each segmented region.

Part of FLIM-FRET Analysis Pipeline
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import argparse

def load_segmented_npz(npz_file_path):
    """
    Load segmented NPZ file and extract TU and full_mask data.
    
    Args:
        npz_file_path (str): Path to segmented NPZ file
        
    Returns:
        dict: Dictionary containing TU data, mask, and metadata, or None if failed
    """
    try:
        data = np.load(npz_file_path, allow_pickle=True)
        
        # Check for required data
        if 'TU' not in data:
            print(f"Warning: No TU data found in {os.path.basename(npz_file_path)}")
            print(f"Available keys: {list(data.keys())}")
            return None
            
        if 'full_mask' not in data:
            print(f"Warning: No full_mask data found in {os.path.basename(npz_file_path)}")
            print(f"Available keys: {list(data.keys())}")
            return None
        
        return {
            'tu_data': data['TU'],
            'mask': data['full_mask'],
            'file_path': npz_file_path,
            'metadata': data.get('metadata', {}),
            'all_data': dict(data)
        }
        
    except Exception as e:
        print(f"Error loading segmented NPZ file {npz_file_path}: {e}")
        return None

def calculate_masked_lifetime_average(tu_data, mask):
    """
    Calculate average lifetime from masked TU data.
    
    Args:
        tu_data (numpy.ndarray): TU (unfiltered lifetime) data
        mask (numpy.ndarray): Binary mask (0 = background, 1 = selected region)
        
    Returns:
        dict: Dictionary containing average lifetime and statistics
    """
    try:
        # Apply mask to get only selected region lifetime values
        masked_lifetime = tu_data * mask
        
        # Get valid lifetime values (non-zero from mask and finite values)
        valid_lifetime = masked_lifetime[masked_lifetime > 0]
        valid_lifetime = valid_lifetime[np.isfinite(valid_lifetime)]
        
        if len(valid_lifetime) == 0:
            return {
                'average_lifetime': np.nan,
                'std_lifetime': np.nan,
                'min_lifetime': np.nan,
                'max_lifetime': np.nan,
                'pixel_count': 0,
                'valid_pixel_count': 0,
                'total_pixels': tu_data.size
            }
        
        # Calculate statistics
        avg_lifetime = np.mean(valid_lifetime)
        std_lifetime = np.std(valid_lifetime)
        min_lifetime = np.min(valid_lifetime)
        max_lifetime = np.max(valid_lifetime)
        
        # Count pixels
        total_pixels = tu_data.size
        masked_pixels = np.sum(mask > 0)
        valid_pixels = len(valid_lifetime)
        
        return {
            'average_lifetime': avg_lifetime,
            'std_lifetime': std_lifetime,
            'min_lifetime': min_lifetime,
            'max_lifetime': max_lifetime,
            'pixel_count': masked_pixels,
            'valid_pixel_count': valid_pixels,
            'total_pixels': total_pixels
        }
        
    except Exception as e:
        print(f"Error calculating masked lifetime average: {e}")
        return None

def process_segmented_npz_file(npz_file_path, output_dir):
    """
    Process a single segmented NPZ file and calculate average lifetime.
    
    Args:
        npz_file_path (str): Path to segmented NPZ file
        output_dir (str): Directory to save results
        
    Returns:
        dict: Results dictionary with filename and statistics, or None if failed
    """
    # Load segmented NPZ data
    data = load_segmented_npz(npz_file_path)
    if data is None:
        return None
    
    # Calculate average lifetime
    stats = calculate_masked_lifetime_average(data['tu_data'], data['mask'])
    if stats is None:
        return None
    
    # Create results dictionary
    base_name = os.path.splitext(os.path.basename(npz_file_path))[0]
    
    results = {
        'filename': base_name,
        'source_file': npz_file_path,
        **stats
    }
    
    # Add metadata if available
    if data['metadata']:
        results['metadata'] = data['metadata']
    
    return results

def process_segmented_npz_directory(npz_dir, output_dir):
    """
    Process all segmented NPZ files in a directory and calculate average lifetimes.
    Supports both manually segmented files (*_manually_segmented.npz) and
    external mask files (*_masked.npz) created by apply_mask.py.
    
    Args:
        npz_dir (str): Directory containing segmented NPZ files
        output_dir (str): Directory to save CSV results
        
    Returns:
        tuple: (success_count, error_count, results_list)
    """
    if not os.path.isdir(npz_dir):
        print(f"Error: Segmented NPZ directory '{npz_dir}' does not exist")
        return 0, 0, []
    
    # Find all segmented NPZ files (both manually segmented and masked files)
    npz_files = glob.glob(os.path.join(npz_dir, "*_manually_segmented.npz"))
    npz_files.extend(glob.glob(os.path.join(npz_dir, "**/*_manually_segmented.npz"), recursive=True))
    
    # Also look for masked NPZ files created by apply_mask.py
    npz_files.extend(glob.glob(os.path.join(npz_dir, "*_masked.npz")))
    npz_files.extend(glob.glob(os.path.join(npz_dir, "**/*_masked.npz"), recursive=True))
    
    # Remove duplicates
    npz_files = list(set(npz_files))
    
    if not npz_files:
        print(f"No segmented NPZ files found in {npz_dir}")
        print(f"Looking for patterns: *_manually_segmented.npz, *_masked.npz")
        return 0, 0, []
    
    print(f"Found {len(npz_files)} segmented NPZ files to process")
    
    success_count = 0
    error_count = 0
    results_list = []
    
    for i, npz_file in enumerate(npz_files):
        print(f"\nProcessing file {i+1}/{len(npz_files)}: {os.path.basename(npz_file)}")
        
        # Process the file
        result = process_segmented_npz_file(npz_file, output_dir)
        
        if result is not None:
            results_list.append(result)
            success_count += 1
            print(f"  Average lifetime: {result['average_lifetime']:.3f} ns")
            print(f"  Valid pixels: {result['valid_pixel_count']}/{result['pixel_count']}")
        else:
            error_count += 1
    
    print(f"\nAverage lifetime calculation complete:")
    print(f"  Successfully processed: {success_count} files")
    print(f"  Errors: {error_count} files")
    
    return success_count, error_count, results_list

def save_results_to_csv(results_list, output_dir):
    """
    Save results to CSV file.
    
    Args:
        results_list (list): List of result dictionaries
        output_dir (str): Directory to save CSV file
        
    Returns:
        str: Path to saved CSV file, or None if failed
    """
    try:
        if not results_list:
            print("No results to save")
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for CSV
        csv_data = []
        for result in results_list:
            # Extract basic statistics
            row = {
                'filename': result['filename'],
                'average_lifetime_ns': result['average_lifetime'],
                'std_lifetime_ns': result['std_lifetime'],
                'min_lifetime_ns': result['min_lifetime'],
                'max_lifetime_ns': result['max_lifetime'],
                'pixel_count': result['pixel_count'],
                'valid_pixel_count': result['valid_pixel_count'],
                'total_pixels': result['total_pixels'],
                'source_file': result['source_file']
            }
            
            # Add metadata if available
            if 'metadata' in result and result['metadata']:
                metadata = result['metadata']
                # Handle metadata as dictionary or numpy array
                if isinstance(metadata, dict):
                    row['pixels_selected'] = metadata.get('pixels_selected', np.nan)
                    row['pixels_thresholded'] = metadata.get('pixels_thresholded', np.nan)
                    row['mask_type'] = metadata.get('mask_type', 'unknown')
                else:
                    # If metadata is not a dictionary, skip it
                    row['pixels_selected'] = np.nan
                    row['pixels_thresholded'] = np.nan
                    row['mask_type'] = 'unknown'
            
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        
        # Sort by filename
        df = df.sort_values('filename')
        
        # Save to CSV
        csv_path = os.path.join(output_dir, 'average_lifetime_results.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"Saved results to: {csv_path}")
        print(f"Processed {len(df)} files")
        
        # Print summary statistics
        valid_results = df[df['average_lifetime_ns'].notna()]
        if len(valid_results) > 0:
            print(f"\nSummary Statistics:")
            print(f"  Overall average lifetime: {valid_results['average_lifetime_ns'].mean():.3f} ± {valid_results['average_lifetime_ns'].std():.3f} ns")
            print(f"  Range: {valid_results['average_lifetime_ns'].min():.3f} - {valid_results['average_lifetime_ns'].max():.3f} ns")
            print(f"  Total valid pixels processed: {valid_results['valid_pixel_count'].sum()}")
        
        return csv_path
        
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        return None

def main(config=None, segmented_npz_dir=None, output_dir=None):
    """
    Main execution function for calculating average lifetime from segmented data.
    
    Args:
        config: Configuration dictionary (optional)
        segmented_npz_dir: Directory containing segmented NPZ files
        output_dir: Directory to save CSV results
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("=== Average Lifetime Calculation from Segmented Data ===")
    
    # Check if running as standalone script
    if config is None:
        parser = argparse.ArgumentParser(description="Calculate average lifetime from segmented NPZ files")
        parser.add_argument("segmented_npz_dir", help="Directory containing segmented NPZ files")
        parser.add_argument("output_dir", help="Directory to save CSV results")
        
        args = parser.parse_args()
        
        segmented_npz_dir = args.segmented_npz_dir
        output_dir = args.output_dir
    
    if not segmented_npz_dir or not output_dir:
        print("Error: Both segmented_npz_dir and output_dir must be specified")
        return False
    
    print(f"Input segmented NPZ directory: {segmented_npz_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process segmented NPZ files
    success_count, error_count, results_list = process_segmented_npz_directory(segmented_npz_dir, output_dir)
    
    if success_count > 0:
        # Save results to CSV
        csv_path = save_results_to_csv(results_list, output_dir)
        
        if csv_path:
            print(f"\n✅ Successfully calculated average lifetime for {success_count} files")
            if error_count > 0:
                print(f"⚠️  {error_count} files had errors")
            return True
        else:
            print(f"\n❌ Failed to save results to CSV")
            return False
    else:
        print(f"\n❌ No average lifetime calculations were performed successfully")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 