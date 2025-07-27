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
from pathlib import Path

def list_npz_files(npz_dir):
    """
    List all NPZ files in the directory.
    
    Args:
        npz_dir (str): Directory containing NPZ files
        
    Returns:
        list: List of NPZ file paths
    """
    # Use recursive glob to find all NPZ files, then remove duplicates
    npz_files = glob.glob(os.path.join(npz_dir, "**/*.npz"), recursive=True)
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for file_path in npz_files:
        if file_path not in seen:
            seen.add(file_path)
            unique_files.append(file_path)
    return sorted(unique_files)

def prompt_file_selection(npz_files):
    """
    Interactive file selection for NPZ files.
    
    Args:
        npz_files (list): List of NPZ file paths
        
    Returns:
        tuple: (selected_files, file_selection_name) where selected_files is list of paths and file_selection_name is the naming variable
    """
    if not npz_files:
        print("No NPZ files found.")
        return [], ""
    
    print(f"\nFound {len(npz_files)} NPZ files:")
    for i, file_path in enumerate(npz_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")
    
    print("\nSelect files to process:")
    print("  [all] Process all files")
    print("  [q] Quit")
    print("  Or enter file numbers separated by spaces (e.g., 1 3 5)")
    
    while True:
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'q':
            return [], ""
        elif choice == 'all':
            return npz_files, "full_dataset"
        else:
            try:
                # Parse space-separated numbers
                indices = [int(x.strip()) - 1 for x in choice.split()]
                
                # Validate indices
                if any(i < 0 or i >= len(npz_files) for i in indices):
                    print("Invalid file number. Please try again.")
                    continue
                
                selected_files = [npz_files[i] for i in indices]
                print(f"\nSelected {len(selected_files)} files:")
                for file_path in selected_files:
                    print(f"  - {os.path.basename(file_path)}")
                
                return selected_files, "partial_dataset"
                
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces, 'all', or 'q'.")

def prompt_mask_source():
    """
    Prompt user to select mask source.
    
    Returns:
        tuple: (mask_source, mask_source_name) where mask_source is the internal value and mask_source_name is the naming variable
    """
    print("\n=== Mask Source Selection ===")
    print("Choose mask source for average lifetime calculation:")
    print("  [1] No mask (use original data)")
    print("  [2] Use masked NPZ files")
    print("  [q] Quit")
    
    while True:
        choice = input("Select option (1, 2, or q): ").strip().lower()
        if choice == 'q':
            return None, ""
        elif choice == '1':
            return 'no-mask', "no-mask"
        elif choice == '2':
            return 'masked', "masked"
        else:
            print("Please enter 1, 2, or q.")

def read_mask_registries(npz_files):
    """
    Read mask registry from NPZ files.
    
    Args:
        npz_files (list): List of NPZ file paths
        
    Returns:
        dict: Dictionary of available masks
    """
    available_masks = {}
    for npz_file in npz_files:
        try:
            npz_data = np.load(npz_file, allow_pickle=True)
            if 'mask_registry' in npz_data:
                mask_registry = npz_data['mask_registry']
                if isinstance(mask_registry, np.ndarray):
                    if mask_registry.dtype == object:
                        mask_registry = mask_registry.item()
                    else:
                        continue
                if isinstance(mask_registry, dict):
                    for mask_name, mask_info in mask_registry.items():
                        if isinstance(mask_info, dict):
                            description = mask_info.get('description', 'No description')
                            mask_type = mask_info.get('type', 'Unknown')
                            created_by = mask_info.get('created_by', 'Unknown')
                            mask_key = f"{mask_name}_{created_by}"
                            if mask_key not in available_masks:
                                available_masks[mask_key] = {
                                    'name': mask_name,
                                    'description': description,
                                    'type': mask_type,
                                    'created_by': created_by,
                                    'files': []
                                }
                            available_masks[mask_key]['files'].append(npz_file)
            npz_data.close()
        except Exception as e:
            print(f"Warning: Could not read mask registry from {npz_file}: {e}")
            continue
    return available_masks

def prompt_mask_selection(available_masks):
    """
    Prompt user to select from available masks.
    
    Args:
        available_masks (dict): Dictionary of available masks
        
    Returns:
        tuple: (selected_mask_name, mask_info) or (None, None) if no selection
    """
    if not available_masks:
        print("No masks found in the NPZ files.")
        return None, None
    
    print(f"\nFound {len(available_masks)} unique masks across all files:")
    mask_list = list(available_masks.items())
    for i, (mask_key, mask_info) in enumerate(mask_list, 1):
        file_count = len(mask_info['files'])
        print(f"  [{i}] {mask_info['name']} ({mask_info['type']})")
        print(f"      Description: {mask_info['description']}")
        print(f"      Created by: {mask_info['created_by']}")
        print(f"      Available in: {file_count} file(s)")
        print()
    
    while True:
        choice = input(f"Select mask (1-{len(mask_list)}) or 'q' to quit: ").strip().lower()
        if choice == 'q':
            return None, None
        elif choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= len(mask_list):
                mask_key, mask_info = mask_list[choice_num - 1]
                return mask_info['name'], mask_info
            else:
                print(f"Please enter a number between 1 and {len(mask_list)}.")
        else:
            print("Please enter a valid number or 'q' to quit.")

def apply_mask_to_data(npz_file, mask_name, data_type='filtered'):
    """
    Apply a mask to NPZ data.
    
    Args:
        npz_file (str): Path to NPZ file
        mask_name (str): Name of mask to apply
        data_type (str): Type of data to use ('filtered' or 'unfiltered')
        
    Returns:
        dict: Modified data dictionary or None if failed
    """
    try:
        npz_data = np.load(npz_file, allow_pickle=True)
        if mask_name not in npz_data:
            print(f"Warning: Mask '{mask_name}' not found in {npz_file}")
            return None
        
        mask = npz_data[mask_name]
        
        # Convert NpzFile to regular dictionary for modification
        data_dict = dict(npz_data)
        
        # Apply mask to TU data
        if 'TU' in data_dict:
            data_dict['TU'] = data_dict['TU'] * mask
            print(f"Applied mask '{mask_name}' to TU data in {os.path.basename(npz_file)}")
            print(f"  Applied mask: {np.sum(mask)} pixels selected out of {mask.size} total")
        
        # Add full_mask key for compatibility with load_segmented_npz
        data_dict['full_mask'] = mask
        
        npz_data.close()
        return data_dict
    except Exception as e:
        print(f"Error applying mask to {npz_file}: {e}")
        return None

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
    
    # Find all NPZ files in the directory
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    npz_files.extend(glob.glob(os.path.join(npz_dir, "**/*.npz"), recursive=True))
    
    # Remove duplicates
    npz_files = list(set(npz_files))
    
    if not npz_files:
        print(f"No NPZ files found in {npz_dir}")
        print(f"Looking for: *.npz")
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

def save_results_to_csv(results_list, output_dir, filename=None):
    """
    Save results to CSV file.
    
    Args:
        results_list (list): List of result dictionaries
        output_dir (str): Directory to save CSV file
        filename (str): Custom filename (optional)
        
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
        if filename is None:
            filename = 'average_lifetime_results.csv'
        csv_path = os.path.join(output_dir, filename)
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

def main(config=None, segmented_npz_dir=None, output_dir=None, select_files=True, mask_source=None, selected_mask_name=None):
    """
    Main execution function for calculating average lifetime from segmented data.
    
    Args:
        config: Configuration dictionary (optional)
        segmented_npz_dir: Directory containing segmented NPZ files
        output_dir: Directory to save CSV results
        select_files: Whether to use interactive file selection
        mask_source: Mask source ('no-mask' or 'masked')
        selected_mask_name: Name of selected mask to apply
        
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
    
    # Get list of NPZ files
    npz_files = list_npz_files(segmented_npz_dir)
    if not npz_files:
        print("No NPZ files found in the directory.")
        return False
    
    # Interactive file selection if requested
    if select_files:
        selected_files, file_selection_name = prompt_file_selection(npz_files)
        if not selected_files:
            print("No files selected. Exiting.")
            return False
    else:
        selected_files = npz_files
        file_selection_name = "full_dataset"
    
    # Interactive mask source selection if not provided
    if mask_source is None:
        mask_source, mask_source_name = prompt_mask_source()
        if mask_source is None:
            print("No mask source selected. Exiting.")
            return False
    else:
        mask_source_name = mask_source
    
    # Handle mask selection if using masked files
    if mask_source == 'masked':
        if selected_mask_name is None:
            available_masks = read_mask_registries(selected_files)
            selected_mask_name, mask_info = prompt_mask_selection(available_masks)
            if selected_mask_name is None:
                print("No mask selected. Exiting.")
                return False
        
        print(f"Selected mask: {selected_mask_name}")
        if mask_info:
            print(f"Description: {mask_info['description']}")
            print(f"Type: {mask_info['type']}")
            print(f"Created by: {mask_info['created_by']}")
    
    # Process selected NPZ files with mask application if needed
    success_count = 0
    error_count = 0
    results_list = []
    
    for npz_file in selected_files:
        try:
            # Apply mask if selected
            if mask_source == 'masked' and selected_mask_name:
                modified_data = apply_mask_to_data(npz_file, selected_mask_name)
                if modified_data is None:
                    print(f"Skipping {os.path.basename(npz_file)} due to mask application error")
                    error_count += 1
                    continue
                
                # Create temporary file with masked data
                temp_file = npz_file.replace('.npz', '_masked_temp.npz')
                np.savez_compressed(temp_file, **modified_data)
                process_file = temp_file
            else:
                process_file = npz_file
            
            # Process the file
            result = process_segmented_npz_file(process_file, output_dir)
            if result:
                results_list.append(result)
                success_count += 1
            else:
                error_count += 1
            
            # Clean up temporary file if created
            if mask_source == 'masked' and selected_mask_name and process_file != npz_file:
                try:
                    os.remove(process_file)
                except:
                    pass
                    
        except Exception as e:
            print(f"Error processing {os.path.basename(npz_file)}: {e}")
            error_count += 1
    
    if success_count > 0:
        # Generate filename with metadata
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"average_lifetime_results_{file_selection_name}_{mask_source_name}_{timestamp}.csv"
        
        # Save results to CSV
        csv_path = save_results_to_csv(results_list, output_dir, filename)
        
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