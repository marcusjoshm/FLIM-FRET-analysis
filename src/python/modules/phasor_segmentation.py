#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phasor Segmentation Module
==========================

This module provides unified phasor segmentation functionality combining
GMM and manual segmentation approaches with interactive file selection.

Part of FLIM-FRET Analysis Pipeline
"""

import os
import sys
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
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

def prompt_segmentation_method():
    """
    Prompt user to select segmentation method.
    
    Returns:
        tuple: (method, method_name) where method is the internal method name and method_name is the naming variable
    """
    print("\n=== Segmentation Method Selection ===")
    print("Choose segmentation method:")
    print("  [1] GMM Segmentation (automated clustering)")
    print("  [2] Manual Segmentation (interactive ellipse)")
    print("  [q] Quit")
    
    while True:
        choice = input("\nSelect option (1, 2, or q): ").strip().lower()
        
        if choice == 'q':
            return 'q', ""
        elif choice == '1':
            return 'gmm', "GMM_segmentation"
        elif choice == '2':
            return 'manual', "manual_segmentation"
        else:
            print("Please enter 1, 2, or q.")

def prompt_data_type():
    """
    Prompt user to select data type for segmentation.
    
    Returns:
        tuple: (data_type, data_type_name) where data_type is the internal type and data_type_name is the naming variable
    """
    print("\n=== Data Type Selection ===")
    print("Choose data type for segmentation:")
    print("  [1] Filtered data (G, S) - recommended")
    print("  [2] Unfiltered data (GU, SU)")
    print("  [q] Quit")
    
    while True:
        choice = input("\nSelect option (1, 2, or q): ").strip().lower()
        
        if choice == 'q':
            return 'q', ""
        elif choice == '1':
            return 'filtered', "filtered"
        elif choice == '2':
            return 'unfiltered', "unfiltered"
        else:
            print("Please enter 1, 2, or q.")

def prompt_mask_source():
    """
    Prompt user to select mask source for segmentation.
    
    Returns:
        tuple: (mask_source, mask_source_name) where mask_source is the internal source and mask_source_name is the naming variable
    """
    print("\n=== Mask Source Selection ===")
    print("Choose mask source for segmentation:")
    print("  [1] No mask (use original data)")
    print("  [2] Use masked NPZ files")
    print("  [q] Quit")
    
    while True:
        choice = input("\nSelect option (1, 2, or q): ").strip().lower()
        
        if choice == 'q':
            return 'q', ""
        elif choice == '1':
            return 'none', "no-mask"
        elif choice == '2':
            return 'masked', "masked"
        else:
            print("Please enter 1, 2, or q.")

def read_mask_registries(npz_files):
    """
    Read mask registries from NPZ files and collect available masks.
    
    Args:
        npz_files (list): List of NPZ file paths
        
    Returns:
        dict: Dictionary mapping mask names to their descriptions and file paths
    """
    available_masks = {}
    
    for npz_file in npz_files:
        try:
            npz_data = np.load(npz_file, allow_pickle=True)
            
            # Check if mask_registry exists
            if 'mask_registry' in npz_data:
                mask_registry = npz_data['mask_registry']
                
                # Handle both dict and numpy array cases
                if isinstance(mask_registry, np.ndarray):
                    # Convert numpy array to dict if needed
                    if mask_registry.dtype == object:
                        mask_registry = mask_registry.item()
                    else:
                        continue  # Skip if it's not a dict-like array
                
                if isinstance(mask_registry, dict):
                    for mask_name, mask_info in mask_registry.items():
                        if isinstance(mask_info, dict):
                            description = mask_info.get('description', 'No description')
                            mask_type = mask_info.get('type', 'Unknown')
                            created_by = mask_info.get('created_by', 'Unknown')
                            
                            # Create a unique key for this mask
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
        tuple: (selected_mask_name, selected_mask_info) or (None, None) if cancelled
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
    Apply a mask to the phasor data in an NPZ file.
    
    Args:
        npz_file (str): Path to NPZ file
        mask_name (str): Name of the mask to apply
        data_type (str): 'filtered' or 'unfiltered'
        
    Returns:
        dict: Masked data dictionary or None if failed
    """
    try:
        npz_data = np.load(npz_file, allow_pickle=True)
        
        # Get the mask
        if mask_name not in npz_data:
            print(f"Warning: Mask '{mask_name}' not found in {npz_file}")
            return None
        
        mask = npz_data[mask_name]
        
        # Get the phasor data based on data_type
        if data_type == 'filtered':
            g_data = npz_data.get('G')
            s_data = npz_data.get('S')
        else:  # unfiltered
            g_data = npz_data.get('GU')
            s_data = npz_data.get('SU')
        
        if g_data is None or s_data is None:
            print(f"Warning: Required phasor data not found in {npz_file}")
            return None
        
        # Apply mask to phasor data
        masked_g = g_data * mask
        masked_s = s_data * mask
        
        # Create masked data dictionary
        masked_data = {
            'G': masked_g,
            'S': masked_s,
            'GU': masked_g,  # Use same masked data for both
            'SU': masked_s,
            'A': npz_data.get('A', np.ones_like(g_data)),  # Keep intensity as is
            'T': npz_data.get('T'),  # Keep lifetime data if available
            'TU': npz_data.get('TU')
        }
        
        npz_data.close()
        return masked_data
        
    except Exception as e:
        print(f"Error applying mask to {npz_file}: {e}")
        return None

def run_gmm_segmentation(selected_files, data_type, mask_source, output_dir, npz_dir, file_selection_name, method_name, data_type_name, mask_source_name, selected_mask_name=None):
    """
    Run GMM segmentation on selected files.
    
    Args:
        selected_files (list): List of NPZ file paths
        data_type (str): 'filtered' or 'unfiltered'
        mask_source (str): 'none' or 'masked'
        output_dir (str): Output directory
        npz_dir (str): Directory containing NPZ files
        file_selection_name (str): Naming variable for file selection
        method_name (str): Naming variable for segmentation method
        data_type_name (str): Naming variable for data type
        mask_source_name (str): Naming variable for mask source
        selected_mask_name (str): Name of the selected mask to apply (if any)
        
    Returns:
        bool: Success status
    """
    try:
        from .GMMSegmentation import main as gmm_main
        
        print(f"\nRunning GMM segmentation on {len(selected_files)} files...")
        print(f"Data type: {data_type}")
        print(f"Mask source: {mask_source}")
        print(f"Naming variables: {file_selection_name}_{method_name}_{data_type_name}_{mask_source_name}")
        
        # Create a basic config for GMM with data type selection
        gmm_config = {
            'gmm_segmentation_params': {
                'data_selection_use_unfiltered_data': (data_type == 'unfiltered'),
                'n_components': 2,
                'covariance_type': 'full',
                'max_iter': 100,
                'random_state': 0,
                'intensity_threshold': 0,
                'threshold_type': 'absolute',
                'combine_datasets': True,
                'radius_ref': 0.5,
                'cov_f': 1.0,
                'shift': 0.0,
                'use_circle_filter': False
            }
        }
        
        # Use the existing GMM module
        success = gmm_main(
            config=gmm_config,
            npz_dir=npz_dir,  # Pass the NPZ directory
            segmented_dir=output_dir,  # Use output_dir directly
            plots_dir=output_dir,  # Use output_dir directly
            lifetime_dir=output_dir,  # Use output_dir directly
            interactive_mode=True,
            naming_variables={
                'file_selection': file_selection_name,
                'method': method_name,
                'data_type': data_type_name,
                'mask_source': mask_source_name
            },
            selected_mask_name=selected_mask_name
        )
        
        return success
        
    except ImportError as e:
        print(f"Error importing GMM segmentation module: {e}")
        return False
    except Exception as e:
        print(f"Error during GMM segmentation: {e}")
        return False

def run_manual_segmentation(selected_files, data_type, mask_source, output_dir, npz_dir, file_selection_name, method_name, data_type_name, mask_source_name, selected_mask_name=None):
    """
    Run manual segmentation on selected files.
    
    Args:
        selected_files (list): List of NPZ file paths
        data_type (str): 'filtered' or 'unfiltered'
        mask_source (str): 'none' or 'masked'
        output_dir (str): Output directory
        npz_dir (str): Directory containing NPZ files
        file_selection_name (str): Naming variable for file selection
        method_name (str): Naming variable for segmentation method
        data_type_name (str): Naming variable for data type
        mask_source_name (str): Naming variable for mask source
        selected_mask_name (str): Name of the selected mask to apply (if any)
        
    Returns:
        bool: Success status
    """
    try:
        from .ManualSegmentation import main as manual_main
        
        print(f"\nRunning manual segmentation on {len(selected_files)} files...")
        print(f"Data type: {data_type}")
        print(f"Mask source: {mask_source}")
        
        # Use the existing manual segmentation module with selected files
        success = manual_main(
            config=None,
            npz_dir=npz_dir,  # Pass the NPZ directory
            output_dir=output_dir,  # Main output directory
            plots_dir=output_dir,  # Keep for compatibility
            lifetime_dir=output_dir,  # Use output_dir directly
            interactive=True,
            selected_files=selected_files,  # Pass the selected files
            data_type=data_type,  # Pass the data type selection
            naming_variables={
                'file_selection': file_selection_name,
                'method': method_name,
                'data_type': data_type_name,
                'mask_source': mask_source_name
            },
            selected_mask_name=selected_mask_name
        )
        
        return success
        
    except ImportError as e:
        print(f"Error importing manual segmentation module: {e}")
        return False
    except Exception as e:
        print(f"Error during manual segmentation: {e}")
        return False

def run_phasor_segmentation(npz_dir, select_files=True):
    """
    Run the interactive phasor segmentation stage.
    
    Args:
        npz_dir (str): Directory containing NPZ files
        select_files (bool): Whether to prompt for file selection
        
    Returns:
        bool: Success status
    """
    print("\n=== Stage 5: Phasor Segmentation ===")
    
    # Check if NPZ directory exists
    if not os.path.isdir(npz_dir):
        print(f"Error: NPZ directory '{npz_dir}' does not exist.")
        print("Please run Stage 2B (processing) first.")
        return False
        
    # List available NPZ files
    npz_files = list_npz_files(npz_dir)
    if not npz_files:
        print("No NPZ files found in the directory.")
        return False
    
    # Interactive file selection
    if select_files:
        selected_files, file_selection_name = prompt_file_selection(npz_files)
        if not selected_files:
            return False
        file_selection = "partial_dataset"
    else:
        # Use all NPZ files if not prompting for selection
        selected_files = npz_files
        file_selection_name = "full_dataset"
        print(f"Using all {len(selected_files)} NPZ files for segmentation.")
        file_selection = "full_dataset"
    
    # Prompt for segmentation method
    method, method_name = prompt_segmentation_method()
    if method == 'q':
        return False
    
    # Prompt for data type
    data_type, data_type_name = prompt_data_type()
    if data_type == 'q':
        return False
    
    # Prompt for mask source
    mask_source, mask_source_name = prompt_mask_source()
    if mask_source == 'q':
        return False
    
    # Handle mask selection and application
    selected_mask_name = None
    if mask_source == 'masked':
        # Read available masks from NPZ files
        available_masks = read_mask_registries(selected_files)
        
        # Prompt user to select a mask
        selected_mask_name, selected_mask_info = prompt_mask_selection(available_masks)
        if selected_mask_name is None:
            print("No mask selected. Exiting.")
            return False
        
        print(f"Selected mask: {selected_mask_name}")
        print(f"Description: {selected_mask_info['description']}")
        print(f"Type: {selected_mask_info['type']}")
        print(f"Created by: {selected_mask_info['created_by']}")
        
        # Update mask_source_name to include the selected mask
        mask_source_name = f"masked_{selected_mask_name}"
    
    # Get the main output directory
    output_dir = os.path.abspath(os.path.join(npz_dir, os.pardir))
    
    print(f"\nStarting {method} segmentation...")
    print(f"Data type: {data_type}")
    print(f"Mask source: {mask_source}")
    if selected_mask_name:
        print(f"Selected mask: {selected_mask_name}")
    print(f"Output directory: {output_dir}")
    
    # Run the selected segmentation method
    if method == 'gmm':
        success = run_gmm_segmentation(selected_files, data_type, mask_source, output_dir, npz_dir, file_selection_name, method_name, data_type_name, mask_source_name, selected_mask_name)
    elif method == 'manual':
        success = run_manual_segmentation(selected_files, data_type, mask_source, output_dir, npz_dir, file_selection_name, method_name, data_type_name, mask_source_name, selected_mask_name)
    else:
        print(f"Unknown method: {method}")
        return False
    
    # Create log file for partial dataset
    if file_selection == "partial_dataset":
        logs_dir = os.path.join(output_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_content = f"Phasor segmentation for partial dataset:\n"
        log_content += f"Method: {method}\n"
        log_content += f"Data type: {data_type}\n"
        log_content += f"Mask source: {mask_source}\n"
        log_content += f"Naming variables: {file_selection_name}_{method_name}_{data_type_name}_{mask_source_name}\n"
        log_content += f"Selected files: {', '.join([os.path.basename(f) for f in selected_files])}\n"
        log_content += f"Total files selected: {len(selected_files)} out of {len(npz_files)} available files\n"
        log_content += f"Success: {success}\n"
        log_content += f"Timestamp: {timestamp}\n"
        
        log_filename = f"phasor_segmentation_{file_selection_name}_{method_name}_{data_type_name}_{mask_source_name}_{timestamp}.txt"
        log_filepath = os.path.join(logs_dir, log_filename)
        
        with open(log_filepath, 'w') as f:
            f.write(log_content)
        print(f"Dataset selection log saved to: {log_filepath}")
    
    if success:
        print(f"\n✅ Segmentation completed successfully!")
    else:
        print(f"\n❌ Segmentation failed!")
    
    return success

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phasor_segmentation.py <npz_dir>")
        sys.exit(1)
        
    npz_dir = sys.argv[1]
    run_phasor_segmentation(npz_dir) 