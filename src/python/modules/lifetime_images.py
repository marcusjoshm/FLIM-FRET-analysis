#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lifetime Images Stage Module
===========================

This module provides interactive lifetime image generation from NPZ files.
It includes file selection options and integrates with the generate_lifetime_images.py logic.

Part of FLIM-FRET Analysis Pipeline
"""

import os
import sys
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imwrite as save_tiff
from pathlib import Path

def load_npz_lifetime_data(npz_file_path):
    """
    Load lifetime data from NPZ file.
    
    Args:
        npz_file_path (str): Path to NPZ file
        
    Returns:
        dict: Dictionary containing lifetime data, or None if failed
    """
    try:
        data = np.load(npz_file_path, allow_pickle=True)
        
        # Check for lifetime data - prioritize TU (unfiltered) over T (filtered)
        lifetime_keys = ['TU', 'lifetime', 'tau_p', 'tau_m', 'lifetime_map']
        available_lifetime = None
        
        for key in lifetime_keys:
            if key in data:
                available_lifetime = key
                break
        
        if available_lifetime is None:
            print(f"Warning: No lifetime data found in {os.path.basename(npz_file_path)}")
            print(f"Available keys: {list(data.keys())}")
            return None
        
        return {
            'lifetime_data': data[available_lifetime],
            'lifetime_key': available_lifetime,
            'file_path': npz_file_path,
            'all_data': dict(data)
        }
        
    except Exception as e:
        print(f"Error loading NPZ file {npz_file_path}: {e}")
        return None

def generate_lifetime_image(lifetime_data, output_path, title=None, cmap='viridis'):
    """
    Generate and save a lifetime image from lifetime data.
    
    Args:
        lifetime_data (numpy.ndarray): Lifetime data array
        output_path (str): Path to save the TIFF image
        title (str): Optional title for the image
        cmap (str): Colormap to use for visualization
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Handle different data types and shapes
        if lifetime_data.dtype == np.dtype('O'):  # Object array
            # Convert object array to float
            lifetime_data = np.array(lifetime_data, dtype=float)
        
        # Remove any NaN or infinite values
        lifetime_data = np.nan_to_num(lifetime_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure 2D array
        if lifetime_data.ndim > 2:
            # Take the first 2D slice if 3D or higher
            lifetime_data = lifetime_data[0] if lifetime_data.ndim == 3 else lifetime_data[0, 0]
        elif lifetime_data.ndim == 1:
            # Reshape 1D array to 2D (assuming square)
            size = int(np.sqrt(lifetime_data.size))
            if size * size == lifetime_data.size:
                lifetime_data = lifetime_data.reshape(size, size)
            else:
                print(f"Warning: Cannot reshape 1D array of size {lifetime_data.size} to square")
                return False
        
        # Save as TIFF
        save_tiff(output_path, lifetime_data)
        
        # Optionally create a preview plot
        if title:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Calculate autoscaling for better visualization
            # Remove zeros and extreme outliers for scaling
            data_for_scaling = lifetime_data[lifetime_data > 0]  # Remove zeros
            if len(data_for_scaling) > 0:
                # Calculate percentiles for robust scaling
                p1, p99 = np.percentile(data_for_scaling, [1, 99])
                vmin = max(0, p1)  # Don't go below 0
                vmax = p99
                
                # If the range is too small, use the full range
                if vmax - vmin < 0.1:
                    vmin = np.min(data_for_scaling)
                    vmax = np.max(data_for_scaling)
            else:
                # Fallback if no non-zero data
                vmin = 0
                vmax = np.max(lifetime_data)
            
            im = ax.imshow(lifetime_data, cmap=cmap, interpolation='nearest', 
                          vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Lifetime (ns)')
            
            # Add scaling info to the plot
            ax.text(0.02, 0.98, f'Range: {vmin:.2f}-{vmax:.2f} ns', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save plot as PNG
            plot_path = output_path.replace('.tiff', '_preview.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved lifetime image: {output_path}")
            print(f"Saved preview plot: {plot_path}")
        else:
            print(f"Saved lifetime image: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error generating lifetime image: {e}")
        return False

def process_npz_file(npz_file_path, output_dir, create_preview=True):
    """
    Process a single NPZ file and generate lifetime images.
    
    Args:
        npz_file_path (str): Path to NPZ file
        output_dir (str): Directory to save lifetime images
        create_preview (bool): Whether to create preview plots
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Load lifetime data
    data = load_npz_lifetime_data(npz_file_path)
    if data is None:
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base filename
    base_name = os.path.splitext(os.path.basename(npz_file_path))[0]
    
    # Generate lifetime image
    lifetime_key = data['lifetime_key']
    lifetime_data = data['lifetime_data']
    
    # Create output filename
    output_filename = f"{base_name}_{lifetime_key}.tiff"
    output_path = os.path.join(output_dir, output_filename)
    
    # Generate title for preview
    title = f"Lifetime Image: {os.path.basename(npz_file_path)}\nKey: {lifetime_key}"
    
    # Generate and save lifetime image
    success = generate_lifetime_image(
        lifetime_data, 
        output_path, 
        title=title if create_preview else None
    )
    
    return success

def list_npz_files(npz_dir):
    """
    List all NPZ files in the directory.
    
    Args:
        npz_dir (str): Directory containing NPZ files
        
    Returns:
        list: List of NPZ file paths
    """
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    npz_files.extend(glob.glob(os.path.join(npz_dir, "**/*.npz"), recursive=True))
    return sorted(npz_files)

def prompt_file_selection(npz_files):
    """
    Interactive file selection for NPZ files.
    
    Args:
        npz_files (list): List of NPZ file paths
        
    Returns:
        list: Selected NPZ file paths, or empty list if cancelled
    """
    if not npz_files:
        print("No NPZ files found.")
        return []
    
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
            return []
        elif choice == 'all':
            return npz_files
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
                
                return selected_files
                
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces, 'all', or 'q'.")

def run_lifetime_images(npz_dir, select_files=True, create_preview=True):
    """
    Run the interactive lifetime images generation stage.
    
    Args:
        npz_dir (str): Directory containing NPZ files
        select_files (bool): Whether to prompt for file selection
        create_preview (bool): Whether to create preview plots
        
    Returns:
        bool: Success status
    """
    print("\n=== Stage 4: Lifetime Images Generation ===")
    
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
        selected_files = prompt_file_selection(npz_files)
        if not selected_files:
            return False
        file_selection = "partial_dataset"
    else:
        # Use all NPZ files if not prompting for selection
        selected_files = npz_files
        print(f"Using all {len(selected_files)} NPZ files for lifetime image generation.")
        file_selection = "full_dataset"
    
    # Create output directory
    output_dir = os.path.abspath(os.path.join(npz_dir, os.pardir))
    lifetime_images_dir = os.path.join(output_dir, 'lifetime_images')
    os.makedirs(lifetime_images_dir, exist_ok=True)
    
    # Use local process_npz_file function
    
    print(f"\nGenerating lifetime images for {len(selected_files)} files...")
    print(f"Output directory: {lifetime_images_dir}")
    
    success_count = 0
    error_count = 0
    
    for i, npz_file in enumerate(selected_files, 1):
        print(f"\nProcessing file {i}/{len(selected_files)}: {os.path.basename(npz_file)}")
        
        # Preserve relative path structure
        rel_path = os.path.relpath(os.path.dirname(npz_file), npz_dir)
        if rel_path == '.':
            rel_path = ''
        
        # Create subdirectory structure
        file_output_dir = os.path.join(lifetime_images_dir, rel_path)
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Process the file
        if process_npz_file(npz_file, file_output_dir, create_preview):
            success_count += 1
        else:
            error_count += 1
    
    print(f"\nLifetime image generation complete:")
    print(f"  Successfully processed: {success_count} files")
    print(f"  Errors: {error_count} files")
    
    # Create log file for partial dataset
    if file_selection == "partial_dataset":
        logs_dir = os.path.join(output_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_content = f"Lifetime image generation for partial dataset:\n"
        log_content += f"Selected files: {', '.join([os.path.basename(f) for f in selected_files])}\n"
        log_content += f"Total files selected: {len(selected_files)} out of {len(npz_files)} available files\n"
        log_content += f"Successfully processed: {success_count} files\n"
        log_content += f"Errors: {error_count} files\n"
        log_content += f"Timestamp: {timestamp}\n"
        
        log_filename = f"lifetime_images_partial_dataset_{timestamp}.txt"
        log_filepath = os.path.join(logs_dir, log_filename)
        
        with open(log_filepath, 'w') as f:
            f.write(log_content)
        print(f"Dataset selection log saved to: {log_filepath}")
    
    return success_count > 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lifetime_images.py <npz_dir>")
        sys.exit(1)
        
    npz_dir = sys.argv[1]
    run_lifetime_images(npz_dir) 