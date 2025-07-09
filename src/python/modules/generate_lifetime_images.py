#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Lifetime Images from NPZ Files
======================================

This module extracts lifetime data from NPZ files and saves them as TIFF images.
It can process both individual NPZ files and entire directories of NPZ datasets.

Part of FLIM-FRET Analysis Pipeline
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imwrite as save_tiff
import argparse

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
            im = ax.imshow(lifetime_data, cmap=cmap, interpolation='nearest')
            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Lifetime (ns)')
            
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

def process_npz_directory(npz_dir, output_dir, create_preview=True):
    """
    Process all NPZ files in a directory and generate lifetime images.
    
    Args:
        npz_dir (str): Directory containing NPZ files
        output_dir (str): Directory to save lifetime images
        create_preview (bool): Whether to create preview plots
        
    Returns:
        tuple: (success_count, error_count)
    """
    if not os.path.isdir(npz_dir):
        print(f"Error: NPZ directory '{npz_dir}' does not exist")
        return 0, 0
    
    # Find all NPZ files
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    npz_files.extend(glob.glob(os.path.join(npz_dir, "**/*.npz"), recursive=True))
    
    if not npz_files:
        print(f"No NPZ files found in {npz_dir}")
        return 0, 0
    
    print(f"Found {len(npz_files)} NPZ files to process")
    
    success_count = 0
    error_count = 0
    
    for i, npz_file in enumerate(npz_files):
        print(f"\nProcessing file {i+1}/{len(npz_files)}: {os.path.basename(npz_file)}")
        
        # Preserve relative path structure
        rel_path = os.path.relpath(os.path.dirname(npz_file), npz_dir)
        if rel_path == '.':
            rel_path = ''
        
        # Create subdirectory structure
        file_output_dir = os.path.join(output_dir, rel_path)
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Process the file
        if process_npz_file(npz_file, file_output_dir, create_preview):
            success_count += 1
        else:
            error_count += 1
    
    print(f"\nLifetime image generation complete:")
    print(f"  Successfully processed: {success_count} files")
    print(f"  Errors: {error_count} files")
    
    return success_count, error_count

def main(config=None, npz_dir=None, output_dir=None, create_preview=True):
    """
    Main execution function for generating lifetime images from NPZ files.
    
    Args:
        config: Configuration dictionary (optional)
        npz_dir: Directory containing NPZ files
        output_dir: Directory to save lifetime images
        create_preview: Whether to create preview plots
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("=== Lifetime Image Generation from NPZ Files ===")
    
    # Check if running as standalone script
    if config is None:
        parser = argparse.ArgumentParser(description="Generate lifetime images from NPZ files")
        parser.add_argument("npz_dir", help="Directory containing NPZ files")
        parser.add_argument("output_dir", help="Directory to save lifetime images")
        parser.add_argument("--no-preview", action="store_true", help="Skip creating preview plots")
        
        args = parser.parse_args()
        
        npz_dir = args.npz_dir
        output_dir = args.output_dir
        create_preview = not args.no_preview
    
    if not npz_dir or not output_dir:
        print("Error: Both npz_dir and output_dir must be specified")
        return False
    
    print(f"Input NPZ directory: {npz_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Create preview plots: {create_preview}")
    
    # Process NPZ files
    success_count, error_count = process_npz_directory(npz_dir, output_dir, create_preview)
    
    if success_count > 0:
        print(f"\n✅ Successfully generated lifetime images for {success_count} files")
        if error_count > 0:
            print(f"⚠️  {error_count} files had errors")
        return True
    else:
        print(f"\n❌ No lifetime images were generated successfully")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 