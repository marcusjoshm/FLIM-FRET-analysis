#!/usr/bin/env python3
"""
TIFF Tile Stitcher

This script recursively processes directories containing 9 TIFF tiles and stitches them
together in the specified 3x3 grid orientation:

s9  s4  s3
s8  s5  s2
s7  s6  s1

Usage:
    python tiff_stitcher.py <parent_directory>
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import argparse


def find_tiles(directory):
    """
    Find all TIFF tiles (s1 through s9) in a directory.
    
    Args:
        directory (str): Path to directory containing tiles
        
    Returns:
        dict: Dictionary mapping tile numbers to file paths, or None if not all tiles found
    """
    tile_pattern = "*_s{}_*.tiff"
    tiles = {}
    
    for i in range(1, 10):  # s1 through s9
        pattern = os.path.join(directory, tile_pattern.format(i))
        matches = glob.glob(pattern)
        
        if len(matches) == 1:
            tiles[i] = matches[0]
        elif len(matches) > 1:
            print(f"Warning: Multiple files found for s{i} in {directory}: {matches}")
            tiles[i] = matches[0]  # Use the first match
        else:
            # No tile found for this number
            return None
    
    return tiles


def stitch_tiles(tiles, output_path):
    """
    Stitch 9 tiles together in the specified 3x3 grid orientation.
    
    Grid layout:
    s9  s4  s3
    s8  s5  s2
    s7  s6  s1
    
    Args:
        tiles (dict): Dictionary mapping tile numbers to file paths
        output_path (str): Path where stitched image will be saved
    """
    # Define the grid layout
    grid_layout = [
        [9, 4, 3],  # Top row
        [8, 5, 2],  # Middle row
        [7, 6, 1]   # Bottom row
    ]
    
    # Load all tiles
    tile_images = {}
    for tile_num, file_path in tiles.items():
        try:
            tile_images[tile_num] = Image.open(file_path)
        except Exception as e:
            print(f"Error loading tile {tile_num} from {file_path}: {e}")
            return False
    
    # Get dimensions of first tile (assuming all tiles are the same size)
    first_tile = tile_images[1]
    tile_width, tile_height = first_tile.size
    
    # Verify all tiles have the same dimensions
    for tile_num, img in tile_images.items():
        if img.size != (tile_width, tile_height):
            print(f"Warning: Tile s{tile_num} has different dimensions: {img.size} vs {(tile_width, tile_height)}")
    
    # Create the stitched image
    stitched_width = tile_width * 3
    stitched_height = tile_height * 3
    
    # Use the mode of the first tile for the stitched image
    stitched_image = Image.new(first_tile.mode, (stitched_width, stitched_height))
    
    # Place tiles according to grid layout
    for row_idx, row in enumerate(grid_layout):
        for col_idx, tile_num in enumerate(row):
            x_offset = col_idx * tile_width
            y_offset = row_idx * tile_height
            
            tile_img = tile_images[tile_num]
            stitched_image.paste(tile_img, (x_offset, y_offset))
    
    # Save the stitched image
    try:
        stitched_image.save(output_path)
        print(f"Successfully created: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving stitched image to {output_path}: {e}")
        return False
    finally:
        # Close all opened images
        for img in tile_images.values():
            img.close()


def get_output_filename(directory, tiles):
    """
    Generate an appropriate output filename based on the directory and tile names.
    
    Args:
        directory (str): Directory containing the tiles
        tiles (dict): Dictionary of tile files
        
    Returns:
        str: Output filename
    """
    # Get the base name pattern from one of the tiles
    sample_tile = tiles[1]
    base_name = os.path.basename(sample_tile)
    
    # Extract prefix (everything before _s1_) and preserve the type suffix
    if '_s1_' in base_name:
        prefix = base_name.split('_s1_')[0]
        suffix_part = base_name.split('_s1_')[1]
        
        # Extract the type designation (g, s, or intensity) and extension
        if suffix_part.startswith('g.'):
            type_suffix = '_g.tiff'
        elif suffix_part.startswith('s.'):
            type_suffix = '_s.tiff'
        elif suffix_part.startswith('intensity.'):
            type_suffix = '_intensity.tiff'
        else:
            # Fallback - try to extract any single character before the extension
            parts = suffix_part.split('.')
            if len(parts) > 1 and len(parts[0]) == 1:
                type_suffix = f"_{parts[0]}.tiff"
            else:
                type_suffix = '.tiff'
        
        output_name = f"{prefix}_stitched{type_suffix}"
    else:
        # Fallback naming
        dir_name = os.path.basename(directory)
        output_name = f"{dir_name}_stitched.tiff"
    
    return os.path.join(directory, output_name)


def process_directory(directory):
    """
    Process a single directory for tile stitching.
    
    Args:
        directory (str): Path to directory to process
        
    Returns:
        bool: True if successful, False otherwise
    """
    tiles = find_tiles(directory)
    
    if tiles is None:
        return False
    
    if len(tiles) != 9:
        print(f"Warning: Found {len(tiles)} tiles in {directory}, expected 9")
        return False
    
    output_path = get_output_filename(directory, tiles)
    
    # Skip if output already exists
    if os.path.exists(output_path):
        print(f"Output already exists, skipping: {output_path}")
        return True
    
    print(f"Processing {directory}...")
    return stitch_tiles(tiles, output_path)


def main():
    parser = argparse.ArgumentParser(description='Stitch TIFF tiles in 3x3 grid orientation')
    parser.add_argument('parent_directory', help='Parent directory to process recursively')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be processed without actually stitching')
    
    args = parser.parse_args()
    
    parent_dir = args.parent_directory
    
    if not os.path.exists(parent_dir):
        print(f"Error: Directory {parent_dir} does not exist")
        sys.exit(1)
    
    # Find all directories that contain tiles
    directories_to_process = []
    
    for root, dirs, files in os.walk(parent_dir):
        # Check if this directory contains tiles
        tiff_files = [f for f in files if f.endswith('.tiff')]
        if len(tiff_files) >= 9:
            tiles = find_tiles(root)
            if tiles and len(tiles) == 9:
                directories_to_process.append(root)
    
    if not directories_to_process:
        print("No directories with complete tile sets found.")
        sys.exit(1)
    
    print(f"Found {len(directories_to_process)} directories to process:")
    for directory in directories_to_process:
        print(f"  {directory}")
    
    if args.dry_run:
        print("\nDry run mode - no files will be created")
        sys.exit(0)
    
    # Process each directory
    successful = 0
    failed = 0
    
    for directory in directories_to_process:
        if process_directory(directory):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()