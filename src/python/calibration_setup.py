#!/usr/bin/env python3
"""
Script to find all .bin files in a directory and save their absolute paths to a CSV file.
"""

import os
import csv
import argparse
from pathlib import Path

def find_bin_files(root_directory):
    """
    Find all .bin files in the given directory and subdirectories.
    
    Args:
        root_directory (str): Path to the root directory to search
        
    Returns:
        list: List of absolute file paths for all .bin files
    """
    bin_files = []
    root_path = Path(root_directory)
    
    # Check if the directory exists
    if not root_path.exists():
        print(f"Error: Directory '{root_directory}' does not exist.")
        return bin_files
    
    # Find all .bin files recursively
    for bin_file in root_path.rglob("*.bin"):
        if bin_file.is_file():
            bin_files.append(str(bin_file.absolute()))
    
    return sorted(bin_files)

def save_to_csv(file_paths, output_file):
    """
    Save the list of file paths to a CSV file.
    
    Args:
        file_paths (list): List of file paths
        output_file (str): Path to the output CSV file
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header with phi and modulation columns
            writer.writerow(['file_path', 'phi', 'modulation'])
            
            # Write each file path as a row (phi and modulation columns will be empty)
            for file_path in file_paths:
                writer.writerow([file_path, '', ''])
        
        print(f"Successfully saved {len(file_paths)} file paths to '{output_file}'")
        
    except Exception as e:
        print(f"Error writing to CSV file: {e}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Find all .bin files and save paths to CSV')
    parser.add_argument('--input', required=True, help='Root directory to search for .bin files')
    
    # Parse arguments
    args = parser.parse_args()
    root_directory = args.input
    
    # Configuration - hardcoded output file
    output_csv = "/Users/joshuamarcus/flimfret/data/calibration.csv"
    
    print(f"Searching for .bin files in: {root_directory}")
    
    # Find all .bin files
    bin_files = find_bin_files(root_directory)
    
    if not bin_files:
        print("No .bin files found in the specified directory.")
        return
    
    print(f"Found {len(bin_files)} .bin files")
    
    # Save to CSV
    save_to_csv(bin_files, output_csv)
    
    # Print first few files as preview
    print(f"\nPreview of first 5 files:")
    for i, file_path in enumerate(bin_files[:5]):
        print(f"  {i+1}. {file_path}")
    
    if len(bin_files) > 5:
        print(f"  ... and {len(bin_files) - 5} more files")

if __name__ == "__main__":
    main()