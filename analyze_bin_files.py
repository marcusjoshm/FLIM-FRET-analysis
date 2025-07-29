#!/usr/bin/env python3
"""
Bin File Structure Analyzer
===========================

This script analyzes .bin files to determine their structure and identify differences
that might cause issues with the ImageJ processing pipeline.

Usage:
    python analyze_bin_files.py <input_directory>

Created by Joshua Marcus
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
import json

def analyze_bin_file(file_path):
    """
    Analyze a single .bin file to determine its structure.
    
    Args:
        file_path (str): Path to the .bin file
        
    Returns:
        dict: Analysis results
    """
    results = {
        'file_path': file_path,
        'file_size': 0,
        'file_size_mb': 0,
        'header_info': {},
        'data_structure': {},
        'possible_dimensions': [],
        'error': None
    }
    
    try:
        # Get file size
        file_size = os.path.getsize(file_path)
        results['file_size'] = file_size
        results['file_size_mb'] = file_size / (1024 * 1024)
        
        with open(file_path, 'rb') as f:
            # Read first 1024 bytes to analyze header
            header_data = f.read(1024)
            
            # Try to detect file format by examining header
            if len(header_data) >= 4:
                # Check for common magic numbers or patterns
                magic_bytes = header_data[:4]
                results['header_info']['magic_bytes'] = magic_bytes.hex()
                
                # Try to interpret as different data types
                try:
                    # Try as 32-bit float
                    float_values = struct.unpack('f' * (len(header_data) // 4), header_data[:len(header_data) - len(header_data) % 4])
                    results['header_info']['first_few_floats'] = float_values[:10]
                    
                    # Check if values are reasonable (not NaN, not inf)
                    valid_floats = [x for x in float_values[:100] if not np.isnan(x) and not np.isinf(x)]
                    if valid_floats:
                        results['header_info']['float_stats'] = {
                            'min': min(valid_floats),
                            'max': max(valid_floats),
                            'mean': np.mean(valid_floats),
                            'std': np.std(valid_floats)
                        }
                except:
                    pass
                
                try:
                    # Try as 32-bit integers
                    int_values = struct.unpack('I' * (len(header_data) // 4), header_data[:len(header_data) - len(header_data) % 4])
                    results['header_info']['first_few_ints'] = int_values[:10]
                except:
                    pass
                
                try:
                    # Try as 16-bit integers
                    short_values = struct.unpack('H' * (len(header_data) // 2), header_data[:len(header_data) - len(header_data) % 2])
                    results['header_info']['first_few_shorts'] = short_values[:10]
                except:
                    pass
            
            # Try to determine possible dimensions
            # Common FLIM data dimensions: 256x256, 512x512, 1024x1024, etc.
            common_dimensions = [
                (256, 256), (512, 512), (1024, 1024), (2048, 2048),
                (128, 128), (64, 64), (32, 32),
                (256, 512), (512, 256), (1024, 512), (512, 1024)
            ]
            
            for width, height in common_dimensions:
                # Try different data types
                for dtype, bytes_per_pixel in [('float32', 4), ('uint16', 2), ('uint32', 4), ('float64', 8)]:
                    expected_size = width * height * bytes_per_pixel
                    if abs(file_size - expected_size) < 1024:  # Allow 1KB tolerance
                        results['possible_dimensions'].append({
                            'width': width,
                            'height': height,
                            'dtype': dtype,
                            'bytes_per_pixel': bytes_per_pixel,
                            'expected_size': expected_size,
                            'difference': file_size - expected_size
                        })
            
            # If no common dimensions match, calculate possible dimensions from file size
            if not results['possible_dimensions']:
                # Try to find dimensions that would give this file size
                for dtype, bytes_per_pixel in [('float32', 4), ('uint16', 2), ('uint32', 4), ('float64', 8)]:
                    total_pixels = file_size // bytes_per_pixel
                    
                    # Try to find reasonable width/height combinations
                    for width in range(64, 4097, 64):  # Common image widths
                        if total_pixels % width == 0:
                            height = total_pixels // width
                            if 64 <= height <= 4096:  # Reasonable height range
                                results['possible_dimensions'].append({
                                    'width': width,
                                    'height': height,
                                    'dtype': dtype,
                                    'bytes_per_pixel': bytes_per_pixel,
                                    'expected_size': width * height * bytes_per_pixel,
                                    'difference': file_size - (width * height * bytes_per_pixel),
                                    'calculated': True
                                })
            
            # Also check for 3D data (width × height × depth)
            if not results['possible_dimensions']:
                for dtype, bytes_per_pixel in [('float32', 4), ('uint16', 2), ('uint32', 4), ('float64', 8)]:
                    total_pixels = file_size // bytes_per_pixel
                    
                    # Common depth values for FLIM data
                    for depth in [132, 256, 512, 1024]:
                        if total_pixels % depth == 0:
                            pixels_per_slice = total_pixels // depth
                            
                            # Try to find width/height combinations
                            for width in range(64, 4097, 64):
                                if pixels_per_slice % width == 0:
                                    height = pixels_per_slice // width
                                    if 64 <= height <= 4096:
                                        results['possible_dimensions'].append({
                                            'width': width,
                                            'height': height,
                                            'depth': depth,
                                            'dtype': dtype,
                                            'bytes_per_pixel': bytes_per_pixel,
                                            'expected_size': width * height * depth * bytes_per_pixel,
                                            'difference': file_size - (width * height * depth * bytes_per_pixel),
                                            'calculated': True,
                                            'is_3d': True
                                        })
            
            # Sort by closest match
            results['possible_dimensions'].sort(key=lambda x: abs(x['difference']))
            
    except Exception as e:
        results['error'] = str(e)
    
    return results

def analyze_directory(input_dir):
    """
    Analyze all .bin files in a directory.
    
    Args:
        input_dir (str): Directory containing .bin files
        
    Returns:
        dict: Analysis results for all files
    """
    input_path = Path(input_dir)
    bin_files = list(input_path.rglob("*.bin"))
    
    if not bin_files:
        print(f"No .bin files found in {input_dir}")
        return {}
    
    print(f"Found {len(bin_files)} .bin files")
    print("=" * 80)
    
    all_results = {}
    file_groups = defaultdict(list)
    
    for i, bin_file in enumerate(bin_files, 1):
        print(f"Analyzing {i}/{len(bin_files)}: {bin_file.name}")
        
        results = analyze_bin_file(str(bin_file))
        all_results[str(bin_file)] = results
        
        # Group files by similar characteristics
        if results['error'] is None:
            # Create a key based on file size and structure
            size_key = f"size_{results['file_size_mb']:.1f}MB"
            file_groups[size_key].append(results)
    
    # Analyze patterns
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Group analysis
    print(f"\nFile Groups (by size):")
    for group_key, files in file_groups.items():
        print(f"\n{group_key}: {len(files)} files")
        if files:
            first_file = files[0]
            print(f"  File size: {first_file['file_size_mb']:.2f} MB")
            
            if first_file['possible_dimensions']:
                best_match = first_file['possible_dimensions'][0]
                print(f"  Most likely dimensions: {best_match['width']}x{best_match['height']} ({best_match['dtype']})")
                print(f"  Size difference: {best_match['difference']} bytes")
            
            if 'float_stats' in first_file['header_info']:
                stats = first_file['header_info']['float_stats']
                print(f"  Data range: {stats['min']:.3f} to {stats['max']:.3f}")
                print(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    
    # Error analysis
    error_files = [r for r in all_results.values() if r['error'] is not None]
    if error_files:
        print(f"\nFiles with errors: {len(error_files)}")
        for result in error_files:
            print(f"  {Path(result['file_path']).name}: {result['error']}")
    
    # Save detailed results
    output_file = input_path / "bin_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return all_results

def compare_with_working_files(working_dir, problematic_dir):
    """
    Compare problematic files with known working files.
    
    Args:
        working_dir (str): Directory with working .bin files
        problematic_dir (str): Directory with problematic .bin files
    """
    print("COMPARING WORKING vs PROBLEMATIC FILES")
    print("=" * 80)
    
    working_results = analyze_directory(working_dir)
    problematic_results = analyze_directory(problematic_dir)
    
    # Extract key characteristics
    working_sizes = [r['file_size_mb'] for r in working_results.values() if r['error'] is None]
    problematic_sizes = [r['file_size_mb'] for r in problematic_results.values() if r['error'] is None]
    
    if working_sizes and problematic_sizes:
        print(f"\nSize comparison:")
        print(f"  Working files: {np.mean(working_sizes):.2f} ± {np.std(working_sizes):.2f} MB")
        print(f"  Problematic files: {np.mean(problematic_sizes):.2f} ± {np.std(problematic_sizes):.2f} MB")
        
        # Check for dimension differences
        working_dims = []
        problematic_dims = []
        
        for results in working_results.values():
            if results['possible_dimensions']:
                working_dims.append(results['possible_dimensions'][0])
        
        for results in problematic_results.values():
            if results['possible_dimensions']:
                problematic_dims.append(results['possible_dimensions'][0])
        
        if working_dims and problematic_dims:
            print(f"\nDimension comparison:")
            working_dims_str = set(f"{d['width']}x{d['height']}" for d in working_dims)
            problematic_dims_str = set(f"{d['width']}x{d['height']}" for d in problematic_dims)
            
            print(f"  Working dimensions: {', '.join(working_dims_str)}")
            print(f"  Problematic dimensions: {', '.join(problematic_dims_str)}")
            
            if working_dims_str != problematic_dims_str:
                print(f"  ⚠️  DIMENSION MISMATCH DETECTED!")

def main():
    parser = argparse.ArgumentParser(description='Analyze .bin files to determine structure differences')
    parser.add_argument('input_dir', help='Directory containing .bin files to analyze')
    parser.add_argument('--compare', help='Directory with working .bin files for comparison')
    parser.add_argument('--output', help='Output file for results (default: bin_analysis_results.json)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    print("Bin File Structure Analyzer")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    
    if args.compare:
        if not os.path.isdir(args.compare):
            print(f"Error: Comparison directory not found: {args.compare}")
            sys.exit(1)
        compare_with_working_files(args.compare, args.input_dir)
    else:
        analyze_directory(args.input_dir)

if __name__ == "__main__":
    main() 