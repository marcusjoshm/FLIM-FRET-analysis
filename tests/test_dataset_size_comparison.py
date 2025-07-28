#!/usr/bin/env python3
"""
Test script to compare phasor plot resolution between single and combined datasets.

This script loads actual NPZ files and compares the pixel sizes
between single file plots and combined dataset plots.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python', 'modules'))

# Import directly to avoid complex import structure
import phasor_plot_utils


def load_npz_data(npz_file_path):
    """
    Load NPZ data from file.
    
    Args:
        npz_file_path: Path to NPZ file
        
    Returns:
        Dictionary of data, or None if failed
    """
    try:
        return dict(np.load(npz_file_path, allow_pickle=True))
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        return None


def extract_phasor_data(data, data_type='filtered'):
    """
    Extract phasor data from NPZ data.
    
    Args:
        data: Dictionary from NPZ file
        data_type: 'filtered' for G/S or 'unfiltered' for GU/SU
        
    Returns:
        tuple: (g_data, s_data, intensity) or (None, None, None) if failed
    """
    if data_type == 'filtered':
        g_data = data.get('G', data.get('g_data', None))
        s_data = data.get('S', data.get('s_data', None))
    elif data_type == 'unfiltered':
        g_data = data.get('GU', data.get('gu_data', None))
        s_data = data.get('SU', data.get('su_data', None))
    else:
        print(f"Warning: Unknown data_type '{data_type}', using filtered data")
        g_data = data.get('G', data.get('g_data', None))
        s_data = data.get('S', data.get('s_data', None))
        
    intensity = data.get('A', data.get('intensity', None))
    
    if g_data is None or s_data is None or intensity is None:
        print(f"Warning: Missing required data")
        return None, None, None
        
    return g_data, s_data, intensity


def find_npz_files(base_dir):
    """
    Find all NPZ files in the directory structure.
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        list: List of NPZ file paths
    """
    npz_files = []
    
    # Search in common output directories
    search_dirs = [
        os.path.join(base_dir, 'npz_datasets'),
        os.path.join(base_dir, 'output'),
        base_dir
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            found_files = glob.glob(os.path.join(search_dir, "*.npz"))
            npz_files.extend(found_files)
            print(f"Found {len(found_files)} NPZ files in {search_dir}")
    
    return sorted(npz_files)


def analyze_dataset_sizes(npz_files, data_type='filtered'):
    """
    Analyze dataset sizes and create comparison plots.
    
    Args:
        npz_files: List of NPZ file paths
        data_type: Data type to use
    """
    print(f"\n=== Dataset Size Analysis ===")
    print(f"Found {len(npz_files)} NPZ files")
    
    if len(npz_files) == 0:
        print("No NPZ files found. Please ensure you have processed data.")
        return
    
    # Load and analyze each file individually
    individual_datasets = []
    combined_g = []
    combined_s = []
    combined_intensity = []
    
    for i, npz_file in enumerate(npz_files):
        print(f"\n--- Analyzing file {i+1}/{len(npz_files)}: {os.path.basename(npz_file)} ---")
        
        data = load_npz_data(npz_file)
        if data is None:
            print(f"  Failed to load {npz_file}")
            continue
            
        g_data, s_data, intensity = extract_phasor_data(data, data_type)
        if g_data is None:
            print(f"  Failed to extract phasor data from {npz_file}")
            continue
        
        # Flatten arrays
        g_flat = g_data.flatten()
        s_flat = s_data.flatten()
        intensity_flat = intensity.flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(g_flat) | np.isnan(s_flat) | np.isnan(intensity_flat))
        g_flat = g_flat[mask]
        s_flat = s_flat[mask]
        intensity_flat = intensity_flat[mask]
        
        if len(g_flat) == 0:
            print(f"  No valid data points in {npz_file}")
            continue
        
        # Store individual dataset
        individual_datasets.append({
            'filename': os.path.basename(npz_file),
            'g_data': g_flat,
            's_data': s_flat,
            'intensity': intensity_flat,
            'n_points': len(g_flat)
        })
        
        # Add to combined dataset
        combined_g.extend(g_flat)
        combined_s.extend(s_flat)
        combined_intensity.extend(intensity_flat)
        
        print(f"  Points: {len(g_flat):,}")
        print(f"  G range: [{g_flat.min():.3f}, {g_flat.max():.3f}]")
        print(f"  S range: [{s_flat.min():.3f}, {s_flat.max():.3f}]")
    
    if len(individual_datasets) == 0:
        print("No valid datasets found.")
        return
    
    # Convert combined data to arrays
    combined_g = np.array(combined_g)
    combined_s = np.array(combined_s)
    combined_intensity = np.array(combined_intensity)
    
    print(f"\n=== Combined Dataset ===")
    print(f"Total points: {len(combined_g):,}")
    print(f"G range: [{combined_g.min():.3f}, {combined_g.max():.3f}]")
    print(f"S range: [{combined_s.min():.3f}, {combined_s.max():.3f}]")
    
    # Test different resolutions
    resolutions = [100, 300, 500]
    
    for resolution in resolutions:
        print(f"\n--- Testing Resolution: {resolution} pixels per unit ---")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot individual datasets (first 3)
        for i, dataset in enumerate(individual_datasets[:3]):
            ax = axes[i//2, i%2]
            
            title = f"Individual: {dataset['filename']}\n({dataset['n_points']:,} points)"
            phasor_plot_utils.create_phasor_plot(
                dataset['g_data'], dataset['s_data'], dataset['intensity'],
                title, ax=ax, target_pixels_per_unit=resolution, show_colorbar=False
            )
            
            # Add dataset info
            ax.text(0.02, 0.98, f"Points: {dataset['n_points']:,}", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot combined dataset
        ax = axes[1, 1]
        title = f"Combined Dataset\n({len(combined_g):,} points)"
        phasor_plot_utils.create_phasor_plot(
            combined_g, combined_s, combined_intensity,
            title, ax=ax, target_pixels_per_unit=resolution, show_colorbar=False
        )
        
        # Add combined dataset info
        ax.text(0.02, 0.98, f"Points: {len(combined_g):,}", 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'dataset_comparison_{resolution}.png', dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot: dataset_comparison_{resolution}.png")
        plt.close()
        
        # Analyze bin sizes for each dataset
        print(f"\n  Bin size analysis:")
        for dataset in individual_datasets[:3]:
            g_range = dataset['g_data'].max() - dataset['g_data'].min()
            s_range = dataset['s_data'].max() - dataset['s_data'].min()
            
            # Estimate bin counts (this is what the algorithm would use)
            iqr_g = np.percentile(dataset['g_data'], 75) - np.percentile(dataset['g_data'], 25)
            iqr_s = np.percentile(dataset['s_data'], 75) - np.percentile(dataset['s_data'], 25)
            
            bin_width_g = 2 * iqr_g * (len(dataset['g_data']) ** (-1/3))
            bin_width_s = 2 * iqr_s * (len(dataset['s_data']) ** (-1/3))
            
            iqr_bins_g = max(1, int(g_range / bin_width_g)) if bin_width_g > 0 else 100
            iqr_bins_s = max(1, int(s_range / bin_width_s)) if bin_width_s > 0 else 70
            
            target_bins_g = int(g_range * resolution)
            target_bins_s = int(s_range * resolution)
            
            # Apply the same logic as in phasor_plot_utils
            dataset_size_factor = min(1.0, len(dataset['g_data']) / 1000000)
            adjusted_iqr_bins_g = int(iqr_bins_g * (1 + (1 - dataset_size_factor) * 2.0))
            adjusted_iqr_bins_s = int(iqr_bins_s * (1 + (1 - dataset_size_factor) * 2.0))
            
            if len(dataset['g_data']) < 500000:
                final_bins_g = min(adjusted_iqr_bins_g, target_bins_g)
                final_bins_s = min(adjusted_iqr_bins_s, target_bins_s)
            else:
                final_bins_g = min(iqr_bins_g, target_bins_g)
                final_bins_s = min(iqr_bins_s, target_bins_s)
            
            final_bins_g = max(50, min(800, final_bins_g))
            final_bins_s = max(35, min(560, final_bins_s))
            
            bin_size_g = g_range / final_bins_g
            bin_size_s = s_range / final_bins_s
            
            print(f"    {dataset['filename']}: {final_bins_g}x{final_bins_s} bins, "
                  f"bin size: {bin_size_g:.6f}x{bin_size_s:.6f}")
        
        # Combined dataset analysis
        g_range_combined = combined_g.max() - combined_g.min()
        s_range_combined = combined_s.max() - combined_s.min()
        
        iqr_g_combined = np.percentile(combined_g, 75) - np.percentile(combined_g, 25)
        iqr_s_combined = np.percentile(combined_s, 75) - np.percentile(combined_s, 25)
        
        bin_width_g_combined = 2 * iqr_g_combined * (len(combined_g) ** (-1/3))
        bin_width_s_combined = 2 * iqr_s_combined * (len(combined_s) ** (-1/3))
        
        iqr_bins_g_combined = max(1, int(g_range_combined / bin_width_g_combined)) if bin_width_g_combined > 0 else 100
        iqr_bins_s_combined = max(1, int(s_range_combined / bin_width_s_combined)) if bin_width_s_combined > 0 else 70
        
        target_bins_g_combined = int(g_range_combined * resolution)
        target_bins_s_combined = int(s_range_combined * resolution)
        
        final_bins_g_combined = min(iqr_bins_g_combined, target_bins_g_combined)
        final_bins_s_combined = min(iqr_bins_s_combined, target_bins_s_combined)
        
        final_bins_g_combined = max(50, min(800, final_bins_g_combined))
        final_bins_s_combined = max(35, min(560, final_bins_s_combined))
        
        bin_size_g_combined = g_range_combined / final_bins_g_combined
        bin_size_s_combined = s_range_combined / final_bins_s_combined
        
        print(f"    Combined: {final_bins_g_combined}x{final_bins_s_combined} bins, "
              f"bin size: {bin_size_g_combined:.6f}x{bin_size_s_combined:.6f}")


def main():
    """
    Main function to run the dataset size comparison test.
    """
    print("Dataset Size Comparison Test")
    print("=" * 50)
    
    # Use the specified directory
    base_dir = "/Volumes/NX-01-A/2025-07-11_analysis_TEST_/npz_datasets"
    npz_files = find_npz_files(base_dir)
    
    if len(npz_files) == 0:
        print("No NPZ files found. Please run some processing first to generate NPZ files.")
        print("Expected locations:")
        print("  - data/npz_datasets/")
        print("  - output/npz_datasets/")
        print("  - Any directory with .npz files")
        return
    
    # Analyze dataset sizes
    analyze_dataset_sizes(npz_files, data_type='filtered')


if __name__ == "__main__":
    main() 