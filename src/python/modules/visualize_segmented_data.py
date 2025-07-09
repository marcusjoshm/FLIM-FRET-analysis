#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Segmented Data
========================

This module visualizes segmented NPZ data by applying masks to filtered/unfiltered
G, S, and intensity values and creating phasor plots using the visualization approach
from phasor_visualization.py.

Part of FLIM-FRET Analysis Pipeline
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from matplotlib import colors
import datetime
import argparse

def list_masked_npz_files(external_mask_npz_dir):
    """
    List all masked NPZ files in the external_mask_npz_datasets directory.
    
    Args:
        external_mask_npz_dir (str): Directory containing masked NPZ files
        
    Returns:
        list: List of masked NPZ file paths
    """
    if not os.path.isdir(external_mask_npz_dir):
        print(f"Error: External mask NPZ directory '{external_mask_npz_dir}' does not exist")
        return []
        
    npz_files = glob.glob(os.path.join(external_mask_npz_dir, "*_masked.npz"))
    return sorted(npz_files)

def prompt_file_selection(npz_files):
    """
    Prompt user to select masked NPZ files for visualization.
    
    Args:
        npz_files (list): List of masked NPZ file paths
        
    Returns:
        list: List of selected NPZ file paths
    """
    if not npz_files:
        print("No masked NPZ files found.")
        return []
        
    print("\nAvailable masked NPZ datasets:")
    for i, file_path in enumerate(npz_files):
        file_name = os.path.basename(file_path)
        print(f"  [{i+1}] {file_name}")
    
    print("\nSelect masked NPZ files to visualize:")
    print("  - Enter numbers separated by commas (e.g., '1,3,5')")
    print("  - Enter 'all' to select all files")
    print("  - Enter 'q' to quit")
    
    selection = input("\nYour selection: ").strip().lower()
    
    if selection == 'q':
        print("Exiting visualization.")
        return []
    
    if selection == 'all':
        print(f"Selected all {len(npz_files)} masked NPZ files.")
        return npz_files
    
    try:
        indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
        selected_files = [npz_files[idx] for idx in indices if 0 <= idx < len(npz_files)]
        
        if not selected_files:
            print("Invalid selection. No files selected.")
            return prompt_file_selection(npz_files)
            
        print(f"Selected {len(selected_files)} masked NPZ files.")
        return selected_files
    except (ValueError, IndexError):
        print("Invalid input format. Please try again.")
        return prompt_file_selection(npz_files)

def load_masked_npz_data(selected_files):
    """
    Load and combine data from selected masked NPZ files.
    
    Args:
        selected_files (list): List of selected masked NPZ file paths
        
    Returns:
        dict: Combined masked data with G, S, GU, SU, and A arrays
    """
    # Initialize empty arrays for combined data
    combined_data = {
        'G': [],       # Filtered G coordinates (masked)
        'S': [],       # Filtered S coordinates (masked)
        'GU': [],      # Unfiltered G coordinates (masked)
        'SU': [],      # Unfiltered S coordinates (masked)
        'A': []        # Intensity values (masked)
    }
    
    total_pixels = 0
    masked_pixels = 0
    file_info = []
    
    for file_path in selected_files:
        try:
            data = np.load(file_path)
            
            # Check if full_mask is present
            if 'full_mask' not in data:
                print(f"Warning: File {os.path.basename(file_path)} has no 'full_mask' - skipping")
                continue
            
            # Check if all required keys are present
            required_keys = ['G', 'S', 'GU', 'SU', 'A']
            missing_keys = [key for key in required_keys if key not in data]
            
            if missing_keys:
                print(f"Warning: File {os.path.basename(file_path)} is missing keys: {', '.join(missing_keys)} - skipping")
                continue
            
            # Load mask and data arrays
            mask = data['full_mask']
            
            # Apply mask to each data array
            masked_data = {}
            for key in required_keys:
                # Multiply data by mask (0 = background, 1 = selected region)
                masked_array = data[key] * mask
                masked_data[key] = masked_array
            
            # Get only non-zero (selected) pixels
            mask_indices = mask > 0
            total_file_pixels = mask.size
            selected_file_pixels = np.sum(mask_indices)
            
            total_pixels += total_file_pixels
            masked_pixels += selected_file_pixels
            
            # Extract selected pixels and add to combined data
            for key in combined_data:
                selected_pixels = masked_data[key][mask_indices]
                combined_data[key].append(selected_pixels)
            
            # Store file info
            file_info.append({
                'file': os.path.basename(file_path),
                'total_pixels': total_file_pixels,
                'selected_pixels': selected_file_pixels,
                'selection_percentage': (selected_file_pixels / total_file_pixels) * 100
            })
            
            print(f"Loaded masked data from {os.path.basename(file_path)}")
            print(f"  - Selected {selected_file_pixels} of {total_file_pixels} pixels ({(selected_file_pixels/total_file_pixels)*100:.1f}%)")
            
        except Exception as e:
            print(f"Error loading {os.path.basename(file_path)}: {str(e)}")
    
    # Combine all data files into single arrays
    for key in combined_data:
        if combined_data[key]:
            combined_data[key] = np.concatenate(combined_data[key])
        else:
            print(f"Warning: No valid data for '{key}' found in selected files.")
            combined_data[key] = np.array([])
    
    # Print summary
    if total_pixels > 0:
        overall_percentage = (masked_pixels / total_pixels) * 100
        print(f"\nCombined masked data summary:")
        print(f"  - Original pixels: {total_pixels}")
        print(f"  - Selected pixels: {masked_pixels} ({overall_percentage:.1f}%)")
        print(f"  - Files processed: {len(file_info)}")
    
    return combined_data

def apply_intensity_threshold(data, threshold=0, auto_percentile=None):
    """
    Apply additional intensity threshold to already masked data.
    
    Args:
        data (dict): Masked data dictionary
        threshold (float): Intensity threshold value
        auto_percentile (float): Percentile value for auto-thresholding
        
    Returns:
        dict: Further filtered data
    """
    # Use auto-thresholding if specified
    if auto_percentile is not None:
        if len(data['A']) > 0:
            threshold = np.percentile(data['A'][data['A'] > 0], auto_percentile)
            print(f"Auto-threshold calculated at {threshold:.2f} (removing bottom {auto_percentile}% of selected pixels)")
        else:
            threshold = 0
    
    # If no thresholding is applied, return original data
    if threshold <= 0:
        return data
        
    # Create mask for pixels with intensity >= threshold
    mask = data['A'] >= threshold
    
    # Apply mask to all data arrays
    filtered_data = {}
    for key in data:
        if len(data[key]) > 0:
            filtered_data[key] = data[key][mask]
        else:
            filtered_data[key] = np.array([])
            
    # Print statistics
    if len(mask) > 0:
        percent_kept = (np.sum(mask) / len(mask)) * 100
        print(f"Applied additional intensity threshold of {threshold:.2f}:")
        print(f"  - Selected pixels: {len(mask)}")
        print(f"  - Pixels retained: {np.sum(mask)} ({percent_kept:.1f}%)")
    else:
        print("No data to threshold.")
        
    return filtered_data

def generate_segmented_phasor_plot(g_data, s_data, intensity, title, contour=True):
    """
    Generate a phasor plot from segmented data.
    
    Args:
        g_data (array): G coordinates (segmented)
        s_data (array): S coordinates (segmented)
        intensity (array): Intensity values (segmented)
        title (str): Plot title
        contour (bool): Whether to use contour style (True) or scatter (False)
        
    Returns:
        figure: Matplotlib figure object
    """
    # Remove any NaN values and zero values
    mask = ~(np.isnan(g_data) | np.isnan(s_data) | np.isnan(intensity)) & (intensity > 0)
    g_data = g_data[mask]
    s_data = s_data[mask]
    intensity = intensity[mask]
    
    # Check for empty data
    if len(g_data) == 0 or len(s_data) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f"{title} - No valid segmented data points")
        ax.set_xlabel("\n$G$")
        ax.set_ylabel("$S$\n")
        ax.grid(True, alpha=0.3)
        return fig
    
    # Create a universal circle for reference
    x = np.linspace(0, 1.0, 100)
    y = np.linspace(0, 0.7, 100)
    X, Y = np.meshgrid(x, y)
    F = (X**2 + Y**2 - X)  # Universal circle equation
    
    # Set plot limits
    x_scale = [-0.005, 1.005]
    y_scale = [0, 0.7]
    
    # Calculate bin widths using IQR or use fixed bins
    if len(g_data) > 4:  # Need at least 4 points for IQR
        iqr_x = np.percentile(g_data, 75) - np.percentile(g_data, 25)
        bin_width_x = 2 * iqr_x * (len(g_data) ** (-1/3))
        bin_width_x = np.nan_to_num(bin_width_x)

        iqr_y = np.percentile(s_data, 75) - np.percentile(s_data, 25)
        bin_width_y = 2 * iqr_y * (len(s_data) ** (-1/3))
        bin_width_y = np.nan_to_num(bin_width_y)
    else:
        bin_width_x = 0
        bin_width_y = 0
    
    # Set a small threshold for bin width to detect impractical values
    min_bin_width = np.finfo(float).eps
    
    # Calculate number of bins, or set manually if bin widths are too small
    if bin_width_x <= min_bin_width or bin_width_y <= min_bin_width or len(g_data) < 100:
        num_bins_x = min(50, max(10, len(g_data) // 10))  # Adaptive bins for small datasets
        num_bins_y = min(50, max(10, len(g_data) // 10))
    else:
        num_bins_x = int(np.ceil((np.max(g_data) - np.min(g_data)) / bin_width_x)) // 2
        num_bins_y = int(np.ceil((np.max(s_data) - np.min(s_data)) / bin_width_y)) // 2
        # Ensure a reasonable number of bins
        num_bins_x = max(20, min(100, num_bins_x))
        num_bins_y = max(20, min(100, num_bins_y))
    
    # Create 2D histogram
    try:
        hist_vals, _, _ = np.histogram2d(g_data, s_data, bins=(num_bins_x, num_bins_y), weights=intensity)
        vmax = hist_vals.max()
        vmin = hist_vals.min()
    except:
        # Fallback for very small datasets
        vmax = np.max(intensity)
        vmin = np.min(intensity)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(g_data) >= 10:  # Use histogram for larger datasets
        # Generate the 2D histogram
        h = ax.hist2d(g_data, s_data, 
                    bins=(num_bins_x, num_bins_y), 
                    weights=intensity, 
                    cmap='nipy_spectral', 
                    norm=colors.SymLogNorm(linthresh=max(1, vmax/1000), linscale=1, vmax=vmax, vmin=vmin), 
                    zorder=1, 
                    cmin=0.01)
        
        # Add the colorbar with custom formatting
        cbar = fig.colorbar(h[3], ax=ax, format=LogFormatter(10, labelOnlyBase=True))
        cbar.set_label('Frequency')
        
    else:  # Use scatter plot for very small datasets
        scatter = ax.scatter(g_data, s_data, c=intensity, cmap='nipy_spectral', 
                           s=30, alpha=0.7, zorder=1, edgecolor='black', linewidth=0.5)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Intensity')
    
    # Set plot properties
    ax.set_facecolor('white')
    ax.set_xlabel('\n$G$')
    ax.set_ylabel('$S$\n')
    ax.set_xlim(x_scale)
    ax.set_ylim(y_scale)
    
    # Add the universal circle contour
    ax.contour(X, Y, F, [0], colors='black', linewidths=1, zorder=2)
    
    # Set title with timestamp and data points count
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f"{title}\n({len(g_data)} segmented pixels, {timestamp})")
    
    fig.tight_layout()
    
    return fig

def save_plot(fig, output_dir, filename):
    """
    Save the figure to a PDF file.
    
    Args:
        fig (figure): Matplotlib figure object
        output_dir (str): Output directory path
        filename (str): Output filename
        
    Returns:
        str: Path to saved file
    """
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(output_dir, "segmented_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Ensure filename has .pdf extension
    if not filename.endswith('.pdf'):
        filename += '.pdf'
        
    # Save figure
    filepath = os.path.join(plots_dir, filename)
    fig.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved plot to {filepath}")
    
    return filepath

def run_segmented_visualization(external_mask_npz_dir, output_dir=None, select_files=True):
    """
    Run the segmented data visualization.
    
    Args:
        external_mask_npz_dir (str): Directory containing masked NPZ files
        output_dir (str): Base output directory (optional)
        select_files (bool): Whether to prompt for file selection
        
    Returns:
        bool: Success status
    """
    print("\n=== Segmented Data Visualization ===")
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(external_mask_npz_dir)
    
    # Check if masked NPZ directory exists
    if not os.path.isdir(external_mask_npz_dir):
        print(f"Error: Masked NPZ directory '{external_mask_npz_dir}' does not exist.")
        print("Please run apply_mask.py first to create masked NPZ files.")
        return False
        
    # List available masked NPZ files
    npz_files = list_masked_npz_files(external_mask_npz_dir)
    if not npz_files:
        print("No masked NPZ files found in the directory.")
        return False
        
    # Interactive loop for visualization
    while True:
        # Prompt user to select NPZ files
        if select_files:
            selected_files = prompt_file_selection(npz_files)
            if not selected_files:
                return False
        else:
            # Use all NPZ files if not prompting for selection
            selected_files = npz_files
            print(f"Using all {len(selected_files)} masked NPZ files for visualization.")
        
        # Load masked NPZ data
        print("\nLoading masked NPZ data...")
        data = load_masked_npz_data(selected_files)
        
        # Check if any data was loaded
        if not data['G'].size or not data['GU'].size:
            print("No valid masked data loaded from selected files.")
            continue
        
        # Prompt for additional intensity threshold
        print(f"\nLoaded {len(data['G'])} segmented pixels total.")
        print("\nAdditional thresholding options (applied to segmented data):")
        print("  [1] No additional threshold")
        print("  [2] Manual threshold (enter a specific value)")
        print("  [3] Auto-threshold (remove bottom 90% of segmented pixels)")
        print("  [4] Custom auto-threshold (specify percentile to remove)")
        print("  [q] Quit visualization")
        
        threshold_choice = input("Select an option: ").strip().lower()
        
        if threshold_choice == 'q':
            return False
        
        threshold = 0
        auto_percentile = None
        threshold_desc = "No additional threshold"
        
        if threshold_choice == '1':
            # No additional threshold
            threshold = 0
            threshold_desc = "No additional threshold"
            
        elif threshold_choice == '2':
            # Manual threshold
            while True:
                threshold_input = input("Enter intensity threshold value: ").strip()
                try:
                    threshold = float(threshold_input)
                    if threshold < 0:
                        print("Threshold must be non-negative. Please try again.")
                        continue
                    threshold_desc = f"Threshold: {threshold}"
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    
        elif threshold_choice == '3':
            # Auto-threshold with default 90%
            auto_percentile = 90
            threshold_desc = f"Auto threshold (90%)"
            
        elif threshold_choice == '4':
            # Custom auto-threshold
            while True:
                percentile_input = input("Enter percentile threshold (1-99): ").strip()
                try:
                    percentile = float(percentile_input)
                    if percentile < 1 or percentile > 99:
                        print("Percentile must be between 1 and 99. Please try again.")
                        continue
                    auto_percentile = percentile
                    threshold_desc = f"Auto threshold ({auto_percentile}%)"
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            print("Invalid choice. Using no additional threshold.")
        
        # Apply additional intensity threshold
        filtered_data = apply_intensity_threshold(data, threshold, auto_percentile)
        
        # Check if we have data after additional filtering
        if len(filtered_data['G']) == 0 or len(filtered_data['GU']) == 0:
            print("No data remaining after additional thresholding. Try a lower threshold.")
            continue
        
        # Create plots for both filtered and unfiltered data
        print(f"\nGenerating phasor plots with {len(filtered_data['G'])} pixels...")
        
        # Fix the extreme values issue by clipping data to reasonable ranges
        # We need to handle the infinity values we saw in the test output
        def clip_data(data_array, name, min_val=-5, max_val=5):
            """Clip data to reasonable ranges and handle extreme values."""
            clipped = np.clip(data_array, min_val, max_val)
            if np.any(np.isinf(data_array)) or np.any(np.abs(data_array) > 100):
                print(f"  Warning: Clipped extreme values in {name} data")
            return clipped
        
        # Handle extreme values in G and S data
        filtered_data_safe = filtered_data.copy()
        filtered_data_safe['G'] = clip_data(filtered_data['G'], 'filtered G')
        filtered_data_safe['S'] = clip_data(filtered_data['S'], 'filtered S') 
        filtered_data_safe['GU'] = clip_data(filtered_data['GU'], 'unfiltered G')
        filtered_data_safe['SU'] = clip_data(filtered_data['SU'], 'unfiltered S')
        
        filtered_fig = generate_segmented_phasor_plot(
            filtered_data_safe['G'], filtered_data_safe['S'], filtered_data_safe['A'],
            f"Segmented Wavelet Filtered Phasor Plot ({threshold_desc})",
            contour=True
        )
        
        unfiltered_fig = generate_segmented_phasor_plot(
            filtered_data_safe['GU'], filtered_data_safe['SU'], filtered_data_safe['A'],
            f"Segmented Unfiltered Phasor Plot ({threshold_desc})",
            contour=True
        )
        
        # Show plots
        plt.ion()  # Turn on interactive mode
        filtered_fig.show()
        unfiltered_fig.show()
        
        # Prompt for next action
        print("\nSegmented Phasor Plots Generated")
        print("What would you like to do next?")
        print("  [1] Save plots and proceed")
        print("  [2] Try a different threshold value")
        print("  [3] Select different NPZ files")
        print("  [q] Quit visualization")
        
        choice = input("Your choice: ").strip().lower()
        
        if choice == '1':
            # Save plots
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_plot(filtered_fig, output_dir, f"segmented_filtered_phasor_{timestamp}.pdf")
            save_plot(unfiltered_fig, output_dir, f"segmented_unfiltered_phasor_{timestamp}.pdf")
            plt.close('all')  # Close all plots
            print("\nSegmented visualization complete.")
            return True
        elif choice == '2':
            plt.close('all')  # Close all plots
            continue  # Return to threshold prompt
        elif choice == '3':
            plt.close('all')  # Close all plots
            select_files = True  # Force file selection on next iteration
            continue
        elif choice == 'q':
            plt.close('all')  # Close all plots
            return False
        else:
            print("Invalid choice. Please try again.")
            plt.close('all')  # Close all plots

def main(config=None, external_mask_npz_dir=None, output_dir=None):
    """
    Main execution function for visualizing segmented data.
    
    Args:
        config: Configuration dictionary (optional)
        external_mask_npz_dir: Directory containing masked NPZ files
        output_dir: Base output directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if running as standalone script
    if config is None:
        parser = argparse.ArgumentParser(description="Visualize segmented NPZ data")
        parser.add_argument("external_mask_npz_dir", help="Directory containing masked NPZ files")
        parser.add_argument("--output-dir", help="Output directory for plots (default: parent of input dir)")
        
        args = parser.parse_args()
        
        external_mask_npz_dir = args.external_mask_npz_dir
        output_dir = args.output_dir
    
    if not external_mask_npz_dir:
        print("Error: external_mask_npz_dir must be specified")
        return False
    
    # Run visualization
    success = run_segmented_visualization(external_mask_npz_dir, output_dir)
    
    if success:
        print("\n✅ Segmented data visualization completed successfully")
        return True
    else:
        print("\n❌ Segmented data visualization failed or was cancelled")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 