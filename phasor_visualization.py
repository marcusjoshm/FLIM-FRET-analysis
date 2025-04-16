"""
Phasor Visualization Stage for FLIM-FRET Analysis Pipeline

This module provides interactive visualization of phasor plots from processed NPZ datasets.
It allows users to:
1. Select specific NPZ files for visualization
2. Apply intensity thresholding to filter low-signal pixels
3. Generate and view phasor plots for both filtered and unfiltered data
4. Save plots for further analysis

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
import re

def list_npz_files(npz_dir):
    """
    List all NPZ files in the specified directory
    
    Args:
        npz_dir (str): Directory containing NPZ files
        
    Returns:
        list: List of NPZ file paths
    """
    if not os.path.isdir(npz_dir):
        print(f"Error: NPZ directory '{npz_dir}' does not exist")
        return []
        
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    return sorted(npz_files)

def prompt_file_selection(npz_files):
    """
    Prompt user to select NPZ files for visualization
    
    Args:
        npz_files (list): List of NPZ file paths
        
    Returns:
        list: List of selected NPZ file paths
    """
    if not npz_files:
        print("No NPZ files found.")
        return []
        
    print("\nAvailable NPZ datasets:")
    for i, file_path in enumerate(npz_files):
        file_name = os.path.basename(file_path)
        print(f"  [{i+1}] {file_name}")
    
    print("\nSelect NPZ files to visualize:")
    print("  - Enter numbers separated by commas (e.g., '1,3,5')")
    print("  - Enter 'all' to select all files")
    print("  - Enter 'q' to quit")
    
    selection = input("\nYour selection: ").strip().lower()
    
    if selection == 'q':
        print("Exiting visualization.")
        return []
    
    if selection == 'all':
        print(f"Selected all {len(npz_files)} NPZ files.")
        return npz_files
    
    try:
        indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
        selected_files = [npz_files[idx] for idx in indices if 0 <= idx < len(npz_files)]
        
        if not selected_files:
            print("Invalid selection. No files selected.")
            return prompt_file_selection(npz_files)
            
        print(f"Selected {len(selected_files)} NPZ files.")
        return selected_files
    except (ValueError, IndexError):
        print("Invalid input format. Please try again.")
        return prompt_file_selection(npz_files)

def load_npz_data(selected_files, individual_threshold=None):
    """
    Load and combine data from selected NPZ files
    
    Args:
        selected_files (list): List of selected NPZ file paths
        individual_threshold (float): Percentile threshold to apply to each dataset individually before combining
        
    Returns:
        dict: Combined data with G, S, GU, SU, and A arrays
    """
    # Initialize empty arrays for combined data
    combined_data = {
        'G': [],   # Filtered G coordinates
        'S': [],   # Filtered S coordinates
        'GU': [],  # Unfiltered G coordinates
        'SU': [],  # Unfiltered S coordinates
        'A': []    # Intensity values
    }
    
    total_pixels = 0
    kept_pixels = 0
    
    for file_path in selected_files:
        try:
            data = np.load(file_path)
            
            # Check if all required keys are present
            required_keys = ['G', 'S', 'GU', 'SU', 'A']
            missing_keys = [key for key in required_keys if key not in data]
            
            if missing_keys:
                print(f"Warning: File {os.path.basename(file_path)} is missing keys: {', '.join(missing_keys)}")
                print("This file will be skipped.")
                continue
            
            # Apply individual thresholding if requested
            if individual_threshold is not None:
                # Extract data for this file
                file_data = {}
                for key in required_keys:
                    file_data[key] = data[key].ravel()
                
                # Calculate threshold for this specific file
                intensity = file_data['A']
                file_threshold = np.percentile(intensity, individual_threshold)
                
                # Create mask for this file
                mask = intensity >= file_threshold
                total_pixels += len(mask)
                kept_pixels += np.sum(mask)
                
                # Apply mask to all arrays for this file
                for key in combined_data:
                    combined_data[key].append(file_data[key][mask])
                
                print(f"Loaded and thresholded data from {os.path.basename(file_path)}")
                print(f"  - Applied individual threshold of {file_threshold:.2f} (removed bottom {individual_threshold}%)")
                print(f"  - Kept {np.sum(mask)} of {len(mask)} pixels ({(np.sum(mask)/len(mask))*100:.1f}%)")
            else:
                # Standard loading without thresholding
                for key in combined_data:
                    # Convert to 1D array before appending
                    arr = data[key].ravel()
                    combined_data[key].append(arr)
                
                print(f"Loaded data from {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"Error loading {os.path.basename(file_path)}: {str(e)}")
    
    # Combine all data files into single arrays
    for key in combined_data:
        if combined_data[key]:
            combined_data[key] = np.concatenate(combined_data[key])
        else:
            print(f"Warning: No valid data for '{key}' found in selected files.")
            combined_data[key] = np.array([])
    
    # Print summary if individual thresholding was applied
    if individual_threshold is not None and total_pixels > 0:
        percent_kept = (kept_pixels / total_pixels) * 100
        print(f"\nIndividual thresholding summary:")
        print(f"  - Original pixels: {total_pixels}")
        print(f"  - Pixels retained: {kept_pixels} ({percent_kept:.1f}%)")
            
    return combined_data

def calculate_auto_threshold(data, percentile=90):
    """
    Calculate a threshold value that removes the bottom X percentile of intensity values
    
    Args:
        data (dict): Combined data dictionary
        percentile (float): Percentile value (0-100) to use for thresholding
        
    Returns:
        float: Calculated threshold value
    """
    if 'A' not in data or len(data['A']) == 0:
        return 0
        
    # Calculate the percentile value of the combined intensity data
    threshold = np.percentile(data['A'], percentile)
    return threshold

def apply_intensity_threshold(data, threshold=0, auto_percentile=None):
    """
    Apply intensity threshold to filter low-signal pixels
    
    Args:
        data (dict): Combined data dictionary
        threshold (float): Intensity threshold value (used if auto_percentile is None)
        auto_percentile (float): Percentile value for auto-thresholding (overrides threshold if set)
        
    Returns:
        dict: Filtered data with intensities >= threshold
    """
    # Use auto-thresholding if specified
    if auto_percentile is not None:
        threshold = calculate_auto_threshold(data, auto_percentile)
        print(f"Auto-threshold calculated at {threshold:.2f} (removing bottom {auto_percentile}% of intensity values)")
    
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
        print(f"Applied intensity threshold of {threshold:.2f}:")
        print(f"  - Original pixels: {len(mask)}")
        print(f"  - Pixels retained: {np.sum(mask)} ({percent_kept:.1f}%)")
    else:
        print("No data to threshold.")
        
    return filtered_data

def generate_phasor_plot(g_data, s_data, intensity, title, contour=True):
    """
    Generate a phasor plot from the provided G and S coordinates
    
    Args:
        g_data (array): G coordinates
        s_data (array): S coordinates
        intensity (array): Intensity values
        title (str): Plot title
        contour (bool): Whether to use contour style (True) or scatter (False)
        
    Returns:
        figure: Matplotlib figure object
    """
    # Remove any NaN values
    mask = ~(np.isnan(g_data) | np.isnan(s_data) | np.isnan(intensity))
    g_data = g_data[mask]
    s_data = s_data[mask]
    intensity = intensity[mask]
    
    # Check for empty data
    if len(g_data) == 0 or len(s_data) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f"{title} - No valid data points")
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
    iqr_x = np.percentile(g_data, 75) - np.percentile(g_data, 25)
    bin_width_x = 2 * iqr_x * (len(g_data) ** (-1/3))
    bin_width_x = np.nan_to_num(bin_width_x)

    iqr_y = np.percentile(s_data, 75) - np.percentile(s_data, 25)
    bin_width_y = 2 * iqr_y * (len(s_data) ** (-1/3))
    bin_width_y = np.nan_to_num(bin_width_y)
    
    # Set a small threshold for bin width to detect impractical values
    min_bin_width = np.finfo(float).eps
    
    # Calculate number of bins, or set manually if bin widths are too small
    if bin_width_x <= min_bin_width or bin_width_y <= min_bin_width:
        num_bins_x = 100  # Default number of bins
        num_bins_y = 100
    else:
        num_bins_x = int(np.ceil((np.max(g_data) - np.min(g_data)) / bin_width_x)) // 2
        num_bins_y = int(np.ceil((np.max(s_data) - np.min(s_data)) / bin_width_y)) // 2
        # Ensure a reasonable number of bins
        num_bins_x = max(50, min(200, num_bins_x))
        num_bins_y = max(50, min(200, num_bins_y))
    
    # Create 2D histogram
    hist_vals, _, _ = np.histogram2d(g_data, s_data, bins=(num_bins_x, num_bins_y), weights=intensity)
    vmax = hist_vals.max()
    vmin = hist_vals.min()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate the 2D histogram
    h = ax.hist2d(g_data, s_data, 
                bins=(num_bins_x, num_bins_y), 
                weights=intensity, 
                cmap='nipy_spectral', 
                norm=colors.SymLogNorm(linthresh=50, linscale=1, vmax=vmax, vmin=vmin), 
                zorder=1, 
                cmin=0.01)
    
    # Set plot properties
    ax.set_facecolor('white')
    ax.set_xlabel('\n$G$')
    ax.set_ylabel('$S$\n')
    ax.set_xlim(x_scale)
    ax.set_ylim(y_scale)
    
    # Add the universal circle contour
    ax.contour(X, Y, F, [0], colors='black', linewidths=1, zorder=2)
    
    # Add the colorbar with custom formatting
    near_zero = 0.1
    cbar = fig.colorbar(h[3], ax=ax, format=LogFormatter(10, labelOnlyBase=True))
    
    # Calculate appropriate ticks for the colorbar
    if vmax > 1:
        ticks = [near_zero] + [10**i for i in range(1, int(np.log10(vmax)) + 1)]
        tick_labels = ['0'] + [f'$10^{i}$' for i in range(1, int(np.log10(vmax)) + 1)]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
    
    cbar.set_label('Frequency')
    
    # Set title with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f"{title}\n({timestamp})")
    
    fig.tight_layout()
    
    return fig

def save_plot(fig, output_dir, filename):
    """
    Save the figure to a PDF file
    
    Args:
        fig (figure): Matplotlib figure object
        output_dir (str): Output directory path
        filename (str): Output filename
        
    Returns:
        str: Path to saved file
    """
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Ensure filename has .pdf extension
    if not filename.endswith('.pdf'):
        filename += '.pdf'
        
    # Save figure
    filepath = os.path.join(plots_dir, filename)
    fig.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved plot to {filepath}")
    
    return filepath

def run_phasor_visualization(output_base_dir, select_files=True):
    """
    Run the interactive phasor visualization stage
    
    Args:
        output_base_dir (str): Base output directory
        select_files (bool): Whether to prompt for file selection
        
    Returns:
        bool: Success status
    """
    print("\n=== Stage 3: Phasor Visualization ===")
    
    # Initialize variables that will be used throughout the function
    threshold = 0
    auto_percentile = None
    individual_percentile = None
    
    # Define directories
    npz_dir = os.path.join(output_base_dir, "npz_datasets")
    
    # Check if NPZ directory exists
    if not os.path.isdir(npz_dir):
        print(f"Error: NPZ directory '{npz_dir}' does not exist.")
        print("Please run Stage 2B (wavelet filtering) first.")
        return False
        
    # List available NPZ files
    npz_files = list_npz_files(npz_dir)
    if not npz_files:
        print("No NPZ files found in the directory.")
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
            print(f"Using all {len(selected_files)} NPZ files for visualization.")
        
        # For individual thresholding, we need to apply thresholds before combining
        if individual_percentile is not None:
            # Load NPZ data with individual thresholding
            data = load_npz_data(selected_files, individual_percentile)
            # No further thresholding needed
            filtered_data = data
        else:
            # Load and combine data from selected files without thresholding
            data = load_npz_data(selected_files)
        
        # Check if any data was loaded
        if not data['G'].size or not data['GU'].size:
            print("No valid data loaded from selected files.")
            continue
            
        # Prompt for intensity threshold method
        print("\nThresholding options:")
        print("  [1] No threshold (use all data)")
        print("  [2] Manual threshold (enter a specific value)")
        print("  [3] Auto-threshold on combined data (remove bottom 90% of intensity values)")
        print("  [4] Custom auto-threshold on combined data (specify percentile to remove)")
        print("  [5] Individual dataset auto-threshold (remove bottom 90% from each dataset)")
        print("  [6] Custom individual dataset auto-threshold (specify percentile to remove from each dataset)")
        print("  [q] Quit visualization")
        
        threshold_choice = input("Select an option: ").strip().lower()
        
        if threshold_choice == 'q':
            return False
        
        threshold = 0
        auto_percentile = None
        individual_percentile = None
        threshold_desc = "0 (No threshold)"
        
        if threshold_choice == '1':
            # No threshold, keep all data
            threshold = 0
            threshold_desc = "0 (No threshold)"
            
        elif threshold_choice == '2':
            # Manual threshold
            while True:
                threshold_input = input("Enter intensity threshold value: ").strip()
                try:
                    threshold = float(threshold_input)
                    if threshold < 0:
                        print("Threshold must be non-negative. Please try again.")
                        continue
                    threshold_desc = str(threshold)
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    
        elif threshold_choice == '3':
            # Auto-threshold with default 90%
            auto_percentile = 90
            threshold_desc = f"Auto ({auto_percentile}%)"
            
        elif threshold_choice == '4':
            # Custom auto-threshold on combined data
            while True:
                percentile_input = input("Enter percentile threshold (1-99): ").strip()
                try:
                    percentile = float(percentile_input)
                    if percentile < 1 or percentile > 99:
                        print("Percentile must be between 1 and 99. Please try again.")
                        continue
                    auto_percentile = percentile
                    threshold_desc = f"Auto combined ({auto_percentile}%)"
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
        elif threshold_choice == '5':
            # Individual dataset auto-threshold (90%)
            individual_percentile = 90
            threshold_desc = f"Individual auto ({individual_percentile}%)"
        elif threshold_choice == '6':
            # Custom individual dataset auto-threshold
            while True:
                percentile_input = input("Enter percentile threshold for individual datasets (1-99): ").strip()
                try:
                    percentile = float(percentile_input)
                    if percentile < 1 or percentile > 99:
                        print("Percentile must be between 1 and 99. Please try again.")
                        continue
                    individual_percentile = percentile
                    threshold_desc = f"Individual auto ({individual_percentile}%)"
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            print("Invalid choice. Using no threshold.")
        
        # Apply intensity threshold (either manual or auto) if not using individual thresholding
        if individual_percentile is None:
            filtered_data = apply_intensity_threshold(data, threshold, auto_percentile)
        
        # Create plots
        filtered_fig = generate_phasor_plot(
            filtered_data['G'], filtered_data['S'], filtered_data['A'],
            f"Wavelet Filtered Phasor Plot (Threshold: {threshold_desc})",
            contour=True
        )
        
        unfiltered_fig = generate_phasor_plot(
            filtered_data['GU'], filtered_data['SU'], filtered_data['A'],
            f"Unfiltered Phasor Plot (Threshold: {threshold_desc})",
            contour=True
        )
        
        # Show plots
        plt.ion()  # Turn on interactive mode
        filtered_fig.show()
        unfiltered_fig.show()
        
        # Prompt for next action
        print("\nPhasor Plots Generated")
        print("What would you like to do next?")
        print("  [1] Save plots and proceed")
        print("  [2] Try a different threshold value")
        print("  [3] Select different NPZ files")
        print("  [q] Quit visualization")
        
        choice = input("Your choice: ").strip().lower()
        
        if choice == '1':
            # Save plots
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_plot(filtered_fig, output_base_dir, f"filtered_phasor_{timestamp}.pdf")
            save_plot(unfiltered_fig, output_base_dir, f"unfiltered_phasor_{timestamp}.pdf")
            plt.close('all')  # Close all plots
            print("\nVisualization complete.")
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phasor_visualization.py <output_base_dir>")
        sys.exit(1)
        
    output_base_dir = sys.argv[1]
    run_phasor_visualization(output_base_dir)
