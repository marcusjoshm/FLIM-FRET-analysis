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

def load_npz_data(selected_files):
    """
    Load and combine data from selected NPZ files
    
    Args:
        selected_files (list): List of selected NPZ file paths
        
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
                
            # Append data to combined arrays
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
            
    return combined_data

def apply_intensity_threshold(data, threshold=0):
    """
    Apply intensity threshold to filter low-signal pixels
    
    Args:
        data (dict): Combined data dictionary
        threshold (float): Intensity threshold value
        
    Returns:
        dict: Filtered data with intensities >= threshold
    """
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
        print(f"Applied intensity threshold of {threshold}:")
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
        
        # Load and combine data from selected files
        data = load_npz_data(selected_files)
        
        # Check if any data was loaded
        if not data['G'].size or not data['GU'].size:
            print("No valid data loaded from selected files.")
            continue
            
        # Prompt for intensity threshold
        while True:
            threshold_input = input("\nEnter intensity threshold (0 for no threshold, 'q' to quit): ").strip().lower()
            
            if threshold_input == 'q':
                return False
                
            try:
                threshold = float(threshold_input)
                if threshold < 0:
                    print("Threshold must be non-negative. Please try again.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # Apply intensity threshold
        filtered_data = apply_intensity_threshold(data, threshold)
        
        # Create plots
        filtered_fig = generate_phasor_plot(
            filtered_data['G'], filtered_data['S'], filtered_data['A'],
            f"Wavelet Filtered Phasor Plot (Threshold: {threshold})",
            contour=True
        )
        
        unfiltered_fig = generate_phasor_plot(
            filtered_data['GU'], filtered_data['SU'], filtered_data['A'],
            f"Unfiltered Phasor Plot (Threshold: {threshold})",
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
