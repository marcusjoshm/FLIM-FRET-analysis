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

# Set matplotlib backend before any matplotlib imports
os.environ['MPLBACKEND'] = 'MacOSX'  # Use MacOSX backend which is more reliable on macOS

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from matplotlib import colors
import datetime
import re

# Import mask selection functions from phasor_segmentation module
try:
    from .phasor_segmentation import read_mask_registries, prompt_mask_selection
except ImportError:
    # Fallback if phasor_segmentation module is not available
    def read_mask_registries(npz_files):
        """Fallback function for reading mask registries"""
        print("Warning: Mask registry reading not available")
        return {}
    
    def prompt_mask_selection(available_masks):
        """Fallback function for mask selection"""
        print("Warning: Mask selection not available")
        return None, None

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

def load_npz_data(selected_files, individual_threshold=None, selected_mask_name=None):
    """
    Load and combine data from selected NPZ files
    
    Args:
        selected_files (list): List of selected NPZ file paths
        individual_threshold (float): Percentile threshold to apply to each dataset individually before combining
        selected_mask_name (str): Name of the selected mask to apply (if any)
        
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
            data = np.load(file_path, allow_pickle=True)
            
            # Check if all required keys are present
            required_keys = ['G', 'S', 'GU', 'SU', 'A']
            missing_keys = [key for key in required_keys if key not in data]
            
            if missing_keys:
                print(f"Warning: File {os.path.basename(file_path)} is missing keys: {', '.join(missing_keys)}")
                print("This file will be skipped.")
                continue
            
            # Apply mask if selected
            if selected_mask_name and selected_mask_name in data:
                print(f"Applying mask '{selected_mask_name}' to {os.path.basename(file_path)}")
                mask = data[selected_mask_name]
                
                # Convert NpzFile to regular dictionary for modification
                data_dict = dict(data)
                
                # Apply mask to all phasor data
                for key in required_keys:
                    if key in data_dict:
                        data_dict[key] = data_dict[key] * mask
                
                print(f"  Applied mask: {np.sum(mask)} pixels selected out of {mask.size} total")
                
                # Use the modified data dictionary
                data = data_dict
            
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

# Import centralized phasor plot utilities
from .phasor_plot_utils import create_phasor_plot, save_phasor_plot

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
    # Use centralized phasor plot creation
    fig, ax = create_phasor_plot(g_data, s_data, intensity, title, figsize=(8, 6))
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
    # Use centralized save function
    return save_phasor_plot(fig, output_dir, filename, format='pdf')

def run_phasor_visualization(npz_dir, select_files=True):
    """
    Run the interactive phasor visualization stage
    
    Args:
        npz_dir (str): Directory containing NPZ files
        select_files (bool): Whether to prompt for file selection
    Returns:
        bool: Success status
    """
    print("\n=== Stage 3: Phasor Visualization ===")
    threshold = 0
    auto_percentile = None
    individual_percentile = None
    
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
            file_selection = "partial_dataset"
        else:
            # Use all NPZ files if not prompting for selection
            selected_files = npz_files
            print(f"Using all {len(selected_files)} NPZ files for visualization.")
            file_selection = "full_dataset"
        
        # Prompt for mask source selection
        print("\n=== Mask Source Selection ===")
        print("Choose mask source for visualization:")
        print("  [1] No mask (use original data)")
        print("  [2] Use masked NPZ files")
        print("  [q] Quit")
        
        mask_source_choice = input("\nSelect option (1, 2, or q): ").strip().lower()
        
        if mask_source_choice == 'q':
            return False
        elif mask_source_choice == '1':
            selected_mask_name = None
            mask_source_name = "no-mask"
            print("Using original data without masks.")
        elif mask_source_choice == '2':
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
        else:
            print("Invalid choice. Using original data without masks.")
            selected_mask_name = None
            mask_source_name = "no-mask"
        
        # Prompt for intensity threshold method before loading data
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
            threshold_name = "no_threshold"
            
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
                    threshold_name = f"manual_threshold_{threshold}"
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    
        elif threshold_choice == '3':
            # Auto-threshold with default 90%
            auto_percentile = 90
            threshold_desc = f"Auto ({auto_percentile}%)"
            threshold_name = "auto_threshold_90"
            
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
                    threshold_name = f"auto_threshold_{auto_percentile}"
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
        elif threshold_choice == '5':
            # Individual dataset auto-threshold (90%)
            individual_percentile = 90
            threshold_desc = f"Individual auto ({individual_percentile}%)"
            threshold_name = "individual_percentile_90"
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
                    threshold_name = f"individual_percentile_{individual_percentile}"
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            print("Invalid choice. Using no threshold.")
            threshold_name = "no_threshold"
            
        # Now that we've determined the thresholding approach, load the data accordingly
        if individual_percentile is not None:
            # Load NPZ data with individual thresholding
            data = load_npz_data(selected_files, individual_percentile, selected_mask_name)
            # No further thresholding needed
            filtered_data = data
        else:
            # Load and combine data from selected files without thresholding
            data = load_npz_data(selected_files, selected_mask_name=selected_mask_name)
        
        # Check if any data was loaded
        if not data['G'].size or not data['GU'].size:
            print("No valid data loaded from selected files.")
            continue
            
        # We've already chosen the thresholding method before loading data
        
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
            # Save plots in 'plots' directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create 'plots' directory in the main output directory
            output_dir = os.path.abspath(os.path.join(npz_dir, os.pardir))
            plots_dir = os.path.join(output_dir, 'phasor_plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Create log directory if it doesn't exist
            logs_dir = os.path.join(output_dir, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            # Save plots with file_selection, threshold, and mask source in filename
            filtered_filename = f"filtered_phasor_{file_selection}_{threshold_name}_{mask_source_name}_{timestamp}.pdf"
            unfiltered_filename = f"unfiltered_phasor_{file_selection}_{threshold_name}_{mask_source_name}_{timestamp}.pdf"
            
            save_plot(filtered_fig, plots_dir, filtered_filename)
            save_plot(unfiltered_fig, plots_dir, unfiltered_filename)
            
            # Create log file for partial dataset
            if file_selection == "partial_dataset":
                log_content = f"Dataset selection for {filtered_filename} and {unfiltered_filename}:\n"
                log_content += f"Selected files: {', '.join(selected_files)}\n"
                log_content += f"Total files selected: {len(selected_files)} out of {len(npz_files)} available files\n"
                log_content += f"Mask source: {mask_source_name}\n"
                if selected_mask_name:
                    log_content += f"Selected mask: {selected_mask_name}\n"
                log_content += f"Timestamp: {timestamp}\n"
                
                log_filename = f"dataset_selection_for_{filtered_filename.replace('.pdf', '.txt')}"
                log_filepath = os.path.join(logs_dir, log_filename)
                
                with open(log_filepath, 'w') as f:
                    f.write(log_content)
                print(f"Dataset selection log saved to: {log_filepath}")
            plt.close('all')  # Close all plots
            print(f"\nVisualization complete. Plots saved to {plots_dir}")
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