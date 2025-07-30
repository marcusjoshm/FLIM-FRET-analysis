#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Exploration Module for FLIM-FRET Analysis
==============================================

This module provides interactive data exploration by allowing users to select
regions on the phasor plot and visualize the corresponding pixels on the intensity image.

Created by Joshua Marcus
"""

import os
import sys
import math
import traceback
import datetime
import numpy as np

# Set matplotlib backend before any matplotlib imports
os.environ['MPLBACKEND'] = 'MacOSX'  # Use MacOSX backend which is more reliable on macOS

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from matplotlib import colors
from tifffile import imwrite as save_tiff

# Import centralized phasor plot utilities
from .phasor_plot_utils import create_phasor_plot, calculate_ellipse_mask


def interactive_file_selection(npz_dir):
    """
    Interactive file selection for NPZ files.
    
    Args:
        npz_dir: Directory containing NPZ files
        
    Returns:
        list: List of selected NPZ file paths
    """
    # Find all NPZ files
    npz_files = []
    for root, _, files in os.walk(npz_dir):
        for file in files:
            if file.endswith('.npz') and not file.endswith('_segmented.npz'):
                npz_path = os.path.join(root, file)
                npz_files.append(npz_path)
    
    if not npz_files:
        print(f"No NPZ files found in {npz_dir}")
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

def are_points_inside_ellipse(points_x, points_y, center_x, center_y, width, height, angle_rad):
    """
    Check if points are inside an ellipse using vectorized operations.
    
    Args:
        points_x: X coordinates of points to check
        points_y: Y coordinates of points to check
        center_x: X coordinate of ellipse center
        center_y: Y coordinate of ellipse center
        width: Width of ellipse
        height: Height of ellipse
        angle_rad: Rotation angle in radians
        
    Returns:
        Boolean array indicating which points are inside the ellipse
    """
    # Use centralized ellipse calculation function
    angle_deg = np.degrees(angle_rad)
    return calculate_ellipse_mask(points_x, points_y, center_x, center_y, width, height, angle_deg)

def process_npz_file_for_exploration(npz_file_path, data_type='filtered', selected_mask_name=None):
    """
    Process a single NPZ file for data exploration.
    
    Args:
        npz_file_path: Path to NPZ file
        data_type: Data type to use ('filtered' for G/S or 'unfiltered' for GU/SU)
        selected_mask_name: Name of the selected mask to apply (if any)
        
    Returns:
        Dictionary containing processed data or None if failed
    """
    data = load_npz_data(npz_file_path)
    if data is None:
        return None
        
    # Extract data based on data_type
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
    lifetime = data.get('lifetime', None)
    
    # Check if required data is available
    if g_data is None or s_data is None or intensity is None:
        print(f"Missing required data in {os.path.basename(npz_file_path)}")
        print(f"  G data: {'Available' if g_data is not None else 'Missing'}")
        print(f"  S data: {'Available' if s_data is not None else 'Missing'}")
        print(f"  Intensity data: {'Available' if intensity is not None else 'Missing'}")
        return None
    
    # Apply mask if specified
    if selected_mask_name and selected_mask_name in data:
        mask = data[selected_mask_name]
        g_data = g_data * mask
        s_data = s_data * mask
        intensity = intensity * mask
        if lifetime is not None:
            lifetime = lifetime * mask
    
    return {
        'npz_data': data,
        'g_data': g_data,
        's_data': s_data,
        'intensity': intensity,
        'lifetime': lifetime,
        'data_type': data_type,
        'original_shape': intensity.shape
    }

def create_interactive_exploration_plot(file_data, data_type, threshold_desc):
    """
    Create an interactive exploration plot with phasor plot and intensity image overlay.
    
    Args:
        file_data: Dictionary containing processed data
        data_type: Data type being used
        threshold_desc: Description of thresholding applied
        
    Returns:
        bool: True if successful, False otherwise
    """
    g_data = file_data['g_data']
    s_data = file_data['s_data']
    intensity = file_data['intensity']
    
    # Get original shape from intensity data if not available
    if 'original_shape' in file_data:
        original_shape = file_data['original_shape']
    else:
        original_shape = intensity.shape
    
    # Flatten arrays for processing
    g_flat = g_data.flatten()
    s_flat = s_data.flatten()
    intensity_flat = intensity.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(g_flat) | np.isnan(s_flat) | np.isnan(intensity_flat))
    g_flat = g_flat[mask]
    s_flat = s_flat[mask]
    intensity_flat = intensity_flat[mask]
    
    # Check for empty data
    if len(g_flat) == 0 or len(s_flat) == 0:
        print("Warning: No valid data points after filtering")
        return False
    
    # Create the figure with two subplots
    fig, (ax_phasor, ax_intensity) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Create the phasor plot using centralized utilities
    title = f"Data Exploration - Phasor Plot ({data_type} data)\n{threshold_desc}"
    create_phasor_plot(g_flat, s_flat, intensity_flat, title, ax=ax_phasor, show_colorbar=True)
    
    # Initial ellipse parameters
    center_x, center_y = 0.5, 0.25
    width, height = 0.2, 0.1
    angle = 0
    
    # Create ellipse
    ellipse = Ellipse(
        xy=(center_x, center_y),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        color='blue',
        linewidth=2
    )
    ax_phasor.add_artist(ellipse)
    
    # Add center marker
    center_point, = ax_phasor.plot([center_x], [center_y], 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
    
    # Display intensity image
    intensity_display = ax_intensity.imshow(intensity, cmap='gray', interpolation='nearest')
    ax_intensity.set_title(f"Intensity Image with ROI Overlay\n{os.path.basename(file_data['npz_data'].get('filename', 'Unknown'))}")
    ax_intensity.set_xlabel('X pixels')
    ax_intensity.set_ylabel('Y pixels')
    
    # Create sliders
    plt.subplots_adjust(bottom=0.25)
    
    # Slider positions
    ax_center_x = plt.axes([0.2, 0.15, 0.6, 0.03])
    ax_center_y = plt.axes([0.2, 0.11, 0.6, 0.03])
    ax_width = plt.axes([0.2, 0.07, 0.6, 0.03])
    ax_height = plt.axes([0.2, 0.03, 0.6, 0.03])
    ax_angle = plt.axes([0.2, 0.19, 0.6, 0.03])
    
    # Create sliders
    s_center_x = Slider(ax_center_x, 'Center X', 0, 1, valinit=center_x)
    s_center_y = Slider(ax_center_y, 'Center Y', 0, 0.5, valinit=center_y)
    s_width = Slider(ax_width, 'Width', 0.01, 0.5, valinit=width)
    s_height = Slider(ax_height, 'Height', 0.01, 0.5, valinit=height)
    s_angle = Slider(ax_angle, 'Angle', 0, 180, valinit=angle)
    
    # Function to update the plot when sliders change
    def update(val):
        # Get values from sliders
        center_x = s_center_x.val
        center_y = s_center_y.val
        width = s_width.val
        height = s_height.val
        angle = s_angle.val
        
        # Update ellipse
        ellipse.set_center((center_x, center_y))
        ellipse.set_width(width)
        ellipse.set_height(height)
        ellipse.set_angle(angle)
        
        # Update center marker
        center_point.set_data([center_x], [center_y])
        
        # Redraw the figure
        fig.canvas.draw_idle()
    
    # Connect sliders to update function
    s_center_x.on_changed(update)
    s_center_y.on_changed(update)
    s_width.on_changed(update)
    s_height.on_changed(update)
    s_angle.on_changed(update)
    
    # Function to apply the ellipse and update intensity overlay
    def apply_roi(event):
        # Get final parameters
        center_x = s_center_x.val
        center_y = s_center_y.val
        width = s_width.val
        height = s_height.val
        angle = s_angle.val
        angle_rad = np.radians(angle)
        
        # Create mask for the selected region
        roi_mask = np.zeros_like(g_data, dtype=bool)
        
        # Get coordinates of all pixels
        mask_indices = np.where(np.ones_like(g_data, dtype=bool))
        mask_g_values = g_data[mask_indices]
        mask_s_values = s_data[mask_indices]
        
        if len(mask_g_values) == 0:
            print("  Warning: No pixels to process")
            return
        
        # Check which points are inside the ellipse
        inside_ellipse = are_points_inside_ellipse(
            mask_g_values, mask_s_values, 
            center_x, center_y, 
            width, height, angle_rad
        )
        
        # Set mask values
        roi_mask[mask_indices[0][inside_ellipse], mask_indices[1][inside_ellipse]] = True
        
        # Create a colored overlay for the ROI - use bright red for visibility
        overlay_colored = np.zeros((*intensity.shape, 4))  # RGBA
        overlay_colored[:, :, 0] = 1.0  # Red channel (bright red)
        overlay_colored[:, :, 1] = 0.0  # Green channel
        overlay_colored[:, :, 2] = 0.0  # Blue channel
        overlay_colored[:, :, 3] = 0.0  # Alpha (transparent by default)
        
        # Add ROI mask as bright red overlay
        overlay_colored[roi_mask, 0] = 1.0  # Bright red for ROI
        overlay_colored[roi_mask, 3] = 0.6  # Semi-transparent red for ROI
        
        # Update intensity image with overlay
        ax_intensity.clear()
        ax_intensity.imshow(intensity, cmap='gray', interpolation='nearest')
        ax_intensity.imshow(overlay_colored, interpolation='nearest')
        ax_intensity.set_title(f"Intensity Image with ROI Overlay\n{os.path.basename(file_data['npz_data'].get('filename', 'Unknown'))}")
        ax_intensity.set_xlabel('X pixels')
        ax_intensity.set_ylabel('Y pixels')
        
        # Print statistics
        total_pixels = roi_mask.size
        selected_pixels = np.sum(roi_mask)
        percentage = (selected_pixels / total_pixels) * 100
        
        print(f"\nROI Statistics:")
        print(f"  Total pixels: {total_pixels}")
        print(f"  Selected pixels: {selected_pixels}")
        print(f"  Percentage: {percentage:.2f}%")
        print(f"  Ellipse center: ({center_x:.3f}, {center_y:.3f})")
        print(f"  Ellipse size: {width:.3f} x {height:.3f}")
        print(f"  Ellipse angle: {angle:.1f}°")
        
        # Redraw the figure
        fig.canvas.draw_idle()
    
    # Create apply button
    ax_apply = plt.axes([0.8, 0.02, 0.1, 0.04])
    button_apply = Button(ax_apply, 'Apply ROI')
    button_apply.on_clicked(apply_roi)
    
    # Add keyboard shortcut for closing
    def on_key(event):
        if event.key == 'escape':
            plt.close('all')
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Note: Use the window's close button (X) or press Escape to close
    # Removed close button to avoid matplotlib widget conflicts
    
    # Show the plot
    plt.show(block=True)
    
    return True

def interactive_data_type_selection():
    """
    Interactive data type selection for phasor plots.
    
    Returns:
        str: Selected data type ('filtered' or 'unfiltered')
    """
    print("\n=== Data Type Selection ===")
    print("Choose which data to use for phasor plots:")
    print("  [1] Filtered data (G/S coordinates) - default")
    print("  [2] Unfiltered data (GU/SU coordinates)")
    print("  [3] Both (will process each file twice)")
    
    while True:
        choice = input("Select option (1, 2, or 3, default: 1): ").strip()
        if choice == "" or choice == "1":
            print("→ Using filtered data (G/S coordinates)")
            return 'filtered'
        elif choice == "2":
            print("→ Using unfiltered data (GU/SU coordinates)")
            return 'unfiltered'
        elif choice == "3":
            print("→ Processing both filtered and unfiltered data")
            return 'both'
        else:
            print("Please enter 1, 2, or 3.")

def interactive_mask_selection(npz_dir):
    """
    Interactive mask source selection for data exploration.
    
    Args:
        npz_dir: Directory containing NPZ files
        
    Returns:
        tuple: (mask_source, selected_mask_name) where mask_source is 'none', 'masked', or None for quit
    """
    print("\n=== Mask Source Selection ===")
    print("Choose mask source for segmentation:")
    print("  [1] No mask (use original data)")
    print("  [2] Use masked NPZ files")
    print("  [q] Quit")
    
    while True:
        choice = input("Select option (1, 2, or q): ").strip().lower()
        
        if choice == 'q':
            print("→ Quitting mask selection")
            return None, None
        elif choice == "1":
            print("→ Using original data (no mask)")
            return 'none', None
        elif choice == "2":
            print("→ Using masked NPZ files")
            # Find available masks in the first NPZ file
            npz_files = []
            for root, _, files in os.walk(npz_dir):
                for file in files:
                    if file.endswith('.npz') and not file.endswith('_segmented.npz'):
                        npz_path = os.path.join(root, file)
                        npz_files.append(npz_path)
                        break  # Just need one file to check masks
                if npz_files:
                    break
            
            if not npz_files:
                print("  No NPZ files found to check for masks")
                return 'none', None
            
            # Load the first NPZ file to check available masks
            data = load_npz_data(npz_files[0])
            if data is None:
                print("  Could not load NPZ file to check masks")
                return 'none', None
            
            # Find mask keys (keys that contain 'mask' in the name)
            mask_keys = [key for key in data.keys() if 'mask' in key.lower()]
            
            if not mask_keys:
                print("  No masks found in NPZ files")
                return 'none', None
            
            print(f"  Found {len(mask_keys)} masks:")
            for i, mask_key in enumerate(mask_keys, 1):
                print(f"    {i}. {mask_key}")
            
            while True:
                mask_choice = input(f"Select mask (1-{len(mask_keys)} or 'all'): ").strip().lower()
                
                if mask_choice == 'all':
                    print("→ Using all available masks")
                    return 'masked', 'all'
                elif mask_choice.isdigit():
                    mask_idx = int(mask_choice) - 1
                    if 0 <= mask_idx < len(mask_keys):
                        selected_mask = mask_keys[mask_idx]
                        print(f"→ Using mask: {selected_mask}")
                        return 'masked', selected_mask
                    else:
                        print(f"Invalid mask number. Please enter 1-{len(mask_keys)} or 'all'.")
                else:
                    print(f"Invalid input. Please enter 1-{len(mask_keys)} or 'all'.")
        else:
            print("Please enter 1, 2, or q.")

def interactive_threshold_selection():
    """
    Interactive thresholding selection for data exploration.
    
    Returns:
        dict: Thresholding configuration with keys 'method', 'value', 'percentile'
    """
    print("\nThresholding options:")
    print("  [1] No threshold (use all data)")
    print("  [2] Manual threshold (enter a specific value)")
    print("  [3] Auto-threshold on combined data (remove bottom 90% of intensity values)")
    print("  [4] Custom auto-threshold on combined data (specify percentile to remove)")
    print("  [5] Individual dataset auto-threshold (remove bottom 90% from each dataset)")
    print("  [6] Custom individual dataset auto-threshold (specify percentile to remove from each dataset)")
    
    while True:
        choice = input("Select option (1-6): ").strip()
        
        if choice == "1":
            print("→ No thresholding applied")
            return {'method': 'none', 'value': None, 'percentile': None}
        elif choice == "2":
            while True:
                try:
                    threshold_value = float(input("Enter threshold value: "))
                    print(f"→ Manual threshold: {threshold_value}")
                    return {'method': 'manual', 'value': threshold_value, 'percentile': None}
                except ValueError:
                    print("Please enter a valid number.")
        elif choice == "3":
            print("→ Auto-threshold on combined data (remove bottom 90%)")
            return {'method': 'auto_combined', 'value': None, 'percentile': 90}
        elif choice == "4":
            while True:
                try:
                    percentile = float(input("Enter percentile to remove (0-100): "))
                    if 0 <= percentile <= 100:
                        print(f"→ Custom auto-threshold on combined data (remove bottom {percentile}%)")
                        return {'method': 'auto_combined', 'value': None, 'percentile': percentile}
                    else:
                        print("Please enter a value between 0 and 100.")
                except ValueError:
                    print("Please enter a valid number.")
        elif choice == "5":
            print("→ Individual dataset auto-threshold (remove bottom 90% from each dataset)")
            return {'method': 'auto_individual', 'value': None, 'percentile': 90}
        elif choice == "6":
            while True:
                try:
                    percentile = float(input("Enter percentile to remove from each dataset (0-100): "))
                    if 0 <= percentile <= 100:
                        print(f"→ Custom individual dataset auto-threshold (remove bottom {percentile}% from each dataset)")
                        return {'method': 'auto_individual', 'value': None, 'percentile': percentile}
                    else:
                        print("Please enter a value between 0 and 100.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            print("Please enter a number between 1 and 6.")

def apply_thresholding(intensity_data, threshold_config):
    """
    Apply thresholding to intensity data based on the configuration.
    
    Args:
        intensity_data: Intensity data array
        threshold_config: Dictionary with thresholding configuration
        
    Returns:
        tuple: (masked_intensity, threshold_desc) where masked_intensity is the thresholded data
    """
    method = threshold_config.get('method', 'none')
    value = threshold_config.get('value')
    percentile = threshold_config.get('percentile')
    
    if method == 'none':
        return intensity_data, "No threshold"
    
    elif method == 'manual':
        if value is None:
            return intensity_data, "No threshold (invalid manual value)"
        mask = intensity_data >= value
        masked_intensity = intensity_data * mask
        return masked_intensity, f"Manual threshold: {value}"
    
    elif method == 'auto_combined':
        if percentile is None:
            return intensity_data, "No threshold (invalid percentile)"
        threshold_value = np.percentile(intensity_data, percentile)
        mask = intensity_data >= threshold_value
        masked_intensity = intensity_data * mask
        return masked_intensity, f"Auto-threshold combined: {percentile}% ({threshold_value:.2f})"
    
    elif method == 'auto_individual':
        if percentile is None:
            return intensity_data, "No threshold (invalid percentile)"
        threshold_value = np.percentile(intensity_data, percentile)
        mask = intensity_data >= threshold_value
        masked_intensity = intensity_data * mask
        return masked_intensity, f"Auto-threshold individual: {percentile}% ({threshold_value:.2f})"
    
    else:
        return intensity_data, "No threshold (unknown method)"

def main(config=None, npz_dir=None, output_dir=None, interactive=True, selected_files=None, data_type='filtered', naming_variables=None, selected_mask_name=None):
    """
    Main execution function for data exploration.
    
    Args:
        config: Configuration dictionary
        npz_dir: Directory containing NPZ files
        output_dir: Main output directory (unused, kept for compatibility)
        interactive: Whether to prompt for user input (default: True)
        selected_files: List of selected NPZ file paths (if None, will find all NPZ files)
        data_type: Data type to use ('filtered' for G/S, 'unfiltered' for GU/SU, or 'both')
        naming_variables: Dictionary containing naming variables for output files (unused)
        selected_mask_name: Name of the selected mask to apply (if any)
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Starting Data Exploration")
    print(f"Input NPZ directory: {npz_dir}")
    
    if not os.path.isdir(npz_dir): 
        print(f"Error: NPZ dir not found: {npz_dir}", file=sys.stderr)
        return False
    
    # Interactive selections if interactive mode is enabled
    mask_source = None
    selected_mask_name = None
    threshold_config = {'method': 'none', 'value': None, 'percentile': None}
    
    if interactive:
        # Data type selection
        if data_type == 'filtered':  # Only prompt if using default
            data_type = interactive_data_type_selection()
        
        # Mask selection
        mask_source, selected_mask_name = interactive_mask_selection(npz_dir)
        if mask_source is None:  # User chose to quit
            print("Exiting data exploration.")
            return True
        
        # Thresholding selection
        threshold_config = interactive_threshold_selection()
    
    # Handle file selection based on the select_files parameter
    # This parameter is passed from the stage class
    if selected_files is None:
        # Find all NPZ files
        npz_files = []
        for root, _, files in os.walk(npz_dir):
            for file in files:
                if file.endswith('.npz') and not file.endswith('_segmented.npz'):
                    npz_path = os.path.join(root, file)
                    npz_files.append(npz_path)
                    
        if not npz_files:
            print(f"No NPZ files found in {npz_dir}")
            return False
            
        print(f"Found {len(npz_files)} NPZ files for data exploration")
        
        # Use all files by default (option 2 behavior)
        selected_files = npz_files
        print(f"Selected {len(selected_files)} files for data exploration")
    else:
        print(f"Using {len(selected_files)} pre-selected files for data exploration")
    
    if not selected_files:
        print("No files selected for data exploration")
        return False
    
    # Determine which data types to process
    data_types_to_process = []
    if data_type == 'both':
        data_types_to_process = ['filtered', 'unfiltered']
    else:
        data_types_to_process = [data_type]
    
    # Process each NPZ file individually
    for npz_path in selected_files:
        print(f"\nProcessing: {os.path.basename(npz_path)}")
        
        for current_data_type in data_types_to_process:
            if data_type == 'both':
                print(f"  Processing {current_data_type} data...")
            
            # Process the NPZ file
            file_data = process_npz_file_for_exploration(npz_path, current_data_type, selected_mask_name)
            if file_data is None:
                print(f"  Skipping {os.path.basename(npz_path)} ({current_data_type}) due to processing error")
                continue
            
            # Add filename to data for display
            file_data['npz_data']['filename'] = os.path.basename(npz_path)
            
            # Apply thresholding to intensity data
            thresholded_intensity, threshold_desc = apply_thresholding(file_data['intensity'], threshold_config)
            file_data['intensity'] = thresholded_intensity
            
            # Create interactive exploration plot
            success = create_interactive_exploration_plot(file_data, current_data_type, threshold_desc)
            if not success:
                print(f"  Failed to create exploration plot for {os.path.basename(npz_path)} ({current_data_type})")
                continue
            
            # Ask if user wants to continue with next data type or file
            if data_type == 'both' and current_data_type == 'filtered':
                continue_choice = input(f"\nContinue to unfiltered data for this file? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    break
            elif len(selected_files) > 1 or (data_type == 'both' and current_data_type == 'unfiltered'):
                continue_choice = input(f"\nContinue to next file? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    return True  # Exit the entire function
    
    return True

if __name__ == "__main__":
    print("This script is intended to be run via the main pipeline")
    print("Please use: python main.py --data-exploration") 