#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Segment From Mask Module for FLIM-FRET Analysis
======================================================

This module provides tools for manual segmentation of phasor data using 
an interactive matplotlib interface with adjustable ellipse parameters.
Unlike the standard ManualSegmentation.py, this module works with NPZ files
that contain full_mask data and multiplies G and S values by the mask
before plotting, allowing segmentation only within the masked regions.

Based on ManualSegmentation.py by Joshua Marcus
"""

import os
import sys
import math
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from matplotlib import colors
from tifffile import imwrite as save_tiff

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
    # For vectorized operations
    points_x = np.asarray(points_x)
    points_y = np.asarray(points_y)
    
    # Calculate semi-axes
    semi_width = width / 2
    semi_height = height / 2
    
    # Pre-compute trig functions
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Translate points
    translated_x = points_x - center_x
    translated_y = points_y - center_y
    
    # Apply rotation transformation
    rotated_x = cos_a * translated_x + sin_a * translated_y
    rotated_y = -sin_a * translated_x + cos_a * translated_y
    
    # Normalize coordinates
    normalized_x = rotated_x / semi_width
    normalized_y = rotated_y / semi_height
    
    # Check which points are inside the ellipse
    return (normalized_x ** 2 + normalized_y ** 2) <= 1

def select_npz_files(npz_files):
    """
    Allow user to select which NPZ files to include in the manual segmentation.
    Uses tkinter GUI if available, otherwise falls back to command line interface.
    
    Args:
        npz_files: List of NPZ file paths
        
    Returns:
        List of selected NPZ file paths
    """
    try:
        # Try to use tkinter if available
        import tkinter as tk
        from tkinter import ttk
        
        # Create tkinter window
        root = tk.Tk()
        root.title("Select NPZ Files for Manual Segmentation From Mask")
        root.geometry("800x600")
        
        # Create frame for instructions
        instruction_frame = ttk.Frame(root, padding="10")
        instruction_frame.pack(fill="x")
        
        ttk.Label(instruction_frame, 
                  text="Select files to include in combined manual segmentation from mask", 
                  font=("Arial", 12, "bold")).pack()
        ttk.Label(instruction_frame, 
                  text="Only masked regions (G*mask, S*mask) will be shown for segmentation", 
                  font=("Arial", 10)).pack()
        
        # Create frame for file list
        list_frame = ttk.Frame(root, padding="10")
        list_frame.pack(fill="both", expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        # Create listbox with checkbuttons
        listbox_frame = ttk.Frame(list_frame)
        listbox_frame.pack(fill="both", expand=True)
        
        # Column headers
        ttk.Label(listbox_frame, text="Select", width=10).grid(row=0, column=0, sticky="w")
        ttk.Label(listbox_frame, text="File Path", width=70).grid(row=0, column=1, sticky="w")
        
        # Variables to track selections
        selections = {}
        
        # Add each file with a checkbox
        for i, npz_path in enumerate(npz_files):
            # Create variable for checkbox
            var = tk.BooleanVar(value=False)
            selections[npz_path] = var
            
            # Add checkbox
            cb = ttk.Checkbutton(listbox_frame, variable=var)
            cb.grid(row=i+1, column=0, sticky="w")
            
            # Add label with file path
            ttk.Label(listbox_frame, text=npz_path).grid(row=i+1, column=1, sticky="w")
        
        # Buttons frame
        button_frame = ttk.Frame(root, padding="10")
        button_frame.pack(fill="x")
        
        # Function to select all
        def select_all():
            for var in selections.values():
                var.set(True)
        
        # Function to deselect all
        def deselect_all():
            for var in selections.values():
                var.set(False)
        
        # Selected files variable
        selected = []
        
        # Function to confirm selection and close window
        def confirm_selection():
            nonlocal selected
            selected = [path for path, var in selections.items() if var.get()]
            root.destroy()
        
        # Add buttons
        ttk.Button(button_frame, text="Select All", command=select_all).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Deselect All", command=deselect_all).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Confirm Selection", command=confirm_selection).pack(side="right", padx=5)
        
        # Run the tkinter event loop
        root.mainloop()
        
        return selected
        
    except ImportError:
        # Fallback to command line interface if tkinter is not available
        print("\ntkinter module not available. Using command line interface for file selection.")
        print(f"Found {len(npz_files)} NPZ files. Select files to include in manual segmentation from mask:")
        
        # Show list of files with indices
        for i, path in enumerate(npz_files):
            print(f"[{i+1}] {path}")
        
        print("\nEnter file numbers to select (comma-separated, e.g. '1,3,5-7'), or 'all' for all files:")
        selection_input = input("> ").strip()
        
        selected = []
        if selection_input.lower() == 'all':
            selected = npz_files
        else:
            # Process comma-separated list with possible ranges (e.g., "1,3,5-7")
            parts = selection_input.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    # Handle range (e.g., "5-7")
                    try:
                        start, end = part.split('-')
                        start_idx = int(start.strip()) - 1  # Convert to 0-based index
                        end_idx = int(end.strip())          # Inclusive end
                        for idx in range(start_idx, end_idx):
                            if 0 <= idx < len(npz_files):
                                selected.append(npz_files[idx])
                    except (ValueError, IndexError):
                        print(f"Warning: Invalid range '{part}', skipping")
                else:
                    # Handle single number
                    try:
                        idx = int(part) - 1  # Convert to 0-based index
                        if 0 <= idx < len(npz_files):
                            selected.append(npz_files[idx])
                        else:
                            print(f"Warning: Index {part} out of range, skipping")
                    except ValueError:
                        print(f"Warning: Invalid input '{part}', skipping")
        
        print(f"\nSelected {len(selected)} files for manual segmentation from mask")
        return selected

def process_combined_npz_files(npz_files, segmented_dir, masks_dir, plots_dir, lifetime_dir=None):
    """
    Process multiple NPZ files for combined manual segmentation from mask.
    
    Args:
        npz_files: List of NPZ file paths
        segmented_dir: Directory to save segmented NPZ files
        masks_dir: Directory to save mask files
        plots_dir: Directory to save plots
        lifetime_dir: Directory to save lifetime images (optional)
        
    Returns:
        bool: Success status
    """
    print(f"\n=== Manual Segmentation From Mask for {len(npz_files)} Combined Files ===")
    
    # Load and combine data from all files
    print("Loading NPZ files...")
    file_data_mapping = {}
    all_g_data = []
    all_s_data = []
    all_intensity_data = []
    
    for npz_path in npz_files:
        data = load_npz_data(npz_path)
        if data is None:
            print(f"Warning: Could not load {npz_path}")
            continue
            
        # Extract data
        g_data = data.get('G', data.get('g_data', None))
        s_data = data.get('S', data.get('s_data', None))
        intensity = data.get('A', data.get('intensity', None))
        lifetime = data.get('lifetime', None)
        
        # *** KEY MODIFICATION: Apply full_mask to G and S data ***
        full_mask = data.get('full_mask', None)
        if full_mask is None:
            print(f"Warning: No full_mask found in {npz_path}")
            print(f"Available keys: {list(data.keys())}")
            continue
            
        print(f"  {os.path.basename(npz_path)}: mask has {np.sum(full_mask)} selected pixels out of {full_mask.size} total")
        
        # Apply mask to G and S data
        g_data_masked = g_data * full_mask
        s_data_masked = s_data * full_mask
        
        if g_data is None or s_data is None or intensity is None:
            print(f"Warning: Missing required data in {npz_path}")
            continue
            
        # Flatten arrays (using masked data for G and S)
        g_flat = g_data_masked.flatten()
        s_flat = s_data_masked.flatten()
        intensity_flat = intensity.flatten()
        
        # Store data for this file
        file_data_mapping[npz_path] = {
            'npz_data': data,
            'g_data': g_data,  # Original G data
            's_data': s_data,  # Original S data
            'g_data_masked': g_data_masked,  # Masked G data
            's_data_masked': s_data_masked,  # Masked S data
            'intensity': intensity,
            'lifetime': lifetime,
            'full_mask': full_mask,
            'g_flat': g_flat,  # Flattened masked G data
            's_flat': s_flat,  # Flattened masked S data
            'intensity_flat': intensity_flat
        }
        
        # Add to combined data (using masked data)
        all_g_data.append(g_flat)
        all_s_data.append(s_flat)
        all_intensity_data.append(intensity_flat)
    
    if not file_data_mapping:
        print("Error: No valid NPZ files with full_mask could be loaded.")
        return False
    
    # Combine all data
    all_g = np.concatenate(all_g_data)
    all_s = np.concatenate(all_s_data)
    all_intensity = np.concatenate(all_intensity_data)
    
    # Remove zero values from masked data (since masked regions are set to 0)
    non_zero_mask = (all_g != 0) | (all_s != 0)
    all_g = all_g[non_zero_mask]
    all_s = all_s[non_zero_mask]
    all_intensity = all_intensity[non_zero_mask]
    
    print(f"Loaded {len(file_data_mapping)} files with {len(all_g)} non-zero masked pixels")
    
    # Apply thresholding before creating the plot (like in phasor_visualization.py)
    print("\nThresholding options:")
    print("  [1] No threshold (use all masked data)")
    print("  [2] Manual threshold (enter a specific value)")
    print("  [3] Auto-threshold on combined data (remove bottom 90% of intensity values)")
    print("  [4] Custom auto-threshold on combined data (specify percentile to remove)")
    print("  [5] Individual dataset auto-threshold (remove bottom 90% from each dataset)")
    print("  [6] Custom individual dataset auto-threshold (specify percentile to remove from each dataset)")
    
    while True:
        threshold_choice = input("Select thresholding option: ").strip()
        
        if threshold_choice == '1':
            # No threshold
            threshold = 0
            auto_percentile = None
            individual_percentile = None
            threshold_desc = "No threshold (masked data only)"
            break
            
        elif threshold_choice == '2':
            # Manual threshold
            while True:
                try:
                    threshold = float(input("Enter intensity threshold value: "))
                    if threshold < 0:
                        print("Threshold must be non-negative. Please try again.")
                        continue
                    auto_percentile = None
                    individual_percentile = None
                    threshold_desc = f"Manual threshold: {threshold} (masked data only)"
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    
        elif threshold_choice == '3':
            # Auto-threshold with default 90%
            threshold = 0
            auto_percentile = 90
            individual_percentile = None
            threshold_desc = f"Auto threshold ({auto_percentile}%, masked data only)"
            break
            
        elif threshold_choice == '4':
            # Custom auto-threshold on combined data
            while True:
                try:
                    percentile = float(input("Enter percentile threshold (1-99): "))
                    if percentile < 1 or percentile > 99:
                        print("Percentile must be between 1 and 99. Please try again.")
                        continue
                    threshold = 0
                    auto_percentile = percentile
                    individual_percentile = None
                    threshold_desc = f"Auto threshold ({auto_percentile}%, masked data only)"
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    
        elif threshold_choice == '5':
            # Individual dataset auto-threshold (90%)
            threshold = 0
            auto_percentile = None
            individual_percentile = 90
            threshold_desc = f"Individual auto threshold ({individual_percentile}%, masked data only)"
            break
            
        elif threshold_choice == '6':
            # Custom individual dataset auto-threshold
            while True:
                try:
                    percentile = float(input("Enter percentile threshold for individual datasets (1-99): "))
                    if percentile < 1 or percentile > 99:
                        print("Percentile must be between 1 and 99. Please try again.")
                        continue
                    threshold = 0
                    auto_percentile = None
                    individual_percentile = percentile
                    threshold_desc = f"Individual auto threshold ({individual_percentile}%, masked data only)"
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            print("Invalid choice. Please select 1-6.")
    
    # Apply thresholding to the already-masked data
    print(f"\nApplying {threshold_desc}...")
    
    if individual_percentile is not None:
        # Apply individual thresholding to each file
        filtered_g = []
        filtered_s = []
        filtered_intensity = []
        
        for npz_path, data in file_data_mapping.items():
            # Use masked data for thresholding
            g_flat = data['g_flat']
            s_flat = data['s_flat']
            intensity_flat = data['intensity_flat']
            
            # Only consider non-zero masked pixels
            non_zero_mask = (g_flat != 0) | (s_flat != 0)
            g_non_zero = g_flat[non_zero_mask]
            s_non_zero = s_flat[non_zero_mask]
            intensity_non_zero = intensity_flat[non_zero_mask]
            
            if len(intensity_non_zero) > 0:
                # Calculate threshold for this specific file
                file_threshold = np.percentile(intensity_non_zero, individual_percentile)
                
                # Create mask for this file
                mask = intensity_non_zero >= file_threshold
                
                if np.sum(mask) > 0:
                    filtered_g.append(g_non_zero[mask])
                    filtered_s.append(s_non_zero[mask])
                    filtered_intensity.append(intensity_non_zero[mask])
                    
                    print(f"  {os.path.basename(npz_path)}: kept {np.sum(mask)} of {len(mask)} masked pixels ({np.sum(mask)/len(mask)*100:.1f}%)")
        
        if filtered_g:
            all_g = np.concatenate(filtered_g)
            all_s = np.concatenate(filtered_s)
            all_intensity = np.concatenate(filtered_intensity)
            print(f"  Total: kept {len(all_g)} pixels from masked regions")
        else:
            print("Warning: No data points remain after individual thresholding on masked data")
            return False
            
    elif auto_percentile is not None:
        # Apply auto-thresholding to combined masked data
        combined_threshold = np.percentile(all_intensity, auto_percentile)
        mask = all_intensity >= combined_threshold
        
        all_g = all_g[mask]
        all_s = all_s[mask]
        all_intensity = all_intensity[mask]
        
        print(f"  Applied threshold of {combined_threshold:.2f}: kept {len(all_g)} of {len(mask)} masked pixels ({len(all_g)/len(mask)*100:.1f}%)")
        
    elif threshold > 0:
        # Apply manual threshold to masked data
        mask = all_intensity >= threshold
        all_g = all_g[mask]
        all_s = all_s[mask]
        all_intensity = all_intensity[mask]
        
        print(f"  Applied threshold of {threshold}: kept {len(all_g)} of {len(mask)} masked pixels ({len(all_g)/len(mask)*100:.1f}%)")
    
    # Create the interactive plot with thresholded masked data using high-quality formatting
    print(f"\nCreating interactive plot with {len(all_g)} thresholded masked pixels...")
    
    # Remove any NaN values
    mask = ~(np.isnan(all_g) | np.isnan(all_s) | np.isnan(all_intensity))
    all_g = all_g[mask]
    all_s = all_s[mask]
    all_intensity = all_intensity[mask]
    
    # Check for empty data
    if len(all_g) == 0 or len(all_s) == 0:
        print("Warning: No valid data points after thresholding masked data")
        return False
    
    # Create a universal circle for reference
    x = np.linspace(0, 1.0, 100)
    y = np.linspace(0, 0.7, 100)
    X, Y = np.meshgrid(x, y)
    F = (X**2 + Y**2 - X)  # Universal circle equation
    
    # Set plot limits
    x_scale = [-0.005, 1.005]
    y_scale = [0, 0.7]
    
    # Calculate bin widths using IQR or use fixed bins
    iqr_x = np.percentile(all_g, 75) - np.percentile(all_g, 25)
    bin_width_x = 2 * iqr_x * (len(all_g) ** (-1/3))
    bin_width_x = np.nan_to_num(bin_width_x)

    iqr_y = np.percentile(all_s, 75) - np.percentile(all_s, 25)
    bin_width_y = 2 * iqr_y * (len(all_s) ** (-1/3))
    bin_width_y = np.nan_to_num(bin_width_y)
    
    # Set a small threshold for bin width to detect impractical values
    min_bin_width = np.finfo(float).eps
    
    # Calculate number of bins, or set manually if bin widths are too small
    if bin_width_x <= min_bin_width or bin_width_y <= min_bin_width:
        num_bins_x = 100  # Default number of bins
        num_bins_y = 100
    else:
        num_bins_x = int(np.ceil((np.max(all_g) - np.min(all_g)) / bin_width_x)) // 2
        num_bins_y = int(np.ceil((np.max(all_s) - np.min(all_s)) / bin_width_y)) // 2
        # Ensure a reasonable number of bins
        num_bins_x = max(50, min(200, num_bins_x))
        num_bins_y = max(50, min(200, num_bins_y))
    
    # Create 2D histogram
    hist_vals, _, _ = np.histogram2d(all_g, all_s, bins=(num_bins_x, num_bins_y), weights=all_intensity)
    vmax = hist_vals.max()
    vmin = hist_vals.min()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate the 2D histogram with high-quality formatting
    h = ax.hist2d(all_g, all_s, 
                bins=(num_bins_x, num_bins_y), 
                weights=all_intensity, 
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
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f"Manual Segmentation From Mask - Combined Dataset ({len(npz_files)} files)\n{threshold_desc}\n({timestamp})")
    
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
    ax.add_artist(ellipse)
    
    # Add center marker
    center_point, = ax.plot([center_x], [center_y], 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
    
    # Create sliders (no threshold slider needed)
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
    
    # Function to apply the ellipse and save segmentation
    def apply_segmentation(event):
        # Get final parameters
        center_x = s_center_x.val
        center_y = s_center_y.val
        width = s_width.val
        height = s_height.val
        angle = s_angle.val
        angle_rad = np.radians(angle)
        
        # Save current state of the plot
        plot_filename = f"manual_segmentation_from_mask_combined_{len(npz_files)}_files.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved combined plot to: {plot_path}")
        
        # Process each file individually
        print(f"\nApplying manual segmentation from mask to {len(file_data_mapping)} files...")
        
        # Metadata for all files
        common_metadata = {
            'data_type': 'manually_segmented_from_mask',
            'ellipse_center': [center_x, center_y],
            'ellipse_width': width,
            'ellipse_height': height,
            'ellipse_angle_degrees': angle,
            'threshold_desc': threshold_desc,
            'ellipse_cond_center': [center_x, center_y],
            'based_on_mask': True
        }
        
        for npz_path, data in file_data_mapping.items():
            print(f"Processing: {os.path.basename(npz_path)}")
            
            npz_data = data['npz_data']
            g_data = data['g_data']  # Original G data
            s_data = data['s_data']  # Original S data
            g_data_masked = data['g_data_masked']  # Masked G data
            s_data_masked = data['s_data_masked']  # Masked S data
            intensity = data['intensity']
            lifetime = data['lifetime']
            original_mask = data['full_mask']
            
            # Apply the same thresholding that was used for the plot
            if individual_percentile is not None:
                # Apply individual thresholding
                g_flat = g_data_masked.flatten()
                s_flat = s_data_masked.flatten()
                intensity_flat = intensity.flatten()
                
                non_zero_mask = (g_flat != 0) | (s_flat != 0)
                intensity_non_zero = intensity_flat[non_zero_mask]
                
                if len(intensity_non_zero) > 0:
                    file_threshold = np.percentile(intensity_non_zero, individual_percentile)
                    threshold_mask = intensity >= file_threshold
                else:
                    threshold_mask = np.ones_like(intensity, dtype=bool)
                    
            elif auto_percentile is not None:
                # Apply auto thresholding
                combined_threshold = np.percentile(all_intensity, auto_percentile)
                threshold_mask = intensity >= combined_threshold
            elif threshold > 0:
                # Apply manual threshold
                threshold_mask = intensity >= threshold
            else:
                # No threshold
                threshold_mask = np.ones_like(intensity, dtype=bool)
            
            # Create ellipse mask: only consider pixels within the original mask AND the threshold
            ellipse_mask = np.zeros_like(g_data, dtype=bool)
            
            # Get coordinates of pixels that are in the original mask and pass threshold
            valid_mask = (original_mask > 0) & threshold_mask
            mask_indices = np.where(valid_mask)
            
            if len(mask_indices[0]) == 0:
                print(f"  Warning: No valid pixels for {os.path.basename(npz_path)}")
                continue
            
            # Use the original (unmasked) G and S values for ellipse calculation
            mask_g_values = g_data[mask_indices]
            mask_s_values = s_data[mask_indices]
            
            # Check which points are inside the ellipse
            inside_ellipse = are_points_inside_ellipse(
                mask_g_values, mask_s_values, 
                center_x, center_y, 
                width, height, angle_rad
            )
            
            # Set mask values
            ellipse_mask[mask_indices[0][inside_ellipse], mask_indices[1][inside_ellipse]] = True
            
            # Create binary mask (0 = background, 1 = selected region)
            # This includes the intersection of original mask, threshold, and ellipse
            full_mask = np.zeros_like(g_data, dtype=np.int32)
            full_mask[ellipse_mask] = 1  # Selected region = 1
            
            # Create output directories (no relative path structure for manual segmentation)
            npz_output_dir = segmented_dir
            mask_output_dir = masks_dir
            plot_output_dir = plots_dir
            os.makedirs(npz_output_dir, exist_ok=True)
            os.makedirs(mask_output_dir, exist_ok=True)
            os.makedirs(plot_output_dir, exist_ok=True)
            
            if lifetime_dir and lifetime is not None:
                lifetime_output_dir = lifetime_dir
                os.makedirs(lifetime_output_dir, exist_ok=True)
            
            # Save the data to a segmented NPZ file
            base_name = os.path.basename(npz_path)
            seg_name = os.path.splitext(base_name)[0] + '_manually_segmented_from_mask.npz'
            seg_path = os.path.join(npz_output_dir, seg_name)
            
            # Combine all data for NPZ file
            save_data = {}
            for key, value in npz_data.items():
                save_data[key] = value
            
            # Add segmentation data
            save_data['mask_component_0'] = ~ellipse_mask  # Background (not selected)
            save_data['mask_component_1'] = ellipse_mask   # Selected region
            save_data['full_mask'] = full_mask  # New mask (intersection of original mask and ellipse)
            save_data['original_mask'] = original_mask  # Store the original mask
            
            # Add metadata
            save_data['metadata'] = {
                **common_metadata,
                'source_file': npz_path,
                'pixels_selected': np.sum(ellipse_mask),
                'pixels_in_original_mask': np.sum(original_mask),
                'pixels_thresholded': np.sum(threshold_mask),
                'total_pixels': g_data.size,
                'mask_type': 'binary'
            }
            
            # Save NPZ file
            np.savez_compressed(seg_path, **save_data)
            print(f"  Saved segmented NPZ: {seg_path}")
            
            # Save mask as TIFF
            mask_name = os.path.splitext(base_name)[0] + '_manually_segmented_from_mask.tiff'
            mask_path = os.path.join(mask_output_dir, mask_name)
            save_tiff(mask_path, full_mask)
            print(f"  Saved mask: {mask_path}")
            
            # Save lifetime image if available
            if lifetime_dir and lifetime is not None:
                lifetime_name = os.path.splitext(base_name)[0] + '_manually_segmented_from_mask_lifetime.tiff'
                lifetime_path = os.path.join(lifetime_output_dir, lifetime_name)
                
                # Apply mask to lifetime
                masked_lifetime = lifetime.copy()
                masked_lifetime[full_mask == 0] = 0  # Background (not selected)
                # Keep original values for selected region (mask == 1)
                
                save_tiff(lifetime_path, masked_lifetime)
                print(f"  Saved lifetime image: {lifetime_path}")
        
        print(f"\nManual segmentation from mask completed for {len(file_data_mapping)} files!")
        plt.close()
    
    # Create apply button
    ax_apply = plt.axes([0.8, 0.02, 0.1, 0.04])
    button_apply = Button(ax_apply, 'Apply')
    button_apply.on_clicked(apply_segmentation)
    
    # Create cancel button
    ax_cancel = plt.axes([0.65, 0.02, 0.1, 0.04])
    button_cancel = Button(ax_cancel, 'Cancel')
    
    def cancel_segmentation(event):
        print("\nManual segmentation from mask cancelled by user.")
        plt.close()
    
    button_cancel.on_clicked(cancel_segmentation)
    
    # Show the plot
    plt.show()
    
    return True

def main(config, npz_dir, segmented_dir, plots_dir, lifetime_dir=None, interactive=True):
    """
    Main function for manual segmentation from mask.
    
    Args:
        config: Configuration dictionary
        npz_dir: Directory containing NPZ files with full_mask data
        segmented_dir: Directory to save segmented NPZ files
        plots_dir: Directory to save plots
        lifetime_dir: Directory to save lifetime images (optional)
        interactive: Whether to run interactively (default: True)
        
    Returns:
        bool: Success status
    """
    print("=== Manual Segmentation From Mask ===")
    print(f"NPZ directory: {npz_dir}")
    print(f"Output directory: {segmented_dir}")
    print(f"Plots directory: {plots_dir}")
    if lifetime_dir:
        print(f"Lifetime directory: {lifetime_dir}")
    
    # Create output directories
    os.makedirs(segmented_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a masks subdirectory
    masks_dir = os.path.join(segmented_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    if lifetime_dir:
        os.makedirs(lifetime_dir, exist_ok=True)
    
    # Find all NPZ files
    npz_files = []
    for root, dirs, files in os.walk(npz_dir):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))
    
    if not npz_files:
        print(f"No NPZ files found in {npz_dir}")
        return False
    
    npz_files.sort()
    print(f"Found {len(npz_files)} NPZ files")
    
    # Check if files have full_mask data
    valid_npz_files = []
    for npz_path in npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True)
            if 'full_mask' in data:
                valid_npz_files.append(npz_path)
            else:
                print(f"Warning: {os.path.basename(npz_path)} does not contain full_mask data")
        except Exception as e:
            print(f"Warning: Could not load {npz_path}: {e}")
    
    if not valid_npz_files:
        print("No NPZ files with full_mask data found")
        return False
    
    print(f"Found {len(valid_npz_files)} NPZ files with full_mask data")
    
    if interactive:
        # Allow user to select files
        selected_files = select_npz_files(valid_npz_files)
        if not selected_files:
            print("No files selected. Exiting manual segmentation from mask.")
            return False
    else:
        # Use all valid files
        selected_files = valid_npz_files
    
    # Process the selected files
    success = process_combined_npz_files(selected_files, segmented_dir, masks_dir, plots_dir, lifetime_dir)
    
    if success:
        print(f"\n✅ Manual segmentation from mask completed successfully!")
        print(f"   Output files saved to: {segmented_dir}")
        print(f"   Plots saved to: {plots_dir}")
        if lifetime_dir:
            print(f"   Lifetime images saved to: {lifetime_dir}")
    else:
        print(f"\n❌ Manual segmentation from mask failed")
    
    return success

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manual segmentation from mask for FLIM-FRET data")
    parser.add_argument("npz_dir", help="Directory containing NPZ files with full_mask data")
    parser.add_argument("segmented_dir", help="Directory to save segmented NPZ files")
    parser.add_argument("plots_dir", help="Directory to save plots")
    parser.add_argument("--lifetime_dir", help="Directory to save lifetime images (optional)")
    
    args = parser.parse_args()
    
    success = main(
        config=None,
        npz_dir=args.npz_dir,
        segmented_dir=args.segmented_dir,
        plots_dir=args.plots_dir,
        lifetime_dir=args.lifetime_dir,
        interactive=True
    )
    
    sys.exit(0 if success else 1) 