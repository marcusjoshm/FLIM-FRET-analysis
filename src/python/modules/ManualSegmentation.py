#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Segmentation Module for FLIM-FRET Analysis
=================================================

This module provides tools for manual segmentation of phasor data using 
an interactive matplotlib interface with adjustable ellipse parameters.

Created by Joshua Marcus
"""

import os
import sys
import math
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Ellipse
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
        root.title("Select NPZ Files for Manual Segmentation")
        root.geometry("800x600")
        
        # Create frame for instructions
        instruction_frame = ttk.Frame(root, padding="10")
        instruction_frame.pack(fill="x")
        
        ttk.Label(instruction_frame, 
                  text="Select files to include in combined manual segmentation", 
                  font=("Arial", 12, "bold")).pack()
        ttk.Label(instruction_frame, 
                  text="All selected datasets will be combined into a single graph for segmentation", 
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
        print(f"Found {len(npz_files)} NPZ files. Select files to include in manual segmentation:")
        
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
        
        print(f"\nSelected {len(selected)} files for manual segmentation")
        return selected

def process_combined_npz_files(npz_files, segmented_dir, masks_dir, plots_dir, lifetime_dir=None):
    """
    Process multiple NPZ files as a combined dataset for manual segmentation.
    
    Args:
        npz_files: List of paths to NPZ files to process as a single combined dataset
        segmented_dir: Output directory for segmented NPZ data
        masks_dir: Output directory for binary masks
        plots_dir: Output directory for plots
        lifetime_dir: Output directory for lifetime images (if any)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not npz_files:
            print("No NPZ files provided for manual segmentation")
            return False
            
        # Create necessary output directories
        os.makedirs(segmented_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        if lifetime_dir:
            os.makedirs(lifetime_dir, exist_ok=True)
        
        # Load and combine data from all selected NPZ files
        print("Loading and combining data from selected files...")
        
        combined_g = []
        combined_s = []
        combined_intensity = []
        file_data_mapping = {}  # Maps file paths to their data for later processing
        
        for npz_path in npz_files:
            # Load NPZ data
            npz_data = load_npz_data(npz_path)
            if npz_data is None:
                print(f"  Error: Could not load NPZ file: {npz_path}")
                continue
            
            # Extract data from NPZ
            g_data = npz_data.get('G', None)
            s_data = npz_data.get('S', None)
            intensity = npz_data.get('intensity', npz_data.get('A', None))
            lifetime = npz_data.get('T', None)
            
            if g_data is None or s_data is None or intensity is None:
                print(f"  Error: Missing required data in NPZ file: {npz_path}")
                continue
            
            # Store original data for later processing
            file_data_mapping[npz_path] = {
                'npz_data': npz_data,
                'g_data': g_data,
                's_data': s_data,
                'intensity': intensity,
                'lifetime': lifetime
            }
            
            # Flatten data for combined phasor plot
            mask = intensity > 0  # Basic mask to filter out zero intensity
            g_flat = g_data[mask].flatten()
            s_flat = s_data[mask].flatten()
            intensity_flat = intensity[mask].flatten()
            
            # Add to combined datasets
            combined_g.append(g_flat)
            combined_s.append(s_flat)
            combined_intensity.append(intensity_flat)
        
        # Concatenate all data
        if not combined_g:
            print("No valid data found in selected NPZ files")
            return False
            
        all_g = np.concatenate(combined_g)
        all_s = np.concatenate(combined_s)
        all_intensity = np.concatenate(combined_intensity)
        
        print(f"Combined dataset contains {len(all_g)} data points")
        
        # Setup initial ellipse parameters
        # Default starting values
        center_x = 0.3
        center_y = 0.45
        width = 0.1
        height = 0.05
        angle = 0
            
        # For applying intensity threshold
        threshold = 0.0  # Default threshold
        
        # Setup the figure and plot initial histogram
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.35)  # Make room for sliders
            
        # Create 2D histogram of the combined phasor data
        h = ax.hist2d(
            all_g, all_s, 
            bins=100, 
            range=[[0, 1], [0, 1]], 
            cmap='nipy_spectral', 
            norm=plt.cm.colors.LogNorm()
        )
        
        # Add colorbar
        plt.colorbar(h[3], ax=ax, format=plt.matplotlib.ticker.LogFormatter(10, labelOnlyBase=True))
            
        # Set up plot properties
        ax.set_xlabel('G')
        ax.set_ylabel('S')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Manual Segmentation - Combined Dataset ({len(npz_files)} files)")
            
        # Add semicircle (universal circle) for reference
        theta = np.linspace(0, np.pi, 100)
        x = 0.5 + 0.5 * np.cos(theta)
        y = 0.5 * np.sin(theta)
        ax.plot(x, y, 'k-', lw=1, alpha=0.5)
        
        # Initial ellipse
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
        center_point, = ax.plot(center_x, center_y, 'bo', markersize=5)
            
        # Create axes for sliders
        ax_center_x = plt.axes([0.15, 0.25, 0.7, 0.03])
        ax_center_y = plt.axes([0.15, 0.20, 0.7, 0.03])
        ax_width = plt.axes([0.15, 0.15, 0.7, 0.03])
        ax_height = plt.axes([0.15, 0.10, 0.7, 0.03])
        ax_angle = plt.axes([0.15, 0.05, 0.7, 0.03])
        ax_threshold = plt.axes([0.15, 0.01, 0.7, 0.03])
        
        # Create sliders
        s_center_x = Slider(ax_center_x, 'Center X', 0.0, 1.0, valinit=center_x)
        s_center_y = Slider(ax_center_y, 'Center Y', 0.0, 1.0, valinit=center_y)
        s_width = Slider(ax_width, 'Width', 0.001, 0.5, valinit=width)
        s_height = Slider(ax_height, 'Height', 0.001, 0.5, valinit=height)
        s_angle = Slider(ax_angle, 'Angle', -180, 180, valinit=angle)
        s_threshold = Slider(ax_threshold, 'Threshold', 0.0, 0.5, valinit=threshold)
            
        # Create button axes and button
        ax_apply = plt.axes([0.15, 0.30, 0.15, 0.04])
        apply_button = Button(ax_apply, 'Apply & Save')
            
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
            center_point.set_data(center_x, center_y)
            
            # Redraw the figure
            fig.canvas.draw_idle()
        
        # Function to update the threshold and replot histogram
        def update_threshold(val):
            # Get threshold value
            threshold = s_threshold.val
            
            # Apply threshold to all data points
            filtered_g = []
            filtered_s = []
            
            for i, npz_path in enumerate(npz_files):
                if npz_path not in file_data_mapping:
                    continue
                    
                data = file_data_mapping[npz_path]
                intensity = data['intensity']
                g_data = data['g_data']
                s_data = data['s_data']
                
                # Apply threshold
                mask = intensity > (np.max(intensity) * threshold if threshold > 0 else 0)
                g_flat = g_data[mask].flatten()
                s_flat = s_data[mask].flatten()
                
                if len(g_flat) > 0:
                    filtered_g.append(g_flat)
                    filtered_s.append(s_flat)
            
            # Concatenate all filtered data
            if not filtered_g:
                print("Warning: No data points remain after applying threshold")
                return
                
            all_filtered_g = np.concatenate(filtered_g)
            all_filtered_s = np.concatenate(filtered_s)
            
            # Clear previous histogram
            ax.clear()
            
            # Plot new histogram with filtered data
            h = ax.hist2d(
                all_filtered_g, all_filtered_s, 
                bins=100, 
                range=[[0, 1], [0, 1]], 
                cmap='nipy_spectral', 
                norm=plt.cm.colors.LogNorm()
            )
            
            # Add colorbar
            plt.colorbar(h[3], ax=ax, format=plt.matplotlib.ticker.LogFormatter(10, labelOnlyBase=True))
            
            # Set up plot properties
            ax.set_xlabel('G')
            ax.set_ylabel('S')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"Manual Segmentation - Combined Dataset ({len(npz_files)} files)")
            
            # Add semicircle (universal circle) for reference
            theta = np.linspace(0, np.pi, 100)
            x = 0.5 + 0.5 * np.cos(theta)
            y = 0.5 * np.sin(theta)
            ax.plot(x, y, 'k-', lw=1, alpha=0.5)
            
            # Redraw ellipse
            ellipse = Ellipse(
                xy=(s_center_x.val, s_center_y.val),
                width=s_width.val,
                height=s_height.val,
                angle=s_angle.val,
                fill=False,
                color='blue',
                linewidth=2
            )
            ax.add_artist(ellipse)
            
            # Add center marker
            center_point, = ax.plot(s_center_x.val, s_center_y.val, 'bo', markersize=5)
            
            # Redraw the figure
            fig.canvas.draw_idle()
        
        # Function to apply the ellipse and save segmentation
        def apply_segmentation(event):
            # Get final parameters
            center_x = s_center_x.val
            center_y = s_center_y.val
            width = s_width.val
            height = s_height.val
            angle = s_angle.val
            threshold = s_threshold.val
            angle_rad = np.radians(angle)
            
            # Save current state of the plot
            plot_filename = f"manual_segmentation_combined_{len(npz_files)}_files.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved combined plot to: {plot_path}")
            
            # Process each file individually
            print(f"\nApplying manual segmentation to {len(file_data_mapping)} files...")
            
            # Metadata for all files
            common_metadata = {
                'data_type': 'manually_segmented',
                'ellipse_center': [center_x, center_y],
                'ellipse_width': width,
                'ellipse_height': height,
                'ellipse_angle_degrees': angle,
                'intensity_threshold': threshold,
                'ellipse_cond_center': [center_x, center_y]
            }
            
            for npz_path, data in file_data_mapping.items():
                print(f"Processing: {os.path.basename(npz_path)}")
                
                npz_data = data['npz_data']
                g_data = data['g_data']
                s_data = data['s_data']
                intensity = data['intensity']
                lifetime = data['lifetime']
                
                # Create ellipse mask
                mask = intensity > (np.max(intensity) * threshold if threshold > 0 else 0)
                ellipse_mask = np.zeros_like(g_data, dtype=bool)
                
                # Get coordinates of all pixels that passed intensity threshold
                mask_indices = np.where(mask)
                mask_g_values = g_data[mask_indices]
                mask_s_values = s_data[mask_indices]
                
                if len(mask_g_values) == 0:
                    print(f"  Warning: No pixels passed threshold for {os.path.basename(npz_path)}")
                    continue
                
                # Check which points are inside the ellipse
                inside_ellipse = are_points_inside_ellipse(
                    mask_g_values, mask_s_values, 
                    center_x, center_y, 
                    width, height, angle_rad
                )
                
                # Set mask values
                ellipse_mask[mask_indices[0][inside_ellipse], mask_indices[1][inside_ellipse]] = True
                
                # Create full mask (1 = background, 2 = condensed phase - matches the GMM segmentation format)
                full_mask = np.zeros_like(g_data, dtype=np.int32)
                full_mask[ellipse_mask] = 2  # Condensed phase = 2
                full_mask[mask & ~ellipse_mask] = 1  # Everything else that passed threshold = 1
                
                # Preserve relative path structure
                rel_path = os.path.relpath(os.path.dirname(npz_path), os.path.dirname(segmented_dir))
                if rel_path == '.':
                    rel_path = ''
                
                # Create output subdirectories for this file's relative path
                npz_output_dir = os.path.join(segmented_dir, rel_path)
                mask_output_dir = os.path.join(masks_dir, rel_path)
                plot_output_dir = os.path.join(plots_dir, rel_path)
                os.makedirs(npz_output_dir, exist_ok=True)
                os.makedirs(mask_output_dir, exist_ok=True)
                os.makedirs(plot_output_dir, exist_ok=True)
                
                if lifetime_dir and lifetime is not None:
                    lifetime_output_dir = os.path.join(lifetime_dir, rel_path)
                    os.makedirs(lifetime_output_dir, exist_ok=True)
                
                # Save the data to a segmented NPZ file
                base_name = os.path.basename(npz_path)
                seg_name = os.path.splitext(base_name)[0] + '_manually_segmented.npz'
                seg_path = os.path.join(npz_output_dir, seg_name)
                
                # Combine all data for NPZ file
                save_data = {}
                for key, value in npz_data.items():
                    save_data[key] = value
                
                # Add segmentation data
                save_data['mask_component_0'] = ellipse_mask
                save_data['full_mask'] = full_mask
                
                # Add metadata
                for key, value in common_metadata.items():
                    save_data[key] = value
                
                try:
                    # Save NPZ file
                    np.savez_compressed(seg_path, **save_data)
                    print(f"  Saved segmented NPZ to: {seg_path}")
                    
                    # Save binary mask as TIFF
                    mask_filename = os.path.splitext(base_name)[0] + '_mask.tiff'
                    mask_path = os.path.join(mask_output_dir, mask_filename)
                    save_tiff(mask_path, ellipse_mask.astype(np.uint8) * 255)
                    print(f"  Saved binary mask to: {mask_path}")
                    
                    # Save individual plot
                    indiv_plot_filename = os.path.splitext(base_name)[0] + '_manual_segmentation.png'
                    indiv_plot_path = os.path.join(plot_output_dir, indiv_plot_filename)
                    
                    # Create individual plot for this file
                    fig_indiv, ax_indiv = plt.subplots(figsize=(8, 6))
                    mask_to_plot = mask.copy()
                    h_indiv = ax_indiv.hist2d(
                        g_data[mask_to_plot].flatten(), s_data[mask_to_plot].flatten(), 
                        bins=100, 
                        range=[[0, 1], [0, 1]], 
                        cmap='nipy_spectral', 
                        norm=plt.cm.colors.LogNorm()
                    )
                    plt.colorbar(h_indiv[3], ax=ax_indiv)
                    
                    # Plot semicircle
                    ax_indiv.plot(x, y, 'k-', lw=1, alpha=0.5)
                    
                    # Plot ellipse
                    ell = Ellipse(
                        xy=(center_x, center_y),
                        width=width,
                        height=height,
                        angle=angle,
                        fill=False,
                        color='blue',
                        linewidth=2
                    )
                    ax_indiv.add_artist(ell)
                    
                    # Set up plot
                    ax_indiv.set_xlabel('G')
                    ax_indiv.set_ylabel('S')
                    ax_indiv.set_xlim(0, 1)
                    ax_indiv.set_ylim(0, 1)
                    ax_indiv.set_title(f"Manual Segmentation - {os.path.basename(npz_path)}")
                    
                    # Save individual plot
                    plt.savefig(indiv_plot_path, dpi=150, bbox_inches='tight')
                    plt.close(fig_indiv)
                    print(f"  Saved individual plot to: {indiv_plot_path}")
                    
                    # If lifetime data is present, save lifetime image
                    if lifetime is not None and lifetime_dir is not None:
                        lifetime_filename = os.path.splitext(base_name)[0] + '_manual_lifetime.tiff'
                        lifetime_path = os.path.join(lifetime_output_dir, lifetime_filename)
                        
                        # Mask lifetime data
                        masked_lifetime = lifetime * (ellipse_mask > 0)
                        
                        # Save as TIFF
                        save_tiff(lifetime_path, masked_lifetime)
                        print(f"  Saved lifetime image to: {lifetime_path}")
                    
                except Exception as e:
                    print(f"  Error saving data for {os.path.basename(npz_path)}: {e}")
                    traceback.print_exc()
            
            print("\nManual segmentation complete. Closing figure.")
            plt.close(fig)  # Close the main figure after saving all files
        
        # Connect callbacks
        s_center_x.on_changed(update)
        s_center_y.on_changed(update)
        s_width.on_changed(update)
        s_height.on_changed(update)
        s_angle.on_changed(update)
        s_threshold.on_changed(update_threshold)
        apply_button.on_clicked(apply_segmentation)
        
        plt.show(block=True)  # Block execution until plot is closed
        
        return True
    except Exception as e:
        print(f"Error in manual segmentation: {e}")
        traceback.print_exc()
        return False

def main(config, npz_dir, segmented_dir, plots_dir, lifetime_dir=None, interactive=True):
    """
    Main execution function for manual ellipse-based phasor segmentation.
    
    Args:
        config: Configuration dictionary
        npz_dir: Directory containing NPZ files
        segmented_dir: Directory to save segmented masks
        plots_dir: Directory to save plots
        lifetime_dir: Directory to save lifetime images (optional)
        interactive: Whether to prompt for user input (default: True)
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Starting Manual Segmentation, Plotting, and Lifetime Saving")
    print(f"Input NPZ directory: {npz_dir}")
    
    # Create manual segmentation subdirectory
    manual_segmentation_dir = os.path.join(os.path.dirname(segmented_dir), "manual_segmentation")
    manual_plots_dir = os.path.join(manual_segmentation_dir, "plots")
    manual_masks_dir = os.path.join(manual_segmentation_dir, "masks")
    manual_npz_dir = os.path.join(manual_segmentation_dir, "segmented_npz")
    
    if lifetime_dir:
        manual_lifetime_dir = os.path.join(manual_segmentation_dir, "lifetime_images")
    else:
        manual_lifetime_dir = None
    
    print(f"Output Manual Segmentation Directory: {manual_segmentation_dir}")
    print(f"Output Segmented NPZ Files: {manual_npz_dir}")
    print(f"Output Binary Masks: {manual_masks_dir}")
    print(f"Output Plots: {manual_plots_dir}")
    if manual_lifetime_dir:
        print(f"Output Lifetime Images: {manual_lifetime_dir}")
    
    if not os.path.isdir(npz_dir): 
        print(f"Error: NPZ dir not found: {npz_dir}", file=sys.stderr)
        return False
        
    # Create necessary output directories
    os.makedirs(manual_segmentation_dir, exist_ok=True)
    os.makedirs(manual_npz_dir, exist_ok=True)
    os.makedirs(manual_masks_dir, exist_ok=True)
    os.makedirs(manual_plots_dir, exist_ok=True)
    if manual_lifetime_dir:
        os.makedirs(manual_lifetime_dir, exist_ok=True)
        
    # Find NPZ files
    npz_files = []
    for root, _, files in os.walk(npz_dir):
        for file in files:
            if file.endswith('.npz') and not file.endswith('_segmented.npz'):
                npz_path = os.path.join(root, file)
                npz_files.append(npz_path)
                
    if not npz_files:
        print(f"No NPZ files found in {npz_dir}")
        return False
        
    print(f"Found {len(npz_files)} NPZ files for possible manual segmentation")
    
    # Let user select which files to use
    selected_files = select_npz_files(npz_files)
    
    if not selected_files:
        print("No files selected for manual segmentation")
        return False
    
    print(f"Selected {len(selected_files)} files for combined manual segmentation")
    
    # Process selected NPZ files as a combined dataset
    success = process_combined_npz_files(selected_files, manual_npz_dir, manual_masks_dir, manual_plots_dir, manual_lifetime_dir)
    
    return success

if __name__ == "__main__":
    print("This script is intended to be run via run_pipeline.py")
    print("Please use: python run_pipeline.py --manual-segment -i <input_dir> -o <output_base_dir> [...]")
