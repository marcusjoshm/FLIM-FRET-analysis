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
import datetime
import numpy as np
import matplotlib
# Set matplotlib backend to ensure interactive plots work properly
try:
    matplotlib.use('TkAgg')  # Try TkAgg backend first
except ImportError:
    try:
        matplotlib.use('Qt5Agg')  # Fallback to Qt5Agg
    except ImportError:
        matplotlib.use('MacOSX')  # Fallback to MacOSX backend
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

# File selection is now handled by phasor_segmentation.py
# This function has been removed to eliminate duplication

def process_combined_npz_files(npz_files, segmented_dir, masks_dir, plots_dir, lifetime_dir=None, data_type='filtered', naming_variables=None, selected_mask_name=None):
    """
    Process multiple NPZ files for combined manual segmentation.
    
    Args:
        npz_files: List of NPZ file paths
        segmented_dir: Directory to save segmented NPZ files
        masks_dir: Directory to save mask files
        plots_dir: Directory to save plots
        lifetime_dir: Directory to save lifetime images (optional)
        data_type: Data type to use ('filtered' for G/S or 'unfiltered' for GU/SU)
        naming_variables: Dictionary containing naming variables for output files
        
    Returns:
        bool: Success status
    """
    print(f"\n=== Manual Segmentation for {len(npz_files)} Combined Files ===")
    
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
        
        if g_data is None or s_data is None or intensity is None:
            print(f"Warning: Missing required data in {npz_path}")
            continue
            
        # Flatten arrays
        g_flat = g_data.flatten()
        s_flat = s_data.flatten()
        intensity_flat = intensity.flatten()
        
        # Apply mask if selected
        if selected_mask_name and selected_mask_name in data:
            print(f"Applying mask '{selected_mask_name}' to {os.path.basename(npz_path)}")
            mask = data[selected_mask_name]
            
            # Apply mask to phasor data
            g_data = g_data * mask
            s_data = s_data * mask
            intensity = intensity * mask
            
            # Re-flatten the masked data
            g_flat = g_data.flatten()
            s_flat = s_data.flatten()
            intensity_flat = intensity.flatten()
            
            print(f"  Applied mask: {np.sum(mask)} pixels selected out of {mask.size} total")
        
        # Store data for this file
        file_data_mapping[npz_path] = {
            'npz_data': data,
            'g_data': g_data,
            's_data': s_data,
            'intensity': intensity,
            'lifetime': lifetime,
            'g_flat': g_flat,
            's_flat': s_flat,
            'intensity_flat': intensity_flat
        }
        
        # Add to combined data
        all_g_data.append(g_flat)
        all_s_data.append(s_flat)
        all_intensity_data.append(intensity_flat)
    
    if not file_data_mapping:
        print("Error: No valid NPZ files could be loaded.")
        return False
    
    # Combine all data
    all_g = np.concatenate(all_g_data)
    all_s = np.concatenate(all_s_data)
    all_intensity = np.concatenate(all_intensity_data)
    
    print(f"Loaded {len(file_data_mapping)} files with {len(all_g)} total pixels")
    
    # Apply thresholding before creating the plot (like in phasor_visualization.py)
    print("\nThresholding options:")
    print("  [1] No threshold (use all data)")
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
            threshold_desc = "No threshold"
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
                    threshold_desc = f"Manual threshold: {threshold}"
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    
        elif threshold_choice == '3':
            # Auto-threshold with default 90%
            threshold = 0
            auto_percentile = 90
            individual_percentile = None
            threshold_desc = f"Auto threshold ({auto_percentile}%)"
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
                    threshold_desc = f"Auto threshold ({auto_percentile}%)"
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    
        elif threshold_choice == '5':
            # Individual dataset auto-threshold (90%)
            threshold = 0
            auto_percentile = None
            individual_percentile = 90
            threshold_desc = f"Individual auto threshold ({individual_percentile}%)"
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
                    threshold_desc = f"Individual auto threshold ({individual_percentile}%)"
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            print("Invalid choice. Please select 1-6.")
    
    # Apply thresholding
    print(f"\nApplying {threshold_desc}...")
    
    if individual_percentile is not None:
        # Apply individual thresholding to each file
        filtered_g = []
        filtered_s = []
        filtered_intensity = []
        
        for npz_path, data in file_data_mapping.items():
            intensity_flat = data['intensity_flat']
            g_flat = data['g_flat']
            s_flat = data['s_flat']
            
            # Calculate threshold for this specific file
            file_threshold = np.percentile(intensity_flat, individual_percentile)
            
            # Create mask for this file
            mask = intensity_flat >= file_threshold
            
            if np.sum(mask) > 0:
                filtered_g.append(g_flat[mask])
                filtered_s.append(s_flat[mask])
                filtered_intensity.append(intensity_flat[mask])
                
                print(f"  {os.path.basename(npz_path)}: kept {np.sum(mask)} of {len(mask)} pixels ({np.sum(mask)/len(mask)*100:.1f}%)")
        
        if filtered_g:
            all_g = np.concatenate(filtered_g)
            all_s = np.concatenate(filtered_s)
            all_intensity = np.concatenate(filtered_intensity)
            print(f"  Total: kept {len(all_g)} of {sum(len(data['g_flat']) for data in file_data_mapping.values())} pixels")
        else:
            print("Warning: No data points remain after individual thresholding")
            return False
            
    elif auto_percentile is not None:
        # Apply auto-thresholding to combined data
        combined_threshold = np.percentile(all_intensity, auto_percentile)
        mask = all_intensity >= combined_threshold
        
        all_g = all_g[mask]
        all_s = all_s[mask]
        all_intensity = all_intensity[mask]
        
        print(f"  Applied threshold of {combined_threshold:.2f}: kept {len(all_g)} of {len(mask)} pixels ({len(all_g)/len(mask)*100:.1f}%)")
        
    elif threshold > 0:
        # Apply manual threshold
        mask = all_intensity >= threshold
        all_g = all_g[mask]
        all_s = all_s[mask]
        all_intensity = all_intensity[mask]
        
        print(f"  Applied threshold of {threshold}: kept {len(all_g)} of {len(mask)} pixels ({len(all_g)/len(mask)*100:.1f}%)")
    
    # Create the interactive plot with thresholded data using high-quality formatting
    print(f"\nCreating interactive plot with {len(all_g)} thresholded pixels...")
    
    # Remove any NaN values
    mask = ~(np.isnan(all_g) | np.isnan(all_s) | np.isnan(all_intensity))
    all_g = all_g[mask]
    all_s = all_s[mask]
    all_intensity = all_intensity[mask]
    
    # Check for empty data
    if len(all_g) == 0 or len(all_s) == 0:
        print("Warning: No valid data points after thresholding")
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
    ax.set_title(f"Manual Segmentation - Combined Dataset ({len(npz_files)} files)\n{threshold_desc}\n({timestamp})")
    
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if naming_variables:
            plot_filename = f"segmentation_phasor_plot_{naming_variables['file_selection']}_{naming_variables['method']}_{naming_variables['data_type']}_{naming_variables['mask_source']}_{timestamp}.png"
        else:
            plot_filename = f"segmentation_phasor_plot_manual_combined_{timestamp}.png"
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
            'threshold_desc': threshold_desc,
            'ellipse_cond_center': [center_x, center_y]
        }
        
        for npz_path, data in file_data_mapping.items():
            print(f"Processing: {os.path.basename(npz_path)}")
            
            npz_data = data['npz_data']
            g_data = data['g_data']
            s_data = data['s_data']
            intensity = data['intensity']
            lifetime = data['lifetime']
            
            # Apply the same thresholding that was used for the plot
            if individual_percentile is not None:
                # Apply individual thresholding
                file_threshold = np.percentile(intensity.flatten(), individual_percentile)
                mask = intensity >= file_threshold
            elif auto_percentile is not None:
                # Apply auto thresholding
                combined_threshold = np.percentile(all_intensity, auto_percentile)
                mask = intensity >= combined_threshold
            elif threshold > 0:
                # Apply manual threshold
                mask = intensity >= threshold
            else:
                # No threshold
                mask = np.ones_like(intensity, dtype=bool)
            
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
            
            # Create binary mask (0 = background, 1 = selected region)
            manual_segmentation_mask = np.zeros_like(g_data, dtype=np.int32)
            manual_segmentation_mask[ellipse_mask] = 1  # Selected region = 1
            
            # Create output directories
            mask_output_dir = masks_dir
            plot_output_dir = plots_dir
            os.makedirs(mask_output_dir, exist_ok=True)
            os.makedirs(plot_output_dir, exist_ok=True)
            
            if lifetime_dir and lifetime is not None:
                lifetime_output_dir = lifetime_dir
                os.makedirs(lifetime_output_dir, exist_ok=True)
            
            # Append mask data to the existing NPZ file
            base_name = os.path.basename(npz_path)
            
            # Convert NPZ data to regular dictionary for modification
            npz_data_dict = dict(npz_data)
            
            # Add segmentation data to existing NPZ data
            npz_data_dict['mask_component_0'] = ~ellipse_mask  # Background (not selected)
            npz_data_dict['mask_component_1'] = ellipse_mask   # Selected region
            npz_data_dict['manual_segmentation_mask'] = manual_segmentation_mask
            
            # Add metadata
            npz_data_dict['segmentation_metadata'] = {
                **common_metadata,
                'source_file': npz_path,
                'pixels_selected': np.sum(ellipse_mask),
                'pixels_thresholded': np.sum(mask),
                'total_pixels': g_data.size,
                'mask_type': 'binary'
            }
            
            # Add mask registry to track available masks
            if 'mask_registry' not in npz_data_dict:
                npz_data_dict['mask_registry'] = {}
            elif not isinstance(npz_data_dict['mask_registry'], dict):
                # If mask_registry exists but is not a dict (e.g., it's a numpy array), replace it
                npz_data_dict['mask_registry'] = {}
            
            npz_data_dict['mask_registry']['manual_segmentation_mask'] = {
                'type': 'binary',
                'description': 'Manual ellipse-based segmentation mask',
                'created_by': 'ManualSegmentation',
                'created_timestamp': datetime.datetime.now().isoformat()
            }
            
            # Save updated NPZ file (overwrite the original)
            np.savez_compressed(npz_path, **npz_data_dict)
            print(f"  Updated NPZ file with mask data: {npz_path}")
            
            # Save mask as TIFF
            if naming_variables:
                mask_name = f"{os.path.splitext(base_name)[0]}_segmentation_mask_{naming_variables['file_selection']}_{naming_variables['method']}_{naming_variables['data_type']}_{naming_variables['mask_source']}_{timestamp}.tiff"
            else:
                mask_name = f"{os.path.splitext(base_name)[0]}_segmentation_mask_manual_{timestamp}.tiff"
            mask_path = os.path.join(mask_output_dir, mask_name)
            save_tiff(mask_path, manual_segmentation_mask)
            print(f"  Saved mask: {mask_path}")
            
            # Save lifetime image if available
            if lifetime_dir and lifetime is not None:
                if naming_variables:
                    lifetime_name = f"{os.path.splitext(base_name)[0]}_segmentation_lifetime_{naming_variables['file_selection']}_{naming_variables['method']}_{naming_variables['data_type']}_{naming_variables['mask_source']}_{timestamp}.tiff"
                else:
                    lifetime_name = f"{os.path.splitext(base_name)[0]}_segmentation_lifetime_manual_{timestamp}.tiff"
                lifetime_path = os.path.join(lifetime_output_dir, lifetime_name)
                
                # Apply mask to lifetime
                masked_lifetime = lifetime.copy()
                masked_lifetime[manual_segmentation_mask == 0] = 0  # Background (not selected)
                # Keep original values for selected region (mask == 1)
                
                save_tiff(lifetime_path, masked_lifetime)
                print(f"  Saved lifetime image: {lifetime_path}")
        
        print(f"\nManual segmentation completed for {len(file_data_mapping)} files!")
        plt.close()
    
    # Create apply button
    ax_apply = plt.axes([0.8, 0.02, 0.1, 0.04])
    button_apply = Button(ax_apply, 'Apply')
    button_apply.on_clicked(apply_segmentation)
    
    # Create cancel button
    ax_cancel = plt.axes([0.65, 0.02, 0.1, 0.04])
    button_cancel = Button(ax_cancel, 'Cancel')
    
    def cancel_segmentation(event):
        plt.close()
    
    button_cancel.on_clicked(cancel_segmentation)
    
    # Show the plot and wait for user interaction
    plt.show(block=True)
    
    return True

def main(config, npz_dir, output_dir, plots_dir, lifetime_dir=None, interactive=True, selected_files=None, data_type='filtered', naming_variables=None, selected_mask_name=None):
    """
    Main execution function for manual ellipse-based phasor segmentation.
    
    Args:
        config: Configuration dictionary
        npz_dir: Directory containing NPZ files
        output_dir: Main output directory (e.g., 2025-07-11_analysis_TEST)
        plots_dir: Directory to save plots (unused, kept for compatibility)
        lifetime_dir: Directory to save lifetime images (optional)
        interactive: Whether to prompt for user input (default: True)
        selected_files: List of selected NPZ file paths (if None, will find all NPZ files)
        data_type: Data type to use ('filtered' for G/S or 'unfiltered' for GU/SU)
        naming_variables: Dictionary containing naming variables for output files
        selected_mask_name: Name of the selected mask to apply (if any)
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Starting Manual Segmentation, Plotting, and Lifetime Saving")
    print(f"Input NPZ directory: {npz_dir}")
    
    # Create simplified output directory structure
    masks_dir = os.path.join(output_dir, "masks")
    phasor_plots_dir = os.path.join(output_dir, "phasor_plots")
    
    print(f"Output Masks Directory: {masks_dir}")
    print(f"Output Phasor Plots Directory: {phasor_plots_dir}")
    
    if not os.path.isdir(npz_dir): 
        print(f"Error: NPZ dir not found: {npz_dir}", file=sys.stderr)
        return False
        
    # Create necessary output directories
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(phasor_plots_dir, exist_ok=True)
    
    # Use provided selected_files or find all NPZ files
    if selected_files is None:
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
        selected_files = npz_files
    else:
        print(f"Using {len(selected_files)} pre-selected files for manual segmentation")
    
    if not selected_files:
        print("No files selected for manual segmentation")
        return False
    
    print(f"Selected {len(selected_files)} files for combined manual segmentation")
    
    # Process selected NPZ files as a combined dataset
    success = process_combined_npz_files(selected_files, npz_dir, masks_dir, phasor_plots_dir, lifetime_dir, data_type, naming_variables, selected_mask_name)
    
    return success

if __name__ == "__main__":
    print("This script is intended to be run via run_pipeline.py")
    print("Please use: python run_pipeline.py --manual-segment -i <input_dir> -o <output_base_dir> [...]")
