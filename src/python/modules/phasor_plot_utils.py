#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phasor Plot Utilities for FLIM-FRET Analysis
============================================

This module provides centralized phasor plot generation functionality
that can be imported and used by other modules to avoid code duplication.

Created by Joshua Marcus
"""

import os
import sys
import datetime
import numpy as np

# Set matplotlib backend before any matplotlib imports
os.environ['MPLBACKEND'] = 'MacOSX'  # Use MacOSX backend which is more reliable on macOS

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from matplotlib import colors


def create_phasor_plot(g_data, s_data, intensity, title, figsize=(8, 6), 
                      ax=None, show_colorbar=True, show_universal_circle=True,
                      timestamp=True, return_histogram_data=False,
                      target_pixels_per_unit=300):
    """
    Create a standardized phasor plot from G and S coordinates.
    
    Args:
        g_data (array): G coordinates
        s_data (array): S coordinates  
        intensity (array): Intensity values
        title (str): Plot title
        figsize (tuple): Figure size (width, height)
        ax (matplotlib.axes.Axes): Existing axes to plot on (if None, creates new figure)
        show_colorbar (bool): Whether to show colorbar
        show_universal_circle (bool): Whether to show universal circle
        timestamp (bool): Whether to add timestamp to title
        return_histogram_data (bool): Whether to return histogram data for further processing
        target_pixels_per_unit (int): Target number of pixels per unit for consistent resolution
        
    Returns:
        tuple: (fig, ax, histogram_data) where histogram_data is None unless return_histogram_data=True
    """
    # Remove any NaN values
    mask = ~(np.isnan(g_data) | np.isnan(s_data) | np.isnan(intensity))
    g_data = g_data[mask]
    s_data = s_data[mask]
    intensity = intensity[mask]
    
    # Check for empty data
    if len(g_data) == 0 or len(s_data) == 0:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        ax.set_title(f"{title} - No valid data points")
        ax.set_xlabel("\n$G$")
        ax.set_ylabel("$S$\n")
        ax.grid(True, alpha=0.3)
        return (fig, ax, None) if return_histogram_data else (fig, ax)
    
    # Create a universal circle for reference
    x = np.linspace(0, 1.0, 100)
    y = np.linspace(0, 0.7, 100)
    X, Y = np.meshgrid(x, y)
    F = (X**2 + Y**2 - X)  # Universal circle equation
    
    # Set plot limits (fixed for all phasor plots)
    x_scale = [-0.005, 1.005]
    y_scale = [0, 0.7]
    
    # Calculate adaptive bin widths using IQR (data-driven approach)
    iqr_x = np.percentile(g_data, 75) - np.percentile(g_data, 25)
    iqr_y = np.percentile(s_data, 75) - np.percentile(s_data, 25)
    
    # Use IQR-based bin widths but normalize to achieve consistent pixel resolution
    # This adapts to data distribution while maintaining consistent pixel sizes
    bin_width_x = 2 * iqr_x * (len(g_data) ** (-1/3))
    bin_width_y = 2 * iqr_y * (len(s_data) ** (-1/3))
    
    # Handle edge cases where IQR is too small
    min_bin_width = np.finfo(float).eps
    if bin_width_x <= min_bin_width:
        bin_width_x = (x_scale[1] - x_scale[0]) / 100  # Default to 100 bins
    if bin_width_y <= min_bin_width:
        bin_width_y = (y_scale[1] - y_scale[0]) / 70   # Default to 70 bins
    
    # Calculate number of bins based on data range and target resolution
    data_range_x = np.max(g_data) - np.min(g_data)
    data_range_y = np.max(s_data) - np.min(s_data)
    
    # Use the larger of IQR-based or target-resolution-based bin counts
    iqr_bins_x = max(1, int(data_range_x / bin_width_x))
    iqr_bins_y = max(1, int(data_range_y / bin_width_y))
    
    # Target resolution bins (for consistent pixel size)
    target_bins_x = int(data_range_x * target_pixels_per_unit)
    target_bins_y = int(data_range_y * target_pixels_per_unit)
    
    # For smaller datasets, be more aggressive with binning to achieve consistent pixel sizes
    # Use a dataset size factor to adjust the IQR calculation
    dataset_size_factor = min(1.0, len(g_data) / 1000000)  # Normalize to 1M points
    adjusted_iqr_bins_x = int(iqr_bins_x * (1 + (1 - dataset_size_factor) * 2.0))  # More bins for smaller datasets
    adjusted_iqr_bins_y = int(iqr_bins_y * (1 + (1 - dataset_size_factor) * 2.0))
    
    # For very small datasets, prioritize target resolution over IQR
    if len(g_data) < 500000:  # Less than 500K points
        num_bins_x = min(adjusted_iqr_bins_x, target_bins_x)
        num_bins_y = min(adjusted_iqr_bins_y, target_bins_y)
    else:
        # For larger datasets, use the more conservative approach
        num_bins_x = min(iqr_bins_x, target_bins_x)
        num_bins_y = min(iqr_bins_y, target_bins_y)
    
    # Ensure reasonable bin counts - much higher limits for better resolution
    num_bins_x = max(50, min(800, num_bins_x))
    num_bins_y = max(35, min(560, num_bins_y))
    
    # Calculate actual bin sizes for information
    actual_bin_size_x = data_range_x / num_bins_x if num_bins_x > 0 else 0
    actual_bin_size_y = data_range_y / num_bins_y if num_bins_y > 0 else 0
    
    # Create 2D histogram with data-adaptive bins
    # This adapts to the data distribution while maintaining reasonable pixel sizes
    hist_vals, _, _ = np.histogram2d(g_data, s_data, 
                                     bins=[num_bins_x, num_bins_y],
                                     weights=intensity)
    vmax = hist_vals.max()
    vmin = hist_vals.min()
    
    # Create or use existing axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Generate the 2D histogram with data-adaptive bins
    h = ax.hist2d(g_data, s_data, 
                bins=[num_bins_x, num_bins_y],
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
    
    # Add the universal circle contour if requested
    if show_universal_circle:
        ax.contour(X, Y, F, [0], colors='black', linewidths=1, zorder=2)
    
    # Add the colorbar with custom formatting if requested
    if show_colorbar:
        near_zero = 0.1
        cbar = fig.colorbar(h[3], ax=ax, format=LogFormatter(10, labelOnlyBase=True))
        
        # Calculate appropriate ticks for the colorbar
        if vmax > 1:
            ticks = [near_zero] + [10**i for i in range(1, int(np.log10(vmax)) + 1)]
            tick_labels = ['0'] + [f'$10^{i}$' for i in range(1, int(np.log10(vmax)) + 1)]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(tick_labels)
        
        cbar.set_label('Frequency')
    
    # Set title with timestamp if requested
    if timestamp:
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.set_title(f"{title}\n({timestamp_str})")
    else:
        ax.set_title(title)
    
    # Prepare return data
    histogram_data = {
        'hist_vals': hist_vals,
        'vmax': vmax,
        'vmin': vmin,
        'num_bins_x': num_bins_x,
        'num_bins_y': num_bins_y,
        'g_data': g_data,
        's_data': s_data,
        'intensity': intensity
    } if return_histogram_data else None
    
    return (fig, ax, histogram_data) if return_histogram_data else (fig, ax)


def create_phasor_plot_with_ellipse(g_data, s_data, intensity, title, 
                                   ellipse_params=None, figsize=(10, 8),
                                   target_pixels_per_unit=300):
    """
    Create a phasor plot with an interactive ellipse overlay.
    
    Args:
        g_data (array): G coordinates
        s_data (array): S coordinates
        intensity (array): Intensity values
        title (str): Plot title
        ellipse_params (dict): Ellipse parameters {'center_x', 'center_y', 'width', 'height', 'angle'}
        figsize (tuple): Figure size
        target_pixels_per_unit (int): Target number of pixels per unit for consistent resolution
        
    Returns:
        tuple: (fig, ax, ellipse) where ellipse is the matplotlib Ellipse object
    """
    from matplotlib.patches import Ellipse
    
    # Create the base phasor plot
    fig, ax = create_phasor_plot(g_data, s_data, intensity, title, figsize=figsize, 
                                 target_pixels_per_unit=target_pixels_per_unit)
    
    # Default ellipse parameters if not provided
    if ellipse_params is None:
        ellipse_params = {
            'center_x': 0.5,
            'center_y': 0.25,
            'width': 0.2,
            'height': 0.1,
            'angle': 0
        }
    
    # Create ellipse
    ellipse = Ellipse(
        xy=(ellipse_params['center_x'], ellipse_params['center_y']),
        width=ellipse_params['width'],
        height=ellipse_params['height'],
        angle=ellipse_params['angle'],
        fill=False,
        color='blue',
        linewidth=2
    )
    ax.add_artist(ellipse)
    
    # Add center marker
    center_point, = ax.plot([ellipse_params['center_x']], [ellipse_params['center_y']], 
                           'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
    
    return fig, ax, ellipse


def save_phasor_plot(fig, output_dir, filename, dpi=300, format='pdf'):
    """
    Save a phasor plot figure to file.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save
        output_dir (str): Output directory path
        filename (str): Output filename
        dpi (int): DPI for saving
        format (str): File format ('pdf', 'png', etc.)
        
    Returns:
        str: Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure filename has correct extension
    if not filename.endswith(f'.{format}'):
        filename += f'.{format}'
        
    # Save figure
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight')
    print(f"Saved phasor plot to {filepath}")
    
    return filepath


def calculate_ellipse_mask(g_data, s_data, center_x, center_y, width, height, angle_deg):
    """
    Calculate which points fall inside an ellipse on the phasor plot.
    
    Args:
        g_data (array): G coordinates
        s_data (array): S coordinates
        center_x (float): Ellipse center X coordinate
        center_y (float): Ellipse center Y coordinate
        width (float): Ellipse width
        height (float): Ellipse height
        angle_deg (float): Ellipse rotation angle in degrees
        
    Returns:
        array: Boolean mask of points inside ellipse
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    
    # Calculate distance from center
    dx = g_data - center_x
    dy = s_data - center_y
    
    # Rotate coordinates
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    x_rot = dx * cos_angle + dy * sin_angle
    y_rot = -dx * sin_angle + dy * cos_angle
    
    # Check if points are inside ellipse
    # (x/a)^2 + (y/b)^2 <= 1
    a = width / 2
    b = height / 2
    
    inside_ellipse = (x_rot / a)**2 + (y_rot / b)**2 <= 1
    
    return inside_ellipse


def are_points_inside_ellipse(points_x, points_y, center_x, center_y, width, height, angle_rad):
    """
    Check which points fall inside an ellipse (legacy function for compatibility).
    
    Args:
        points_x (array): X coordinates of points
        points_y (array): Y coordinates of points
        center_x (float): Ellipse center X coordinate
        center_y (float): Ellipse center Y coordinate
        width (float): Ellipse width
        height (float): Ellipse height
        angle_rad (float): Ellipse rotation angle in radians
        
    Returns:
        array: Boolean mask of points inside ellipse
    """
    # Convert angle to degrees for the new function
    angle_deg = np.degrees(angle_rad)
    return calculate_ellipse_mask(points_x, points_y, center_x, center_y, width, height, angle_deg)


def get_phasor_plot_resolution_info(target_pixels_per_unit=300):
    """
    Get information about the pixel resolution for phasor plots.
    
    Args:
        target_pixels_per_unit (int): Target number of pixels per unit in G/S space
        
    Returns:
        dict: Information about the resolution settings
    """
    # Phasor plot dimensions (fixed for all plots)
    g_range = 1.01  # -0.005 to 1.005
    s_range = 0.7   # 0 to 0.7
    
    # Calculate target bin counts for consistent pixel sizes
    target_bins_x = int(g_range * target_pixels_per_unit)
    target_bins_y = int(s_range * target_pixels_per_unit)
    
    # Apply much higher limits for better resolution
    target_bins_x = max(100, min(1000, target_bins_x))
    target_bins_y = max(70, min(700, target_bins_y))
    
    # Calculate target bin sizes
    target_bin_size_x = g_range / target_bins_x
    target_bin_size_y = s_range / target_bins_y
    
    return {
        'target_pixels_per_unit': target_pixels_per_unit,
        'target_g_bins': target_bins_x,
        'target_s_bins': target_bins_y,
        'target_total_bins': target_bins_x * target_bins_y,
        'target_bin_size_g': target_bin_size_x,
        'target_bin_size_s': target_bin_size_y,
        'g_range': g_range,
        's_range': s_range,
        'target_pixels_per_unit_g': target_bins_x / g_range,
        'target_pixels_per_unit_s': target_bins_y / s_range,
        'description': 'Hybrid approach: IQR-based binning with target resolution limits'
    }


def get_preset_resolutions():
    """
    Get preset resolution configurations for different use cases.
    
    Returns:
        dict: Dictionary of preset configurations
    """
    return {
        'low': {
            'target_pixels_per_unit': 100,
            'description': 'Low resolution, fast rendering',
            'total_bins': 7070  # 100 * 1.01 * 100 * 0.7
        },
        'medium': {
            'target_pixels_per_unit': 300,
            'description': 'Medium resolution, balanced performance',
            'total_bins': 63630  # 300 * 1.01 * 300 * 0.7
        },
        'high': {
            'target_pixels_per_unit': 500,
            'description': 'High resolution, detailed visualization',
            'total_bins': 176750  # 500 * 1.01 * 500 * 0.7
        },
        'ultra': {
            'target_pixels_per_unit': 800,
            'description': 'Ultra high resolution, maximum detail',
            'total_bins': 452480  # 800 * 1.01 * 800 * 0.7
        }
    } 