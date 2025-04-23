#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phasor Transform Module

A standalone module for processing FLIM data and generating phasor plots
without GUI dependencies. This module provides functions to:

1. Load TIFF stacks containing FLIM data 
2. Calculate phasor coordinates (G and S) through FFT
3. Save output files (G, S, intensity) as TIFF images
"""

import os
import sys
import numpy as np
from skimage import io
import tifffile
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal
import time

# Path to the original FLUTE installation (for importing if needed)
FLUTE_PATH = "/Users/joshuamarcus/FLUTE"

def load_tiff_stack(filename):
    """
    Load TIFF stack containing FLIM data.
    
    Args:
        filename (str): Path to the TIFF stack or BIN file
        
    Returns:
        numpy.ndarray: 3D array of shape (time_bins, height, width)
    """
    try:
        # Try loading as TIFF
        if filename.lower().endswith(('.tif', '.tiff')):
            try:
                # Try using tifffile first
                data = tifffile.imread(filename)
                
                # Make sure shape is (time_bins, height, width)
                if data.ndim == 3:
                    # Check if time is the first dimension
                    if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                        # Time is likely the first dimension (correct)
                        return data
                    else:
                        # Transpose to get time as first dimension
                        print(f"Transposing data to get time as first dimension. Original shape: {data.shape}")
                        return np.transpose(data, (2, 0, 1))
                else:
                    raise ValueError(f"Expected 3D data but got shape {data.shape}")
            except Exception as e:
                print(f"Error loading with tifffile: {e}, trying with skimage.io...")
                
                # Fallback to skimage.io
                data = io.imread(filename)
                if data.ndim == 3:
                    return data
                else:
                    raise ValueError(f"Expected 3D data but got shape {data.shape}")
        # Add handling for BIN files if needed
        elif filename.lower().endswith('.bin'):
            raise NotImplementedError("BIN file loading not implemented yet")
        else:
            raise ValueError(f"Unsupported file type: {filename}")
            
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        raise
    
def calculate_phasor(data, freq_mhz=80.0, harmonic=1, bin_width_ns=0.2208, phi_cal=0.0, m_cal=1.0):
    """
    Calculate phasor coordinates (G, S) for FLIM data using FFT.
    Applies calibration corrections for instrument response.
    
    Args:
        data (numpy.ndarray): FLIM data as 3D array (time_bins, height, width)
        freq_mhz (float): Laser frequency in MHz
        harmonic (int): Harmonic to use for phasor calculation
        bin_width_ns (float): Width of each time bin in nanoseconds
        phi_cal (float): Phase shift calibration value
        m_cal (float): Modulation calibration value
        
    Returns:
        tuple: (G, S, intensity) as 2D numpy arrays
    """
    # Get dimensions
    nt, ny, nx = data.shape
    
    # Create output arrays
    g_img = np.zeros((ny, nx), dtype=np.float32)
    s_img = np.zeros((ny, nx), dtype=np.float32)
    intensity = np.zeros((ny, nx), dtype=np.float32)
    
    # Calculate omega (angular frequency)
    period_ns = 1000.0 / freq_mhz  # Convert MHz to ns period
    omega = 2.0 * np.pi * harmonic / period_ns  # radians/ns
    
    # Create time axis
    time_ns = np.arange(nt) * bin_width_ns
    
    # Calculate cosine and sine references for correlation
    cos_ref = np.cos(omega * time_ns)
    sin_ref = np.sin(omega * time_ns)
    
    # Calculate phasor coordinates for each pixel
    for y in range(ny):
        for x in range(nx):
            # Get decay curve for this pixel
            decay = data[:, y, x].astype(np.float32)
            
            # Calculate total intensity (sum of all time bins)
            total_intensity = np.sum(decay)
            intensity[y, x] = total_intensity
            
            # Skip pixels with no signal
            if total_intensity <= 0:
                g_img[y, x] = 0
                s_img[y, x] = 0
                continue
            
            # Normalize decay
            decay_norm = decay / total_intensity
            
            # Calculate G and S coordinates (via correlation)
            g = 2.0 * np.sum(decay_norm * cos_ref)
            s = 2.0 * np.sum(decay_norm * sin_ref)
            
            # Apply calibration correction
            g_cal = (g * np.cos(phi_cal) - s * np.sin(phi_cal)) / m_cal
            s_cal = (g * np.sin(phi_cal) + s * np.cos(phi_cal)) / m_cal
            
            g_img[y, x] = g_cal
            s_img[y, x] = s_cal
    
    return g_img, s_img, intensity

def calculate_phasor_vectorized(data, freq_mhz=80.0, harmonic=1, bin_width_ns=0.2208, phi_cal=0.0, m_cal=1.0):
    """
    Vectorized version of calculate_phasor for better performance.
    
    Args:
        data (numpy.ndarray): FLIM data as 3D array (time_bins, height, width)
        freq_mhz (float): Laser frequency in MHz
        harmonic (int): Harmonic to use for phasor calculation
        bin_width_ns (float): Width of each time bin in nanoseconds
        phi_cal (float): Phase shift calibration value
        m_cal (float): Modulation calibration value
        
    Returns:
        tuple: (G, S, intensity) as 2D numpy arrays
    """
    # Get dimensions
    nt, ny, nx = data.shape
    
    # Calculate omega (angular frequency)
    period_ns = 1000.0 / freq_mhz  # Convert MHz to ns period
    omega = 2.0 * np.pi * harmonic / period_ns  # radians/ns
    
    # Create time axis
    time_ns = np.arange(nt) * bin_width_ns
    
    # Calculate cosine and sine references for correlation
    cos_ref = np.cos(omega * time_ns)
    sin_ref = np.sin(omega * time_ns)
    
    # Reshape references to allow broadcasting
    cos_ref = cos_ref.reshape(-1, 1, 1)
    sin_ref = sin_ref.reshape(-1, 1, 1)
    
    # Calculate total intensity
    intensity = np.sum(data, axis=0)
    
    # Create a mask for zero intensity pixels
    nonzero_mask = intensity > 0
    
    # Initialize G and S images
    g_img = np.zeros((ny, nx), dtype=np.float32)
    s_img = np.zeros((ny, nx), dtype=np.float32)
    
    # Normalize data where intensity is non-zero
    normalized_data = np.zeros_like(data, dtype=np.float32)
    for t in range(nt):
        normalized_data[t, nonzero_mask] = data[t, nonzero_mask] / intensity[nonzero_mask]
    
    # Calculate G and S through correlation
    g_img = 2.0 * np.sum(normalized_data * cos_ref, axis=0)
    s_img = 2.0 * np.sum(normalized_data * sin_ref, axis=0)
    
    # Apply calibration correction
    g_cal = (g_img * np.cos(phi_cal) - s_img * np.sin(phi_cal)) / m_cal
    s_cal = (g_img * np.sin(phi_cal) + s_img * np.cos(phi_cal)) / m_cal
    
    return g_cal, s_cal, intensity

def perform_fft(image, bin_width_ns, freq_mhz, harmonic, phi_cal=0, m_cal=1):
    """
    Performs FFT on the image data to get the g and s coordinates.
    This is the core function from FLUTE's ImageHandler class.
    
    Args:
        image: 3D FLIM image stack (time bins, height, width)
        bin_width_ns: Width of each time bin in nanoseconds
        freq_mhz: Laser frequency in MHz
        harmonic: Harmonic number to use (usually 1)
        phi_cal: Phase calibration value
        m_cal: Modulation calibration value
        
    Returns:
        g_map: 2D array of g (x) coordinates
        s_map: 2D array of s (y) coordinates
        intensity_map: 2D array of total intensity
    """
    # Ensure image is in the right format (time bins as last dimension)
    if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
        # Time bins are already in first dimension, move to last
        image = np.moveaxis(image, 0, 2)
    
    bins = image.shape[2]  # Number of time bins
    t_arr = np.linspace(bin_width_ns / 2, bin_width_ns * (bins - 1/2), bins)
    
    # Calculate total intensity (sum over time bins)
    intensity = np.sum(image, axis=2).astype(float)
    
    # Avoid division by zero
    intensity[intensity == 0] = 0.00001
    
    # Calculate g and s coordinates using FFT formula
    g = np.sum(image * np.cos(2 * np.pi * freq_mhz / 1000 * harmonic * t_arr[:]), axis=2) / intensity
    s = np.sum(image * np.sin(2 * np.pi * freq_mhz / 1000 * harmonic * t_arr[:]), axis=2) / intensity
    
    # Apply calibration (phase rotation and modulation scaling)
    R = np.array(((np.cos(phi_cal), -np.sin(phi_cal)), (np.sin(phi_cal), np.cos(phi_cal))))
    
    # Pre-allocate arrays for g and s
    g_cal = np.zeros_like(g)
    s_cal = np.zeros_like(s)
    
    # Apply rotation matrix and scaling to each pixel
    for y in range(g.shape[0]):
        for x in range(g.shape[1]):
            gs_rotated = R.dot([g[y, x], s[y, x]]) * m_cal
            g_cal[y, x] = gs_rotated[0]
            s_cal[y, x] = gs_rotated[1]
    
    return g_cal, s_cal, intensity

def apply_median_filter(g_map, s_map, num_filters=1):
    """
    Applies median filtering to g and s maps.
    
    Args:
        g_map: 2D array of g coordinates
        s_map: 2D array of s coordinates
        num_filters: Number of times to apply the filter
        
    Returns:
        g_filtered: Filtered g map
        s_filtered: Filtered s map
    """
    g_filtered = g_map.copy()
    s_filtered = s_map.copy()
    
    for _ in range(num_filters):
        g_filtered = signal.medfilt(g_filtered)
        s_filtered = signal.medfilt(s_filtered)
    
    return g_filtered, s_filtered

def calculate_lifetime_maps(g_map, s_map, freq_mhz, harmonic=1):
    """
    Calculate lifetime maps from g and s coordinates.
    
    Args:
        g_map: 2D array of g coordinates
        s_map: 2D array of s coordinates
        freq_mhz: Laser frequency in MHz
        harmonic: Harmonic number (usually 1)
        
    Returns:
        tau_p: Phase lifetime map
        tau_m: Modulation lifetime map
    """
    omega = 2 * np.pi * freq_mhz / 1000 * harmonic
    
    # Calculate phase lifetime (TauP)
    angle_arr = s_map / g_map
    tau_p = np.zeros_like(angle_arr)
    valid_mask = (g_map > 0)
    tau_p[valid_mask] = 1/omega * np.arctan(angle_arr[valid_mask])
    
    # Calculate modulation lifetime (TauM)
    distance_arr = np.sqrt(s_map**2 + g_map**2)
    tau_m = np.zeros_like(distance_arr)
    valid_mask = (distance_arr > 0) & (distance_arr <= 1)
    tau_m[valid_mask] = 1/omega * np.sqrt(1/np.power(distance_arr[valid_mask], 2) - 1)
    
    return tau_p, tau_m

def process_flim_file(input_file, output_dir, phi_cal=0, m_cal=1, bin_width_ns=0.2208, freq_mhz=80, harmonic=1,
                     apply_filter=1, threshold_min=0, threshold_max=None):
    """
    Process a FLIM file (TIF or BIN) and generate phasor maps.
    
    Args:
        input_file: Path to input FLIM file (.tif or .bin)
        output_dir: Directory to save output files
        phi_cal: Phase calibration value
        m_cal: Modulation calibration value
        bin_width_ns: Width of each time bin in nanoseconds
        freq_mhz: Laser frequency in MHz
        harmonic: Harmonic number (usually 1)
        apply_filter: Number of times to apply median filter
        threshold_min: Minimum intensity threshold
        threshold_max: Maximum intensity threshold (None for no upper limit)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Processing FLIM file: {input_file}")
        start_time = time.time()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get file basename for output files
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Load image data
        if input_file.endswith(('.tif', '.tiff')):
            print("  Loading TIFF file...")
            # For TIFF files, use skimage.io.imread
            image = io.imread(input_file)
        elif input_file.endswith('.bin'):
            print("  Loading BIN file... (Not implemented yet)")
            # TODO: Implement BIN file loading - this would require understanding the binary format
            # For now, we assume the file has been converted to TIF first
            return False
        else:
            print(f"  Unsupported file format: {input_file}")
            return False
        
        print(f"  Image loaded: shape={image.shape}, dtype={image.dtype}")
        
        # Perform FFT to get phasor coordinates
        print(f"  Calculating phasor coordinates with bin_width={bin_width_ns}ns, freq={freq_mhz}MHz, harmonic={harmonic}")
        g_map, s_map, intensity_map = perform_fft(image, bin_width_ns, freq_mhz, harmonic, phi_cal, m_cal)
        
        # Apply median filter if requested
        if apply_filter > 0:
            print(f"  Applying median filter ({apply_filter} iterations)...")
            g_map, s_map = apply_median_filter(g_map, s_map, apply_filter)
        
        # Apply intensity threshold if specified
        if threshold_min > 0 or threshold_max is not None:
            print(f"  Applying intensity threshold: min={threshold_min}, max={threshold_max}")
            mask = intensity_map < threshold_min
            if threshold_max is not None:
                mask |= intensity_map > threshold_max
            
            # Set masked values to 0 (or another value that indicates "invalid")
            g_map[mask] = 0
            s_map[mask] = 0
        
        # Calculate lifetime maps
        print("  Calculating lifetime maps...")
        tau_p, tau_m = calculate_lifetime_maps(g_map, s_map, freq_mhz, harmonic)
        
        # Save output files
        print("  Saving output files...")
        
        # Define output paths
        output_g_path = os.path.join(output_dir, f"{base_name}_g.tiff")
        output_s_path = os.path.join(output_dir, f"{base_name}_s.tiff")
        output_intensity_path = os.path.join(output_dir, f"{base_name}_intensity.tiff")
        
        # Convert to float32 for consistent saving
        g_map = g_map.astype(np.float32)
        s_map = s_map.astype(np.float32)
        intensity_map = intensity_map.astype(np.float32)
        
        # Save G, S, and intensity maps
        tifffile.imwrite(output_g_path, g_map)
        tifffile.imwrite(output_s_path, s_map)
        tifffile.imwrite(output_intensity_path, intensity_map)
        
        # Optionally save lifetime maps as well
        tifffile.imwrite(os.path.join(output_dir, f"{base_name}_tau_p.tiff"), tau_p.astype(np.float32))
        tifffile.imwrite(os.path.join(output_dir, f"{base_name}_tau_m.tiff"), tau_m.astype(np.float32))
        
        elapsed_time = time.time() - start_time
        print(f"  Processing completed in {elapsed_time:.2f} seconds")
        print(f"  Output files saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error processing FLIM file: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Example usage:
    if len(sys.argv) != 3:
        print("Usage: python phasor_transform.py <input_tiff_file> <output_directory>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Example values - in real usage these would come from calibration
    phi_cal = 0.0
    m_cal = 1.0
    
    success = process_flim_file(
        input_file=input_file,
        output_dir=output_dir,
        phi_cal=phi_cal,
        m_cal=m_cal,
        bin_width_ns=0.2208,
        freq_mhz=80,
        harmonic=1,
        apply_filter=1
    )
    
    sys.exit(0 if success else 1) 