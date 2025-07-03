#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLIM-FRET Automated Analysis Script

This script automates the FLIM-FRET analysis workflow without requiring the FLUTE GUI.
It performs the same FFT calculations with calibration and saves output in the same format.
"""

import os
import sys
import numpy as np
import pandas as pd
from skimage import io
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage

def find_matching_calibration(input_file, calibration_values):
    """
    Find the most appropriate calibration values for a given input file.
    
    This function handles several special cases, particularly when calibration files
    and input files have different paths or extensions.
    
    Args:
        input_file (str): Path to the input file
        calibration_values (dict): Dictionary with calibration data
        
    Returns:
        tuple: (phi_cal, m_cal, source_description)
    """
    # Default values if no match is found
    default_values = (0.0, 1.0, "default values")
    
    if not calibration_values or not input_file:
        return default_values
        
    # Extract components from the input file path
    file_basename = os.path.basename(input_file)
    file_basename_no_ext = os.path.splitext(file_basename)[0]
    file_dirname = os.path.dirname(input_file)
    
    # 1. Direct exact match
    if input_file in calibration_values['full_paths']:
        return calibration_values['full_paths'][input_file] + ("exact path match",)
    
    # 2. Directory match
    if file_dirname in calibration_values['roots']:
        return calibration_values['roots'][file_dirname] + ("directory match",)
    
    # 3. Basename match (with extension)
    if file_basename in calibration_values['basenames']:
        return calibration_values['basenames'][file_basename] + ("basename match",)
    
    # 4. Basename match (without extension)
    if file_basename_no_ext in calibration_values['basenames']:
        return calibration_values['basenames'][file_basename_no_ext] + ("basename (no ext) match",)
    
    # 5. Special case for different extensions with same basename
    for cal_basename in calibration_values['basenames']:
        cal_basename_no_ext = os.path.splitext(cal_basename)[0]
        if cal_basename_no_ext == file_basename_no_ext:
            return calibration_values['basenames'][cal_basename] + ("basename (different ext) match",)
    
    # 6. Special case for FLIM paths: Match pattern like 
    #    when input is: /Volumes/NX-01-A/FLIM_workflow_test_data_analysis/output/Dish_1_Post-Rapa/R1/R_1_s1.tif
    #    cal file is:   /Volumes/NX-01-A/FLIM_workflow_test_data/Dish_1_Post-Rapa/R1/R_1_s1.bin
    for cal_path in calibration_values['full_paths']:
        # Extract the dish, region, and sample pattern (e.g., "Dish_1_Post-Rapa/R1/R_1_s1")
        cal_parts = cal_path.split('/')
        input_parts = input_file.split('/')
        
        # Look for matching dish, region and sample parts in both paths
        for i in range(len(input_parts) - 2):
            # Try to find a pattern like "Dish_X/RY/R_Y_sZ" in both paths
            if (i+2 < len(input_parts) and i+2 < len(cal_parts) and 
                input_parts[i].startswith("Dish_") and 
                input_parts[i+1].startswith("R") and 
                input_parts[i+2].startswith("R_")):  
                
                if (input_parts[i] == cal_parts[i] and 
                    input_parts[i+1] == cal_parts[i+1] and 
                    os.path.splitext(input_parts[i+2])[0] == os.path.splitext(cal_parts[i+2])[0]):
                    
                    return calibration_values['full_paths'][cal_path] + ("pattern match (Dish/Region/Sample)",)
    
    # If we get here, no match was found
    return default_values

def read_calibration_csv(csv_file):
    """
    Read calibration values from a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file containing calibration values
        
    Returns:
        dict: Dictionary with keys: 'full_paths', 'roots', 'basenames', each mapping to calibration tuples
    """
    try:
        df = pd.read_csv(csv_file)
        
        # Validate required columns
        required_cols = ['file_path', 'phi_cal', 'm_cal']
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: CSV file must contain a '{col}' column")
                return {}
        
        # Build dictionaries of calibration values by different path components
        calibration_values = {
            'full_paths': {},  # Full paths as keys
            'roots': {},      # Directory paths as keys
            'basenames': {}   # Base filenames as keys
        }
        
        for _, row in df.iterrows():
            full_path = row['file_path']
            phi_cal = float(row['phi_cal'])
            m_cal = float(row['m_cal'])
            cal_tuple = (phi_cal, m_cal)
            
            # Store by full path
            calibration_values['full_paths'][full_path] = cal_tuple
            
            # Store by root directory
            root_dir = os.path.dirname(full_path)
            if root_dir not in calibration_values['roots']:
                calibration_values['roots'][root_dir] = cal_tuple
            
            # Store by basename (filename without directory)
            basename = os.path.basename(full_path)
            basename_no_ext = os.path.splitext(basename)[0]
            calibration_values['basenames'][basename] = cal_tuple
            calibration_values['basenames'][basename_no_ext] = cal_tuple
        
        print(f"Loaded calibration values for {len(calibration_values['full_paths'])} files from {csv_file}")
        return calibration_values
        
    except Exception as e:
        print(f"Error reading calibration CSV file: {e}")
        return {'full_paths': {}, 'roots': {}, 'basenames': {}}

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
        # Add handling for BIN files if needed
        elif filename.lower().endswith('.bin'):
            # This is a placeholder - implement BIN file loading as needed
            raise NotImplementedError("BIN file loading not implemented yet")
        else:
            raise ValueError(f"Unsupported file type: {filename}")
            
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        raise

def perform_fft(image, bin_width_ns, freq_mhz, harmonic, phi_cal=0, m_cal=1):
    """
    Performs FFT on the image data to get the g and s coordinates.
    
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
    # Get dimensions
    nt, ny, nx = image.shape
    
    # Create time array
    t_arr = np.linspace(bin_width_ns / 2, bin_width_ns * (nt - 1/2), nt)
    
    # Calculate total intensity (sum over time bins)
    intensity = np.sum(image, axis=0).astype(float)
    
    # Avoid division by zero
    intensity_safe = np.maximum(intensity, 0.00001)
    
    # Pre-allocate arrays for g and s
    g = np.zeros((ny, nx), dtype=np.float32)
    s = np.zeros((ny, nx), dtype=np.float32)
    
    # Angular frequency
    omega = 2 * np.pi * freq_mhz / 1000 * harmonic
    
    # Calculate g and s coordinates using FFT formula
    for t in range(nt):
        cos_val = np.cos(omega * t_arr[t])
        sin_val = np.sin(omega * t_arr[t])
        g += image[t] * cos_val
        s += image[t] * sin_val
    
    g = g / intensity_safe
    s = s / intensity_safe
    
    # Apply calibration (phase rotation and modulation scaling)
    cos_phi = np.cos(phi_cal)
    sin_phi = np.sin(phi_cal)
    
    g_cal = (g * cos_phi - s * sin_phi) / m_cal
    s_cal = (g * sin_phi + s * cos_phi) / m_cal
    
    return g_cal, s_cal, intensity

def apply_median_filter(g_map, s_map, filter_size=1):
    """
    Applies median filtering to g and s maps.
    
    Args:
        g_map: 2D array of g coordinates
        s_map: 2D array of s coordinates
        filter_size: Size of the median filter
        
    Returns:
        g_filtered: Filtered g map
        s_filtered: Filtered s map
    """
    if filter_size <= 0:
        return g_map, s_map
    
    # Apply median filter with kernel size determined by filter_size
    kernel_size = 2 * filter_size + 1
    g_filtered = ndimage.median_filter(g_map, size=kernel_size)
    s_filtered = ndimage.median_filter(s_map, size=kernel_size)
    
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
    # Angular frequency in radians/ns
    omega = 2 * np.pi * freq_mhz / 1000 * harmonic
    
    # Calculate phase angle
    phi = np.arctan2(s_map, g_map)
    
    # Calculate modulation
    m = np.sqrt(g_map**2 + s_map**2)
    
    # Calculate phase lifetime
    tau_p = np.zeros_like(phi)
    valid_phi = phi < 0  # Only calculate where phase is negative
    tau_p[valid_phi] = -np.tan(phi[valid_phi]) / omega
    
    # Calculate modulation lifetime
    tau_m = np.zeros_like(m)
    valid_m = (m > 0) & (m < 1)  # Only calculate where 0 < m < 1
    tau_m[valid_m] = np.sqrt(1/(m[valid_m]**2) - 1) / omega
    
    return tau_p, tau_m

def apply_thresholds(g_map, s_map, intensity, threshold_min=0, threshold_max=None, 
                     angle_min=0, angle_max=90, circle_min=0, circle_max=120):
    """
    Apply thresholds to g and s maps based on intensity, angle and modulation.
    
    Args:
        g_map: 2D array of g coordinates
        s_map: 2D array of s coordinates
        intensity: 2D array of intensity values
        threshold_min: Minimum intensity threshold
        threshold_max: Maximum intensity threshold
        angle_min: Minimum angle (degrees)
        angle_max: Maximum angle (degrees)
        circle_min: Minimum modulation lifetime (ns)
        circle_max: Maximum modulation lifetime (ns)
        
    Returns:
        mask: Boolean mask of valid pixels
    """
    # Initialize mask
    mask = np.ones_like(g_map, dtype=bool)
    
    # Apply intensity thresholds
    mask = mask & (intensity >= threshold_min)
    if threshold_max is not None:
        mask = mask & (intensity <= threshold_max)
    
    # Calculate angle (in degrees)
    angle = np.arctan2(s_map, g_map) * 180 / np.pi
    # Convert to 0-90 range
    angle = -angle  # Negate angle to match FLUTE convention
    angle[angle < 0] += 180  # Convert negative angles to 0-180 range
    
    # Apply angle thresholds
    mask = mask & (angle >= angle_min) & (angle <= angle_max)
    
    # Calculate modulation
    modulation = np.sqrt(g_map**2 + s_map**2)
    
    # Calculate modulation lifetime
    omega = 2 * np.pi * 80 / 1000  # Assuming 80 MHz, harmonic=1
    tau_m = np.zeros_like(modulation)
    valid_m = (modulation > 0) & (modulation < 1)
    tau_m[valid_m] = np.sqrt(1/(modulation[valid_m]**2) - 1) / omega
    
    # Apply modulation lifetime thresholds
    if circle_min > 0:
        mask = mask & (tau_m >= circle_min)
    if circle_max < float('inf'):
        mask = mask & (tau_m <= circle_max)
    
    return mask

def save_output_files(output_dir, base_filename, g_map, s_map, intensity, mask,
                      tau_p=None, tau_m=None, phasor_plot=True):
    """
    Save output files in the same format as FLUTE.
    
    Args:
        output_dir: Directory to save output files
        base_filename: Base name for output files
        g_map: 2D array of g coordinates
        s_map: 2D array of s coordinates
        intensity: 2D array of intensity values
        mask: Boolean mask of valid pixels
        tau_p: Phase lifetime map (optional)
        tau_m: Modulation lifetime map (optional)
        phasor_plot: Whether to create a phasor plot image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save masked g, s maps
    g_masked = np.copy(g_map)
    s_masked = np.copy(s_map)
    g_masked[~mask] = 0
    s_masked[~mask] = 0
    
    # Save intensity map
    intensity_filename = os.path.join(output_dir, f"{base_filename}_intensity.tif")
    tifffile.imwrite(intensity_filename, intensity.astype(np.float32))
    
    # Save g and s maps
    g_filename = os.path.join(output_dir, f"{base_filename}_g.tif")
    s_filename = os.path.join(output_dir, f"{base_filename}_s.tif")
    tifffile.imwrite(g_filename, g_masked.astype(np.float32))
    tifffile.imwrite(s_filename, s_masked.astype(np.float32))
    
    # Save mask
    mask_filename = os.path.join(output_dir, f"{base_filename}_mask.tif")
    tifffile.imwrite(mask_filename, mask.astype(np.uint8) * 255)
    
    # Save lifetime maps if available
    if tau_p is not None:
        tau_p_masked = np.copy(tau_p)
        tau_p_masked[~mask] = 0
        tau_p_filename = os.path.join(output_dir, f"{base_filename}_taup.tif")
        tifffile.imwrite(tau_p_filename, tau_p_masked.astype(np.float32))
    
    if tau_m is not None:
        tau_m_masked = np.copy(tau_m)
        tau_m_masked[~mask] = 0
        tau_m_filename = os.path.join(output_dir, f"{base_filename}_taum.tif")
        tifffile.imwrite(tau_m_filename, tau_m_masked.astype(np.float32))
    
    # Create phasor plot
    if phasor_plot:
        plt.figure(figsize=(8, 8))
        
        # Plot unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        plt.plot(x, y, 'k-', linewidth=1)
        
        # Plot universal circle
        x_universal = 0.5 + 0.5 * np.cos(theta)
        y_universal = 0.5 * np.sin(theta)
        plt.plot(x_universal, y_universal, 'k--', linewidth=1)
        
        # Plot phasor points
        valid_points = mask & (intensity > 0)
        plt.scatter(g_map[valid_points], s_map[valid_points], s=1, alpha=0.05)
        
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.2, 1.2)
        plt.xlabel('G')
        plt.ylabel('S')
        plt.title('Phasor Plot')
        plt.grid(True)
        
        # Save phasor plot
        plot_filename = os.path.join(output_dir, f"{base_filename}_phasor.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close()
    
    print(f"Output files saved to {output_dir}")

def process_flim_file(input_file, output_dir, phi_cal=0, m_cal=1, bin_width_ns=0.2208, 
                      freq_mhz=80, harmonic=1, filter_size=0, 
                      threshold_min=0, threshold_max=1000000, 
                      angle_min=0, angle_max=90, 
                      circle_min=0, circle_max=120):
    """
    Process a FLIM file and generate phasor maps, matching FLUTE's workflow.
    
    Args:
        input_file: Path to input FLIM file (.tif or .bin)
        output_dir: Directory to save output files
        phi_cal: Phase calibration value
        m_cal: Modulation calibration value
        bin_width_ns: Width of each time bin in nanoseconds
        freq_mhz: Laser frequency in MHz
        harmonic: Harmonic number (usually 1)
        filter_size: Size of median filter to apply (0 for no filter)
        threshold_min: Minimum intensity threshold
        threshold_max: Maximum intensity threshold
        angle_min: Minimum angle (degrees)
        angle_max: Maximum angle (degrees)
        circle_min: Minimum modulation lifetime (ns)
        circle_max: Maximum modulation lifetime (ns)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Processing {input_file}...")
        print(f"  Using calibration: phi_cal={phi_cal}, m_cal={m_cal}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the FLIM data
        data = load_tiff_stack(input_file)
        
        # Step 1: Perform FFT to get g and s maps
        g_map, s_map, intensity = perform_fft(data, bin_width_ns, freq_mhz, harmonic, phi_cal, m_cal)
        
        # Step 2: Apply median filter (convolution in FLUTE)
        if filter_size > 0:
            g_map, s_map = apply_median_filter(g_map, s_map, filter_size)
        
        # Step 3: Calculate lifetime maps
        tau_p, tau_m = calculate_lifetime_maps(g_map, s_map, freq_mhz, harmonic)
        
        # Step 4: Apply thresholds
        mask = apply_thresholds(g_map, s_map, intensity, 
                                threshold_min, threshold_max, 
                                angle_min, angle_max, 
                                circle_min, circle_max)
        
        # Step 5: Save output files
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        save_output_files(output_dir, base_filename, g_map, s_map, intensity, mask, tau_p, tau_m)
        
        print(f"Processing of {input_file} completed successfully.")
        return True
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False

def process_folder(input_folder, output_folder, calibration_values=None, default_phi_cal=0.0, default_m_cal=1.0, 
                   bin_width_ns=0.2208, freq_mhz=80, harmonic=1, filter_size=0,
                   threshold_min=0, threshold_max=1000000, 
                   angle_min=0, angle_max=90, 
                   circle_min=0, circle_max=120):
    """
    Process all TIFF files in the input folder and save results to the output folder.
    
    Args:
        input_folder: Path to folder containing TIFF files
        output_folder: Path to save output files
        calibration_values: Dictionary with calibration information organized by path components
        default_phi_cal: Default phase calibration value if not found in calibration_values
        default_m_cal: Default modulation calibration value if not found in calibration_values
        bin_width_ns: Width of each time bin in nanoseconds
        freq_mhz: Laser frequency in MHz
        harmonic: Harmonic to use for phasor calculation
        filter_size: Size of median filter to apply (0 for no filter)
        threshold_min: Minimum intensity threshold
        threshold_max: Maximum intensity threshold
        angle_min: Minimum angle (degrees)
        angle_max: Maximum angle (degrees)
        circle_min: Minimum modulation lifetime (ns)
        circle_max: Maximum modulation lifetime (ns)
    """
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create FLUTE_output directory inside output_folder
    flute_output_dir = os.path.join(output_folder, "FLUTE_output")
    os.makedirs(flute_output_dir, exist_ok=True)
    
    print(f"Processing all TIFF files in {input_folder}")
    print(f"Saving results to {flute_output_dir}")
    
    # Get all TIFF files in the input folder
    tiff_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                tiff_files.append(os.path.join(root, file))
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Process each TIFF file
    success_count = 0
    for i, tiff_file in enumerate(tiff_files):
        # Get calibration values for this file using our advanced matching function
        if calibration_values:
            phi_cal, m_cal, cal_source = find_matching_calibration(tiff_file, calibration_values)
        else:
            phi_cal, m_cal = default_phi_cal, default_m_cal
            cal_source = "default values"
        
        print(f"Processing file {i+1}/{len(tiff_files)}: {tiff_file}")
        print(f"Using calibration from {cal_source}: phi={phi_cal}, mod={m_cal}")
        
        success = process_flim_file(
            tiff_file, 
            flute_output_dir, 
            phi_cal, 
            m_cal, 
            bin_width_ns, 
            freq_mhz, 
            harmonic,
            filter_size,
            threshold_min,
            threshold_max,
            angle_min,
            angle_max,
            circle_min,
            circle_max
        )
        if success:
            success_count += 1
    
    print(f"\nProcessing complete. Successfully processed {success_count}/{len(tiff_files)} files.")
    print(f"Results saved to {flute_output_dir}")

def main():
    """Main function to process command line arguments."""
    if len(sys.argv) < 3:
        print("Usage:")
        print("  For a single file: python flim_fft_automated.py --file <input_tiff_file> <output_directory>")
        print("  For a folder:      python flim_fft_automated.py --folder <input_folder> <output_directory>")
        print("\nOptional parameters:")
        print("  --calibration <csv_file> CSV file with calibration values (file_path,phi_cal,m_cal)")
        print("  --phi <phi_cal>          Phase calibration value (default: 0.0)")
        print("  --mod <m_cal>            Modulation calibration value (default: 1.0)")
        print("  --bin <bin_width>        Bin width in nanoseconds (default: 0.2208)")
        print("  --freq <freq>            Laser frequency in MHz (default: 80)")
        print("  --harmonic <harmonic>    Harmonic to use (default: 1)")
        print("  --filter <filter_size>   Median filter size (default: 0)")
        print("  --threshold <min> <max>  Intensity threshold range (default: 0 1000000)")
        print("  --angle <min> <max>      Angle range in degrees (default: 0 90)")
        print("  --circle <min> <max>     Modulation lifetime range in ns (default: 0 120)")
        sys.exit(1)
        
    # Default values
    phi_cal = 0.0
    m_cal = 1.0
    bin_width = 0.097
    freq = 78
    harmonic = 1
    filter_size = 0
    threshold_min = 0
    threshold_max = 1000000
    angle_min = 0
    angle_max = 90
    circle_min = 0
    circle_max = 120
    calibration_file = None
    calibration_values = None
    
    # Parse command line arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--calibration" and i+1 < len(sys.argv):
            calibration_file = sys.argv[i+1]
            calibration_values = read_calibration_csv(calibration_file)
            i += 2
        elif sys.argv[i] == "--phi" and i+1 < len(sys.argv):
            phi_cal = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--mod" and i+1 < len(sys.argv):
            m_cal = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--bin" and i+1 < len(sys.argv):
            bin_width = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--freq" and i+1 < len(sys.argv):
            freq = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--harmonic" and i+1 < len(sys.argv):
            harmonic = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--filter" and i+1 < len(sys.argv):
            filter_size = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--threshold" and i+2 < len(sys.argv):
            threshold_min = float(sys.argv[i+1])
            threshold_max = float(sys.argv[i+2])
            i += 3
        elif sys.argv[i] == "--angle" and i+2 < len(sys.argv):
            angle_min = float(sys.argv[i+1])
            angle_max = float(sys.argv[i+2])
            i += 3
        elif sys.argv[i] == "--circle" and i+2 < len(sys.argv):
            circle_min = float(sys.argv[i+1])
            circle_max = float(sys.argv[i+2])
            i += 3
        elif sys.argv[i] == "--file" and i+1 < len(sys.argv) and i+2 < len(sys.argv):
            input_file = sys.argv[i+1]
            output_dir = sys.argv[i+2]
            
            # Check if we have calibration values for this file
            if calibration_values:
                    # Use our specialized matching function to find calibration values
                phi_cal, m_cal, cal_source = find_matching_calibration(input_file, calibration_values)
                
                print(f"Using calibration from {cal_source}: phi={phi_cal}, mod={m_cal}")
            
            # Create FLUTE_output directory inside output_dir
            flute_output_dir = os.path.join(output_dir, "FLUTE_output")
            os.makedirs(flute_output_dir, exist_ok=True)
            
            process_flim_file(
                input_file, 
                flute_output_dir, 
                phi_cal, 
                m_cal, 
                bin_width, 
                freq, 
                harmonic,
                filter_size,
                threshold_min,
                threshold_max,
                angle_min,
                angle_max,
                circle_min,
                circle_max
            )
            i += 3
        elif sys.argv[i] == "--folder" and i+1 < len(sys.argv) and i+2 < len(sys.argv):
            input_folder = sys.argv[i+1]
            output_dir = sys.argv[i+2]
            process_folder(
                input_folder, 
                output_dir, 
                calibration_values, 
                phi_cal, 
                m_cal, 
                bin_width, 
                freq, 
                harmonic,
                filter_size,
                threshold_min,
                threshold_max,
                angle_min,
                angle_max,
                circle_min,
                circle_max
            )
            i += 3
        else:
            i += 1

if __name__ == "__main__":
    main()
