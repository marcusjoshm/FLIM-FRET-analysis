#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComplexWaveletFilter_v2_0.py

Advanced implementation of Complex Wavelet Filtering for FLIM-FRET analysis.
This script processes G, S, and intensity files using dual-tree complex wavelet transforms
to reduce noise while preserving edge information.

The workflow:
1. Loads G, S, and intensity TIFF files from preprocessed directory
2. Applies Anscombe transform to stabilize noise variance
3. Performs DTCWT on transformed data
4. Calculates local noise variance and applies filtering 
5. Reconstructs filtered data with inverse transform
6. Calculates lifetimes from both filtered and unfiltered data
7. Saves all data to NPZ files with the specified structure
"""

import os
import sys
import numpy as np
from PIL import Image
import dtcwt
import time
import warnings
import traceback
import math

# Suppress RuntimeWarning for division by zero, etc.
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Reference fluorophore coordinates and filtering level
REF_G = 0.30227996721890404  # G coordinate of reference fluorophore
REF_S = 0.4592458920992018   # S coordinate of reference fluorophore
DEFAULT_FLEVEL = 9           # Default filtering level

def main(config, preprocessed_dir, npz_dir):
    """
    Main execution function for complex wavelet filtering.
    
    Args:
        config (dict): Configuration dictionary with parameters
        preprocessed_dir (str): Path to preprocessed TIFF files
        npz_dir (str): Path to output NPZ files
        
    Returns:
        bool: True if successful, False if failed
    """
    # Initialize counters
    processed_count = 0
    skipped_count = 0
    
    # Validate inputs
    if not os.path.isdir(preprocessed_dir):
        print(f"Error: Input directory does not exist: {preprocessed_dir}", file=sys.stderr)
        return False
        
    if not os.path.isdir(npz_dir):
        os.makedirs(npz_dir, exist_ok=True)
        
    # Extract parameters from config
    try:
        flevel = config.get("wavelet_params", {}).get("filter_level", 9)
        ref_g = config.get("wavelet_params", {}).get("reference_g", 0.30227996721890404)
        ref_s = config.get("wavelet_params", {}).get("reference_s", 0.4592458920992018)
        
        # Get microscope parameters
        microscope_params = config.get("microscope_params", {})
        freq_mhz = microscope_params.get("frequency", 78.0)
        harmonic = microscope_params.get("harmonic", 1)
    except Exception as e:
        print(f"Error extracting parameters from config: {e}", file=sys.stderr)
        flevel = 9  # Default filter level
        ref_g = 0.30227996721890404  # Default reference G
        ref_s = 0.4592458920992018  # Default reference S
        freq_mhz = 78.0  # Default frequency in MHz
        harmonic = 1  # Default harmonic
        
    # Calculate angular frequency
    omega_rad_per_ns = 2 * np.pi * freq_mhz * 1e6 * harmonic * 1e-9
    
    print("Starting Complex Wavelet Filtering with advanced implementation")
    print(f"Input directory: {preprocessed_dir}")
    print(f"Output directory: {npz_dir}")
    print(f"Filter level: {flevel}")
    print(f"Reference fluorophore: G={ref_g}, S={ref_s}")
    print(f"Calculated Omega (rad/ns): {omega_rad_per_ns:.4f}\n")
    print("Looking for FLIM data files in the preprocessed directory...")
    
    # Create output directory if it doesn't exist
    os.makedirs(npz_dir, exist_ok=True)
    
    if not os.path.isdir(preprocessed_dir):
        print(f"Error: Input directory not found: {preprocessed_dir}", file=sys.stderr)
        return False
    
    # Initialize dataset dictionary to store matched G/S/intensity files
    datasets = {}
    
    # First, look for subdirectories with the expected structure (G_unfiltered, S_unfiltered, intensity)
    sample_dirs = []
    
    for root, dirs, files in os.walk(preprocessed_dir):
        # Check if this directory has our target subdirectories
        if 'G_unfiltered' in dirs and 'S_unfiltered' in dirs and 'intensity' in dirs:
            sample_dirs.append(root)
    
    if not sample_dirs:
        # If we didn't find the expected directory structure, try looking directly for files
        print("Did not find standard directory structure, searching for individual files...")
        
        for root, dirs, files in os.walk(preprocessed_dir):
            # Look for our target files
            g_files = [f for f in files if f.lower().endswith('.g.tiff') or f.lower().endswith('.g.tif') or '_g.tiff' in f.lower() or '_g.tif' in f.lower()]
            s_files = [f for f in files if f.lower().endswith('.s.tiff') or f.lower().endswith('.s.tif') or '_s.tiff' in f.lower() or '_s.tif' in f.lower()]
            intensity_files = [f for f in files if f.lower().endswith('.intensity.tiff') or f.lower().endswith('.intensity.tif') or '_intensity.tiff' in f.lower() or '_intensity.tif' in f.lower()]
            
            # Skip directories without any target files
            if not g_files and not s_files and not intensity_files:
                continue
            
            print(f"Found files in: {os.path.relpath(root, preprocessed_dir) if root != preprocessed_dir else 'root'}")
            print(f"  {len(g_files)} G files, {len(s_files)} S files, {len(intensity_files)} intensity files")
            
            # Process the files in this directory
            for g_file in g_files:
                if '_g.tiff' in g_file.lower() or '_g.tif' in g_file.lower():
                    # Extract base name (everything before _g)
                    base_name = os.path.splitext(g_file)[0][:-2]  # Remove _g
                else:
                    base_name = os.path.splitext(g_file)[0]
                datasets.setdefault(base_name, {})['g'] = os.path.join(root, g_file)
            
            for s_file in s_files:
                if '_s.tiff' in s_file.lower() or '_s.tif' in s_file.lower():
                    # Extract base name (everything before _s)
                    base_name = os.path.splitext(s_file)[0][:-2]  # Remove _s
                else:
                    base_name = os.path.splitext(s_file)[0]
                datasets.setdefault(base_name, {})['s'] = os.path.join(root, s_file)
            
            for int_file in intensity_files:
                if '_intensity.tiff' in int_file.lower() or '_intensity.tif' in int_file.lower() or '_wavelet_intensity.tiff' in int_file.lower():
                    # Extract base name (everything before _intensity or _wavelet_intensity)
                    if '_wavelet_intensity' in int_file:
                        base_name = int_file.split('_wavelet_intensity')[0]
                    else:
                        base_name = int_file.split('_intensity')[0]
                else:
                    base_name = os.path.splitext(int_file)[0]
                datasets.setdefault(base_name, {})['int'] = os.path.join(root, int_file)
    else:
        # Process samples with standard directory structure
        for sample_dir in sample_dirs:
            print(f"Found sample directory: {os.path.relpath(sample_dir, preprocessed_dir) if sample_dir != preprocessed_dir else 'root'}")
            
            # Get the G files
            g_dir = os.path.join(sample_dir, 'G_unfiltered')
            g_files = [f for f in os.listdir(g_dir) if f.endswith('.tif') or f.endswith('.tiff')]
            
            # Get the S files
            s_dir = os.path.join(sample_dir, 'S_unfiltered')
            s_files = [f for f in os.listdir(s_dir) if f.endswith('.tif') or f.endswith('.tiff')]
            
            # Get the intensity files
            intensity_dir = os.path.join(sample_dir, 'intensity')
            intensity_files = [f for f in os.listdir(intensity_dir) if f.endswith('.tif') or f.endswith('.tiff')]
            
            print(f"  {len(g_files)} G files, {len(s_files)} S files, {len(intensity_files)} intensity files")
            
            # Process all G files
            for g_file in g_files:
                # Extract the base name from the G filename
                if '_g.tiff' in g_file:
                    base_name = g_file.split('_g.tiff')[0]
                elif '_g.tif' in g_file:
                    base_name = g_file.split('_g.tif')[0]
                else:
                    base_name = os.path.splitext(g_file)[0]
                
                # Look for matching S and intensity files
                s_match = None
                int_match = None
                
                # Find matching S file
                for s_file in s_files:
                    if '_s.tiff' in s_file and s_file.split('_s.tiff')[0] == base_name:
                        s_match = s_file
                        break
                    elif '_s.tif' in s_file and s_file.split('_s.tif')[0] == base_name:
                        s_match = s_file
                        break
                
                # Find matching intensity file
                for int_file in intensity_files:
                    if '_intensity.tiff' in int_file and int_file.split('_intensity.tiff')[0] == base_name:
                        int_match = int_file
                        break
                    elif '_intensity.tif' in int_file and int_file.split('_intensity.tif')[0] == base_name:
                        int_match = int_file
                        break
                    elif '_wavelet_intensity.tiff' in int_file and int_file.split('_wavelet_intensity.tiff')[0] == base_name:
                        int_match = int_file
                        break
                
                # Add the complete dataset to our dictionary
                if s_match and int_match:
                    full_base_name = f"{os.path.basename(sample_dir)}_{base_name}" if sample_dir != preprocessed_dir else base_name
                    datasets[full_base_name] = {
                        'g': os.path.join(g_dir, g_file),
                        's': os.path.join(s_dir, s_match),
                        'int': os.path.join(intensity_dir, int_match)
                    }
        
        # Process each complete dataset
        for base_name, file_paths in datasets.items():
            if 'g' in file_paths and 's' in file_paths and 'int' in file_paths:
                print(f"Processing dataset: {base_name}")
                
                try:
                    # Process the dataset with complex wavelet filtering
                    process_start = time.time()
                    result = process_dataset(
                        file_paths['g'], 
                        file_paths['s'], 
                        file_paths['int'],
                        flevel,
                        omega_rad_per_ns,
                        ref_g,
                        ref_s
                    )
                    process_end = time.time()
                    
                    if result is not None:
                        # Save directly to npz_dir without subdirectories
                        # Extract the sample name and base filename for a clean output file
                        sample_name = os.path.basename(os.path.dirname(os.path.dirname(file_paths.get('g', '')))) if '/' in file_paths.get('g', '') else ''
                        
                        # Combine sample name with base name for a descriptive filename
                        if sample_name and sample_name not in base_name:
                            npz_filename = f"{sample_name}_{base_name}.npz"
                        else:
                            npz_filename = f"{base_name}.npz"
                        
                        # Save npz file directly to the main npz_dir
                        output_path = os.path.join(npz_dir, npz_filename)
                        np.savez(output_path, **result)
                        
                        print(f"  Saving NPZ to: {output_path}")
                        print(f"  Processing completed in {process_end - process_start:.2f} seconds")
                        processed_count += 1
                    else:
                        print(f"  Error processing dataset {base_name}")
                        skipped_count += 1
                        
                except Exception as e:
                    print(f"  Error processing {base_name}: {e}")
                    traceback.print_exc()
                    skipped_count += 1
            else:
                print(f"  Incomplete dataset for {base_name}. Missing: {', '.join(k for k in ['g', 's', 'int'] if k not in file_paths)}")
                skipped_count += 1
                
        # Prevent recursing into subdirectories
        dirs[:] = []
    
    print(f"\nComplex Wavelet Filtering completed.")
    print(f"Successfully processed datasets: {processed_count}")
    print(f"Skipped datasets: {skipped_count}")
    
    if processed_count > 0:
        return True
    else:
        print("Warning: No datasets were successfully processed.")
        return False

def load_tiff(file_path):
    """
    Load a TIFF file and convert it to a numpy array.
    
    Args:
        file_path (str): Path to the TIFF file
        
    Returns:
        tuple: (numpy.ndarray, bool) - The image array and whether it's a dummy
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"  File not found: {file_path}")
            return np.zeros((10, 10)), True
            
        # Check if file is empty or very small
        if os.path.getsize(file_path) < 100:
            print(f"  Empty or very small file: {file_path}")
            return np.zeros((10, 10)), True
            
        # Load the image
        image = Image.open(file_path)
        image_array = np.array(image).astype(np.float64)
        
        return image_array, False
        
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return np.zeros((10, 10)), True

def anscombe_transform(data):
    """
    Apply Anscombe transform to stabilize noise variance.
    
    Args:
        data (numpy.ndarray): Input data
        
    Returns:
        numpy.ndarray: Transformed data
    """
    return 2 * np.sqrt(np.maximum(data + (3/8), 0))

def reverse_anscombe_transform(y):
    """
    Apply inverse Anscombe transform.
    
    Args:
        y (numpy.ndarray): Anscombe-transformed data
        
    Returns:
        numpy.ndarray: Original-scale data
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.maximum(y, 1e-6)  # Avoid division by zero
    
    inverse = (
        (y**2 / 4) +
        (np.sqrt(3/2) * (1/y) / 4) -
        (11 / (8 * y**2)) +
        (np.sqrt(5/2) * (1/y**3) / 8) -
        (1 / (8 * y**4))
    )
    return np.maximum(inverse, 0)  # Ensure non-negative values

def perform_dtcwt_transform(data, N):
    """
    Perform dual-tree complex wavelet transform.
    
    Args:
        data (numpy.ndarray): Input data
        N (int): Number of decomposition levels
        
    Returns:
        tuple: (transform result, transform object)
    """
    transform = dtcwt.Transform2d(biort='near_sym_a', qshift='qshift_a')
    transformed_data = transform.forward(data, nlevels=N)
    return transformed_data, transform

def calculate_median_values(transformed_data):
    """
    Calculate median absolute values of wavelet coefficients.
    
    Args:
        transformed_data: DTCWT transformed data
        
    Returns:
        float: Mean of median absolute values
    """
    median_values = []
    for level in range(len(transformed_data.highpasses)):
        highpasses = transformed_data.highpasses[level]
        for band in range(highpasses.shape[2]):
            coeffs = highpasses[:, :, band]
            median_absolute = np.median(np.abs(coeffs.flatten()))
            median_values.append(median_absolute)
    return np.mean(median_values)

def calculate_local_noise_variance(transformed_data, N):
    """
    Calculate local noise variance for each coefficient.
    
    Args:
        transformed_data: DTCWT transformed data
        N (int): Window size for local variance calculation or filter level
        
    Returns:
        list: Matrices of local noise variance for each level and band
    """
    sigma_n_squared_matrices = []

    # If N is a filter level rather than window size, set window size to a constant value
    window_size = 3 if N > 10 else N

    def local_noise_variance(coeffs, window_size):
        sigma_n_squared = np.zeros_like(coeffs, dtype=float)
        height, width = coeffs.shape
        for x in range(width):
            for y in range(height):
                x_min, x_max = max(0, x - window_size), min(width, x + window_size + 1)
                y_min, y_max = max(0, y - window_size), min(height, y + window_size + 1)
                window = coeffs[y_min:y_max, x_min:x_max]
                local_variance = np.mean(np.abs(window)**2)
                sigma_n_squared[y, x] = local_variance
        return sigma_n_squared

    num_levels = len(transformed_data.highpasses)
    for level in range(num_levels):
        highpasses = transformed_data.highpasses[level]
        for band in range(highpasses.shape[2]):
            coeffs = highpasses[:, :, band]
            sigma_n_squared = local_noise_variance(coeffs, window_size)
            sigma_n_squared_matrices.append((level, band, sigma_n_squared))

    return sigma_n_squared_matrices

def compute_phi_prime(mandrill_t, sigma_g_squared, sigma_n_squared_matrices):
    """
    Compute modified wavelet coefficients for denoising.
    
    Args:
        mandrill_t: DTCWT transformed data
        sigma_g_squared (float): Signal variance estimate
        sigma_n_squared_matrices (list): Local noise variance matrices
        
    Returns:
        list: Updated wavelet coefficients
    """
    updated_coefficients = []
    max_level = len(mandrill_t.highpasses) - 1
    local_term = np.sqrt(3) * np.sqrt(sigma_g_squared)

    for level in range(max_level):
        highpasses_l = mandrill_t.highpasses[level]
        highpasses_l_plus_1 = mandrill_t.highpasses[level + 1]
        level_coefficients = []

        for band in range(highpasses_l.shape[2]):
            phi_l_b = highpasses_l[:, :, band]
            phi_l_plus_1_b = highpasses_l_plus_1[:, :, band]

            _, _, sigma_n_squared = sigma_n_squared_matrices[level * 6 + band]
            phi_prime = np.zeros_like(phi_l_b, dtype=complex)

            # Account for potential size mismatch due to downsampling
            if sigma_n_squared.shape != phi_l_b.shape:
                downsample_factor = max(1, phi_l_b.shape[0] // sigma_n_squared.shape[0])
            else:
                downsample_factor = 1

            for x in range(phi_l_b.shape[1]):
                for y in range(phi_l_b.shape[0]):
                    x_half = x // 2
                    y_half = y // 2

                    x_downsampled = min(x // downsample_factor, sigma_n_squared.shape[1]-1)
                    y_downsampled = min(y // downsample_factor, sigma_n_squared.shape[0]-1)

                    # Handle potential index errors
                    if y_half >= highpasses_l_plus_1.shape[0] or x_half >= highpasses_l_plus_1.shape[1]:
                        continue

                    phi_squared_sum = np.abs(phi_l_b[y, x])**2
                    if y_half < phi_l_plus_1_b.shape[0] and x_half < phi_l_plus_1_b.shape[1]:
                        phi_squared_sum += np.abs(phi_l_plus_1_b[y_half, x_half])**2

                    if sigma_n_squared[y_downsampled, x_downsampled] > 0 and phi_squared_sum > 0:
                        denominator = np.sqrt(phi_squared_sum + local_term)
                        factor = 1 - local_term / denominator
                    else:
                        factor = 0

                    factor = max(factor, 0)
                    phi_prime[y, x] = factor * phi_l_b[y, x]

            level_coefficients.append(phi_prime)
        updated_coefficients.append(level_coefficients)

    return updated_coefficients

def update_coefficients(mandrill_t, phi_prime_matrices):
    """
    Update wavelet coefficients with filtered values.
    
    Args:
        mandrill_t: DTCWT transformed data
        phi_prime_matrices (list): Modified wavelet coefficients
    """
    for level, level_matrices in enumerate(phi_prime_matrices):
        for band, phi_prime in enumerate(level_matrices):
            if band < mandrill_t.highpasses[level].shape[2]:
                mandrill_t.highpasses[level][:, :, band] = phi_prime

def perform_inverse_dtcwt_transform(transformed_data):
    """
    Perform inverse DTCWT transform.
    
    Args:
        transformed_data: DTCWT transformed data
        
    Returns:
        numpy.ndarray: Reconstructed image
    """
    transform = dtcwt.Transform2d(biort='near_sym_a', qshift='qshift_a')
    return transform.inverse(transformed_data)

def calculate_lifetime(g, s, omega):
    """
    Calculate fluorescence lifetime from G and S phasor coordinates.
    
    Args:
        g (numpy.ndarray): G phasor coordinates
        s (numpy.ndarray): S phasor coordinates
        omega (float): Angular frequency in rad/ns
        
    Returns:
        numpy.ndarray: Lifetime values in nanoseconds
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # Phase lifetime calculation
        lifetime = s / (g * omega)
        
        # Exclude invalid values
        lifetime[~np.isfinite(lifetime)] = np.nan
        lifetime[lifetime < 0] = np.nan
        lifetime[lifetime > 20] = np.nan  # Cap at reasonable value
        
    return lifetime

def process_dataset(g_file, s_file, int_file, flevel, omega, ref_g, ref_s):
    """
    Process a dataset with complex wavelet filtering.
    
    Args:
        g_file (str): Path to G file
        s_file (str): Path to S file
        int_file (str): Path to intensity file
        flevel (int): Filter level
        omega (float): Angular frequency
        ref_g (float): Reference G coordinate
        ref_s (float): Reference S coordinate
        
    Returns:
        dict: Processed data dictionary or None if error
    """
    try:
        # Step 1: Load files
        print(f"  Loading files...")
        g_data, g_dummy = load_tiff(g_file)
        s_data, s_dummy = load_tiff(s_file)
        int_data, int_dummy = load_tiff(int_file)
        
        # Check if all files are valid
        if g_dummy or s_dummy or int_dummy:
            print("  Warning: One or more input files could not be loaded properly.")
            if g_dummy and s_dummy and int_dummy:
                print("  All files are invalid. Cannot proceed.")
                return None
        
        # Step 2: Make sure dimensions match
        if g_data.shape != s_data.shape or g_data.shape != int_data.shape:
            print(f"  Warning: Dimensions don't match: G={g_data.shape}, S={s_data.shape}, Int={int_data.shape}")
            min_rows = min(g_data.shape[0], s_data.shape[0], int_data.shape[0])
            min_cols = min(g_data.shape[1], s_data.shape[1], int_data.shape[1])
            g_data = g_data[:min_rows, :min_cols]
            s_data = s_data[:min_rows, :min_cols]
            int_data = int_data[:min_rows, :min_cols]
        
        # Step 3: Calculate unfiltered lifetime
        print(f"  Calculating unfiltered lifetime...")
        lifetime_unfiltered = calculate_lifetime(g_data, s_data, omega)
        
        # Step 4: Apply complex wavelet filtering
        print(f"  Applying complex wavelet filtering (level {flevel})...")

        # Compute Fourier coefficients
        Freal_rescale = g_data * int_data
        Fimag_rescale = s_data * int_data

        # Freal transformations and filtering
        print(f"  Processing real Fourier coefficients...")
        Freal_ans = anscombe_transform(Freal_rescale)
        Freal_transformed, Freal_transformed_object = perform_dtcwt_transform(Freal_ans, flevel)
        median_values = calculate_median_values(Freal_transformed)
        sigma_g_squared = median_values / 0.6745
        sigma_n_squared = calculate_local_noise_variance(Freal_transformed, flevel)
        phi_prime = compute_phi_prime(Freal_transformed, sigma_g_squared, sigma_n_squared)
        update_coefficients(Freal_transformed, phi_prime)
        Freal_reconstructed_filtered = perform_inverse_dtcwt_transform(Freal_transformed)
        Freal_filtered = reverse_anscombe_transform(Freal_reconstructed_filtered)

        # Fimag transformations and filtering
        print(f"  Processing imaginary Fourier coefficients...")
        Fimag_ans = anscombe_transform(Fimag_rescale)
        Fimag_transformed, Fimag_transformed_object = perform_dtcwt_transform(Fimag_ans, flevel)
        median_values = calculate_median_values(Fimag_transformed)
        sigma_g_squared = median_values / 0.6745
        sigma_n_squared = calculate_local_noise_variance(Fimag_transformed, flevel)
        phi_prime = compute_phi_prime(Fimag_transformed, sigma_g_squared, sigma_n_squared)
        update_coefficients(Fimag_transformed, phi_prime)
        Fimag_reconstructed_filtered = perform_inverse_dtcwt_transform(Fimag_transformed)
        Fimag_filtered = reverse_anscombe_transform(Fimag_reconstructed_filtered)

        # Intensity transformations and filtering
        print(f"  Processing intensity values...")
        Intensity_ans = anscombe_transform(int_data)
        Intensity_transformed, Intensity_transformed_object = perform_dtcwt_transform(Intensity_ans, flevel)
        median_values = calculate_median_values(Intensity_transformed)
        sigma_g_squared = median_values / 0.6745
        sigma_n_squared = calculate_local_noise_variance(Intensity_transformed, flevel)
        phi_prime = compute_phi_prime(Intensity_transformed, sigma_g_squared, sigma_n_squared)
        update_coefficients(Intensity_transformed, phi_prime)
        Intensity_reconstructed_filtered = perform_inverse_dtcwt_transform(Intensity_transformed)
        Intensity_filtered = reverse_anscombe_transform(Intensity_reconstructed_filtered)

        # Calculate filtered G and S by dividing filtered Fourier coefficients by filtered intensity
        with np.errstate(divide='ignore', invalid='ignore'):
            g_filtered = Freal_filtered / Intensity_filtered
            s_filtered = Fimag_filtered / Intensity_filtered

        # Handle NaN values
        g_filtered_clean = np.nan_to_num(g_filtered)
        s_filtered_clean = np.nan_to_num(s_filtered)
        
        # Step 5: Calculate filtered lifetime using filtered G and S
        print(f"  Calculating filtered lifetime...")
        lifetime_filtered = calculate_lifetime(g_filtered_clean, s_filtered_clean, omega)
        
        # Step 6: Create output dictionary with required structure
        result = {
            'G': g_filtered_clean,      # Wavelet-filtered G
            'S': s_filtered_clean,      # Wavelet-filtered S
            'A': int_data,              # Original intensity (not filtered)
            'T': lifetime_filtered,     # Filtered lifetime (from filtered G and S)
            'GU': g_data,               # Unfiltered G
            'SU': s_data,               # Unfiltered S
            'TU': lifetime_unfiltered   # Unfiltered lifetime
        }
        
        # Add metadata
        result['ref_g'] = ref_g
        result['ref_s'] = ref_s
        result['flevel'] = flevel
        result['omega'] = omega
        
        return result
        
    except Exception as e:
        print(f"  Error in complex wavelet filtering: {e}")
        traceback.print_exc()
        return None

def save_npz(file_path, data_dict):
    """Save processed data to NPZ file.
    
    Args:
        file_path (str): Output file path
        data_dict (dict): Data dictionary to save
        
    Returns:
        bool: True if successful
    """
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the file
        np.savez(file_path, **data_dict)
        return True
    except Exception as e:
        print(f"Error saving NPZ file: {e}", file=sys.stderr)
        return False
