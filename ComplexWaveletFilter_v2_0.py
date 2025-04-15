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
    # Extract parameters from config, using defaults if not found
    try:
        freq_mhz = config["microscope_params"]["frequency"]
        harmonic = config["microscope_params"]["harmonic"]
    except KeyError as e:
        print(f"Warning: Missing parameter in config: {e}. Using defaults.")
        freq_mhz = 78.0  # Default frequency in MHz
        harmonic = 1     # Default harmonic
    
    # Hard-code filter level to 9 as specified
    flevel = DEFAULT_FLEVEL
    
    # Calculate angular frequency
    omega_rad_per_ns = 2 * np.pi * freq_mhz * 1e6 * harmonic * 1e-9
    
    print(f"\nStarting Complex Wavelet Filtering with advanced implementation")
    print(f"Input directory: {preprocessed_dir}")
    print(f"Output directory: {npz_dir}")
    print(f"Filter level: {flevel}")
    print(f"Reference fluorophore: G={REF_G}, S={REF_S}")
    print(f"Calculated Omega (rad/ns): {omega_rad_per_ns:.4f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(npz_dir, exist_ok=True)
    
    if not os.path.isdir(preprocessed_dir):
        print(f"Error: Input directory not found: {preprocessed_dir}", file=sys.stderr)
        return False
    
    # Process files
    processed_count = 0
    skipped_count = 0
    
    # Walk through input directory to find G, S, and intensity files
    for root, dirs, files in os.walk(preprocessed_dir):
        # Look for our target files
        g_files = [f for f in files if f.lower().endswith('.g.tiff') or f.lower().endswith('.g.tif')]
        s_files = [f for f in files if f.lower().endswith('.s.tiff') or f.lower().endswith('.s.tif')]
        intensity_files = [f for f in files if f.lower().endswith('.intensity.tiff') or f.lower().endswith('.intensity.tif')]
        
        # Skip directories without our target files
        if not g_files and not s_files and not intensity_files:
            continue
        
        relative_path = os.path.relpath(root, preprocessed_dir)
        print(f"\nFound files in: {relative_path if relative_path != '.' else 'root'}")
        print(f"  {len(g_files)} G files, {len(s_files)} S files, {len(intensity_files)} intensity files")
        
        # Group files by base name
        datasets = {}
        
        # Process G files - extract base name without extension
        for g_file in g_files:
            base_name = g_file[:-len(".g.tiff")] if g_file.endswith('.g.tiff') else g_file[:-len(".g.tif")]
            datasets.setdefault(base_name, {})['g'] = os.path.join(root, g_file)
            
        # Process S files
        for s_file in s_files:
            base_name = s_file[:-len(".s.tiff")] if s_file.endswith('.s.tiff') else s_file[:-len(".s.tif")]
            datasets.setdefault(base_name, {})['s'] = os.path.join(root, s_file)
            
        # Process intensity files
        for int_file in intensity_files:
            base_name = int_file[:-len(".intensity.tiff")] if int_file.endswith('.intensity.tiff') else int_file[:-len(".intensity.tif")]
            datasets.setdefault(base_name, {})['int'] = os.path.join(root, int_file)
        
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
                        REF_G,
                        REF_S
                    )
                    process_end = time.time()
                    
                    if result is not None:
                        # Construct output path mirroring input structure
                        output_dir = os.path.join(npz_dir, relative_path)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Save result to NPZ file
                        output_file = os.path.join(output_dir, f"{base_name}_processed.npz")
                        print(f"  Saving NPZ to: {output_file}")
                        save_npz(output_file, result)
                        
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
        N (int): Window size for local variance calculation
        
    Returns:
        list: Matrices of local noise variance for each level and band
    """
    sigma_n_squared_matrices = []

    def local_noise_variance(coeffs, N):
        sigma_n_squared = np.zeros_like(coeffs, dtype=float)
        height, width = coeffs.shape
        for x in range(width):
            for y in range(height):
                x_min, x_max = max(0, x - N), min(width, x + N + 1)
                y_min, y_max = max(0, y - N), min(height, y + N + 1)
                window = coeffs[y_min:y_max, x_min:x_max]
                local_variance = np.mean(np.abs(window)**2)
                sigma_n_squared[y, x] = local_variance
        return sigma_n_squared

    num_levels = len(transformed_data.highpasses)
    for level in range(num_levels):
        highpasses = transformed_data.highpasses[level]
        for band in range(highpasses.shape[2]):
            coeffs = highpasses[:, :, band]
            sigma_n_squared = local_noise_variance(coeffs, N)
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
        
        # Step 3: Create mask for valid pixels (intensity threshold)
        intensity_threshold = 5  # Minimum intensity to consider valid
        valid_mask = int_data > intensity_threshold
        
        # Apply mask to input data
        g_masked = np.copy(g_data)
        s_masked = np.copy(s_data)
        g_masked[~valid_mask] = 0
        s_masked[~valid_mask] = 0
        
        # Step 4: Calculate unfiltered lifetime
        print(f"  Calculating unfiltered lifetime...")
        lifetime_unfiltered = calculate_lifetime(g_data, s_data, omega)
        
        # Step 5: Apply complex wavelet filtering
        print(f"  Applying complex wavelet filtering (level {flevel})...")
        
        # Apply Anscombe transform
        g_anscombe = anscombe_transform(g_masked)
        s_anscombe = anscombe_transform(s_masked)
        
        # Perform DTCWT transform
        g_transform, g_transform_obj = perform_dtcwt_transform(g_anscombe, flevel)
        s_transform, s_transform_obj = perform_dtcwt_transform(s_anscombe, flevel)
        
        # Calculate noise parameters
        local_window_size = 3
        sigma_g_squared_g = calculate_median_values(g_transform) ** 2
        sigma_g_squared_s = calculate_median_values(s_transform) ** 2
        
        # Calculate local noise variance
        g_noise_var = calculate_local_noise_variance(g_transform, local_window_size)
        s_noise_var = calculate_local_noise_variance(s_transform, local_window_size)
        
        # Compute modified coefficients
        g_updated_coeffs = compute_phi_prime(g_transform, sigma_g_squared_g, g_noise_var)
        s_updated_coeffs = compute_phi_prime(s_transform, sigma_g_squared_s, s_noise_var)
        
        # Update coefficients
        update_coefficients(g_transform, g_updated_coeffs)
        update_coefficients(s_transform, s_updated_coeffs)
        
        # Perform inverse transform
        g_filtered_anscombe = perform_inverse_dtcwt_transform(g_transform)
        s_filtered_anscombe = perform_inverse_dtcwt_transform(s_transform)
        
        # Reverse Anscombe transform
        g_filtered = reverse_anscombe_transform(g_filtered_anscombe)
        s_filtered = reverse_anscombe_transform(s_filtered_anscombe)
        
        # Restore masked areas
        g_filtered[~valid_mask] = np.nan
        s_filtered[~valid_mask] = np.nan
        
        # Step 6: Calculate filtered lifetime
        print(f"  Calculating filtered lifetime...")
        lifetime_filtered = calculate_lifetime(g_filtered, s_data, omega)
        
        # Step 7: Create output dictionary with required structure
        result = {
            'G': g_filtered,          # Wavelet-filtered G
            'S': s_filtered,          # Wavelet-filtered S
            'A': int_data,            # Intensity 
            'T': lifetime_filtered,   # Filtered lifetime
            'GU': g_data,             # Unfiltered G
            'SU': s_data,             # Unfiltered S
            'TU': lifetime_unfiltered # Unfiltered lifetime
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
    """
    Save processed data to NPZ file.
    
    Args:
        file_path (str): Output file path
        data_dict (dict): Data dictionary to save
        
    Returns:
        bool: True if successful
    """
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Extract arrays and metadata
        arrays = {}
        metadata = {}
        
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray) and value.size > 1:
                arrays[key] = value
            else:
                metadata[key] = value
        
        # Save the arrays
        np.savez(file_path, **arrays)
        
        # Optionally save metadata separately
        metadata_path = file_path.replace('.npz', '_metadata.npz')
        if metadata:
            np.savez(metadata_path, **metadata)
            
        return True
        
    except Exception as e:
        print(f"  Error saving NPZ file: {e}")
        return False

if __name__ == "__main__":
    print("This script is intended to be run via run_pipeline.py")
    print("Please use: python run_pipeline.py --filter -i <input_dir> -o <output_base_dir>")
    sys.exit(1)
