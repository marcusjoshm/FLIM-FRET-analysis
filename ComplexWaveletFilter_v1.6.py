#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:45:05 2024

@author: joshuamarcus
"""

import os
import numpy as np
from PIL import Image
import dtcwt
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from matplotlib import colors
import math
import tifffile as tiff
import time
import json
import sys
import warnings

# COMPLEX WAVELET FILTER v1.6

# Batch processing multiple datasets at one time

# File paths and parameters
main_preprocessed_directory = '/Users/leelab/Documents/JM_data/analysis/preprocessed'
output_base_directory = '/Users/leelab/Documents/JM_data/analysis/processed'
Gc = 0.30227996721890404  # G coordinate for the reference lifetime
Sc = 0.4592458920992018  # S coordinate for the reference lifetime
flevel = 9  # levels of complex wavelet filtering
processing_conditions_filter = "G3BP1-Sac1_FRET_CWFlevels=9"
processing_conditions_unfiltered = "G3BP1-Sac1_FRET_unfiltered"

# File handling functions
def load_and_process_image(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    image = Image.open(file_path)
    image_array = np.array(image)
    return np.nan_to_num(image_array)

# Complex wavelet filter calculation functions
def anscombe_transform(data):
    return 2 * np.sqrt(data + (3/8))

def perform_dtcwt_transform(data, N):
    transform = dtcwt.Transform2d(biort='Legall', qshift='qshift_a')
    transformed_data = transform.forward(data, nlevels=N, include_scale=False)
    return transformed_data, transform

def calculate_median_values(transformed_data):
    median_values = []
    for level in range(len(transformed_data.highpasses)):
        highpasses = transformed_data.highpasses[level if level else 0]
        for band in range(6):
            coeffs = highpasses[:, :, band]
            median_absolute = np.median(np.abs(coeffs.flatten()))
            median_values.append(median_absolute)
    return np.mean(median_values)

def calculate_local_noise_variance(transformed_data, N):
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
    num_bands = 6

    for level in range(num_levels):
        highpasses = transformed_data.highpasses[level]
        for band in range(num_bands):
            coeffs = highpasses[:, :, band]
            sigma_n_squared = local_noise_variance(coeffs, N)
            sigma_n_squared_matrices.append((level, band, sigma_n_squared))

    return sigma_n_squared_matrices

def compute_phi_prime(mandrill_t, sigma_g_squared, sigma_n_squared_matrices):
    updated_coefficients = []

    max_level = len(mandrill_t.highpasses) - 1
    local_term = np.sqrt(3) * np.sqrt(sigma_g_squared)

    for level in range(max_level):
        highpasses_l = mandrill_t.highpasses[level]
        highpasses_l_plus_1 = mandrill_t.highpasses[level + 1]
        level_coefficients = []

        for band in range(6):
            phi_l_b = highpasses_l[:, :, band]
            phi_l_plus_1_b = highpasses_l_plus_1[:, :, band]

            _, _, sigma_n_squared = sigma_n_squared_matrices[level * 6 + band]
            phi_prime = np.zeros_like(phi_l_b, dtype=complex)

            downsample_factor = phi_l_b.shape[0] // sigma_n_squared.shape[0]

            for x in range(phi_l_b.shape[1]):
                for y in range(phi_l_b.shape[0]):
                    x_half = x // 2
                    y_half = y // 2

                    x_downsampled = x // downsample_factor
                    y_downsampled = y // downsample_factor

                    phi_squared_sum = np.abs(phi_l_b[y, x])**2 + np.abs(phi_l_plus_1_b[y_half, x_half])**2

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
    for level, level_matrices in enumerate(phi_prime_matrices):
        for band, phi_prime in enumerate(level_matrices):
            mandrill_t.highpasses[level][:, :, band] = phi_prime

def perform_inverse_dtcwt_transform(transformed_data):
    transform = dtcwt.Transform2d(biort='Legall', qshift='qshift_a')
    return transform.inverse(transformed_data)
    
def reverse_anscombe_transform(y):
    y = np.asarray(y, dtype=np.float64)
    inverse = (
        (y**2 / 4) +
        (np.sqrt(3/2) * (1/y) / 4) -
        (11 / (8 * y**2)) +
        (np.sqrt(5/2) * (1/y**3) / 8) -
        (1 / (8 * y**4))
    )
    return inverse

# GMM filtering functions
def is_point_inside_circle(point, center, radius):
    distance = math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
    if distance <= radius:
        return True
    else:
        return False

def are_points_inside_circle(points, center, radius):
    results = []
    for point in points:
        results.append(is_point_inside_circle(point, center, radius))
    return results

# Function #4: check if points are inside the ellipse
def is_points_inside_rotated_ellipse(center_x, center_y, semi_major_axis, semi_minor_axis, angle_degrees, points):
    # Calculate the distance between each point and the center of the ellipse
    distances = [(point[0] - center_x)**2 + (point[1] - center_y)**2 for point in points]
    
    # Check if the ellipse is a circle (semi-major and semi-minor axes are equal)
    is_circle = math.isclose(semi_major_axis, semi_minor_axis)
    
    results = []
    
    if is_circle:
        # If it's a circle, check if each point is inside the circle
        for distance in distances:
            results.append(distance <= semi_major_axis**2)
    else:
        # Calculate the rotation angle of the ellipse
        angle_radians = math.radians(angle_degrees)
        cos_a = math.cos(angle_radians)
        sin_a = math.sin(angle_radians)

        for i, point in enumerate(points):
            point_x, point_y = point

            # Translate the point to the ellipse's coordinate system
            translated_x = point_x - center_x
            translated_y = point_y - center_y

            # Apply the rotation transformation
            rotated_x = cos_a * translated_x + sin_a * translated_y
            rotated_y = -sin_a * translated_x + cos_a * translated_y

            # Calculate the normalized coordinates
            normalized_x = rotated_x / semi_major_axis
            normalized_y = rotated_y / semi_minor_axis

            # Check if the transformed point is inside the unrotated ellipse
            results.append(normalized_x ** 2 + normalized_y ** 2 <= 1)

    return results

def check_either_value_greater_than_zero(list1, list2):
    results = [x > 0 or y > 0 for x, y in zip(list1, list2)]
    return results

def convert_list_to_array_with_dimensions(lst, rows, columns):
    array = np.array(lst)
    array_with_dimensions = array.reshape(rows, columns)
    return array_with_dimensions

# Complex wavelet filter batch processing function
def process_files(file_paths, G_combined, S_combined, I_combined):
    G_unfil = load_and_process_image(file_paths["G_unfiltered"])
    S_unfil = load_and_process_image(file_paths["S_unfiltered"])
    Intensity = load_and_process_image(file_paths["intensity"])

    if G_unfil is None or S_unfil is None or Intensity is None:
        print("One or more files could not be loaded. Skipping this replicate.")
        return G_combined, S_combined, I_combined
    
    # Compute Fourier coefficients
    Freal_rescale = G_unfil * Intensity
    Fimag_rescale = S_unfil * Intensity

    # Freal transformations and filtering
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
    Intensity_ans = anscombe_transform(Intensity)
    Intensity_transformed, Intensity_transformed_object = perform_dtcwt_transform(Intensity_ans, flevel)
    median_values = calculate_median_values(Intensity_transformed)
    sigma_g_squared = median_values / 0.6745
    sigma_n_squared = calculate_local_noise_variance(Intensity_transformed, flevel)
    phi_prime = compute_phi_prime(Intensity_transformed, sigma_g_squared, sigma_n_squared)
    update_coefficients(Intensity_transformed, phi_prime)
    Intensity_reconstructed_filtered = perform_inverse_dtcwt_transform(Intensity_transformed)
    Intensity_filtered = reverse_anscombe_transform(Intensity_reconstructed_filtered)

    G_wavelet_filtered = Freal_filtered / Intensity_filtered
    S_wavelet_filtered = Fimag_filtered / Intensity_filtered

    threshold = 0
    
    G_array = np.nan_to_num(G_wavelet_filtered)
    S_array = np.nan_to_num(S_wavelet_filtered)
    I_array = np.nan_to_num(Intensity)

    threshold_array = I_array > threshold
    G001_array = np.clip(G_array * threshold_array, -0.1, 1.1)
    S001_array = np.clip(S_array * threshold_array, -0.1, 1.1)
    
    # Append to combined arrays
    if G_combined.size == 0:
        G_combined = G001_array
    else:
        G_combined = np.hstack((G_combined, G001_array))

    if S_combined.size == 0:
        S_combined = S001_array
    else:
        S_combined = np.hstack((S_combined, S001_array))

    if I_combined.size == 0:
        I_combined = I_array
    else:
        I_combined = np.hstack((I_combined, I_array))
    
    return G_combined, S_combined, I_combined

# Complex wavelet filter batch processing function
def process_unfil_files(file_paths, G_combined, S_combined, I_combined):
    G_unfil = load_and_process_image(file_paths["G_unfiltered"])
    S_unfil = load_and_process_image(file_paths["S_unfiltered"])
    Intensity = load_and_process_image(file_paths["intensity"])

    if G_unfil is None or S_unfil is None or Intensity is None:
        print("One or more files could not be loaded. Skipping this replicate.")
        return G_combined, S_combined, I_combined

    threshold = 0
    
    G_array = np.nan_to_num(G_unfil)
    S_array = np.nan_to_num(S_unfil)
    I_array = np.nan_to_num(Intensity)

    threshold_array = I_array > threshold
    G001_array = np.clip(G_array * threshold_array, -0.1, 1.1)
    S001_array = np.clip(S_array * threshold_array, -0.1, 1.1)
    
    # Append to combined arrays
    if G_combined.size == 0:
        G_combined = G001_array
    else:
        G_combined = np.hstack((G_combined, G001_array))

    if S_combined.size == 0:
        S_combined = S001_array
    else:
        S_combined = np.hstack((S_combined, S001_array))

    if I_combined.size == 0:
        I_combined = I_array
    else:
        I_combined = np.hstack((I_combined, I_array))
    
    return G_combined, S_combined, I_combined

# Phasor and lifetime calculation from batch processed data
def plot_combined_data(G_combined, S_combined, I_combined, phasor_output_path):
    G_combined_flat = G_combined.ravel()
    S_combined_flat = S_combined.ravel()
    I_combined_flat = I_combined.ravel().astype(int)

    x_scale = [-0.005, 1.005]
    y_scale = [0, 0.9]

    G001_weighted = np.repeat(G_combined_flat, I_combined_flat)
    S001_weighted = np.repeat(S_combined_flat, I_combined_flat)
    G001_weighted = np.nan_to_num(G001_weighted)
    S001_weighted = np.nan_to_num(S001_weighted)

    x = np.linspace(0, 1.0, 100)
    y = np.linspace(0, 1.0, 100)
    X, Y = np.meshgrid(x, y)
    F = (X**2 + Y**2 - X)

    iqr_x = np.percentile(G001_weighted, 75) - np.percentile(G001_weighted, 25)
    bin_width_x = 2 * iqr_x * (len(G001_weighted) ** (-1/3))
    bin_width_x = np.nan_to_num(bin_width_x)

    iqr_y = np.percentile(S001_weighted, 75) - np.percentile(S001_weighted, 25)
    bin_width_y = 2 * iqr_y * (len(S001_weighted) ** (-1/3))
    bin_width_y = np.nan_to_num(bin_width_y)

    num_bins_x_G001 = int(np.ceil((np.max(G001_weighted) - np.min(G001_weighted)) / bin_width_x)) // 4
    num_bins_y_G001 = int(np.ceil((np.max(S001_weighted) - np.min(S001_weighted)) / bin_width_y)) // 4

    hist_vals, _, _ = np.histogram2d(G_combined_flat, S_combined_flat, bins=(num_bins_x_G001, num_bins_y_G001), weights=I_combined_flat)
    vmax = hist_vals.max()
    vmin = hist_vals.min()

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.hist2d(G_combined_flat, S_combined_flat, bins=(num_bins_x_G001, num_bins_y_G001), weights=I_combined_flat, cmap='nipy_spectral', norm=colors.SymLogNorm(linthresh=50, linscale=1, vmax=vmax, vmin=vmin), zorder=1, cmin=0.01)
    ax.set_facecolor('white')
    ax.set_xlabel('\n$G$')
    ax.set_ylabel('$S$\n')
    ax.set_xlim(x_scale)
    ax.set_ylim(y_scale)
    ax.contour(X, Y, F, [0], colors='black', linewidths=1, zorder=2)

    near_zero = 0.1
    cbar = fig.colorbar(h[3], ax=ax, format=LogFormatter(10, labelOnlyBase=True))
    ticks = [near_zero] + [10**i for i in range(1, int(np.log10(vmax)) + 1)]
    cbar.set_ticks(ticks)
    tick_labels = ['0'] + [f'$10^{i}$' for i in range(1, int(np.log10(vmax)) + 1)]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Frequency')

    fig.tight_layout()
    fig.savefig(phasor_output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
def calculate_and_plot_lifetime(G_combined, S_combined):
    """
    Calculate the fluorescence lifetime from combined G and S components and plot the result.

    Parameters:
    G_combined (numpy.ndarray): Combined G component array
    S_combined (numpy.ndarray): Combined S component array

    Returns:
    numpy.ndarray: Calculated fluorescence lifetime array
    """
    # Calculate phase angle theta
    SoverG = S_combined / G_combined
    theta1 = np.arctan(SoverG)

    # Calculate tangent of phase angle
    tantheta = np.tan(theta1)

    # 78mHz = period of laser - needs to be converted to nanoseconds (value below)
    ns = 12.820512820513

    # Calculate angular frequency (w) from period in nanoseconds (ns)
    w = (2 * np.pi) / ns

    # Fluorescence lifetime of image or first ROI
    T = tantheta / w

    # Plot fluorescence lifetime with colorbar (T)
    T[np.isnan(T)] = 0
    plt.imshow(T)
    plt.colorbar()
    plt.show()

    return T

# --- Configuration Loading ---
def load_config(config_path="config.json"):
    """Loads configuration from a JSON file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        # Basic validation
        required_keys = ["preprocessed_dir", "filtered_dir", "wavelet_filter_params"]
        if not all(key in config for key in required_keys):
            raise ValueError("Config file missing required keys: preprocessed_dir, filtered_dir, wavelet_filter_params")
        if not all(key in config["wavelet_filter_params"] for key in ["Gc", "Sc", "flevel"]):
             raise ValueError("Config file missing required wavelet_filter_params keys: Gc, Sc, flevel")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Configuration file {config_path} is not valid JSON.", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error in configuration file: {e}", file=sys.stderr)
        sys.exit(1)

# --- Main Processing Function ---
def apply_wavelet_filter(g_unfil_path, s_unfil_path, int_path, flevel):
    """
    Applies DT-CWT filtering to a set of G, S, and Intensity images.

    Args:
        g_unfil_path (str): Path to the unfiltered G-coordinate TIFF file.
        s_unfil_path (str): Path to the unfiltered S-coordinate TIFF file.
        int_path (str): Path to the intensity TIFF file.
        flevel (int): Number of wavelet decomposition levels.

    Returns:
        tuple: (G_filtered, S_filtered, Intensity) numpy arrays, or (None, None, None) on error.
    """
    print(f"  Loading G: {os.path.basename(g_unfil_path)}, S: {os.path.basename(s_unfil_path)}, I: {os.path.basename(int_path)}")
    G_unfil = load_and_process_image(g_unfil_path)
    S_unfil = load_and_process_image(s_unfil_path)
    Intensity = load_and_process_image(int_path)

    if G_unfil is None or S_unfil is None or Intensity is None:
        print("  Error: One or more input files could not be loaded. Skipping.", file=sys.stderr)
        return None, None, None
        
    if not (G_unfil.shape == S_unfil.shape == Intensity.shape):
         print(f"  Error: Input images for {os.path.basename(g_unfil_path)} do not have matching dimensions. Skipping.", file=sys.stderr)
         return None, None, None

    print(f"  Applying filter (flevel={flevel})...")
    # Compute rescaled Fourier coefficients (real and imaginary parts)
    Freal_rescale = G_unfil * Intensity
    Fimag_rescale = S_unfil * Intensity

    # --- Process Freal (G * Intensity) ---
    # 1. Anscombe transform
    Freal_ans = anscombe_transform(Freal_rescale)
    # 2. Forward DTCWT
    Freal_transformed = perform_dtcwt_transform(Freal_ans, flevel)
    # 3. Estimate noise variance (using simple global estimate for now)
    sigma_g_sq_freal = calculate_sigma_g_squared(Freal_transformed)
    # 4. Apply Thresholding (using simple soft thresholding)
    simple_soft_thresholding(Freal_transformed, sigma_g_sq_freal, flevel) 
    # 5. Inverse DTCWT
    Freal_reconstructed_filtered_ans = perform_inverse_dtcwt_transform(Freal_transformed)
    # 6. Inverse Anscombe transform
    Freal_filtered = reverse_anscombe_transform(Freal_reconstructed_filtered_ans)

    # --- Process Fimag (S * Intensity) ---
    # 1. Anscombe transform
    Fimag_ans = anscombe_transform(Fimag_rescale)
    # 2. Forward DTCWT
    Fimag_transformed = perform_dtcwt_transform(Fimag_ans, flevel)
    # 3. Estimate noise variance
    sigma_g_sq_fimag = calculate_sigma_g_squared(Fimag_transformed)
    # 4. Apply Thresholding
    simple_soft_thresholding(Fimag_transformed, sigma_g_sq_fimag, flevel)
    # 5. Inverse DTCWT
    Fimag_reconstructed_filtered_ans = perform_inverse_dtcwt_transform(Fimag_transformed)
    # 6. Inverse Anscombe transform
    Fimag_filtered = reverse_anscombe_transform(Fimag_reconstructed_filtered_ans)

    # --- Calculate final filtered G and S ---
    # Handle division by zero or near-zero intensity
    intensity_threshold = 1e-9 # Use a small threshold
    valid_intensity_mask = Intensity > intensity_threshold

    G_filtered = np.full_like(Intensity, np.nan, dtype=np.float64) 
    S_filtered = np.full_like(Intensity, np.nan, dtype=np.float64)

    # Use np.divide with the 'where' clause for safe division
    np.divide(Freal_filtered, Intensity, out=G_filtered, where=valid_intensity_mask)
    np.divide(Fimag_filtered, Intensity, out=S_filtered, where=valid_intensity_mask)

    # Replace any remaining NaNs (from division by zero) with 0.0
    G_filtered = np.nan_to_num(G_filtered)
    S_filtered = np.nan_to_num(S_filtered)

    print("  Filtering complete.")
    return G_filtered, S_filtered, Intensity # Return original intensity for saving

def main():
    """Main execution function: loads config, finds files, processes, saves results."""
    config = load_config()
    
    preprocessed_dir = config["preprocessed_dir"]
    filtered_dir = config["filtered_dir"]
    flevel = config["wavelet_filter_params"]["flevel"]
    # Gc, Sc are loaded but not used in this refactored version (removed plotting/GMM)
    # Gc = config["wavelet_filter_params"]["Gc"] 
    # Sc = config["wavelet_filter_params"]["Sc"]

    print(f"Starting Complex Wavelet Filtering")
    print(f"Input (preprocessed) directory: {preprocessed_dir}")
    print(f"Output (filtered) directory: {filtered_dir}")
    print(f"Wavelet levels (flevel): {flevel}")

    if not os.path.isdir(preprocessed_dir):
        print(f"Error: Preprocessed directory not found: {preprocessed_dir}", file=sys.stderr)
        sys.exit(1)
        
    processed_count = 0
    skipped_count = 0

    # Iterate through condition subdirectories
    for condition_subdir_name in os.listdir(preprocessed_dir):
        condition_subdir_path = os.path.join(preprocessed_dir, condition_subdir_name)
        if not os.path.isdir(condition_subdir_path):
            continue # Skip files, only process directories

        print(f"\nProcessing condition: {condition_subdir_name}")
        
        g_unfil_dir = os.path.join(condition_subdir_path, "G_unfiltered")
        s_unfil_dir = os.path.join(condition_subdir_path, "S_unfiltered")
        int_dir = os.path.join(condition_subdir_path, "intensity")

        if not (os.path.isdir(g_unfil_dir) and os.path.isdir(s_unfil_dir) and os.path.isdir(int_dir)):
            print(f" Warning: Missing G_unfiltered, S_unfiltered, or intensity subfolder in {condition_subdir_name}. Skipping.", file=sys.stderr)
            skipped_count += 1
            continue

        # Find unique numerical filenames (assuming '.tiff' extension)
        try:
             # Use intensity dir as reference, assuming all dirs have matching files
            file_list = [f for f in os.listdir(int_dir) if f.lower().endswith(".tiff")]
            num_names = sorted(list(set(os.path.splitext(f)[0] for f in file_list)))
            if not num_names:
                 print(f" Warning: No .tiff files found in intensity subfolder for {condition_subdir_name}. Skipping.")
                 skipped_count +=1
                 continue
        except OSError as e:
             print(f" Error listing files in {int_dir}: {e}. Skipping condition.", file=sys.stderr)
             skipped_count += 1
             continue
             
        print(f" Found {len(num_names)} numerical file sets to process: {', '.join(num_names)}")

        # Process each numerical file set
        for num_name in num_names:
            g_unfil_path = os.path.join(g_unfil_dir, f"{num_name}.tiff")
            s_unfil_path = os.path.join(s_unfil_dir, f"{num_name}.tiff")
            int_path = os.path.join(int_dir, f"{num_name}.tiff")

            # Check if all three files for this number exist before processing
            if not (os.path.exists(g_unfil_path) and os.path.exists(s_unfil_path) and os.path.exists(int_path)):
                 print(f" Warning: Missing G, S, or Intensity file for '{num_name}' in {condition_subdir_name}. Skipping this set.", file=sys.stderr)
                 skipped_count += 1
                 continue

            try:
                # Apply the filter
                G_filtered, S_filtered, Intensity_orig = apply_wavelet_filter(
                    g_unfil_path, s_unfil_path, int_path, flevel
                )

                if G_filtered is not None: # Check if processing was successful
                    # Define output paths
                    output_condition_dir = os.path.join(filtered_dir, condition_subdir_name)
                    g_filt_path = os.path.join(output_condition_dir, f"{num_name}_G_CWF.tiff")
                    s_filt_path = os.path.join(output_condition_dir, f"{num_name}_S_CWF.tiff")
                    int_out_path = os.path.join(output_condition_dir, f"{num_name}_Intensity.tiff") # Save original intensity too

                    print(f"  Saving filtered outputs to: {output_condition_dir}")
                    # Save the results
                    save_image(g_filt_path, G_filtered)
                    save_image(s_filt_path, S_filtered)
                    save_image(int_out_path, Intensity_orig) 
                    processed_count += 1
                else:
                    skipped_count += 1 # Processing failed within apply_wavelet_filter

            except Exception as e:
                print(f" Error processing file set '{num_name}' in {condition_subdir_name}: {e}", file=sys.stderr)
                skipped_count += 1

    print(f"\nComplex Wavelet Filtering finished.")
    print(f"Successfully processed file sets: {processed_count}")
    print(f"Skipped/failed file sets: {skipped_count}")


if __name__ == "__main__":
    # Add basic warning filtering if desired
    # warnings.simplefilter('ignore', category=RuntimeWarning) 
    main()

