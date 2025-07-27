#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCSPC Preprocessing Script (with auto calibration)
- Runs ImageJ macros to convert BIN files to TIF
- Processes TIF files using custom phasor transformation
- Organizes files into directories for wavelet filtering
"""

import os
import pandas as pd
import subprocess
import json
import sys
import shutil
import glob
from typing import Optional
import numpy as np
from PIL import Image
import traceback

def prompt_file_selection(input_dir, file_extension='.bin'):
    """
    Prompt user to select specific files for processing from the input directory.
    
    Args:
        input_dir (str): Directory to search for files
        file_extension (str): File extension to search for (default: '.bin')
    
    Returns:
        list: List of selected file paths, or None if cancelled
    """
    print(f"\n=== File Selection for Preprocessing ===")
    print(f"Searching for {file_extension} files in: {input_dir}")
    
    # Find all files with the specified extension
    found_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(file_extension.lower()):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, input_dir)
                found_files.append((full_path, rel_path))
    
    if not found_files:
        print(f"No {file_extension} files found in {input_dir}")
        return None
    
    # Sort files by relative path for consistent display
    found_files.sort(key=lambda x: x[1])
    
    print(f"\nFound {len(found_files)} {file_extension} files:")
    for i, (full_path, rel_path) in enumerate(found_files, 1):
        file_size = os.path.getsize(full_path)
        size_mb = file_size / (1024 * 1024)
        print(f"  [{i:2d}] {rel_path} ({size_mb:.1f} MB)")
    
    print(f"\nSelection options:")
    print(f"  - Enter specific numbers (e.g., 1,3,5 or 1-5)")
    print(f"  - Enter 'all' to select all files")
    print(f"  - Enter 'cancel' to cancel selection")
    
    while True:
        try:
            user_input = input(f"\nSelect files to process: ").strip()
            
            if user_input.lower() == 'cancel':
                print("File selection cancelled.")
                return None
            
            if user_input.lower() == 'all':
                selected_files = [full_path for full_path, _ in found_files]
                print(f"Selected all {len(selected_files)} files.")
                return selected_files
            
            # Parse selection (numbers, ranges, and comma-separated values)
            selected_indices = set()
            
            # Split by comma
            parts = user_input.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    # Handle range (e.g., "1-5")
                    try:
                        start, end = map(int, part.split('-'))
                        for i in range(start, end + 1):
                            if 1 <= i <= len(found_files):
                                selected_indices.add(i)
                    except ValueError:
                        print(f"Invalid range format: {part}")
                        continue
                else:
                    # Handle single number
                    try:
                        index = int(part)
                        if 1 <= index <= len(found_files):
                            selected_indices.add(index)
                        else:
                            print(f"Index {index} out of range (1-{len(found_files)})")
                    except ValueError:
                        print(f"Invalid number: {part}")
                        continue
            
            if not selected_indices:
                print("No valid files selected. Please try again.")
                continue
            
            # Convert indices to file paths
            selected_files = []
            for index in sorted(selected_indices):
                full_path, rel_path = found_files[index - 1]
                selected_files.append(full_path)
            
            print(f"\nSelected {len(selected_files)} files:")
            for i, file_path in enumerate(selected_files, 1):
                rel_path = os.path.relpath(file_path, input_dir)
                print(f"  [{i}] {rel_path}")
            
            # Confirm selection
            confirm = input(f"\nProceed with these {len(selected_files)} files? (y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                return selected_files
            else:
                print("Selection cancelled. Please select again.")
                continue
                
        except KeyboardInterrupt:
            print("\nFile selection cancelled.")
            return None
        except Exception as e:
            print(f"Error during file selection: {e}")
            print("Please try again.")
            continue

def create_calibration_file_from_selection(selected_files, original_calibration_file, temp_calibration_file_path):
    """
    Create a calibration CSV file from selected files using values from the original calibration file.
    
    Args:
        selected_files (list): List of selected file paths
        original_calibration_file (str): Path to the original calibration CSV file
        temp_calibration_file_path (str): Path where to save the filtered calibration file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the original calibration file
        try:
            original_df = pd.read_csv(original_calibration_file, dtype={'file_path': str})
        except Exception as e:
            print(f"Error reading original calibration file {original_calibration_file}: {e}")
            return False
        
        # Normalize paths for comparison
        normalized_selected = [os.path.normpath(f) for f in selected_files]
        
        # Find matching entries in the original calibration file
        filtered_data = []
        found_files = set()
        
        for index, row in original_df.iterrows():
            cal_file_path = str(row['file_path']).strip()
            normalized_cal_path = os.path.normpath(cal_file_path)
            
            # Add .bin extension if not present for comparison
            if not normalized_cal_path.endswith('.bin'):
                normalized_cal_path += '.bin'
            
            # Check if this calibration entry matches any selected file
            if normalized_cal_path in normalized_selected:
                filtered_data.append(row.to_dict())
                found_files.add(normalized_cal_path)
        
        # Check for selected files that don't have calibration entries
        missing_files = set(normalized_selected) - found_files
        
        # Add entries for missing files with default values
        default_phi = 0.0
        default_m = 1.0
        
        # Try to determine default values from existing calibration data
        if len(filtered_data) > 0:
            # Use the most common values from the existing calibration as defaults
            phi_values = []
            m_values = []
            for entry in filtered_data:
                if 'phi_cal' in entry:
                    phi_values.append(float(entry['phi_cal']))
                elif 'phi' in entry:
                    phi_values.append(float(entry['phi']))
                
                if 'm_cal' in entry:
                    m_values.append(float(entry['m_cal']))
                elif 'modulation' in entry:
                    m_values.append(float(entry['modulation']))
            
            if phi_values:
                default_phi = phi_values[0]  # Use first available value
            if m_values:
                default_m = m_values[0]  # Use first available value
        
        for missing_file in missing_files:
            print(f"Warning: No calibration entry found for {missing_file}, using defaults (phi_cal={default_phi}, m_cal={default_m})")
            filtered_data.append({
                'file_path': missing_file,
                'phi_cal': default_phi,
                'm_cal': default_m
            })
        
        if not filtered_data:
            print("Error: No matching calibration data found for selected files")
            return False
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(filtered_data)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(temp_calibration_file_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(temp_calibration_file_path, index=False)
        
        print(f"Created filtered calibration file: {temp_calibration_file_path}")
        print(f"Contains {len(filtered_data)} files:")
        print(f"  - {len(found_files)} files with original calibration values")
        print(f"  - {len(missing_files)} files with default values")
        
        return True
        
    except Exception as e:
        print(f"Error creating filtered calibration file: {e}")
        return False

def run_imagej(imagej_path, macro_file, *args):
    """
    Run ImageJ with the specified macro file and arguments.
    Prints detailed debug output and performs error checking.
    
    Args:
        imagej_path (str): Path to ImageJ executable
        macro_file (str): Path to the macro file (.ijm)
        *args: Additional arguments to pass to the macro
    
    Returns:
        bool: True if successful, False if failed
    """
    # Try headless mode first, fall back to regular mode if it fails
    command_headless = [
        imagej_path,
        '-batch',      # Batch mode for faster processing
        '-macro', macro_file, ",".join(args)
    ]
    
    command_regular = [
        imagej_path,
        '-macro', macro_file, ",".join(args)
    ]
    
    print(f"Attempting ImageJ in batch mode: {' '.join(command_headless)}")
    
    try:
        # Try batch mode first
        result = subprocess.run(command_headless, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        print(f"ImageJ Command Output (batch mode): {result.stdout}")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Batch mode failed: {e}")
        print(f"Falling back to regular mode...")
        
        try:
            # Fall back to regular mode
            print(f"Running ImageJ in regular mode: {' '.join(command_regular)}")
            result = subprocess.run(command_regular, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
            print(f"ImageJ Command Output (regular mode): {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the macro: {e}")
            print(f"Exit code: {e.returncode}")
            print(f"Output: {e.stdout}")
            print(f"Error: {e.stderr}")
            return False
        except subprocess.TimeoutExpired as e:
            print(f"ImageJ macro timed out: {e}")
            return False
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check the path to the ImageJ executable.")
        return False
    except PermissionError as e:
        print(f"Permission error: {e}")
        print("Please check the permissions of the ImageJ executable.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def create_empty_tiffs_from_bins(calibration_file, input_dir, output_dir, raw_data_root):
    """
    Create empty .tif files in the output directory structure when ImageJ fails.
    This allows the phasor transformation to find the files it needs to process.
    
    Args:
        calibration_file (str): Path to the CSV file with bin file paths
        input_dir (str): The input directory
        output_dir (str): Where to create tif files
        raw_data_root (str): Root directory for raw data
    """
    print("Creating empty TIFF files from BIN files...")
    
    try:
        # Read calibration data
        calibration_data = pd.read_csv(calibration_file, dtype={'file_path': str})
    except Exception as e:
        print(f"Error reading calibration file: {e}")
        return False
    
    # Get normalized root path
    normalized_raw_root = os.path.normpath(raw_data_root)
    if not normalized_raw_root.endswith(os.path.sep):
        normalized_raw_root += os.path.sep
    
    created_count = 0
    skipped_count = 0
    
    # Process each bin file listed in the calibration CSV
    for index, row in calibration_data.iterrows():
        try:
            bin_file_full_path = str(row['file_path']).strip()
            normalized_bin_path = os.path.normpath(bin_file_full_path)
            
            # Check if path already has .bin extension
            if not normalized_bin_path.endswith('.bin'):
                normalized_bin_path += '.bin'
                print(f"Added .bin extension to path: {normalized_bin_path}")
            
            # Determine if bin file exists
            if not os.path.exists(normalized_bin_path):
                print(f"Warning: BIN file not found at: {normalized_bin_path}. Skipping.")
                skipped_count += 1
                continue
            else:
                print(f"Found BIN file at: {normalized_bin_path}")
                
            # Get relative path and construct target tif path
            if not normalized_bin_path.startswith(normalized_raw_root):
                print(f"Warning: BIN path {bin_file_full_path} does not start with raw_data_root {raw_data_root}. Skipping.")
                skipped_count += 1
                continue
                
            relative_path = os.path.relpath(normalized_bin_path, normalized_raw_root)
            relative_tif_path = os.path.splitext(relative_path)[0] + ".tif"
            target_tif_path = os.path.join(output_dir, relative_tif_path)
            
            # Create the output directory for this file if it doesn't exist
            os.makedirs(os.path.dirname(target_tif_path), exist_ok=True)
            
            # Create an empty tif file
            print(f"Creating empty TIF file at: {target_tif_path}")
            with open(target_tif_path, 'w') as f:
                f.write("# Empty TIF file created by preprocessing script\n")
            
            created_count += 1
            
        except Exception as e:
            print(f"Error processing bin file {bin_file_full_path}: {e}")
            skipped_count += 1
    
    print(f"Created {created_count} empty TIFF files, skipped {skipped_count}")
    return created_count > 0

def manually_copy_files_to_preprocessed(output_dir, preprocessed_dir):
    """
    Manually copy G, S, and intensity files from output to preprocessed directory
    maintaining the same directory structure.
    
    Args:
        output_dir (str): Source directory containing _g.tiff, _s.tiff, and _intensity.tiff files
        preprocessed_dir (str): Target directory for copied files
    """
    print("Manually copying files from output to preprocessed directory...")
    
    # Walk through the output directory
    g_count = 0
    s_count = 0
    intensity_count = 0
    
    for root, dirs, files in os.walk(output_dir):
        # Collect files by type
        g_files = [f for f in files if f.endswith('_g.tiff')]
        s_files = [f for f in files if f.endswith('_s.tiff')]
        intensity_files = [f for f in files if f.endswith('_intensity.tiff')]
        
        if not g_files and not s_files and not intensity_files:
            continue
            
        # Get relative path from output_dir
        rel_path = os.path.relpath(root, output_dir)
        print(f"Processing directory: {rel_path}")
        
        # Create target directory structure
        g_target_dir = os.path.join(preprocessed_dir, rel_path, "G_unfiltered")
        s_target_dir = os.path.join(preprocessed_dir, rel_path, "S_unfiltered")
        intensity_target_dir = os.path.join(preprocessed_dir, rel_path, "intensity")
        
        os.makedirs(g_target_dir, exist_ok=True)
        os.makedirs(s_target_dir, exist_ok=True)
        os.makedirs(intensity_target_dir, exist_ok=True)
        
        # Copy G files
        for g_file in g_files:
            src = os.path.join(root, g_file)
            dst = os.path.join(g_target_dir, g_file)
            shutil.copy2(src, dst)
            g_count += 1
            print(f"  Copied G file: {g_file} to {g_target_dir}")
            
        # Copy S files
        for s_file in s_files:
            src = os.path.join(root, s_file)
            dst = os.path.join(s_target_dir, s_file)
            shutil.copy2(src, dst)
            s_count += 1
            print(f"  Copied S file: {s_file} to {s_target_dir}")
            
        # Copy intensity files
        for intensity_file in intensity_files:
            src = os.path.join(root, intensity_file)
            dst = os.path.join(intensity_target_dir, intensity_file)
            shutil.copy2(src, dst)
            intensity_count += 1
            print(f"  Copied intensity file: {intensity_file} to {intensity_target_dir}")
    
    print(f"Copied {g_count} G files, {s_count} S files, and {intensity_count} intensity files to preprocessed directory.")
    return g_count > 0 and s_count > 0

def process_flim_file(input_file: str, output_dir: str, phi_cal: float, m_cal: float, 
                     bin_width_ns: float, freq_mhz: float, harmonic: int = 1,
                     apply_filter: int = 1, threshold_min: float = 0, 
                     threshold_max: Optional[float] = None) -> bool:
    """
    Process a FLIM file to generate phasor coordinates.
    (Moved from phasor_transform.py)
    """
    try:
        print(f"    Processing FLIM file: {input_file}")
        # Determine file type and load data
        if input_file.endswith('.bin'):
            print(f"    Error: Direct .bin file processing not implemented yet")
            return False
        elif input_file.endswith(('.tif', '.tiff')):
            try:
                with Image.open(input_file) as img:
                    if hasattr(img, 'n_frames'):
                        frames = []
                        for i in range(img.n_frames):
                            img.seek(i)
                            frames.append(np.array(img))
                        flim_data = np.stack(frames, axis=-1)
                    else:
                        flim_data = np.array(img)
                        if flim_data.ndim == 2:
                            flim_data = flim_data[:, :, np.newaxis]
            except Exception as e:
                print(f"    Error loading TIFF file: {e}")
                return False
        else:
            print(f"    Error: Unsupported file format: {input_file}")
            return False
        print(f"    Loaded data shape: {flim_data.shape}")
        omega = 2 * np.pi * freq_mhz * 1e6 * bin_width_ns * 1e-9 * harmonic
        if flim_data.ndim == 3:
            height, width, n_bins = flim_data.shape
        else:
            height, width = flim_data.shape
            n_bins = 1
            flim_data = flim_data[:, :, np.newaxis]
        print(f"    Calculating phasor coordinates (omega={omega:.6f})")
        g_coords = np.zeros((height, width), dtype=np.float32)
        s_coords = np.zeros((height, width), dtype=np.float32)
        intensity = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                pixel_data = flim_data[y, x, :]
                pixel_intensity = np.sum(pixel_data)
                intensity[y, x] = pixel_intensity
                if pixel_intensity > threshold_min and (threshold_max is None or pixel_intensity < threshold_max):
                    time_indices = np.arange(n_bins)
                    cos_component = np.sum(pixel_data * np.cos(omega * time_indices))
                    g_raw = cos_component / pixel_intensity if pixel_intensity > 0 else 0
                    sin_component = np.sum(pixel_data * np.sin(omega * time_indices))
                    s_raw = sin_component / pixel_intensity if pixel_intensity > 0 else 0
                    g_corrected = (g_raw * m_cal * np.cos(phi_cal) - s_raw * m_cal * np.sin(phi_cal))
                    s_corrected = (g_raw * m_cal * np.sin(phi_cal) + s_raw * m_cal * np.cos(phi_cal))
                    g_coords[y, x] = g_corrected
                    s_coords[y, x] = s_corrected
                else:
                    g_coords[y, x] = 0
                    s_coords[y, x] = 0
        if apply_filter > 0:
            try:
                from scipy.ndimage import median_filter
                for _ in range(apply_filter):
                    g_coords = median_filter(g_coords, size=3)
                    s_coords = median_filter(s_coords, size=3)
                    intensity = median_filter(intensity, size=3)
                print(f"    Applied {apply_filter} passes of median filtering")
            except ImportError:
                print(f"    Warning: scipy not available, skipping median filtering")
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        g_output = os.path.join(output_dir, f"{base_name}_g.tiff")
        s_output = os.path.join(output_dir, f"{base_name}_s.tiff")
        intensity_output = os.path.join(output_dir, f"{base_name}_intensity.tiff")
        try:
            g_img = Image.fromarray(g_coords.astype(np.float32))
            g_img.save(g_output)
            s_img = Image.fromarray(s_coords.astype(np.float32))
            s_img.save(s_output)
            intensity_img = Image.fromarray(intensity.astype(np.float32))
            intensity_img.save(intensity_output)
            print(f"    Saved output files:")
            print(f"      G: {g_output}")
            print(f"      S: {s_output}")
            print(f"      Intensity: {intensity_output}")
            return True
        except Exception as e:
            print(f"    Error saving output files: {e}")
            return False
    except Exception as e:
        print(f"    Error in phasor transformation: {e}")
        traceback.print_exc()
        return False


    """
    Processes specific TIFF files to generate phasor coordinates (G, S, intensity).
    Maps the full .bin file path from CSV to the expected .tif path in base_output_dir.
    Uses our custom phasor_transform module instead of FLUTE to avoid GUI dependencies.

    Args:
        calibration_file (str): Path to the CSV file ('file_path', 'phi', 'modulation').
        base_output_dir (str): The main output directory where corresponding TIFF files exist.
        raw_data_root (str): The root directory of the raw input data.
        microscope_params (dict): Dictionary containing 'bin_width_ns', 'freq_mhz', 'harmonic'.
    """
    
    # --- Phasor transformation function is now local; call process_flim_file directly ---
    try:
        # Read calibration data
        calibration_data = pd.read_csv(calibration_file, dtype={'file_path': str})
    except FileNotFoundError:
        print(f"Error: Calibration file not found: {calibration_file}")
        return False
    except pd.errors.EmptyDataError:
        print(f"Error: Calibration file is empty: {calibration_file}")
        return False
    except pd.errors.ParserError as e:
        print(f"Error parsing calibration CSV: {e}")
        return False
        
    # Ensure base output directory exists
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Extract params
    try:
        bin_width_ns = microscope_params["bin_width_ns"]
        freq_mhz = microscope_params["freq_mhz"]
        harmonic = microscope_params["harmonic"]
    except KeyError as e:
        print(f"Error: Missing microscope parameter in config: {e}. Cannot run phasor transform.")
        return False
        
    normalized_raw_root = os.path.normpath(raw_data_root)
    if not normalized_raw_root.endswith(os.path.sep):
         normalized_raw_root += os.path.sep

    print(f"Starting phasor transformation using calibration from: {calibration_file}")
    print(f"Mapping from raw root: {normalized_raw_root} to output base: {base_output_dir}")
    processed_count = 0
    skipped_count = 0

    for index, row in calibration_data.iterrows():
        try:
            bin_file_full_path = str(row['file_path']).strip()
            
            # Handle different possible column names for phi calibration
            if 'phi_cal' in row:
                phi_cal = float(row['phi_cal'])
            elif 'phi' in row:
                phi_cal = float(row['phi'])
            else:
                print(f"Error: Row {index}: Missing column: 'phi_cal' or 'phi'. Skipping.")
                skipped_count += 1
                continue
            
            # Handle different possible column names for modulation calibration
            if 'm_cal' in row:
                m_cal = float(row['m_cal'])
            elif 'modulation' in row:
                m_cal = float(row['modulation'])
            else:
                print(f"Error: Row {index}: Missing column: 'm_cal' or 'modulation'. Skipping.")
                skipped_count += 1
                continue
            
            # Ensure normalized path and add .bin extension if needed
            normalized_bin_path = os.path.normpath(bin_file_full_path)
            if not normalized_bin_path.endswith('.bin'):
                normalized_bin_path += '.bin'
            
            if not normalized_bin_path.startswith(normalized_raw_root):
                print(f"Warning: Row {index}: .bin path {bin_file_full_path} does not start with raw_data_root {raw_data_root}. Skipping.")
                skipped_count += 1
                continue
                
            # Get relative paths based on bin file location
            relative_path = os.path.relpath(normalized_bin_path, normalized_raw_root)
            relative_dir = os.path.dirname(relative_path)
            bin_filename = os.path.basename(normalized_bin_path)
            tif_filename = os.path.splitext(bin_filename)[0] + ".tif"
            
            # Construct output paths
            output_subdir = os.path.join(base_output_dir, relative_dir)
            target_tif_path = os.path.join(output_subdir, tif_filename)
            
            print(f"\nProcessing entry {index}: BIN={bin_filename} with Phi={phi_cal}, Mod={m_cal}")
            print(f"  Relative path: {relative_path}")
            print(f"  Target TIFF path: {target_tif_path}")

            # Create output subdirectory if it doesn't exist
            os.makedirs(output_subdir, exist_ok=True)
            
            # Define output paths for g, s, intensity
            output_g_path = os.path.join(output_subdir, os.path.splitext(tif_filename)[0] + "_g.tiff")
            output_s_path = os.path.join(output_subdir, os.path.splitext(tif_filename)[0] + "_s.tiff")
            output_intensity_path = os.path.join(output_subdir, os.path.splitext(tif_filename)[0] + "_intensity.tiff")
            
            try:
                # Process either the bin file directly or tif file if it exists
                input_file = None
                if os.path.exists(target_tif_path):
                    input_file = target_tif_path
                    print(f"  Processing existing TIF file: {input_file}")
                elif os.path.exists(normalized_bin_path):
                    input_file = normalized_bin_path
                    print(f"  Processing BIN file directly: {input_file}")
                else:
                    print(f"  Error: Neither TIF file nor BIN file found. Skipping.")
                    skipped_count += 1
                    continue
                
                # Process the file using our custom phasor transform
                try:
                    print(f"  Processing with phasor_transform: bin_width={bin_width_ns}, freq={freq_mhz}, harmonic={harmonic}")
                    success = process_flim_file(
                        input_file=input_file,
                        output_dir=output_subdir,
                        phi_cal=phi_cal,
                        m_cal=m_cal,
                        bin_width_ns=bin_width_ns,
                        freq_mhz=freq_mhz,
                        harmonic=harmonic
                    )
                    
                    if success:
                        print(f"  Phasor transformation completed successfully for {input_file}")
                    else:
                        print(f"  Phasor transformation failed for {input_file}")
                        skipped_count += 1
                        continue
                        
                except Exception as transform_error:
                    print(f"  Error in phasor transformation: {transform_error}")
                    skipped_count += 1
                    continue
                
                # Check if output files were created
                files_exist = (
                    os.path.exists(output_g_path) and 
                    os.path.exists(output_s_path) and
                    os.path.exists(output_intensity_path)
                )
                
                if files_exist:
                    print(f"    Successfully processed with phasor_transform. Output files created.")
                    processed_count += 1
                else:
                    print(f"    Warning: Some expected output files not created:")
                    if not os.path.exists(output_g_path):
                        print(f"      Missing: {output_g_path}")
                    if not os.path.exists(output_s_path):
                        print(f"      Missing: {output_s_path}")
                    if not os.path.exists(output_intensity_path):
                        print(f"      Missing: {output_intensity_path}")
                    skipped_count += 1
            except Exception as e:
                print(f"    Error processing file: {e}")
                skipped_count += 1
        except KeyError as e: 
            print(f"Error: Row {index}: Missing column: {e}. Skipping.") 
            skipped_count += 1
        except ValueError as e: 
            print(f"Error: Row {index}: Invalid numerical/path value: {e}. Skipping.") 
            skipped_count += 1
        except Exception as e: 
            print(f"Error processing row {index} ({bin_file_full_path}): {e}") 
            skipped_count += 1

    print(f"\nPhasor transformation complete.")
    print(f"Successfully processed files listed in CSV: {processed_count}")
    print(f"Skipped/failed CSV entries: {skipped_count}")
    
    return processed_count > 0

def run_bin_to_tiff_conversion(config, input_dir, output_dir, calibration_file, raw_data_root, interactive_file_selection=False):
    """
    Stage 1: Convert .bin files to .tiff files using ImageJ macros
    
    Args:
        config (dict): Configuration dictionary
        input_dir (str): Input directory containing .bin files
        output_dir (str): Output directory for TIFF files
        calibration_file (str): Path to calibration CSV file
        raw_data_root (str): Root directory for raw data
        interactive_file_selection (bool): If True, prompt user to select specific files
    
    Returns:
        tuple: (success, temp_calibration_file or calibration_file)
    """
    # Extract required config params
    try:
        imagej_path = config["imagej_path"]
        macro_files = config["macro_files"]
    except KeyError as e:
        print(f"Error: Missing required key in config data: {e}. Cannot run bin to TIFF conversion.")
        return False, calibration_file
    
    # Handle interactive file selection if requested
    temp_calibration_file = calibration_file
    if interactive_file_selection:
        print("\n=== Interactive File Selection Mode ===")
        selected_files = prompt_file_selection(input_dir, '.bin')
        
        if selected_files is None:
            print("File selection cancelled. Conversion aborted.")
            return False, calibration_file
        
        # Create a temporary calibration file with selected files
        temp_calibration_file = os.path.join(os.path.dirname(calibration_file), 'temp_calibration.csv')
        
        print("\n=== Using Original Calibration Values ===")
        print(f"Reading calibration values from: {calibration_file}")
        print("Creating filtered calibration file for selected files...")
        
        if not create_calibration_file_from_selection(selected_files, calibration_file, temp_calibration_file):
            print("Error creating filtered calibration file. Conversion aborted.")
            return False, calibration_file
        
        print(f"Using temporary calibration file: {temp_calibration_file}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
        
    # Check if ImageJ executable exists
    if not os.path.exists(imagej_path):
        print(f"Error: ImageJ executable not found at {imagej_path}")
        return False, temp_calibration_file
    
    # Check if macros exist
    for i, macro_file in enumerate(macro_files):
        if not os.path.exists(macro_file):
            print(f"Error: Macro file {i+1} not found at {macro_file}")
            return False, temp_calibration_file
    
    # === Run ImageJ Macros for TIF conversion ===
    if interactive_file_selection:
        # Create file list for ImageJ macro
        print("Creating file list for ImageJ macro from selected files...")
        
        try:
            selected_df = pd.read_csv(temp_calibration_file, dtype={'file_path': str})
            selected_file_paths = [str(row['file_path']).strip() for _, row in selected_df.iterrows()]
            
            file_list_path = os.path.join(os.path.dirname(temp_calibration_file), 'selected_files.txt')
            with open(file_list_path, 'w') as f:
                for file_path in selected_file_paths:
                    if not file_path.endswith('.bin'):
                        file_path += '.bin'
                    f.write(file_path + '\n')
            
            print(f"Created file list: {file_list_path}")
            print(f"Contains {len(selected_file_paths)} selected files")
            
            # Run ImageJ Macro with file list
            print("Running ImageJ Macro with selected files...")
            macro_success = run_imagej(imagej_path, macro_files[0], input_dir, output_dir, file_list_path)
            
            # Clean up file list
            try:
                if os.path.exists(file_list_path):
                    os.remove(file_list_path)
                    print(f"Cleaned up file list: {file_list_path}")
            except Exception as e:
                print(f"Warning: Could not clean up file list: {e}")
                
        except Exception as e:
            print(f"Error creating file list for ImageJ: {e}")
            print("Falling back to processing all files...")
            macro_success = run_imagej(imagej_path, macro_files[0], input_dir, output_dir)
        
    else:
        print("Running ImageJ Macro for all .bin files...")
        macro_success = run_imagej(imagej_path, macro_files[0], input_dir, output_dir)
        if not macro_success:
            print("Warning: ImageJ Macro failed. Continuing anyway...")
    
    # Check if any .tif files were created
    tif_files_exist = False
    for root, dirs, files in os.walk(output_dir):
        if any(f.endswith('.tif') or f.endswith('.tiff') for f in files):
            tif_files_exist = True
            break
    
    if not tif_files_exist:
        print("Warning: No .tif files were created by ImageJ macros.")
    else:
        print("Successfully converted BIN files to TIFF format.")
    
    return True, temp_calibration_file

def run_phasor_transformation_stage(config, output_dir, calibration_file, raw_data_root):
    """
    Stage 2: Perform phasor transformation on TIFF files
    
    Args:
        config (dict): Configuration dictionary
        output_dir (str): Directory containing TIFF files and where G/S/intensity files will be saved
        calibration_file (str): Path to calibration CSV file
        raw_data_root (str): Root directory for raw data
    
    Returns:
        bool: True if successful
    """
    try:
        microscope_params = config["microscope_params"]
    except KeyError as e:
        print(f"Error: Missing required microscope_params in config: {e}")
        return False
    
    print("=== Stage 2: Phasor Transformation ===")
    print(f"Processing TIFF files in: {output_dir}")
    print(f"Using calibration file: {calibration_file}")
    
    # --- Phasor transformation function is now local; call process_flim_file directly ---
    try:
        # Read calibration data
        calibration_data = pd.read_csv(calibration_file, dtype={'file_path': str})
    except FileNotFoundError:
        print(f"Error: Calibration file not found: {calibration_file}")
        return False
    except pd.errors.EmptyDataError:
        print(f"Error: Calibration file is empty: {calibration_file}")
        return False
    except pd.errors.ParserError as e:
        print(f"Error parsing calibration CSV: {e}")
        return False
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract params
    try:
        bin_width_ns = microscope_params["bin_width_ns"]
        freq_mhz = microscope_params["freq_mhz"]
        harmonic = microscope_params["harmonic"]
    except KeyError as e:
        print(f"Error: Missing microscope parameter in config: {e}. Cannot run phasor transform.")
        return False
        
    normalized_raw_root = os.path.normpath(raw_data_root)
    if not normalized_raw_root.endswith(os.path.sep):
         normalized_raw_root += os.path.sep

    print(f"Starting phasor transformation using calibration from: {calibration_file}")
    print(f"Mapping from raw root: {normalized_raw_root} to output base: {output_dir}")
    processed_count = 0
    skipped_count = 0

    for index, row in calibration_data.iterrows():
        try:
            bin_file_full_path = str(row['file_path']).strip()
            
            # Handle different possible column names for phi calibration
            if 'phi_cal' in row:
                phi_cal = float(row['phi_cal'])
            elif 'phi' in row:
                phi_cal = float(row['phi'])
            else:
                print(f"Error: Row {index}: Missing column: 'phi_cal' or 'phi'. Skipping.")
                skipped_count += 1
                continue
            
            # Handle different possible column names for modulation calibration
            if 'm_cal' in row:
                m_cal = float(row['m_cal'])
            elif 'modulation' in row:
                m_cal = float(row['modulation'])
            else:
                print(f"Error: Row {index}: Missing column: 'm_cal' or 'modulation'. Skipping.")
                skipped_count += 1
                continue
            
            # Ensure normalized path and add .bin extension if needed
            normalized_bin_path = os.path.normpath(bin_file_full_path)
            if not normalized_bin_path.endswith('.bin'):
                normalized_bin_path += '.bin'
            
            if not normalized_bin_path.startswith(normalized_raw_root):
                print(f"Warning: Row {index}: .bin path {bin_file_full_path} does not start with raw_data_root {raw_data_root}. Skipping.")
                skipped_count += 1
                continue
                
            # Get relative paths based on bin file location
            relative_path = os.path.relpath(normalized_bin_path, normalized_raw_root)
            relative_dir = os.path.dirname(relative_path)
            bin_filename = os.path.basename(normalized_bin_path)
            tif_filename = os.path.splitext(bin_filename)[0] + ".tif"
            
            # Construct output paths
            output_subdir = os.path.join(output_dir, relative_dir)
            target_tif_path = os.path.join(output_subdir, tif_filename)
            
            print(f"\nProcessing entry {index}: BIN={bin_filename} with Phi={phi_cal}, Mod={m_cal}")
            print(f"  Target TIFF path: {target_tif_path}")

            # Create output subdirectory if it doesn't exist
            os.makedirs(output_subdir, exist_ok=True)
            
            try:
                # Process either the bin file directly or tif file if it exists
                input_file = None
                if os.path.exists(target_tif_path):
                    input_file = target_tif_path
                    print(f"  Processing existing TIF file: {input_file}")
                elif os.path.exists(normalized_bin_path):
                    input_file = normalized_bin_path
                    print(f"  Processing BIN file directly: {input_file}")
                else:
                    print(f"  Error: Neither TIF file nor BIN file found. Skipping.")
                    skipped_count += 1
                    continue
                
                # Process the file using our custom phasor transform
                try:
                    print(f"  Processing with phasor_transform: bin_width={bin_width_ns}, freq={freq_mhz}, harmonic={harmonic}")
                    success = process_flim_file(
                        input_file=input_file,
                        output_dir=output_subdir,
                        phi_cal=phi_cal,
                        m_cal=m_cal,
                        bin_width_ns=bin_width_ns,
                        freq_mhz=freq_mhz,
                        harmonic=harmonic
                    )
                    
                    if success:
                        print(f"  Phasor transformation completed successfully for {input_file}")
                        processed_count += 1
                    else:
                        print(f"  Phasor transformation failed for {input_file}")
                        skipped_count += 1
                        
                except Exception as transform_error:
                    print(f"  Error in phasor transformation: {transform_error}")
                    skipped_count += 1
                    
            except Exception as e:
                print(f"    Error processing file: {e}")
                skipped_count += 1
        except KeyError as e: 
            print(f"Error: Row {index}: Missing column: {e}. Skipping.") 
            skipped_count += 1
        except ValueError as e: 
            print(f"Error: Row {index}: Invalid numerical/path value: {e}. Skipping.") 
            skipped_count += 1
        except Exception as e: 
            print(f"Error processing row {index} ({bin_file_full_path}): {e}") 
            skipped_count += 1

    print(f"\nPhasor transformation complete.")
    print(f"Successfully processed files: {processed_count}")
    print(f"Skipped/failed entries: {skipped_count}")
    
    return processed_count > 0

def run_file_organization(output_dir, preprocessed_dir):
    """
    Stage 3: Organize phasor transformation output files into preprocessed directory structure
    
    Args:
        output_dir (str): Directory containing G, S, and intensity files
        preprocessed_dir (str): Target directory for organized files
    
    Returns:
        bool: True if successful
    """
    print("=== Stage 3: File Organization ===")
    print(f"Organizing files from {output_dir} to {preprocessed_dir}")
    
    # Ensure preprocessed directory exists
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    try:
        # Use the organize_output_files module
        modules_dir = os.path.dirname(os.path.abspath(__file__))
        if modules_dir not in sys.path:
            sys.path.insert(0, modules_dir)
            
        import organize_output_files
        print("Successfully imported organize_output_files module")
        
        # Call the organize_output_files function directly
        success_count, error_count = organize_output_files.organize_output_files(output_dir, preprocessed_dir)
        
        if success_count > 0:
            print(f"Successfully organized {success_count} files into preprocessed directory")
            if error_count > 0:
                print(f"Warning: {error_count} files could not be organized")
            organization_success = True
        else:
            print(f"Error: No files were successfully organized. Checking for fallback method...")
            organization_success = False
            
    except ImportError as e:
        print(f"Error importing organize_output_files module: {e}")
        organization_success = False
    except Exception as e:
        print(f"Error organizing files: {e}")
        organization_success = False
    
    # If the Python organization failed, try the manual file copying as a fallback
    if not organization_success:
        print("Python file organization failed. Using manual file copying as fallback...")
        copy_success = manually_copy_files_to_preprocessed(output_dir, preprocessed_dir)
        if not copy_success:
            print("Error: Manual file copying failed. Pipeline cannot continue.")
            return False
    
    # Verify that files exist in the preprocessed directory
    g_files_exist = False
    s_files_exist = False
    intensity_files_exist = False
    
    for root, dirs, files in os.walk(preprocessed_dir):
        if "G_unfiltered" in root and any(f.endswith('.tiff') for f in files):
            g_files_exist = True
        if "S_unfiltered" in root and any(f.endswith('.tiff') for f in files):
            s_files_exist = True
        if "intensity" in root and any(f.endswith('.tiff') for f in files):
            intensity_files_exist = True
    
    if g_files_exist and s_files_exist and intensity_files_exist:
        print("Successfully organized all required file types in preprocessed directory.")
        return True
    else:
        print("Warning: Some file types are missing in the preprocessed directory.")
        if not g_files_exist:
            print("  - Missing G_unfiltered files")
        if not s_files_exist:
            print("  - Missing S_unfiltered files")
        if not intensity_files_exist:
            print("  - Missing intensity files")
        return False

def run_preprocessing(config, input_dir, output_dir, preprocessed_dir, calibration_file, raw_data_root, interactive_file_selection=False):
    """
    Runs the full TCSPC preprocessing pipeline (all 3 stages).
    
    Args:
        config (dict): Configuration dictionary
        input_dir (str): Input directory containing .bin files
        output_dir (str): Output directory for processed files
        preprocessed_dir (str): Directory for organized preprocessed files
        calibration_file (str): Path to calibration CSV file
        raw_data_root (str): Root directory for raw data
        interactive_file_selection (bool): If True, prompt user to select specific files
    
    Returns:
        bool: True if successful, False if failed
    """
    print("\n=== Full Preprocessing Pipeline ===")
    
    # Stage 1: BIN to TIFF conversion
    conversion_success, temp_calibration_file = run_bin_to_tiff_conversion(
        config, input_dir, output_dir, calibration_file, raw_data_root, interactive_file_selection
    )
    if not conversion_success:
        print("Error: BIN to TIFF conversion stage failed.")
        return False
    
    # Stage 2: Phasor transformation
    phasor_success = run_phasor_transformation_stage(config, output_dir, temp_calibration_file, raw_data_root)
    if not phasor_success:
        print("Error: Phasor transformation stage failed.")
        return False
    
    # Stage 3: File organization
    organization_success = run_file_organization(output_dir, preprocessed_dir)
    if not organization_success:
        print("Error: File organization stage failed.")
        return False
    
    # Clean up temporary calibration file if it was created
    if interactive_file_selection and temp_calibration_file != calibration_file:
        try:
            if os.path.exists(temp_calibration_file):
                os.remove(temp_calibration_file)
                print(f"Cleaned up temporary calibration file: {temp_calibration_file}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary calibration file: {e}")
    
    print("Full preprocessing pipeline completed successfully.")
    return True
