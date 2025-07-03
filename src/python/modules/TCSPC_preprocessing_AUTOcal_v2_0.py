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
    command = [
        imagej_path,
        '-macro', macro_file, ",".join(args)
    ]
    print(f"Running ImageJ command: {' '.join(command)}")
    
    try:
        # Run ImageJ with stdout and stderr captured
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"ImageJ Command Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the macro: {e}")
        print(f"Exit code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
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

def process_tiffs_with_phasor_transform(calibration_file, base_output_dir, raw_data_root, microscope_params):
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
    
    # --- Import our custom phasor transform module ---
    try:
        # Add the modules directory to the path if not already there
        modules_dir = os.path.dirname(os.path.abspath(__file__))
        if modules_dir not in sys.path:
            sys.path.insert(0, modules_dir)
        
        import phasor_transform
        print("Successfully imported phasor_transform module")
    except ImportError as e:
        print(f"Error importing phasor_transform module: {e}")
        print("Cannot proceed with phasor processing.")
        return False # Exit this function if import fails
    
    # --- Continue with the rest of the function --- 
    try:
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
                    success = phasor_transform.process_flim_file(
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

def run_preprocessing(config, input_dir, output_dir, preprocessed_dir, calibration_file, raw_data_root):
    """
    Runs the full TCSPC preprocessing pipeline (ImageJ + phasor transformation).
    
    Returns:
        bool: True if successful, False if failed
    """
    
    # Extract required config params
    try:
        imagej_path = config["imagej_path"]
        macro_files = config["macro_files"]
        microscope_params = config["microscope_params"]
    except KeyError as e:
        print(f"Error: Missing required key in config data: {e}. Cannot run preprocessing.")
        return False # Indicate failure
    
    # Ensure input, output and preprocessed directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
        
    # Check if ImageJ executable exists
    if not os.path.exists(imagej_path):
        print(f"Error: ImageJ executable not found at {imagej_path}")
        print("You may need to update the 'imagej_path' in config.json")
        return False
    
    # Check if macros exist (only macros 1 and 2 are used)
    for i, macro_file in enumerate(macro_files):
        if not os.path.exists(macro_file):
            print(f"Error: Macro file {i+1} not found at {macro_file}")
            print("Please check the 'macro_files' paths in config.json")
            return False
    
    # === Run ImageJ Macros for TIF conversion ===
    print("Running ImageJ Macro 1 for FITC.bin...")
    macro1_success = run_imagej(imagej_path, macro_files[0], input_dir, output_dir)
    if not macro1_success:
        print("Warning: ImageJ Macro 1 (FITC.bin) failed. Continuing anyway...")

    print("Running ImageJ Macro 2 for all .bin files...")
    macro2_success = run_imagej(imagej_path, macro_files[1], input_dir, output_dir)
    if not macro2_success:
        print("Warning: ImageJ Macro 2 (All BIN files) failed. Continuing anyway...")
    
    # Check if any .tif files were created in the output directory
    tif_files_exist = False
    for root, dirs, files in os.walk(output_dir):
        if any(f.endswith('.tif') or f.endswith('.tiff') for f in files):
            tif_files_exist = True
            break
    
    if not tif_files_exist:
        print("Warning: No .tif files were created by ImageJ macros.")
        print("Proceeding directly to FLUTE processing...")
    
    # === Run phasor transformation processing ===
    print("Starting phasor transformation processing...")
    phasor_success = process_tiffs_with_phasor_transform(
        calibration_file, 
        output_dir, 
        raw_data_root, 
        microscope_params
    )
    
    if not phasor_success:
        print("Error: Phasor transformation processing failed. Cannot continue pipeline.")
        return False
    
    # === Organize output files using Python script ===
    # This replaces the ImageJ Macros 3, 4, and 5 for better reliability
    print("Organizing output files with Python script...")
    
    try:
        # Use the new organize_output_files.py script to organize files
        # Add the modules directory to the path if not already there
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
        if "Intensity" in root and any(f.endswith('.tiff') for f in files):
            intensity_files_exist = True
    
    if g_files_exist and s_files_exist and intensity_files_exist:
        print("Successfully organized all required file types in preprocessed directory.")
    else:
        print("Warning: Some file types are missing in the preprocessed directory.")
        if not g_files_exist:
            print("  - Missing G_unfiltered files")
        if not s_files_exist:
            print("  - Missing S_unfiltered files")
        if not intensity_files_exist:
            print("  - Missing Intensity files")
    
    # Final verification
    for root, dirs, files in os.walk(preprocessed_dir):
        if any(f.endswith('.tiff') for f in files):
            print("Preprocessing pipeline complete.")
            return True
    
    print("Error: No files found in preprocessed directory after pipeline.")
    return False

# === Main script execution block (for running standalone) ===
if __name__ == "__main__":
    print("This script is intended to be run via run_pipeline.py")
    print("Please use: python run_pipeline.py --preprocess -i <input_dir> -o <output_base_dir> [-c <calib.csv>]")
    sys.exit(1) 