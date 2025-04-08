import os
import pandas as pd
from FLUTE.ImageHandler import ImageHandler

def process_tiffs_with_flute(calibration_file, base_output_dir, raw_data_root, bin_width_ns, freq_mhz, harmonic):
    """
    Processes specific TIFF files using FLUTE's ImageHandler based on calibration data.
    Maps the full .bin file path from CSV to the expected .tif path in base_output_dir.

    Args:
        calibration_file (str): Path to the CSV file with calibration data.
                                Expected columns: 'file_path' (full path to .bin file),
                                'phi', 'modulation'.
        base_output_dir (str): The main output directory where corresponding TIFF files exist.
        raw_data_root (str): The root directory corresponding to the start of relative paths
                             mirrored between raw data and output data.
        bin_width_ns (float): Temporal bin width in nanoseconds.
        freq_mhz (float): Laser repetition frequency in MHz.
        harmonic (int): Harmonic number.
    """
    try:
        # Read calibration data, ensure file_path is string
        calibration_data = pd.read_csv(calibration_file, dtype={'file_path': str})
    except FileNotFoundError:
        print(f"Error: Calibration file not found at {calibration_file}")
        return
    except Exception as e:
        print(f"Error reading calibration file {calibration_file}: {e}")
        return

    # Normalize raw_data_root path for reliable comparison
    normalized_raw_root = os.path.normpath(raw_data_root)
    if not normalized_raw_root.endswith(os.path.sep):
         normalized_raw_root += os.path.sep

    print(f"Starting FLUTE processing using calibration from: {calibration_file}")
    print(f"Mapping from raw root: {normalized_raw_root} to output base: {base_output_dir}")
    processed_count = 0
    skipped_count = 0

    # Process each specific file defined in the calibration file
    for index, row in calibration_data.iterrows():
        try:
            bin_file_full_path = str(row['file_path']).strip()
            phi_cal = float(row['phi'])
            m_cal = float(row['modulation'])
            
            # Normalize bin_file_full_path for comparison
            normalized_bin_path = os.path.normpath(bin_file_full_path)
            
            # --- Map .bin path to expected .tif path in output_dir --- 
            if not normalized_bin_path.startswith(normalized_raw_root):
                print(f"Warning: Row {index}: .bin path {bin_file_full_path} does not start with raw_data_root {raw_data_root}. Skipping.")
                skipped_count += 1
                continue
                
            # Get the relative path from the raw root
            relative_path = os.path.relpath(normalized_bin_path, normalized_raw_root)
            
            # Change extension from .bin to .tif
            relative_tif_path = os.path.splitext(relative_path)[0] + ".tif"
            
            # Construct the full path to the expected .tif file in the output directory
            target_tif_path = os.path.join(base_output_dir, relative_tif_path)
            target_tif_filename = os.path.basename(target_tif_path)
            
            print(f"\nProcessing entry {index}: Target TIFF = {relative_tif_path} with Phi={phi_cal}, Mod={m_cal}")

            # --- Check if target TIFF file exists --- 
            if not os.path.exists(target_tif_path):
                print(f"  Warning: Target TIFF file not found at mapped location: {target_tif_path}. Skipping.")
                print(f"  (Derived from: {bin_file_full_path})")
                skipped_count += 1
                continue
            
            print(f"  Found target TIFF at: {target_tif_path}")
            
            # Define output paths for _g and _s files (in the same directory as the found TIFF)
            tif_dir = os.path.dirname(target_tif_path)
            output_g_path = os.path.join(tif_dir, os.path.splitext(target_tif_filename)[0] + "_g.tiff")
            output_s_path = os.path.join(tif_dir, os.path.splitext(target_tif_filename)[0] + "_s.tiff")
            
            # --- Process the found TIFF file --- 
            try:
                handler = ImageHandler(
                    filename=target_tif_path,
                    phi_cal=phi_cal,
                    m_cal=m_cal,
                    bin_width=bin_width_ns,
                    freq=freq_mhz,
                    harmonic=harmonic
                )
                handler.save_data(file=tif_dir, save_type=None) 
                
                if os.path.exists(output_g_path) and os.path.exists(output_s_path):
                    print(f"    Successfully processed and saved output for {target_tif_filename}")
                    processed_count += 1
                else:
                    print(f"    Error: Output files (_g.tiff, _s.tiff) not created for {target_tif_filename}. Check FLUTE logs.")
                    skipped_count += 1
                    
            except Exception as e:
                print(f"    Error processing file {target_tif_filename} with ImageHandler: {e}")
                skipped_count += 1

        except KeyError as e:
            print(f"Error: Row {index}: Missing expected column in {calibration_file}: {e}. Skipping row.")
            skipped_count += 1
        except ValueError as e:
            print(f"Error: Row {index}: Invalid numerical/path value in {calibration_file}: {e}. Skipping.")
            skipped_count += 1
        except Exception as e:
            print(f"An unexpected error occurred while processing row {index} ({bin_file_full_path}): {e}")
            skipped_count += 1

    print(f"\nFLUTE processing complete.")
    print(f"Successfully processed files listed in CSV: {processed_count}")
    print(f"Skipped/failed CSV entries: {skipped_count}")

def run_preprocessing():
    """Runs the full TCSPC preprocessing pipeline (ImageJ + FLUTE)."""
    # Load config inside the function
    try:
        with open("config.json", "r") as config_file:
            config = json.load(config_file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
         print(f"Error loading config.json: {e}")
         return # Cannot proceed without config

    # Extract paths and params (handle potential missing keys)
    try:
        imagej_path = config["imagej_path"]
        flute_path = config["flute_path"]
        macro_files = config["macro_files"]
        input_dir = config["input_dir"]
        output_dir = config["output_dir"]
        preprocessed_dir = config["preprocessed_dir"]
        raw_data_root = config["raw_data_root"] # Get the raw data root
        bin_width_ns = config["microscope_params"]["bin_width_ns"]
        freq_mhz = config["microscope_params"]["freq_mhz"]
        harmonic = config["microscope_params"]["harmonic"]
    except KeyError as e:
        print(f"Error: Missing required key in config.json: {e}")
        return
        
    # === Run ImageJ Macros ===
    print("Running ImageJ Macro 1...")
    run_imagej(macro_files[0], input_dir, output_dir)
    print("ImageJ Macro 1 finished.")

    print("Running ImageJ Macro 2...")
    run_imagej(macro_files[1], input_dir, output_dir)
    print("ImageJ Macro 2 finished.")

    # === Run FLUTE processing ===
    print("Starting FLUTE processing...")
    process_tiffs_with_flute(
        'calibration.csv', 
        output_dir, 
        raw_data_root, # Pass raw_data_root
        bin_width_ns, 
        freq_mhz, 
        harmonic
    )

    # === Run Post-FLUTE ImageJ Macros ===
    print("Running ImageJ Macro 3...")
    run_imagej(macro_files[2], output_dir, preprocessed_dir)
    print("ImageJ Macro 3 finished.")

    print("Running ImageJ Macro 4...")
    run_imagej(macro_files[3], output_dir, preprocessed_dir)
    print("ImageJ Macro 4 finished.")

    print("Running ImageJ Macro 5...")
    run_imagej(macro_files[4], preprocessed_dir)
    print("ImageJ Macro 5 finished.")

    print("Preprocessing pipeline complete.") 