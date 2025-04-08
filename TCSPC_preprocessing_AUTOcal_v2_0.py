import os
import pandas as pd
from FLUTE.ImageHandler import ImageHandler
import subprocess
import json
import sys

def run_imagej(imagej_path, macro_file, *args):
    command = [
        imagej_path,
        '-macro', macro_file, ",".join(args)
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the macro: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check the path to the ImageJ executable.")
    except PermissionError as e:
        print(f"Permission error: {e}")
        print("Please check the permissions of the ImageJ executable.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def process_tiffs_with_flute(calibration_file, base_output_dir, raw_data_root, microscope_params):
    """
    Processes specific TIFF files using FLUTE's ImageHandler based on calibration data.
    Maps the full .bin file path from CSV to the expected .tif path in base_output_dir.

    Args:
        calibration_file (str): Path to the CSV file ('file_path', 'phi', 'modulation').
        base_output_dir (str): The main output directory where corresponding TIFF files exist.
        raw_data_root (str): The root directory of the raw input data.
        microscope_params (dict): Dictionary containing 'bin_width_ns', 'freq_mhz', 'harmonic'.
    """
    try:
        calibration_data = pd.read_csv(calibration_file, dtype={'file_path': str})
    except FileNotFoundError:
        print(f"Error: Calibration file not found at {calibration_file}")
        return
    except Exception as e:
        print(f"Error reading calibration file {calibration_file}: {e}")
        return

    # Extract params
    try:
        bin_width_ns = microscope_params["bin_width_ns"]
        freq_mhz = microscope_params["freq_mhz"]
        harmonic = microscope_params["harmonic"]
    except KeyError as e:
        print(f"Error: Missing microscope parameter in config: {e}. Cannot run FLUTE.")
        return
        
    normalized_raw_root = os.path.normpath(raw_data_root)
    if not normalized_raw_root.endswith(os.path.sep):
         normalized_raw_root += os.path.sep

    print(f"Starting FLUTE processing using calibration from: {calibration_file}")
    print(f"Mapping from raw root: {normalized_raw_root} to output base: {base_output_dir}")
    processed_count = 0
    skipped_count = 0

    for index, row in calibration_data.iterrows():
        try:
            bin_file_full_path = str(row['file_path']).strip()
            phi_cal = float(row['phi'])
            m_cal = float(row['modulation'])
            normalized_bin_path = os.path.normpath(bin_file_full_path)
            
            if not normalized_bin_path.startswith(normalized_raw_root):
                print(f"Warning: Row {index}: .bin path {bin_file_full_path} does not start with raw_data_root {raw_data_root}. Skipping.")
                skipped_count += 1; continue
                
            relative_path = os.path.relpath(normalized_bin_path, normalized_raw_root)
            relative_tif_path = os.path.splitext(relative_path)[0] + ".tif"
            target_tif_path = os.path.join(base_output_dir, relative_tif_path)
            target_tif_filename = os.path.basename(target_tif_path)
            
            print(f"\nProcessing entry {index}: Target TIFF = {relative_tif_path} with Phi={phi_cal}, Mod={m_cal}")

            if not os.path.exists(target_tif_path):
                print(f"  Warning: Target TIFF file not found at mapped location: {target_tif_path}. Skipping.")
                skipped_count += 1; continue
            
            print(f"  Found target TIFF at: {target_tif_path}")
            tif_dir = os.path.dirname(target_tif_path)
            output_g_path = os.path.join(tif_dir, os.path.splitext(target_tif_filename)[0] + "_g.tiff")
            output_s_path = os.path.join(tif_dir, os.path.splitext(target_tif_filename)[0] + "_s.tiff")
            
            try:
                handler = ImageHandler(
                    filename=target_tif_path, phi_cal=phi_cal, m_cal=m_cal,
                    bin_width=bin_width_ns, freq=freq_mhz, harmonic=harmonic
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
        except KeyError as e: print(f"Error: Row {index}: Missing column: {e}. Skipping."); skipped_count += 1
        except ValueError as e: print(f"Error: Row {index}: Invalid numerical/path value: {e}. Skipping."); skipped_count += 1
        except Exception as e: print(f"Error processing row {index} ({bin_file_full_path}): {e}"); skipped_count += 1

    print(f"\nFLUTE processing complete.")
    print(f"Successfully processed files listed in CSV: {processed_count}")
    print(f"Skipped/failed CSV entries: {skipped_count}")

def run_preprocessing(config, input_dir, output_dir, preprocessed_dir, calibration_file, raw_data_root):
    """Runs the full TCSPC preprocessing pipeline (ImageJ + FLUTE)."""
    
    # Extract required config params
    try:
        imagej_path = config["imagej_path"]
        flute_path = config["flute_path"] # Still needed for ImageHandler import path
        macro_files = config["macro_files"]
        microscope_params = config["microscope_params"]
    except KeyError as e:
        print(f"Error: Missing required key in config data: {e}. Cannot run preprocessing.")
        return False # Indicate failure
        
    # Add FLUTE directory to sys.path to allow importing ImageHandler
    # (This needs to happen before ImageHandler is used in process_tiffs_with_flute)
    flute_dir = os.path.dirname(flute_path)
    if flute_dir not in sys.path:
        sys.path.append(flute_dir)
    try:
        from ImageHandler_noGUI import ImageHandler # Ensure import happens
    except ImportError as e:
        print(f"Error importing ImageHandler from {flute_dir}: {e}")
        return False # Indicate failure
        
    # === Run ImageJ Macros ===
    print("Running ImageJ Macro 1...")
    run_imagej(imagej_path, macro_files[0], input_dir, output_dir)
    print("ImageJ Macro 1 finished.")

    print("Running ImageJ Macro 2...")
    run_imagej(imagej_path, macro_files[1], input_dir, output_dir)
    print("ImageJ Macro 2 finished.")

    # === Run FLUTE processing ===
    print("Starting FLUTE processing...")
    process_tiffs_with_flute(
        calibration_file, 
        output_dir, 
        raw_data_root, 
        microscope_params
    )

    # === Run Post-FLUTE ImageJ Macros ===
    print("Running ImageJ Macro 3...")
    run_imagej(imagej_path, macro_files[2], output_dir, preprocessed_dir)
    print("ImageJ Macro 3 finished.")

    print("Running ImageJ Macro 4...")
    run_imagej(imagej_path, macro_files[3], output_dir, preprocessed_dir)
    print("ImageJ Macro 4 finished.")

    print("Running ImageJ Macro 5...")
    run_imagej(imagej_path, macro_files[4], preprocessed_dir)
    print("ImageJ Macro 5 finished.")

    print("Preprocessing pipeline complete.")
    return True # Indicate success

# === Main script execution block (for running standalone) ===
if __name__ == "__main__":
    print("This script is intended to be run via run_pipeline.py")
    print("Please use: python run_pipeline.py --preprocess -i <input_dir> -o <output_base_dir> [-c <calib.csv>]")
    # Example of how it *could* be run standalone, but discouraged
    # config = load_pipeline_config() # Need a config loader here too
    # if config:
    #    # Define input/output for standalone test
    #    input_dir = "/path/to/your/input"
    #    output_base_dir = "/path/to/your/output_base"
    #    calibration_file = "calibration.csv"
    #    raw_data_root = "/path/to/your/input" # Often same as input_dir
    #    output_dir = os.path.join(output_base_dir, 'output')
    #    preprocessed_dir = os.path.join(output_base_dir, 'preprocessed')
    #    os.makedirs(output_dir, exist_ok=True)
    #    os.makedirs(preprocessed_dir, exist_ok=True)
    #    run_preprocessing(config, input_dir, output_dir, preprocessed_dir, calibration_file, raw_data_root)
    sys.exit(1) 