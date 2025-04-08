import os
import sys
import numpy as np

def main(config, preprocessed_dir, npz_dir):
    """
    Main execution function: loads params from config, finds files, processes, saves results to NPZ.
    """
    # config = load_config() # Config is passed as argument
    
    # Paths are passed as arguments
    # preprocessed_dir = config["preprocessed_dir"]
    # npz_dir = config["npz_dir"]
    
    # Extract params from config dict
    try:
        flevel = config["wavelet_filter_params"]["flevel"]
        freq_mhz = config["microscope_params"]["freq_mhz"]
        harmonic = config["microscope_params"]["harmonic"]
    except KeyError as e:
        print(f"Error: Missing required parameter key in config data: {e}. Cannot run wavelet filter.")
        return False # Indicate failure
        
    omega_rad_per_ns = 2 * np.pi * freq_mhz * 1e6 * harmonic * 1e-9

    print(f"Starting Complex Wavelet Filtering and Data Processing")
    print(f"Input (preprocessed TIFFs) directory: {preprocessed_dir}")
    print(f"Output (NPZ datasets) directory: {npz_dir}")
    print(f"Wavelet levels (flevel): {flevel}")
    print(f"Calculated Omega (rad/ns): {omega_rad_per_ns:.4f}")

    if not os.path.isdir(preprocessed_dir):
        print(f"Error: Preprocessed directory not found: {preprocessed_dir}", file=sys.stderr)
        return False # Indicate failure
        
    processed_count = 0
    skipped_count = 0

    # --- Use os.walk to find relevant files --- 
    for root, dirs, files in os.walk(preprocessed_dir):
        # Logic: Process files if we are in a directory containing _g.tiff, _s.tiff, _Intensity.tiff files
        # We assume these files correspond to a single acquisition (e.g., R_1_s1)
        # and are all located in the same directory (e.g., .../Dish_1_Post-Rapa/R1/)
        
        # Check if required file types are present in the current directory `root`
        found_g = any(f.lower().endswith("_g.tiff") for f in files)
        found_s = any(f.lower().endswith("_s.tiff") for f in files)
        # Check for the intensity file name created by ImageJ Macro 4
        # Let's assume it's just {base_name}_Intensity.tiff or similar - adjust if needed
        found_int = any("_intensity.tiff" in f.lower() for f in files)

        # If not all types are found, continue walking
        if not (found_g and found_s and found_int):
            continue
        
        relative_path = os.path.relpath(root, preprocessed_dir)
        print(f"\nFound potential dataset in: {relative_path if relative_path != '.' else 'root'}")
        
        # Group files by base name (e.g., 'R_1_s1')
        datasets = {}
        intensity_suffix = "_intensity.tiff" # Adjust this if Macro 4 names it differently
        for f in files:
            if f.lower().endswith("_g.tiff"):
                base_name = f[:-len("_g.tiff")]
                datasets.setdefault(base_name, {})['g'] = os.path.join(root, f)
            elif f.lower().endswith("_s.tiff"):
                base_name = f[:-len("_s.tiff")]
                datasets.setdefault(base_name, {})['s'] = os.path.join(root, f)
            elif intensity_suffix in f.lower(): 
                base_name = f.lower().replace(intensity_suffix, "")
                datasets.setdefault(base_name, {})['int'] = os.path.join(root, f)
        
        # Process each complete dataset found in this directory
        for base_name, file_paths in datasets.items():
            if 'g' in file_paths and 's' in file_paths and 'int' in file_paths:
                print(f" Processing dataset: {base_name}")
                try:
                    # Process the data
                    processed_data = process_and_filter_set(
                        file_paths['g'], file_paths['s'], file_paths['int'], 
                        flevel, omega_rad_per_ns
                    )

                    if processed_data is not None:
                        # Construct output path mirroring the input structure relative to preprocessed_dir
                        output_npz_condition_dir = os.path.join(npz_dir, relative_path)
                        npz_out_path = os.path.join(output_npz_condition_dir, f"{base_name}_processed.npz")
                        
                        print(f"  Saving processed data to: {npz_out_path}")
                        save_npz(npz_out_path, processed_data)
                        processed_count += 1
                    else:
                        skipped_count += 1

                except Exception as e:
                    print(f" Error processing dataset '{base_name}' in {root}: {e}", file=sys.stderr)
                    skipped_count += 1
            else:
                print(f" Warning: Incomplete dataset for '{base_name}' in {root}. Skipping.", file=sys.stderr)
                skipped_count += 1
                
        # Prevent os.walk from going deeper into already processed subdirs (like G_unfiltered etc. if they existed)
        dirs[:] = [] # Clear the list of directories to visit further down this path

    print(f"\nData Processing and Filtering finished.")
    print(f"Successfully processed and saved NPZ file sets: {processed_count}")
    print(f"Skipped/failed file sets: {skipped_count}")
    return True # Indicate success

if __name__ == "__main__":
    print("This script is intended to be run via run_pipeline.py")
    print("Please use: python run_pipeline.py --filter -i <input_dir> -o <output_base_dir> [...]")
    sys.exit(1) 