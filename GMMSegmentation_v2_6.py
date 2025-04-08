import os
import sys
import numpy as np

def main(config, npz_dir, segmented_dir, plots_dir, lifetime_dir):
    """
    Main execution: Load params from config, find NPZ, run GMM, save outputs.
    """
    # config = load_config() # Config passed as argument
    # npz_dir = config["npz_dir"] # Path passed as argument
    # segmented_dir = config["segmented_dir"]
    # plots_dir = config["plots_dir"]
    # lifetime_dir = config["lifetime_dir"]
    
    # Extract params from config dict
    try:
        gmm_params = config["gmm_segmentation_params"]
    except KeyError as e:
        print(f"Error: Missing required gmm_segmentation_params in config data: {e}")
        return False # Indicate failure

    print(f"Starting GMM Segmentation, Plotting, and Lifetime Saving")
    print(f"Input NPZ directory: {npz_dir}")
    print(f"Output Segmented Masks: {segmented_dir}")
    print(f"Output Plots: {plots_dir}")
    print(f"Output Lifetime Images: {lifetime_dir}")
    print(f"GMM Parameters: {gmm_params}")

    if not os.path.isdir(npz_dir): 
        print(f"Error: NPZ dir not found: {npz_dir}", file=sys.stderr)
        return False # Indicate failure
        
    processed_count = 0
    skipped_count = 0

    # os.walk logic remains the same...
    for root, dirs, files in os.walk(npz_dir):
        npz_files_in_dir = sorted([
            f for f in files if f.lower().endswith("_processed.npz")
        ])
        if not npz_files_in_dir: continue
            
        relative_path = os.path.relpath(root, npz_dir)
        print(f"\nProcessing directory: {relative_path if relative_path != '.' else 'root'}")
        print(f" Found {len(npz_files_in_dir)} NPZ files to process.")

        for npz_filename in npz_files_in_dir:
            npz_file_path = os.path.join(root, npz_filename)
            base_name = npz_filename[:-len("_processed.npz")]
            print(f"\n Processing file: {npz_filename}")
            
            try:
                npz_data = load_npz_data(npz_file_path)
                if npz_data is None: skipped_count += 1; continue 

                output_masks, phasor_plot_fig = perform_gmm_segmentation(npz_data, gmm_params)

                output_condition_mask_dir = os.path.join(segmented_dir, relative_path)
                output_condition_plot_dir = os.path.join(plots_dir, relative_path)
                output_condition_lifetime_dir = os.path.join(lifetime_dir, relative_path)

                # Save Masks
                if output_masks is not None:
                    print(f"  Saving output masks to: {output_condition_mask_dir}")
                    for mask_type, mask_data in output_masks.items():
                        mask_filename = f"{base_name}_{mask_type}_mask.tiff"
                        mask_out_path = os.path.join(output_condition_mask_dir, mask_filename)
                        save_tiff(mask_out_path, mask_data, dtype=np.uint8)
                else: print("  Skipping mask saving due to segmentation error.")
                
                # Save Plot
                if phasor_plot_fig is not None:
                    plot_filename = f"{base_name}_phasor.png"
                    plot_out_path = os.path.join(output_condition_plot_dir, plot_filename)
                    print(f"  Saving phasor plot to: {plot_out_path}")
                    save_plot(phasor_plot_fig, plot_out_path)
                else: print("  Skipping plot saving due to error.")
                
                # Save Lifetime Images
                t_unfil_filename = f"{base_name}_T_unfiltered.tiff"
                t_cwf_filename = f"{base_name}_T_CWF.tiff"
                t_unfil_out_path = os.path.join(output_condition_lifetime_dir, t_unfil_filename)
                t_cwf_out_path = os.path.join(output_condition_lifetime_dir, t_cwf_filename)
                print(f"  Saving lifetime images to: {output_condition_lifetime_dir}")
                save_tiff(t_unfil_out_path, npz_data['T'], dtype=np.float32)
                save_tiff(t_cwf_out_path, npz_data['TCWF'], dtype=np.float32)
                
                if output_masks is not None and phasor_plot_fig is not None:
                    processed_count += 1
                else: 
                    if npz_data: skipped_count += 1

            except Exception as e: print(f" Error processing {npz_filename}: {e}"); skipped_count += 1

    print(f"\nGMM Segmentation, Plotting, and Lifetime Saving finished.")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped/failed: {skipped_count}")
    return True # Indicate success

if __name__ == "__main__":
    print("This script is intended to be run via run_pipeline.py")
    print("Please use: python run_pipeline.py --segment -i <input_dir> -o <output_base_dir> [...]")
    sys.exit(1) 