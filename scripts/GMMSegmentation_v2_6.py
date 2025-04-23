import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from PIL import Image
import traceback

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
        
    # Create output directories if they don't exist
    os.makedirs(segmented_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(lifetime_dir, exist_ok=True)
        
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

                # Create output directories mirroring input structure
                output_segmented_dir = os.path.join(segmented_dir, relative_path)
                os.makedirs(output_segmented_dir, exist_ok=True)
                
                output_plots_dir = os.path.join(plots_dir, relative_path)
                os.makedirs(output_plots_dir, exist_ok=True)
                
                output_lifetime_dir = os.path.join(lifetime_dir, relative_path)
                os.makedirs(output_lifetime_dir, exist_ok=True)
                
                # Output paths
                segmented_mask_path = os.path.join(output_segmented_dir, f"{base_name}_mask.tiff")
                plot_path = os.path.join(output_plots_dir, f"{base_name}_plot.png")
                lifetime_path = os.path.join(output_lifetime_dir, f"{base_name}_lifetime.tiff")

                # Save Masks
                if output_masks is not None:
                    print(f"  Saving output masks to: {output_segmented_dir}")
                    for mask_type, mask_data in output_masks.items():
                        mask_filename = f"{base_name}_{mask_type}_mask.tiff"
                        mask_out_path = os.path.join(output_segmented_dir, mask_filename)
                        save_tiff(mask_out_path, mask_data, dtype=np.uint8)
                else: print("  Skipping mask saving due to segmentation error.")
                
                # Save Plot
                if phasor_plot_fig is not None:
                    plot_filename = f"{base_name}_phasor.png"
                    plot_out_path = os.path.join(output_plots_dir, plot_filename)
                    print(f"  Saving phasor plot to: {plot_out_path}")
                    save_plot(phasor_plot_fig, plot_out_path)
                else: print("  Skipping plot saving due to error.")
                
                # Save Lifetime Images
                t_unfil_filename = f"{base_name}_T_unfiltered.tiff"
                t_cwf_filename = f"{base_name}_T_CWF.tiff"
                t_unfil_out_path = os.path.join(output_lifetime_dir, t_unfil_filename)
                t_cwf_out_path = os.path.join(output_lifetime_dir, t_cwf_filename)
                print(f"  Saving lifetime images to: {output_lifetime_dir}")
                save_tiff(t_unfil_out_path, npz_data['T'], dtype=np.float32)
                save_tiff(t_cwf_out_path, npz_data['TCWF'], dtype=np.float32)
                
                if output_masks is not None and phasor_plot_fig is not None:
                    processed_count += 1
                else: 
                    if npz_data: skipped_count += 1

            except Exception as e: 
                print(f" Error processing {npz_filename}: {e}")
                traceback.print_exc()
                skipped_count += 1

    print(f"\nGMM Segmentation, Plotting, and Lifetime Saving finished.")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped/failed: {skipped_count}")
    return True # Indicate success

def load_npz_data(npz_file_path):
    """Load NPZ file and return its contents as a dictionary"""
    try:
        if not os.path.exists(npz_file_path):
            print(f"Error: NPZ file not found: {npz_file_path}")
            return None
            
        # Load the NPZ file
        data = np.load(npz_file_path)
        
        # Convert to dictionary for easier access
        npz_dict = {}
        for key in data.files:
            npz_dict[key] = data[key]
            
        print(f"  Loaded NPZ with keys: {list(npz_dict.keys())}")
        return npz_dict
        
    except Exception as e:
        print(f"Error loading NPZ file {npz_file_path}: {e}")
        return None

def perform_gmm_segmentation(npz_data, gmm_params):
    """
    Perform GMM segmentation on phasor plot data.
    
    Args:
        npz_data: Dictionary containing G, S, and intensity data
        gmm_params: Dictionary containing GMM parameters
        
    Returns:
        Tuple of (mask_dict, plot_figure)
    """
    try:
        # Extract data and parameters
        G = npz_data.get('G', None)
        S = npz_data.get('S', None)
        intensity = npz_data.get('intensity', None)
        GCWF = npz_data.get('GCWF', None)
        SCWF = npz_data.get('SCWF', None)
        
        if G is None or S is None or intensity is None:
            print("  Error: Missing required G, S, or intensity data in NPZ file")
            return None, None
            
        # Use CWF versions if available, otherwise use unfiltered
        g_data = GCWF if GCWF is not None else G
        s_data = SCWF if SCWF is not None else S
        
        # Set up parameters for GMM
        n_components = gmm_params.get('n_components', 3)
        threshold = gmm_params.get('intensity_threshold', 0.1)
        max_iter = gmm_params.get('max_iter', 100)
        
        # Threshold data to include only pixels with intensity above threshold
        mask = intensity > np.max(intensity) * threshold
        g_flat = g_data[mask].flatten()
        s_flat = s_data[mask].flatten()
        
        # Prepare data for GMM
        X = np.column_stack((g_flat, s_flat))
        
        # Check if we have enough points for GMM
        if len(X) < n_components * 2:
            print(f"  Warning: Not enough data points ({len(X)}) for {n_components} GMM components")
            if len(X) <= 10:  # Really not enough data
                return None, None
            n_components = min(n_components, max(2, len(X) // 2))
            print(f"  Reducing to {n_components} components")
        
        # Perform GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=gmm_params.get('covariance_type', 'full'),
            max_iter=max_iter,
            random_state=gmm_params.get('random_state', 0)
        )
        
        # Fit the GMM
        gmm.fit(X)
        
        # Predict labels for all pixels
        full_mask = np.zeros_like(g_data, dtype=np.int32)
        predictions = gmm.predict(X)
        
        # Create a mask for each component
        height, width = g_data.shape
        mask_indices = np.where(mask)
        
        for i in range(len(mask_indices[0])):
            y, x = mask_indices[0][i], mask_indices[1][i]
            idx = i % len(predictions)  # Ensure we don't go out of bounds
            full_mask[y, x] = predictions[idx] + 1  # +1 to avoid 0 (background)
        
        # Create individual binary masks for each component
        masks = {}
        for i in range(n_components):
            component_mask = (full_mask == (i + 1)).astype(np.uint8) * 255
            masks[f"component_{i+1}"] = component_mask
        
        # Also include the full mask
        masks["full"] = full_mask.astype(np.uint8)
        
        # Create phasor plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot histogram of G and S values
        hist, xedges, yedges = np.histogram2d(g_flat, s_flat, bins=100, range=[[0, 1], [0, 1]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        # Plot phasor histogram with logarithmic colorscale
        plt.imshow(hist.T, extent=extent, origin='lower', cmap='viridis', 
                  aspect='auto', norm=plt.cm.colors.LogNorm())
        
        # Plot GMM components
        for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
            v, w = np.linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan2(u[1], u[0])
            angle = np.degrees(angle)
            ellipse = plt.matplotlib.patches.Ellipse(mean, v[0], v[1], 180. + angle, 
                                                   color=f'C{i}', alpha=0.5, label=f'Component {i+1}')
            ax.add_artist(ellipse)
            
            # Mark the center of the component
            ax.scatter(mean[0], mean[1], color=f'C{i}', s=100, marker='x')
            
        # Add a semicircle for reference
        theta = np.linspace(0, np.pi, 100)
        x = 0.5 + 0.5 * np.cos(theta)
        y = 0.5 * np.sin(theta)
        ax.plot(x, y, 'k--', alpha=0.3)
        
        # Add gridlines
        ax.grid(True, alpha=0.3)
        
        # Set labels and title
        ax.set_xlabel('G')
        ax.set_ylabel('S')
        ax.set_title('Phasor Plot with GMM Components')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.7)
        ax.legend()
        
        plt.colorbar(label='Pixel Count (log scale)')
        
        return masks, fig
        
    except Exception as e:
        print(f"  Error in GMM segmentation: {e}")
        traceback.print_exc()
        return None, None

def save_tiff(file_path, data, dtype=None):
    """Save data as a TIFF file using PIL"""
    try:
        # Ensure data is in the right type
        if dtype is not None:
            data = data.astype(dtype)
            
        # Scale data for uint8 images
        if dtype == np.uint8 and data.max() <= 1:
            data = (data * 255).astype(np.uint8)
            
        # Create Image and save
        img = Image.fromarray(data)
        img.save(file_path)
        return True
    except Exception as e:
        print(f"Error saving TIFF file {file_path}: {e}")
        traceback.print_exc()
        return False

def save_plot(figure, file_path):
    """Save matplotlib figure to file"""
    try:
        figure.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close(figure)  # Close to free memory
        return True
    except Exception as e:
        print(f"Error saving plot to {file_path}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("This script is intended to be run via run_pipeline.py")
    print("Please use: python run_pipeline.py --segment -i <input_dir> -o <output_base_dir> [...]")
    sys.exit(1) 