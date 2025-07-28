import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatter
from sklearn.mixture import GaussianMixture
from PIL import Image
import traceback
import glob
import argparse
import datetime

# Import centralized phasor plot utilities
from .phasor_plot_utils import create_phasor_plot

# Import error tracking
try:
    from error_tracker import create_error_tracker
except ImportError:
    # Fallback if error tracker is not available
    def create_error_tracker(module_name, log_dir=None):
        class DummyTracker:
            def log_error(self, error, context="", file_path=""):
                print(f"ERROR in {module_name}: {context} - {error}")
            def log_warning(self, message, context=""):
                print(f"WARNING in {module_name}: {context} - {message}")
            def log_info(self, message):
                print(f"{module_name}: {message}")
            def error_context(self, context="", file_path=""):
                import contextlib
                @contextlib.contextmanager
                def dummy_context():
                    try:
                        yield
                    except Exception as e:
                        self.log_error(e, context, file_path)
                        raise
                return dummy_context()
        return DummyTracker()

def is_point_inside_circle(point, center, radius):
    """
    Check if a point is inside a circle.
    
    Args:
        point: Tuple (x, y) representing the coordinates of the point.
        center: Tuple (x, y) representing the coordinates of the circle's center.
        radius: Float representing the radius of the circle.
    
    Returns:
        True if the point is inside the circle, False otherwise.
    """
    distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
    return distance <= radius

def are_points_inside_circle(points, center, radius):
    """
    Check if a list of points are inside a circle.
    
    Args:
        points: List of tuples representing the coordinates of the points.
        center: Tuple (x, y) representing the coordinates of the circle's center.
        radius: Float representing the radius of the circle.
    
    Returns:
        List of booleans indicating whether each point is inside the circle.
    """
    results = []
    for point in points:
        results.append(is_point_inside_circle(point, center, radius))
    return results

def is_points_inside_rotated_ellipse(center_x, center_y, semi_major_axis, semi_minor_axis, angle_degrees, points):
    """
    Check if points are inside a rotated ellipse.
    
    Args:
        center_x, center_y: Center coordinates of the ellipse
        semi_major_axis, semi_minor_axis: Ellipse axes
        angle_degrees: Rotation angle in degrees
        points: List of (x, y) points to check
    
    Returns:
        List of booleans indicating whether each point is inside the ellipse
    """
    # Calculate the rotation angle of the ellipse
    angle_radians = np.radians(angle_degrees)
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)

    results = []

    for point in points:
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

def load_gmm_config(config_file_path):
    """
    Load GMM configuration from JSON file and flatten the nested structure.
    
    Args:
        config_file_path: Path to the GMM config JSON file
        
    Returns:
        Dictionary with flattened parameter structure
    """
    try:
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        
        # Extract the gmm_segmentation_params section
        gmm_config = config.get('gmm_segmentation_params', {})
        
        # Flatten the nested structure to get actual parameter values
        flattened_params = {}
        
        # Helper function to extract values from nested structure
        def extract_values(section, prefix=""):
            for key, value in section.items():
                if key == "description" or key == "type" or key == "min" or key == "max" or key == "options":
                    continue
                if isinstance(value, dict) and "value" in value:
                    # This is a parameter with a value
                    param_key = f"{prefix}{key}" if prefix else key
                    flattened_params[param_key] = value["value"]
                elif isinstance(value, dict):
                    # This is a subsection, recurse
                    new_prefix = f"{prefix}{key}_" if prefix else f"{key}_"
                    extract_values(value, new_prefix)
        
        extract_values(gmm_config)
        
        return flattened_params
        
    except Exception as e:
        print(f"Error loading GMM config file {config_file_path}: {e}")
        return {}

def parse_gmm_arguments():
    """Parse command-line arguments for GMM segmentation."""
    parser = argparse.ArgumentParser(description="GMM Segmentation for FLIM-FRET Analysis")
    
    # Required arguments
    parser.add_argument("--npz-dir", required=True, help="Directory containing NPZ files")
    parser.add_argument("--output-dir", required=True, help="Base output directory")
    
    # Configuration file argument
    parser.add_argument("--config-file", type=str, help="Path to GMM configuration JSON file (overrides other arguments)")
    
    # GMM parameters
    parser.add_argument("--n-components", type=int, default=2, help="Number of GMM components (default: 2)")
    parser.add_argument("--intensity-threshold", type=float, default=0.0, help="Intensity threshold value (default: 0.0)")
    parser.add_argument("--threshold-type", choices=['relative', 'absolute', 'percentile'], default='relative', 
                       help="Threshold type: relative to max, absolute value, or percentile (default: relative)")
    parser.add_argument("--covariance-type", choices=['full', 'tied', 'diag', 'spherical'], default='full',
                       help="GMM covariance type (default: full)")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum GMM iterations (default: 100)")
    parser.add_argument("--random-state", type=int, default=0, help="Random state for reproducibility (default: 0)")
    
    # Advanced segmentation parameters
    parser.add_argument("--radius-ref", type=float, default=0.05, help="Radius for reference circle filtering (default: 0.05)")
    parser.add_argument("--cov-f", type=float, default=6.0, help="Covariance multiplier for ellipse size (default: 6.0)")
    parser.add_argument("--shift", type=float, default=0.0, help="Shift multiplier for ellipse position (default: 0.0)")
    parser.add_argument("--use-circle-filter", action="store_true", help="Use circle filtering around reference center")
    
    # Data source options
    parser.add_argument("--use-unfiltered", action="store_true", help="Use unfiltered data (GU/SU) instead of filtered (G/S)")
    parser.add_argument("--combine-datasets", action="store_true", help="Process all NPZ files as a combined dataset")
    
    # Reference center options
    parser.add_argument("--ref-g", type=float, help="Reference center G coordinate")
    parser.add_argument("--ref-s", type=float, help="Reference center S coordinate")
    
    # Interactive mode
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode to select options")
    
    # Output options
    parser.add_argument("--segmented-dir", help="Directory for segmented masks (default: output_dir/segmented)")
    parser.add_argument("--plots-dir", help="Directory for plots (default: output_dir/plots)")
    parser.add_argument("--lifetime-dir", help="Directory for lifetime images (default: output_dir/lifetime_images)")
    
    return parser.parse_args()

def interactive_gmm_setup(npz_dir=None):
    """Interactive setup for GMM parameters with list-based selection."""
    print("\n" + "="*60)
    print("GMM Segmentation Interactive Setup")
    print("="*60)
    
    params = {}
    
    # 1. Data source selection
    print("\n1. DATA SOURCE SELECTION:")
    print("   1. Filtered data (G/S)")
    print("   2. Unfiltered data (GU/SU)")
    
    while True:
        data_choice = input("   Select data source (1-2, default: 1): ").strip()
        if data_choice == "" or data_choice == "1":
            params['use_unfiltered_data'] = False
            print("   → Using filtered data (G/S)")
            break
        elif data_choice == "2":
            params['use_unfiltered_data'] = True
            print("   → Using unfiltered data (GU/SU)")
            break
        else:
            print("   Please select 1 or 2.")
    
    # 2. NPZ file selection (if npz_dir is provided)
    if npz_dir and os.path.exists(npz_dir):
        print(f"\n2. NPZ FILE SELECTION:")
        print(f"   Directory: {npz_dir}")
        
        # Find all NPZ files
        npz_files = []
        for root, dirs, files in os.walk(npz_dir):
            for file in files:
                if file.lower().endswith(".npz") and not file.lower().endswith("_segmented.npz"):
                    npz_files.append(os.path.join(root, file))
        
        if npz_files:
            print(f"   Found {len(npz_files)} NPZ files:")
            for i, file_path in enumerate(npz_files, 1):
                rel_path = os.path.relpath(file_path, npz_dir)
                print(f"   {i:2d}. {rel_path}")
            
            print("   Options:")
            print("   - Enter 'all' to process all files as combined dataset")
            print("   - Enter 'individual' to process all files individually")
            print("   - Enter comma-separated numbers (e.g., 1,3,5) for specific files")
            print("   - Enter 'combined' to process as combined dataset")
            
            while True:
                file_choice = input("   Select files (default: all): ").strip()
                if file_choice == "" or file_choice.lower() == "all":
                    params['selected_files'] = "all"
                    params['combine_datasets'] = True  # Changed to True for combined processing
                    print("   → Processing all files as combined dataset")
                    break
                elif file_choice.lower() == "individual":
                    params['selected_files'] = "all"
                    params['combine_datasets'] = False
                    print("   → Processing all files individually")
                    break
                elif file_choice.lower() == "combined":
                    params['selected_files'] = "all"
                    params['combine_datasets'] = True
                    print("   → Processing all files as combined dataset")
                    break
                else:
                    try:
                        # Parse comma-separated numbers
                        indices = [int(x.strip()) for x in file_choice.split(",")]
                        if all(1 <= i <= len(npz_files) for i in indices):
                            params['selected_files'] = indices
                            params['combine_datasets'] = False
                            selected_names = [os.path.relpath(npz_files[i-1], npz_dir) for i in indices]
                            print(f"   → Selected files: {', '.join(selected_names)}")
                            break
                        else:
                            print(f"   Please enter numbers between 1 and {len(npz_files)}")
                    except ValueError:
                        print("   Please enter valid numbers separated by commas")
        else:
            print("   No NPZ files found in directory")
            params['selected_files'] = "all"
            params['combine_datasets'] = False
    else:
        # Default if no npz_dir provided
        params['selected_files'] = "all"
        params['combine_datasets'] = False
    
    # 3. Threshold selection
    print("\n3. THRESHOLD SELECTION:")
    print("   Threshold options:")
    print("   1. Relative to maximum intensity")
    print("   2. Absolute value")
    print("   3. Percentile")
    
    while True:
        threshold_choice = input("   Select threshold type (1-3, default: 1): ").strip()
        if threshold_choice == "" or threshold_choice == "1":
            params['threshold_type'] = 'relative'
            print("   → Using relative threshold")
            break
        elif threshold_choice == "2":
            params['threshold_type'] = 'absolute'
            print("   → Using absolute threshold")
            break
        elif threshold_choice == "3":
            params['threshold_type'] = 'percentile'
            print("   → Using percentile threshold")
            break
        else:
            print("   Please select 1, 2, or 3.")
    
    # Threshold value
    while True:
        try:
            threshold_value = input("   Threshold value (default: 0.0): ").strip()
            if threshold_value == "":
                params['intensity_threshold'] = 0.0
                break
            params['intensity_threshold'] = float(threshold_value)
            break
        except ValueError:
            print("   Please enter a valid number.")
    
    # 4. Advanced segmentation parameters
    print("\n4. ADVANCED SEGMENTATION PARAMETERS:")
    
    # Circle filtering
    print("   Circle filtering around reference center:")
    print("   1. Enable circle filtering")
    print("   2. Disable circle filtering")
    
    while True:
        circle_filter_choice = input("   Select circle filtering (1-2, default: 2): ").strip()
        if circle_filter_choice == "" or circle_filter_choice == "2":
            params['use_circle_filter'] = False
            print("   → Circle filtering disabled")
            break
        elif circle_filter_choice == "1":
            params['use_circle_filter'] = True
            print("   → Circle filtering enabled")
            break
        else:
            print("   Please select 1 or 2.")
    
    # Radius reference
    while True:
        try:
            radius_ref = input("   Reference circle radius (default: 0.05): ").strip()
            if radius_ref == "":
                params['radius_ref'] = 0.05
                break
            params['radius_ref'] = float(radius_ref)
            break
        except ValueError:
            print("   Please enter a valid number.")
    
    # Covariance multiplier for ellipse
    while True:
        try:
            cov_f = input("   Ellipse covariance multiplier (default: 6.0): ").strip()
            if cov_f == "":
                params['cov_f'] = 6.0
                break
            params['cov_f'] = float(cov_f)
            break
        except ValueError:
            print("   Please enter a valid number.")
    
    # Shift multiplier
    while True:
        try:
            shift = input("   Position shift multiplier (default: 0.0): ").strip()
            if shift == "":
                params['shift'] = 0.0
                break
            params['shift'] = float(shift)
            break
        except ValueError:
            print("   Please enter a valid number.")
    
    # Note: GMM components, covariance type, and reference center are set from config file
    # and cannot be modified in interactive mode
    
    # Other parameters
    params['max_iter'] = 100
    params['random_state'] = 0
    
    print("\n" + "="*60)
    print("SELECTED PARAMETERS:")
    for key, value in params.items():
        if key != 'selected_files':  # Don't show the full file list in summary
            print(f"  {key}: {value}")
    if 'selected_files' in params:
        if params['selected_files'] == "all":
            print(f"  selected_files: all files")
        else:
            print(f"  selected_files: {len(params['selected_files'])} specific files")
    print("="*60)
    
    return params

def main(config=None, npz_dir=None, segmented_dir=None, plots_dir=None, lifetime_dir=None, interactive_mode=False, naming_variables=None, selected_mask_name=None):
    """
    Main execution: Load params from config, find NPZ, run GMM, save outputs.
    
    Args:
        config: Configuration dictionary (optional if using command-line args)
        npz_dir: NPZ directory (optional if using command-line args)
        segmented_dir: Segmented output directory (optional if using command-line args)
        plots_dir: Plots output directory (optional if using command-line args)
        lifetime_dir: Lifetime output directory (optional if using command-line args)
        interactive_mode: Whether to run in interactive mode (for pipeline integration)
        naming_variables: Dictionary containing naming variables for output files
        selected_mask_name: Name of the selected mask to apply (if any)
    """
    # Check if running as standalone script
    if config is None:
        args = parse_gmm_arguments()
        
        # Set up directories
        npz_dir = args.npz_dir
        output_dir = args.output_dir
        
        if args.segmented_dir:
            segmented_dir = args.segmented_dir
        else:
            segmented_dir = os.path.join(output_dir, 'segmented')
            
        if args.plots_dir:
            plots_dir = args.plots_dir
        else:
            plots_dir = os.path.join(output_dir, 'plots')
            
        if args.lifetime_dir:
            lifetime_dir = args.lifetime_dir
        else:
            lifetime_dir = os.path.join(output_dir, 'lifetime_images')
        
        # Build config from command-line arguments or config file
        if args.config_file:
            # Load parameters from config file
            print(f"Loading GMM parameters from config file: {args.config_file}")
            gmm_params = load_gmm_config(args.config_file)
            if not gmm_params:
                print("Failed to load config file, using default parameters")
                gmm_params = {}
        elif args.interactive:
            gmm_params = interactive_gmm_setup(npz_dir)
        else:
            gmm_params = {
                'n_components': args.n_components,
                'intensity_threshold': args.intensity_threshold,
                'threshold_type': args.threshold_type,
                'covariance_type': args.covariance_type,
                'max_iter': args.max_iter,
                'random_state': args.random_state,
                'use_unfiltered_data': args.use_unfiltered,
                'combine_datasets': args.combine_datasets,
                'radius_ref': args.radius_ref,
                'cov_f': args.cov_f,
                'shift': args.shift,
                'use_circle_filter': args.use_circle_filter
            }
            
            if args.ref_g is not None and args.ref_s is not None:
                gmm_params['reference_center_G'] = args.ref_g
                gmm_params['reference_center_S'] = args.ref_s
        
        config = {'gmm_segmentation_params': gmm_params}
    
    # Check if interactive mode is requested (either from command line or pipeline)
    if interactive_mode and config and 'gmm_segmentation_params' in config:
        # Get new parameters interactively with npz_dir for file selection
        new_params = interactive_gmm_setup(npz_dir)
        
        # Preserve original config values for restricted parameters
        original_params = config['gmm_segmentation_params']
        preserved_params = {
            'n_components': original_params.get('n_components', 2),
            'covariance_type': original_params.get('covariance_type', 'full'),
            'max_iter': original_params.get('max_iter', 100),
            'random_state': original_params.get('random_state', 0)
        }
        
        # Preserve reference center if it exists
        if 'reference_center_G' in original_params:
            preserved_params['reference_center_G'] = original_params['reference_center_G']
        if 'reference_center_S' in original_params:
            preserved_params['reference_center_S'] = original_params['reference_center_S']
        
        # Update config with new parameters, preserving restricted ones
        config['gmm_segmentation_params'].update(new_params)
        config['gmm_segmentation_params'].update(preserved_params)
        
        print("\nParameters updated! (GMM components, covariance type, and reference center preserved from config)")
    
    # Initialize error tracker
    tracker = create_error_tracker("GMMSegmentation", os.path.join(os.path.dirname(segmented_dir), 'logs'))
    
    # Extract params from config dict
    try:
        gmm_params = config["gmm_segmentation_params"]
    except KeyError as e:
        error_msg = f"Missing required gmm_segmentation_params in config data: {e}"
        tracker.log_error(Exception(error_msg), "Config loading")
        return False # Indicate failure

    tracker.log_info("Starting GMM Segmentation, Plotting, and Lifetime Saving")
    tracker.log_info(f"Input NPZ directory: {npz_dir}")
    tracker.log_info(f"Output Segmented Masks: {segmented_dir}")
    tracker.log_info(f"Output Plots: {plots_dir}")
    tracker.log_info(f"Output Lifetime Images: {lifetime_dir}")
    tracker.log_info(f"GMM Parameters: {gmm_params}")

    if not os.path.isdir(npz_dir): 
        error_msg = f"NPZ dir not found: {npz_dir}"
        tracker.log_error(Exception(error_msg), "Directory validation")
        return False # Indicate failure
        
    # Create output directories if they don't exist
    masks_dir = os.path.join(segmented_dir, 'masks')
    phasor_plots_dir = os.path.join(segmented_dir, 'phasor_plots')
    lifetime_images_dir = os.path.join(segmented_dir, 'lifetime_images')
    
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(phasor_plots_dir, exist_ok=True)
    os.makedirs(lifetime_images_dir, exist_ok=True)
        
    processed_count = 0
    skipped_count = 0

    # Check if we should process files individually or as combined datasets
    # Map flattened parameter names to expected names
    combine_datasets = gmm_params.get('data_selection_combine_datasets', gmm_params.get('combine_datasets', False))
    selected_files = gmm_params.get('data_selection_selected_files', gmm_params.get('selected_files', "all"))
    
    # Debug: Print the actual values being used
    tracker.log_info(f"DEBUG: combine_datasets = {combine_datasets}")
    tracker.log_info(f"DEBUG: selected_files = {selected_files}")
    tracker.log_info(f"DEBUG: All gmm_params keys = {list(gmm_params.keys())}")
    
    # Get list of all available NPZ files
    all_npz_files = []
    for root, dirs, files in os.walk(npz_dir):
        for file in files:
            if file.lower().endswith(".npz") and not file.lower().endswith("_segmented.npz"):
                all_npz_files.append(os.path.join(root, file))
    
    # Filter files based on selection
    if selected_files == "all":
        files_to_process = all_npz_files
        tracker.log_info(f"Processing all {len(files_to_process)} NPZ files")
    elif isinstance(selected_files, list):
        # User selected specific files by index
        files_to_process = [all_npz_files[i-1] for i in selected_files if 1 <= i <= len(all_npz_files)]
        tracker.log_info(f"Processing {len(files_to_process)} selected NPZ files")
    else:
        files_to_process = all_npz_files
        tracker.log_info(f"Processing all {len(files_to_process)} NPZ files (no selection specified)")
    
    if combine_datasets:
        # Process selected NPZ files as a single combined dataset
        tracker.log_info("Processing selected NPZ files as combined dataset")
        success = process_combined_datasets_with_selection(config, npz_dir, segmented_dir, plots_dir, lifetime_dir, tracker, files_to_process, naming_variables, selected_mask_name)
        return success
    else:
        # Process files individually
        tracker.log_info("Processing NPZ files individually")
        
        # Group files by directory for processing
        files_by_dir = {}
        for file_path in files_to_process:
            dir_path = os.path.dirname(file_path)
            if dir_path not in files_by_dir:
                files_by_dir[dir_path] = []
            files_by_dir[dir_path].append(os.path.basename(file_path))
        
        for root, npz_files_in_dir in files_by_dir.items():
            if not npz_files_in_dir: continue
                
            relative_path = os.path.relpath(root, npz_dir)
            tracker.log_info(f"Processing directory: {relative_path if relative_path != '.' else 'root'}")
            tracker.log_info(f"Found {len(npz_files_in_dir)} NPZ files to process.")

            for npz_filename in npz_files_in_dir:
                npz_file_path = os.path.join(root, npz_filename)
                base_name = npz_filename[:-len(".npz")]  # Remove .npz extension
                tracker.log_info(f"Processing file: {npz_filename}")
                
                with tracker.error_context("Processing NPZ file", npz_file_path):
                    npz_data = load_npz_data(npz_file_path, tracker)
                    if npz_data is None: 
                        skipped_count += 1
                        continue 

                    output_masks, phasor_plot_fig = perform_gmm_segmentation(npz_data, gmm_params, tracker)

                    # Create output directories for masks, plots, and lifetime images
                    output_masks_dir = os.path.join(masks_dir, relative_path)
                    output_phasor_plots_dir = os.path.join(phasor_plots_dir, relative_path)
                    output_lifetime_dir = os.path.join(lifetime_images_dir, relative_path)
                    
                    os.makedirs(output_masks_dir, exist_ok=True)
                    os.makedirs(output_phasor_plots_dir, exist_ok=True)
                    os.makedirs(output_lifetime_dir, exist_ok=True)

                    # Get output options from config
                    save_individual_masks = gmm_params.get('output_options_save_individual_masks', gmm_params.get('save_individual_masks', True))
                    save_individual_plots = gmm_params.get('output_options_save_individual_plots', gmm_params.get('save_individual_plots', False))
                    save_lifetime_images = gmm_params.get('output_options_save_lifetime_images', gmm_params.get('save_lifetime_images', True))
                    
                    # Save Masks
                    if output_masks is not None and save_individual_masks:
                        tracker.log_info(f"Saving output masks to: {output_masks_dir}")
                        for mask_type, mask_data in output_masks.items():
                            # Save as TIFF file
                            mask_filename = f"{base_name}_{mask_type}_mask.tiff"
                            mask_out_path = os.path.join(output_masks_dir, mask_filename)
                            if save_tiff(mask_out_path, mask_data, dtype=np.uint8):
                                tracker.log_info(f"Saved {mask_filename}")
                            else:
                                tracker.log_warning(f"Failed to save {mask_filename}")
                            
                            # Append to NPZ file
                            try:
                                # Load existing NPZ data
                                existing_data = np.load(npz_file_path, allow_pickle=True)
                                npz_data_dict = dict(existing_data)
                                
                                # Add GMM component masks to NPZ data
                                # Ensure mask_data is properly formatted as uint8
                                mask_data_uint8 = mask_data.astype(np.uint8)
                                
                                if mask_type == "ellipse_segmentation_component_1":
                                    npz_data_dict['GMM_segmentation_component_1_mask'] = mask_data_uint8
                                elif mask_type == "ellipse_segmentation_component_2":
                                    npz_data_dict['GMM_segmentation_component_2_mask'] = mask_data_uint8
                                elif mask_type == "ellipse_segmentation_combined":
                                    npz_data_dict['GMM_segmentation_mask'] = mask_data_uint8
                                
                                # Add mask registry if it doesn't exist
                                if 'mask_registry' not in npz_data_dict:
                                    npz_data_dict['mask_registry'] = {}
                                
                                # Add metadata for this mask
                                mask_key = None
                                if mask_type == "ellipse_segmentation_component_1":
                                    mask_key = 'GMM_segmentation_component_1_mask'
                                elif mask_type == "ellipse_segmentation_component_2":
                                    mask_key = 'GMM_segmentation_component_2_mask'
                                elif mask_type == "ellipse_segmentation_combined":
                                    mask_key = 'GMM_segmentation_mask'
                                
                                if mask_key:
                                    npz_data_dict['mask_registry'][mask_key] = {
                                        'type': 'binary',
                                        'description': f'GMM segmentation {mask_type}',
                                        'created_by': 'GMMSegmentation',
                                        'created_timestamp': datetime.datetime.now().isoformat()
                                    }
                                
                                # Save updated NPZ file
                                np.savez_compressed(npz_file_path, **npz_data_dict)
                                tracker.log_info(f"Updated NPZ file with {mask_key}: {npz_file_path}")
                                
                            except Exception as e:
                                tracker.log_error(e, f"Failed to update NPZ file {npz_file_path}")
                    elif output_masks is None:
                        tracker.log_warning("Skipping mask saving due to segmentation error")
                    else:
                        tracker.log_info("Skipping individual mask saving (disabled in config)")
                    
                    # Save Plot
                    if phasor_plot_fig is not None and save_individual_plots:
                        # Use naming variables if provided, otherwise use default naming
                        if naming_variables:
                            plot_filename = f"{base_name}_{naming_variables['file_selection']}_{naming_variables['method']}_{naming_variables['data_type']}_{naming_variables['mask_source']}_phasor.png"
                        else:
                            plot_filename = f"{base_name}_phasor.png"
                        plot_out_path = os.path.join(output_phasor_plots_dir, plot_filename)
                        tracker.log_info(f"Saving phasor plot to: {plot_out_path}")
                        if save_plot(phasor_plot_fig, plot_out_path):
                            tracker.log_info(f"Saved {plot_filename}")
                        else:
                            tracker.log_warning(f"Failed to save {plot_filename}")
                    elif phasor_plot_fig is None:
                        tracker.log_warning("Skipping plot saving due to error")
                    else:
                        tracker.log_info("Skipping individual plot saving (disabled in config)")
                    
                    # Save Lifetime Images
                    if save_lifetime_images:
                        # Check what lifetime data is available
                        lifetime_keys = [key for key in ['T', 'TU'] if key in npz_data]
                        for lifetime_key in lifetime_keys:
                            lifetime_filename = f"{base_name}_{lifetime_key}.tiff"
                            lifetime_out_path = os.path.join(output_lifetime_dir, lifetime_filename)
                            tracker.log_info(f"Saving lifetime image {lifetime_key} to: {lifetime_out_path}")
                            if save_tiff(lifetime_out_path, npz_data[lifetime_key], dtype=np.float32):
                                tracker.log_info(f"Saved {lifetime_filename}")
                            else:
                                tracker.log_warning(f"Failed to save {lifetime_filename}")
                    else:
                        tracker.log_info("Skipping individual lifetime image saving (disabled in config)")
                    
                    if output_masks is not None and phasor_plot_fig is not None:
                        processed_count += 1
                    else: 
                        if npz_data: 
                            skipped_count += 1

        tracker.log_info("GMM Segmentation, Plotting, and Lifetime Saving finished")
        tracker.log_info(f"Successfully processed: {processed_count}")
        tracker.log_info(f"Skipped/failed: {skipped_count}")
        return True # Indicate success

def process_combined_datasets_with_selection(config, npz_dir, segmented_dir, plots_dir, lifetime_dir, tracker, files_to_process, naming_variables=None, selected_mask_name=None):
    """
    Process selected NPZ files as a single combined dataset for GMM segmentation.
    """
    if not files_to_process:
        tracker.log_error(Exception("No NPZ files selected for processing"), "Combined dataset processing")
        return False
    
    # Create output directories for combined processing
    masks_dir = os.path.join(segmented_dir, 'masks')
    phasor_plots_dir = os.path.join(segmented_dir, 'phasor_plots')
    lifetime_images_dir = os.path.join(segmented_dir, 'lifetime_images')
    
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(phasor_plots_dir, exist_ok=True)
    os.makedirs(lifetime_images_dir, exist_ok=True)
    
    tracker.log_info(f"Processing {len(files_to_process)} selected NPZ files as combined dataset")
    
    # Load and combine data from selected NPZ files
    combined_data = {}
    file_info = []
    cumulative_sizes = [0]  # Track cumulative sizes for proper mask extraction
    
    for npz_file in files_to_process:
        with tracker.error_context("Loading NPZ file for combined processing", npz_file):
            npz_data = load_npz_data(npz_file, tracker)
            if npz_data is None:
                continue
            
            # Apply mask if selected
            if selected_mask_name and selected_mask_name in npz_data:
                tracker.log_info(f"Applying mask '{selected_mask_name}' to {os.path.basename(npz_file)}")
                mask = npz_data[selected_mask_name]
                
                # Apply mask to all phasor data
                for key in ['G', 'S', 'A', 'T', 'GU', 'SU', 'TU']:
                    if key in npz_data:
                        npz_data[key] = npz_data[key] * mask
                
                tracker.log_info(f"  Applied mask: {np.sum(mask)} pixels selected out of {mask.size} total")
            
            # Get the shape of this file for tracking
            g_data = npz_data.get('G', npz_data.get('GU'))
            if g_data is None:
                continue
            
            file_shape = g_data.shape
            current_size = file_shape[0]  # Number of rows
            
            # Store file info for later reference
            base_name = os.path.basename(npz_file)[:-4]  # Remove .npz
            file_info.append({
                'path': npz_file,
                'base_name': base_name,
                'data': npz_data,
                'start_idx': cumulative_sizes[-1],  # Starting index in combined dataset
                'end_idx': cumulative_sizes[-1] + current_size,  # Ending index in combined dataset
                'shape': file_shape
            })
            
            # Update cumulative sizes
            cumulative_sizes.append(cumulative_sizes[-1] + current_size)
            
            # Combine data
            for key in ['G', 'S', 'A', 'T', 'GU', 'SU', 'TU']:
                if key in npz_data:
                    if key not in combined_data:
                        combined_data[key] = []
                    combined_data[key].append(npz_data[key])
    
    if not combined_data:
        tracker.log_error(Exception("No valid data found in NPZ files"), "Combined dataset processing")
        return False
    
    # Concatenate all data
    for key in combined_data:
        combined_data[key] = np.concatenate(combined_data[key], axis=0)
    
    tracker.log_info(f"Combined dataset shape: {combined_data.get('G', combined_data.get('GU')).shape}")
    
    # Perform GMM segmentation on combined data
    output_masks, phasor_plot_fig = perform_gmm_segmentation(combined_data, config["gmm_segmentation_params"], tracker)
    
    if output_masks is None or phasor_plot_fig is None:
        tracker.log_error(Exception("GMM segmentation failed on combined dataset"), "Combined dataset processing")
        return False
    
    # Save combined outputs
    combined_base_name = f"combined_{len(file_info)}_files"
    
    # Get output options from config
    gmm_params = config["gmm_segmentation_params"]
    save_individual_masks = gmm_params.get('output_options_save_individual_masks', gmm_params.get('save_individual_masks', True))
    save_combined_masks = gmm_params.get('output_options_save_combined_masks', gmm_params.get('save_combined_masks', True))
    save_individual_plots = gmm_params.get('output_options_save_individual_plots', gmm_params.get('save_individual_plots', False))
    save_combined_plot = gmm_params.get('output_options_save_combined_plot', gmm_params.get('save_combined_plot', True))
    save_lifetime_images = gmm_params.get('output_options_save_lifetime_images', gmm_params.get('save_lifetime_images', True))
    
    # Save combined masks
    if save_combined_masks:
        tracker.log_info(f"Saving combined output masks to: {masks_dir}")
        for mask_type, mask_data in output_masks.items():
            mask_filename = f"{combined_base_name}_{mask_type}_mask.tiff"
            mask_out_path = os.path.join(masks_dir, mask_filename)
            if save_tiff(mask_out_path, mask_data, dtype=np.uint8):
                tracker.log_info(f"Saved {mask_filename}")
            else:
                tracker.log_warning(f"Failed to save {mask_filename}")
    else:
        tracker.log_info("Skipping combined mask saving (disabled in config)")
    
    # Save combined plot
    if save_combined_plot:
        # Use naming variables if provided, otherwise use default naming
        if naming_variables:
            plot_filename = f"{combined_base_name}_{naming_variables['file_selection']}_{naming_variables['method']}_{naming_variables['data_type']}_{naming_variables['mask_source']}_phasor.png"
        else:
            plot_filename = f"{combined_base_name}_phasor.png"
        plot_out_path = os.path.join(phasor_plots_dir, plot_filename)
        tracker.log_info(f"Saving combined phasor plot to: {plot_out_path}")
        if save_plot(phasor_plot_fig, plot_out_path):
            tracker.log_info(f"Saved {plot_filename}")
        else:
            tracker.log_warning(f"Failed to save {plot_filename}")
    else:
        tracker.log_info("Skipping combined plot saving (disabled in config)")
    
    # Save combined lifetime images
    if save_lifetime_images:
        lifetime_keys = [key for key in ['T', 'TU'] if key in combined_data]
        for lifetime_key in lifetime_keys:
            lifetime_filename = f"{combined_base_name}_{lifetime_key}.tiff"
            lifetime_out_path = os.path.join(lifetime_images_dir, lifetime_filename)
            tracker.log_info(f"Saving combined lifetime image {lifetime_key} to: {lifetime_out_path}")
            if save_tiff(lifetime_out_path, combined_data[lifetime_key], dtype=np.float32):
                tracker.log_info(f"Saved {lifetime_filename}")
            else:
                tracker.log_warning(f"Failed to save {lifetime_filename}")
    else:
        tracker.log_info("Skipping combined lifetime image saving (disabled in config)")
    
    # Apply combined masks to individual files
    tracker.log_info("Applying combined segmentation to individual files...")
    for file_info_item in file_info:
        apply_combined_mask_to_file(file_info_item, output_masks, masks_dir, phasor_plots_dir, lifetime_images_dir, tracker)
    
    tracker.log_info("Combined dataset processing completed successfully")
    return True

def process_combined_datasets(config, npz_dir, segmented_dir, plots_dir, lifetime_dir, tracker):
    """
    Process all NPZ files as a single combined dataset for GMM segmentation.
    (Legacy function for backward compatibility)
    """
    # Find all NPZ files
    npz_files = []
    for root, dirs, files in os.walk(npz_dir):
        for file in files:
            if file.lower().endswith(".npz") and not file.lower().endswith("_segmented.npz"):
                npz_files.append(os.path.join(root, file))
    
    if not npz_files:
        tracker.log_error(Exception("No NPZ files found"), "Combined dataset processing")
        return False
    
    tracker.log_info(f"Found {len(npz_files)} NPZ files for combined processing")
    
    # Use the new function with all files
    return process_combined_datasets_with_selection(config, npz_dir, segmented_dir, plots_dir, lifetime_dir, tracker, npz_files, None, None)

def apply_combined_mask_to_file(file_info_item, combined_masks, masks_dir, phasor_plots_dir, lifetime_images_dir, tracker):
    """
    Apply the combined segmentation masks to individual files.
    """
    npz_data = file_info_item['data']
    base_name = file_info_item['base_name']
    start_idx = file_info_item['start_idx']
    end_idx = file_info_item['end_idx']
    file_shape = file_info_item['shape']
    
    # Extract the corresponding portion from combined masks
    individual_masks = {}
    for mask_type, combined_mask in combined_masks.items():
        if combined_mask.shape[0] >= end_idx:
            # Extract the portion corresponding to this file
            individual_mask = combined_mask[start_idx:end_idx]
            
            # Ensure the extracted mask has the correct shape
            if individual_mask.shape != file_shape:
                tracker.log_warning(f"Mask shape mismatch for {base_name}: expected {file_shape}, got {individual_mask.shape}")
                # Reshape if possible
                if individual_mask.size == file_shape[0] * file_shape[1]:
                    individual_mask = individual_mask.reshape(file_shape)
                else:
                    continue
            
            individual_masks[mask_type] = individual_mask
        else:
            tracker.log_warning(f"Combined mask too small for {base_name}: {combined_mask.shape[0]} < {end_idx}")
    
    # Save individual masks and append to NPZ files
    for mask_type, mask_data in individual_masks.items():
        # Save as TIFF file
        mask_filename = f"{base_name}_{mask_type}_from_combined.tiff"
        mask_out_path = os.path.join(masks_dir, mask_filename)
        if save_tiff(mask_out_path, mask_data, dtype=np.uint8):
            tracker.log_info(f"Saved individual mask: {mask_filename}")
        else:
            tracker.log_warning(f"Failed to save individual mask: {mask_filename}")
        
        # Append to NPZ file
        npz_file_path = file_info_item['path']
        try:
            # Load existing NPZ data
            existing_data = np.load(npz_file_path, allow_pickle=True)
            npz_data_dict = dict(existing_data)
            
            # Add GMM component masks to NPZ data
            # Ensure mask_data is properly formatted as uint8
            mask_data_uint8 = mask_data.astype(np.uint8)
            
            if mask_type == "ellipse_segmentation_component_1":
                npz_data_dict['GMM_segmentation_component_1_mask'] = mask_data_uint8
            elif mask_type == "ellipse_segmentation_component_2":
                npz_data_dict['GMM_segmentation_component_2_mask'] = mask_data_uint8
            elif mask_type == "ellipse_segmentation_combined":
                npz_data_dict['GMM_segmentation_mask'] = mask_data_uint8
            
            # Add mask registry if it doesn't exist
            if 'mask_registry' not in npz_data_dict:
                npz_data_dict['mask_registry'] = {}
            
            # Add metadata for this mask
            mask_key = None
            if mask_type == "ellipse_segmentation_component_1":
                mask_key = 'GMM_segmentation_component_1_mask'
            elif mask_type == "ellipse_segmentation_component_2":
                mask_key = 'GMM_segmentation_component_2_mask'
            elif mask_type == "ellipse_segmentation_combined":
                mask_key = 'GMM_segmentation_mask'
            
            if mask_key:
                npz_data_dict['mask_registry'][mask_key] = {
                    'type': 'binary',
                    'description': f'GMM segmentation {mask_type}',
                    'created_by': 'GMMSegmentation',
                    'created_timestamp': datetime.datetime.now().isoformat()
                }
            
            # Save updated NPZ file
            np.savez_compressed(npz_file_path, **npz_data_dict)
            tracker.log_info(f"Updated NPZ file with {mask_key}: {npz_file_path}")
            
        except Exception as e:
            tracker.log_error(e, f"Failed to update NPZ file {npz_file_path}")

def load_npz_data(npz_file_path, tracker=None):
    """Load NPZ file and return its contents as a dictionary"""
    try:
        if not os.path.exists(npz_file_path):
            error_msg = f"NPZ file not found: {npz_file_path}"
            if tracker:
                tracker.log_error(Exception(error_msg), "File existence check")
            return None
            
        # Load the NPZ file
        data = np.load(npz_file_path, allow_pickle=True)
        
        # Convert to dictionary for easier access
        npz_dict = {}
        for key in data.files:
            npz_dict[key] = data[key]
            
        if tracker:
            tracker.log_info(f"Loaded NPZ with keys: {list(npz_dict.keys())}")
        return npz_dict
        
    except Exception as e:
        if tracker:
            tracker.log_error(e, f"Loading NPZ file {npz_file_path}")
        return None

def perform_gmm_segmentation(npz_data, gmm_params, tracker=None):
    """
    Perform GMM segmentation on phasor plot data with circle filtering and elliptical segmentation.
    
    Args:
        npz_data: Dictionary containing G, S, and intensity data
        gmm_params: Dictionary containing GMM parameters
        tracker: Error tracker instance
        
    Returns:
        Tuple of (mask_dict, plot_figure)
    """
    try:
        # Extract data - handle both wavelet filter output and other formats
        G = npz_data.get('G', None)
        S = npz_data.get('S', None)
        intensity = npz_data.get('A', npz_data.get('intensity', None))  # 'A' is intensity from wavelet filter
        
        # Also check for unfiltered versions
        GU = npz_data.get('GU', None)
        SU = npz_data.get('SU', None)
        
        if G is None or S is None or intensity is None:
            error_msg = "Missing required G, S, or intensity data in NPZ file"
            if tracker:
                tracker.log_error(Exception(error_msg), "Data validation")
            return None, None
        
        # Determine which data source to use
        use_unfiltered = gmm_params.get('data_selection_use_unfiltered_data', gmm_params.get('use_unfiltered_data', False))
        
        if use_unfiltered and GU is not None and SU is not None:
            g_data = GU
            s_data = SU
            if tracker:
                tracker.log_info("Using unfiltered data (GU, SU)")
        else:
            g_data = G
            s_data = S
            if tracker:
                tracker.log_info("Using filtered data (G, S)")
        
        if g_data is None or s_data is None:
            error_msg = "No valid G or S data found"
            if tracker:
                tracker.log_error(Exception(error_msg), "Data validation")
            return None, None
        
        # Set up parameters for GMM
        n_components = gmm_params.get('gmm_model_n_components', gmm_params.get('n_components', 2))
        max_iter = gmm_params.get('gmm_model_max_iter', gmm_params.get('max_iter', 100))
        covariance_type = gmm_params.get('gmm_model_covariance_type', gmm_params.get('covariance_type', 'full'))
        random_state = gmm_params.get('gmm_model_random_state', gmm_params.get('random_state', 0))
        
        # Get reference center if available
        ref_g = gmm_params.get('reference_center_reference_center_G', gmm_params.get('reference_center_G', None))
        ref_s = gmm_params.get('reference_center_reference_center_S', gmm_params.get('reference_center_S', None))
        
        # Advanced segmentation parameters
        radius_ref = gmm_params.get('circle_filtering_radius_ref', gmm_params.get('radius_ref', 0.05))
        cov_f = gmm_params.get('elliptical_segmentation_cov_f', gmm_params.get('cov_f', 6.0))
        shift = gmm_params.get('elliptical_segmentation_shift', gmm_params.get('shift', 0.0))
        use_circle_filter = gmm_params.get('circle_filtering_use_circle_filter', gmm_params.get('use_circle_filter', False))
        
        # Enhanced thresholding options
        threshold_type = gmm_params.get('thresholding_threshold_type', gmm_params.get('threshold_type', 'relative'))
        threshold_value = gmm_params.get('thresholding_intensity_threshold', gmm_params.get('intensity_threshold', 0.0))
        
        # Apply thresholding based on type
        if threshold_type == 'relative':
            if threshold_value > 0:
                mask = intensity > np.max(intensity) * threshold_value
            else:
                mask = intensity > 0
        elif threshold_type == 'absolute':
            mask = intensity > threshold_value
        elif threshold_type == 'percentile':
            percentile = max(0, min(100, threshold_value))
            threshold_val = np.percentile(intensity[intensity > 0], percentile)
            mask = intensity > threshold_val
        else:
            if threshold_value > 0:
                mask = intensity > np.max(intensity) * threshold_value
            else:
                mask = intensity > 0
        
        if tracker:
            tracker.log_info(f"Thresholding: {threshold_type}, value: {threshold_value}")
            tracker.log_info(f"Pixels above threshold: {np.sum(mask)} / {mask.size}")
        
        # Apply thresholding to G and S data
        g_thresholded = g_data * mask
        s_thresholded = s_data * mask
        
        # Flatten the data for processing
        g_flat = g_thresholded.ravel()
        s_flat = s_thresholded.ravel()
        intensity_flat = intensity.ravel()
        
        # Create points list for filtering
        points = list(zip(g_flat, s_flat))
        
        # Apply circle filtering if enabled and reference center is available
        if use_circle_filter and ref_g is not None and ref_s is not None:
            center_cond = (ref_g, ref_s)
            results_cond = are_points_inside_circle(points, center_cond, radius_ref)
            
            # Reshape results to match original dimensions
            height, width = g_data.shape
            matrix_for_cond_gmm = np.reshape(results_cond, (height, width))
            
            # Apply circle filter to G and S data
            g_for_gmm = matrix_for_cond_gmm * g_thresholded
            s_for_gmm = matrix_for_cond_gmm * s_thresholded
            
            # Flatten filtered data
            g_gmm = g_for_gmm.ravel()
            s_gmm = s_for_gmm.ravel()
            
            # Remove zero values
            non_zero_mask = (g_gmm != 0) | (s_gmm != 0)
            g_gmm = g_gmm[non_zero_mask]
            s_gmm = s_gmm[non_zero_mask]
            
            if tracker:
                tracker.log_info(f"Circle filtering applied with radius {radius_ref}")
                tracker.log_info(f"Points after circle filtering: {len(g_gmm)}")
        else:
            # Use all thresholded data
            g_gmm = g_flat
            s_gmm = s_flat
        
        # Prepare data for GMM
        X = np.column_stack((g_gmm, s_gmm))
        
        # Check if we have enough points for GMM
        if len(X) < n_components * 2:
            warning_msg = f"Not enough data points ({len(X)}) for {n_components} GMM components"
            if tracker:
                tracker.log_warning(warning_msg, "Data validation")
            if len(X) <= 10:
                return None, None
            n_components = min(n_components, max(2, len(X) // 2))
            if tracker:
                tracker.log_info(f"Reducing to {n_components} components")
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state
        )
        gmm.fit(X)
        
        # Get cluster centers and covariances
        cluster_centers = gmm.means_
        cov_matrices = gmm.covariances_
        
        if tracker:
            tracker.log_info(f"GMM fitted with {n_components} components")
            for i, center in enumerate(cluster_centers):
                tracker.log_info(f"Component {i+1} center: {center}")
        
        # Create segmentation masks for all components
        masks = {}
        
        # Create points for the full dataset
        points_full = list(zip(g_flat, s_flat))
        height, width = g_data.shape
        
        # Process each GMM component
        for i in range(n_components):
            cluster_center = cluster_centers[i]
            cov_matrix = cov_matrices[i]
            
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Calculate ellipse parameters
            if eigenvectors[0, 1] > 0 and eigenvectors[1, 1] > 0:
                angle_cond = np.arctan2(-eigenvectors[1, 1], -eigenvectors[0, 1])
            else:
                angle_cond = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
            angle_degrees_cond = np.degrees(angle_cond)
            
            # Calculate ellipse width and height with covariance multiplier
            width_cond = cov_f * np.sqrt(eigenvalues[1])
            height_cond = cov_f * np.sqrt(eigenvalues[0])
            
            # Calculate shift
            dx_cond = shift * width_cond * np.cos(angle_cond)
            dy_cond = shift * width_cond * np.sin(angle_cond)
            
            # Calculate new center coordinates
            center_cond_x = cluster_center[0] + dx_cond
            center_cond_y = cluster_center[1] + dy_cond
            roi_center_cond = np.array([center_cond_x, center_cond_y])
            
            if tracker:
                tracker.log_info(f"Component {i+1} ellipse parameters: width={width_cond:.4f}, height={height_cond:.4f}, angle={angle_degrees_cond:.2f}°")
                tracker.log_info(f"Component {i+1} ROI center: {roi_center_cond}")
            
            # Elliptical segmentation for this component
            results_cluster_cond = is_points_inside_rotated_ellipse(
                roi_center_cond[0], roi_center_cond[1], width_cond, height_cond, angle_degrees_cond, points_full
            )
            matrix_cluster_cond = np.reshape(results_cluster_cond, (height, width))
            
            # Create binary mask for this component
            component_mask = matrix_cluster_cond.astype(np.uint8) * 255
            masks[f"ellipse_segmentation_component_{i+1}"] = component_mask
        
        # Create combined mask (union of all components)
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for i in range(n_components):
            combined_mask = np.maximum(combined_mask, masks[f"ellipse_segmentation_component_{i+1}"])
        
        masks["ellipse_segmentation_combined"] = combined_mask
        masks["gmm_segmentation_mask"] = combined_mask  # Renamed for clarity
        
        if tracker:
            tracker.log_info(f"Created elliptical segmentation masks for {n_components} components")
        
        # Create phasor plot using centralized utilities
        title = f"Phasor Plot with GMM Ellipse Segmentation ({data_source} Data)"
        fig, ax = create_phasor_plot(g_flat, s_flat, intensity_flat, title, figsize=(8, 6), show_colorbar=False)
        
        # Plot GMM components
        for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
            v, w = np.linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan2(u[1], u[0])
            angle = np.degrees(angle)
            ellipse = plt.matplotlib.patches.Ellipse(xy=mean, width=v[0], height=v[1], angle=180. + angle, 
                                                   color=f'C{i}', alpha=0.7, linewidth=0.5,
                                                   label=f'Component {i+1}', zorder=3)
            ax.add_artist(ellipse)
            
            # Mark the center of the component
            ax.scatter(mean[0], mean[1], color=f'C{i}', s=80, marker='x', linewidth=0.5, 
                      edgecolor='white', zorder=4, label=f'Component {i+1} Center')
        
        # Plot the segmentation ellipses for all components
        for i in range(n_components):
            cluster_center = cluster_centers[i]
            cov_matrix = cov_matrices[i]
            
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Calculate ellipse parameters
            if eigenvectors[0, 1] > 0 and eigenvectors[1, 1] > 0:
                angle_cond = np.arctan2(-eigenvectors[1, 1], -eigenvectors[0, 1])
            else:
                angle_cond = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
            angle_degrees_cond = np.degrees(angle_cond)
            
            # Calculate ellipse width and height with covariance multiplier
            width_cond = cov_f * np.sqrt(eigenvalues[1])
            height_cond = cov_f * np.sqrt(eigenvalues[0])
            
            # Calculate shift
            dx_cond = shift * width_cond * np.cos(angle_cond)
            dy_cond = shift * width_cond * np.sin(angle_cond)
            
            # Calculate new center coordinates
            center_cond_x = cluster_center[0] + dx_cond
            center_cond_y = cluster_center[1] + dy_cond
            roi_center_cond = np.array([center_cond_x, center_cond_y])
            
            # Plot the segmentation ellipse for this component
            seg_ellipse = plt.matplotlib.patches.Ellipse(
                xy=roi_center_cond, 
                width=2 * width_cond, 
                height=2 * height_cond, 
                angle=angle_degrees_cond, 
                color=f'C{i+2}',  # Different color for each component
                fill=False, 
                linewidth=0.5,
                label=f'Segmentation Ellipse {i+1}',
                zorder=5
            )
            ax.add_patch(seg_ellipse)
        
        # Add reference center if available
        if ref_g is not None and ref_s is not None:
            ax.scatter(ref_g, ref_s, color='red', s=150, marker='*', linewidth=0.5, 
                      edgecolor='white', zorder=4, label='Reference Center')
            
            # Add reference circle if circle filtering is enabled
            if use_circle_filter:
                ref_circle = plt.matplotlib.patches.Circle(
                    xy=(ref_g, ref_s), 
                    radius=radius_ref, 
                    color='red', 
                    fill=False, 
                    linewidth=0.5,
                    linestyle='--',
                    alpha=0.7,
                    label='Reference Circle',
                    zorder=3
                )
                ax.add_patch(ref_circle)
        
        # Add the colorbar with custom formatting
        near_zero = 0.1
        h = ax.get_children()[0]  # Get the histogram object
        cbar = fig.colorbar(h, ax=ax, format=LogFormatter(10, labelOnlyBase=True))
        
        # Calculate appropriate ticks for the colorbar
        if hasattr(h, 'get_array') and h.get_array() is not None:
            vmax = h.get_array().max()
            if vmax > 1:
                ticks = [near_zero] + [10**i for i in range(1, int(np.log10(vmax)) + 1)]
                tick_labels = ['0'] + [f'$10^{i}$' for i in range(1, int(np.log10(vmax)) + 1)]
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(tick_labels)
        
        cbar.set_label('Frequency')
        
        # Set title with timestamp and data source
        data_source = "Unfiltered" if use_unfiltered else "Filtered"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.set_title(f"Phasor Plot with GMM Ellipse Segmentation ({data_source} Data)\n({timestamp})")
        
        # Add legend
        ax.legend(loc='upper right', framealpha=0.9)
        
        fig.tight_layout()
        
        return masks, fig
        
    except Exception as e:
        if tracker:
            tracker.log_error(e, "GMM segmentation")
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
    try:
        # Run as standalone script
        success = main()
        if success:
            print("GMM segmentation completed successfully!")
            sys.exit(0)
        else:
            print("GMM segmentation failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nGMM segmentation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running GMM segmentation: {e}")
        sys.exit(1) 