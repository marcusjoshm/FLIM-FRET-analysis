#!/usr/bin/env python3
# FLUTE command logger

import os
import sys
import inspect
import time
import json
from functools import wraps

# Path to the original FLUTE installation
FLUTE_PATH = "/Users/joshuamarcus/FLUTE"

# Add FLUTE to the sys.path
if FLUTE_PATH not in sys.path:
    sys.path.append(FLUTE_PATH)

# Log file path
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "flute_commands.log")
JSON_LOG_FILE = os.path.join(LOG_DIR, "flute_commands.json")

# Dictionary to store logged calls
logged_calls = []

# Load existing log if available
if os.path.exists(JSON_LOG_FILE):
    try:
        with open(JSON_LOG_FILE, 'r') as f:
            logged_calls = json.load(f)
    except (json.JSONDecodeError, IOError):
        # If file is empty or invalid, start fresh
        logged_calls = []

def log_method_call(func):
    """Decorator to log method calls with arguments and return value."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get class name if it's a method
        if len(args) > 0 and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
        else:
            class_name = "None"

        # Log method call
        call_info = {
            "timestamp": time.time(),
            "class": class_name,
            "method": func.__name__,
            "args": [str(arg) for arg in args[1:]] if class_name != "None" else [str(arg) for arg in args],
            "kwargs": {k: str(v) for k, v in kwargs.items()}
        }
        
        print(f"CALL: {class_name}.{func.__name__}()")
        
        # Execute the original function
        result = func(*args, **kwargs)
        
        # Add result to log (as string to ensure serializability)
        call_info["result"] = str(result)
        logged_calls.append(call_info)
        
        # Save to JSON after each call
        with open(JSON_LOG_FILE, 'w') as f:
            json.dump(logged_calls, f, indent=2)
            
        # Also log to text file
        with open(LOG_FILE, 'a') as f:
            f.write(f"{call_info['timestamp']} - {call_info['class']}.{call_info['method']}() - "
                   f"args: {call_info['args']}, kwargs: {call_info['kwargs']}, result: {call_info['result']}\n")
        
        return result
    return wrapper

# Import original FLUTE modules
try:
    import ImageHandler
    import main as flute_main
    import Calibration
    
    # Store original classes
    OriginalImageHandler = ImageHandler.ImageHandler
    original_perform_fft = OriginalImageHandler.perform_fft
    original_convolution = OriginalImageHandler.convolution
    original_update_threshold = OriginalImageHandler.update_threshold
    original_update_circle_range = OriginalImageHandler.update_circle_range
    original_update_angle_range = OriginalImageHandler.update_angle_range
    original_fraction_lifetime_map = OriginalImageHandler.fraction_lifetime_map
    original_save_data = OriginalImageHandler.save_data
    
    # Patch methods with logging
    OriginalImageHandler.perform_fft = log_method_call(original_perform_fft)
    OriginalImageHandler.convolution = log_method_call(original_convolution)
    OriginalImageHandler.update_threshold = log_method_call(original_update_threshold)
    OriginalImageHandler.update_circle_range = log_method_call(original_update_circle_range)
    OriginalImageHandler.update_angle_range = log_method_call(original_update_angle_range)
    OriginalImageHandler.fraction_lifetime_map = log_method_call(original_fraction_lifetime_map)
    OriginalImageHandler.save_data = log_method_call(original_save_data)
    
    # Patch the calibration function
    original_get_calibration_parameters = Calibration.get_calibration_parameters
    Calibration.get_calibration_parameters = log_method_call(original_get_calibration_parameters)
    
    print(f"=== FLUTE Command Logger initialized ===")
    print(f"Logging to: {LOG_FILE}")
    print(f"JSON log: {JSON_LOG_FILE}")
    
except ImportError as e:
    print(f"Error importing FLUTE modules: {e}")
    print(f"Make sure FLUTE_PATH is set correctly: {FLUTE_PATH}")
    sys.exit(1)

def run_flute():
    """
    Runs the original FLUTE application with logging enabled.
    This allows you to use FLUTE normally and capture all the function calls.
    """
    # Initialize the log files only when running FLUTE, not when generating a script
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n=== FLUTE Command Logger Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    print("Starting FLUTE with command logging...")
    
    # Reset command arguments to match what FLUTE expects
    sys.argv = [os.path.join(FLUTE_PATH, "main.py")]
    
    # Start FLUTE's main window
    if hasattr(flute_main, 'MainWindow'):
        flute_main.app = flute_main.QtWidgets.QApplication(sys.argv)
        window = flute_main.MainWindow()
        window.show()
        flute_main.app.exec_()
    else:
        print("Error: Could not find MainWindow class in FLUTE")

def generate_code_from_log():
    """
    Generates Python code from the logged commands that can be used
    to recreate the operations performed in the FLUTE GUI.
    """
    if not os.path.exists(JSON_LOG_FILE):
        print(f"Error: Log file not found: {JSON_LOG_FILE}")
        return
        
    with open(JSON_LOG_FILE, 'r') as f:
        calls = json.load(f)
    
    if not calls:
        print("No logged calls found.")
        return
        
    print(f"Found {len(calls)} logged calls. Generating processing script...")
        
    # Generate code for a standalone Python script
    code = [
        "# Generated FLUTE processing script",
        "# This script recreates the operations performed in the FLUTE GUI",
        "",
        "import os",
        "import sys",
        "import numpy as np",
        "from skimage import io",
        "import tifffile",
        "",
        "# Add the FLUTE directory to Python path",
        f"FLUTE_PATH = \"{FLUTE_PATH}\"",
        "if FLUTE_PATH not in sys.path:",
        "    sys.path.append(FLUTE_PATH)",
        "",
        "# Import FLUTE modules",
        "import ImageHandler",
        "import Calibration",
        ""
    ]
    
    # Parse key operations
    file_operations = []
    calibration_ops = []
    processing_ops = []
    
    for call in calls:
        if call["method"] == "perform_fft":
            continue  # Skip internal FFT calls as they're part of initialization
        
        if call["method"] == "get_calibration_parameters":
            # Extract calibration parameters
            calibration_ops.append(f"# Calibration parameters")
            calibration_ops.append(f"calibration_file = {call['args'][0]}")
            calibration_ops.append(f"bin_width = {call['args'][1]}")
            calibration_ops.append(f"freq = {call['args'][2]}")
            calibration_ops.append(f"harmonic = {call['args'][3]}")
            calibration_ops.append(f"tau_ref = {call['args'][4]}")
            calibration_ops.append(f"phi_cal, m_cal = Calibration.get_calibration_parameters(calibration_file, bin_width, freq, harmonic, tau_ref)")
            calibration_ops.append(f"print(f\"Calibration: phi_cal={phi_cal}, m_cal={m_cal}\")")
            calibration_ops.append("")
        
        elif call["method"] == "save_data":
            # Track save operations
            file_operations.append(f"# Save data for image")
            file_operations.append(f"image_handler.save_data(\"{call['args'][0]}\", \"{call['args'][1]}\")")
            file_operations.append("")
            
        elif call["method"] in ["update_threshold", "convolution", "update_circle_range", 
                              "update_angle_range", "fraction_lifetime_map"]:
            # Add processing steps
            if call["method"] == "update_threshold":
                processing_ops.append(f"# Set intensity threshold")
                processing_ops.append(f"image_handler.update_threshold({call['args'][0]}, {call['args'][1]})")
            elif call["method"] == "convolution":
                processing_ops.append(f"# Apply median filter")
                processing_ops.append(f"image_handler.convolution({call['args'][0]})")
            elif call["method"] == "update_circle_range":
                processing_ops.append(f"# Set TauM range")
                processing_ops.append(f"image_handler.update_circle_range({call['args'][0]}, {call['args'][1]})")
            elif call["method"] == "update_angle_range":
                processing_ops.append(f"# Set TauP angle range")
                processing_ops.append(f"image_handler.update_angle_range({call['args'][0]}, {call['args'][1]})")
            elif call["method"] == "fraction_lifetime_map":
                processing_ops.append(f"# Set fraction bound lifetime")
                processing_ops.append(f"image_handler.fraction_lifetime_map({call['args'][0]})")
            processing_ops.append("")
    
    # Build the main script
    if calibration_ops:
        code.extend(calibration_ops)
    
    code.extend([
        "def process_tiff_file(input_file, output_dir, phi_cal, m_cal, bin_width=0.2208, freq=80, harmonic=1):",
        "    \"\"\"",
        "    Process a TIFF file with the parameters specified through the FLUTE GUI.",
        "    \"\"\"",
        "    print(f\"Processing {input_file}...\")",
        "    os.makedirs(output_dir, exist_ok=True)",
        "",
        "    # Initialize ImageHandler",
        "    image_handler = ImageHandler.ImageHandler(input_file, phi_cal, m_cal, bin_width, freq, harmonic)",
        ""
    ])
    
    # Add processing operations
    if processing_ops:
        for op in processing_ops:
            code.append(f"    {op}")
    
    # Add save operations
    if file_operations:
        for op in file_operations:
            code.append(f"    {op}")
    else:
        code.extend([
            "    # Save processed data",
            "    output_path = os.path.join(output_dir, os.path.basename(input_file).split('.')[0])",
            "    image_handler.save_data(output_dir, 'all')",
            ""
        ])
    
    code.extend([
        "    print(f\"Processing of {input_file} completed successfully.\")",
        "    return True",
        "",
        "if __name__ == \"__main__\":",
        "    # Example usage",
        "    if len(sys.argv) != 3:",
        "        print(\"Usage: python script.py <input_tiff_file> <output_directory>\")",
        "        sys.exit(1)",
        "",
        "    input_file = sys.argv[1]",
        "    output_dir = sys.argv[2]",
        "",
        "    process_tiff_file(input_file, output_dir, phi_cal, m_cal, bin_width, freq, harmonic)",
        ""
    ])
    
    # Write the generated script to file
    output_file = os.path.join(LOG_DIR, "flute_processing_script.py")
    with open(output_file, 'w') as f:
        f.write("\n".join(code))
    
    print(f"Generated processing script: {output_file}")
    print(f"You can now use this script to process FLIM files without the GUI.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-script":
        generate_code_from_log()
    else:
        run_flute() 