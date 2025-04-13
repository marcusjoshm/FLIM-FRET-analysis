#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify FLUTE integration works correctly with the virtual environment.
"""

import os
import sys
import json
import subprocess

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def test_flute_venv():
    """Test if FLUTE virtual environment is configured correctly"""
    
    # Load config to get FLUTE paths
    config = load_config()
    flute_path = config.get("flute_path")
    flute_python_path = config.get("flute_python_path")
    
    if not flute_path or not os.path.exists(flute_path):
        print(f"Error: flute_path not found or invalid: {flute_path}")
        return False
    
    if not flute_python_path or not os.path.exists(flute_python_path):
        print(f"Error: flute_python_path not found or invalid: {flute_python_path}")
        return False
    
    print(f"FLUTE path: {flute_path}")
    print(f"FLUTE Python path: {flute_python_path}")
    
    # Get FLUTE directory
    flute_dir = os.path.dirname(flute_path)
    
    # Create a test script to verify ImageHandler import
    test_script = "test_flute_import_temp.py"
    
    with open(test_script, "w") as f:
        f.write(f"""
import os
import sys

# Add FLUTE directory to Python path
flute_dir = "{flute_dir}"
if flute_dir not in sys.path:
    sys.path.append(flute_dir)

try:
    from ImageHandler import ImageHandler
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("Successfully imported ImageHandler from:", flute_dir)

    # List available attributes
    handler_attrs = [attr for attr in dir(ImageHandler) if not attr.startswith('__')]
    print("\\nImageHandler attributes:")
    for attr in sorted(handler_attrs)[:10]:  # Show first 10 attributes
        print(f"  - {{attr}}")
        
    print("\\nFLUTE integration test successful!")
except ImportError as e:
    print(f"Error importing ImageHandler: {{e}}")
    print("Python path:")
    for p in sys.path:
        print(f"  {{p}}")
    print("\\nFLUTE integration test failed.")
    sys.exit(1)
""")
    
    print("\nRunning test script with FLUTE virtual environment...")
    try:
        # Run the test script with the FLUTE virtual environment's Python
        result = subprocess.run(
            [flute_python_path, test_script], 
            check=False, 
            capture_output=True, 
            text=True
        )
        print("\nOutput from FLUTE environment:")
        print(result.stdout)
        
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
        
        # Clean up
        if os.path.exists(test_script):
            os.remove(test_script)
            
        return "FLUTE integration test successful" in result.stdout
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        
        # Clean up
        if os.path.exists(test_script):
            os.remove(test_script)
            
        return False

if __name__ == "__main__":
    print("Testing FLUTE integration with virtual environment...")
    if test_flute_venv():
        print("\nSuccess! FLUTE integration is working correctly.")
    else:
        print("\nFailed! There are issues with the FLUTE integration setup.") 