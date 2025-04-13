#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify FLUTE imports work correctly.
"""

import os
import sys
import json

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def test_flute_import():
    """Test if we can import from FLUTE correctly"""
    # Load config to get FLUTE path
    config = load_config()
    flute_path = config["flute_path"]
    
    # Print information
    print(f"Testing FLUTE import from: {flute_path}")
    
    # Get FLUTE directory
    flute_dir = os.path.dirname(flute_path)
    
    # Check if directory exists
    if not os.path.exists(flute_dir):
        print(f"Error: FLUTE directory not found at {flute_dir}")
        return False
    
    # Add to path
    print(f"Adding to sys.path: {flute_dir}")
    if flute_dir not in sys.path:
        sys.path.append(flute_dir)
    
    # Try different import approaches
    print("\nAttempting imports:")
    
    # Approach 1: Try direct import
    try:
        print("- Trying: from ImageHandler import ImageHandler")
        from ImageHandler import ImageHandler
        print("  Success! ImageHandler imported directly")
        return True
    except ImportError as e:
        print(f"  Failed: {e}")
    
    # Approach 2: Try from FLUTE module  
    try:
        print("- Trying: from FLUTE.ImageHandler import ImageHandler")
        from FLUTE.ImageHandler import ImageHandler
        print("  Success! ImageHandler imported from FLUTE module")
        return True
    except ImportError as e:
        print(f"  Failed: {e}")
    
    # Approach 3: List directory contents to find the correct module
    print("\nListing files in FLUTE directory to find ImageHandler:")
    try:
        files = os.listdir(flute_dir)
        py_files = [f for f in files if f.endswith('.py')]
        print(f"Python files in {flute_dir}:")
        for py_file in py_files:
            print(f"  - {py_file}")
    except Exception as e:
        print(f"Error listing files: {e}")
    
    print("\nFLUTE import test failed. Please check the FLUTE installation and path.")
    return False

if __name__ == "__main__":
    test_flute_import() 