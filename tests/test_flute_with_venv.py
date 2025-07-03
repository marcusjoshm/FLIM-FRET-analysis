#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify FLUTE modules can be accessed from the project venv.
"""

import os
import sys
import json

def main():
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    flute_path = config.get("flute_path")
    flute_python_path = config.get("flute_python_path")
    
    print(f"Testing FLUTE integration with project venv")
    print(f"Python executable: {sys.executable}")
    print(f"FLUTE path: {flute_path}")
    print(f"FLUTE Python path in config: {flute_python_path}")
    
    # Get FLUTE directory
    flute_dir = os.path.dirname(flute_path)
    
    # Add FLUTE directory to Python path
    if flute_dir not in sys.path:
        sys.path.append(flute_dir)
        print(f"Added FLUTE directory to Python path: {flute_dir}")
    
    # Try to import ImageHandler
    try:
        from ImageHandler import ImageHandler
        print("\n✓ Successfully imported ImageHandler from FLUTE")
        
        # List available methods
        print("\nImageHandler has the following methods (first 10):")
        handler_attrs = [attr for attr in dir(ImageHandler) if not attr.startswith('__')]
        for attr in sorted(handler_attrs)[:10]:
            print(f"  - {attr}")
            
        return True
    except ImportError as e:
        print(f"\n❌ Failed to import ImageHandler: {e}")
        print("\nPython path:")
        for p in sys.path:
            print(f"  {p}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed. Please check the output for details.") 