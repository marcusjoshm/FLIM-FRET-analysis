#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLIM-FRET Analysis Pipeline Setup Script

This script validates the installation and generates a config.json file.
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section"""
    print(f"\n--- {title} ---")

def check_python_version():
    """Check if Python version is compatible"""
    print_section("Python Version Check")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def find_imagej():
    """Find ImageJ/Fiji installation"""
    print_section("ImageJ/Fiji Detection")
    
    # Common ImageJ paths
    common_paths = [
        "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",  # macOS
        "/Applications/Fiji.app/Contents/MacOS/ImageJ",         # macOS alternative
        "/usr/local/bin/imagej",                                # Linux
        "/opt/Fiji.app/ImageJ-linux64",                         # Linux alternative
        "C:\\Program Files\\Fiji.app\\ImageJ-win64.exe",        # Windows
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            print(f"✅ Found ImageJ at: {path}")
            return path
    
    print("❌ ImageJ/Fiji not found in common locations")
    print("Please install Fiji from: https://fiji.sc/")
    print("Or provide the path manually when prompted.")
    return None

def check_required_packages():
    """Check if required Python packages are installed"""
    print_section("Required Python Packages")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print(f"✅ Running in virtual environment: {sys.prefix}")
    else:
        print("⚠️  Warning: Not running in a virtual environment")
    
    required_packages = [
        "numpy",
        "scipy", 
        "matplotlib",
        "pandas",
        "scikit-image",
        "tifffile",
        "opencv-python"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "opencv-python":
                __import__("cv2")
            elif package == "scikit-image":
                __import__("skimage")
            else:
                __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please activate your virtual environment and install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_macro_files():
    """Check if ImageJ macro files exist"""
    print_section("ImageJ Macro Files")
    
    macro_files = [
        "src/scripts/imagej/FLIM_processing_macro_1.ijm",
        "src/scripts/imagej/FLIM_processing_macro_2.ijm"
    ]
    
    missing_macros = []
    
    for macro in macro_files:
        if os.path.exists(macro):
            print(f"✅ {macro}")
        else:
            print(f"❌ {macro} - MISSING")
            missing_macros.append(macro)
    
    if missing_macros:
        print(f"\n❌ Missing macro files: {', '.join(missing_macros)}")
        return False
    
    print("✅ All macro files are present")
    return True



def generate_config():
    """Generate config.json file"""
    print_header("Configuration Generation")
    
    config = {}
    
    # ImageJ path
    imagej_path = find_imagej()
    if not imagej_path:
        print("❌ Error: ImageJ/Fiji not found. Please install Fiji from https://fiji.sc/")
        return None
    config["imagej_path"] = imagej_path
    print(f"✅ Using ImageJ: {imagej_path}")
    
    # Macro files (these should be relative to the project)
    config["macro_files"] = [
        "src/scripts/imagej/FLIM_processing_macro_1.ijm",
        "src/scripts/imagej/FLIM_processing_macro_2.ijm"
    ]
    print("✅ Using macro files from project directory")
    
    # Microscope parameters (use defaults)
    print_section("Microscope Parameters")
    config["microscope_params"] = {
        "bin_width_ns": 0.097,
        "freq_mhz": 78.0,
        "harmonic": 1
    }
    print("✅ Using default microscope parameters")
    
    # Wavelet filter parameters (use defaults)
    print_section("Wavelet Filter Parameters")
    config["wavelet_filter_params"] = {
        "Gc": 0.30227996721890404,
        "Sc": 0.4592458920992018,
        "flevel": 9
    }
    print("✅ Using default wavelet filter parameters")
    
    # GMM segmentation parameters (use defaults)
    print_section("GMM Segmentation Parameters")
    config["gmm_segmentation_params"] = {
        "intensity_threshold": 0.0,
        "threshold_type": "relative",
        "n_components": 2,
        "covariance_type": "full",
        "max_iter": 100,
        "random_state": 0,
        "reference_center_G": 0.30227996721890404,
        "reference_center_S": 0.4592458920992018,
        "use_unfiltered_data": False,
        "combine_datasets": False
    }
    print("✅ Using default GMM segmentation parameters")
    
    return config

def save_config(config, config_path="config/config.json"):
    """Save config to file"""
    print_section("Saving Configuration")
    
    # Create config directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✅ Configuration saved to: {config_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

def test_imagej_connection(imagej_path):
    """Test ImageJ connection"""
    print_section("ImageJ Connection Test")
    
    try:
        # Try to run ImageJ with version flag
        result = subprocess.run([imagej_path, "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ ImageJ connection successful")
            return True
        else:
            print("❌ ImageJ connection failed")
            return False
    except subprocess.TimeoutExpired:
        print("❌ ImageJ connection timed out")
        return False
    except Exception as e:
        print(f"❌ ImageJ connection error: {e}")
        return False

def main():
    """Main setup function"""
    print_header("FLIM-FRET Analysis Pipeline Setup")
    
    # Check if we're in the right directory
    if not os.path.exists("run_pipeline.py"):
        print("❌ Error: Please run this script from the FLIM-FRET-analysis directory")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_required_packages():
        print("\nPlease install missing packages and run setup again.")
        sys.exit(1)
    
    if not check_macro_files():
        print("\nPlease ensure all macro files are present and run setup again.")
        sys.exit(1)
    
    # Generate configuration
    config = generate_config()
    if not config:
        print("❌ Failed to generate configuration.")
        sys.exit(1)
    
    # Test ImageJ connection (non-blocking)
    if not test_imagej_connection(config["imagej_path"]):
        print("⚠️  Warning: ImageJ connection test failed. You may need to check the ImageJ path.")
        print("   Continuing with setup...")
    
    # Save configuration
    if save_config(config):
        print_header("Setup Complete!")
        print("✅ FLIM-FRET Analysis Pipeline is ready to use!")
        print(f"📁 Configuration saved to: config/config.json")
        print("\nNext steps:")
        print("1. Test the pipeline with: python run_pipeline.py --help")
        print("2. Run preprocessing with: python run_pipeline.py --preprocess --input <data_dir> --output <output_dir>")
    else:
        print("❌ Setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 