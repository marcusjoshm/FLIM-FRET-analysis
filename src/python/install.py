#!/usr/bin/env python3
"""
FLIM-FRET Installation Script

This script provides a complete installation and setup for the FLIM-FRET package.
It handles:
1. Virtual environment creation
2. Package installation
3. Configuration setup
4. Software path detection
5. Package installation in development mode

Usage:
    python install.py
"""

import os
import sys
import json
import subprocess
import venv
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

# Global variable for Python command
python_cmd = 'python3'

def print_status(message: str, color: str = Colors.GREEN):
    """Print a status message with color."""
    print(f"{color}[INFO]{Colors.NC} {message}")

def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")

def check_python_version():
    """Check if Python 3.8+ is available."""
    # Try to find Python 3.8+ specifically
    python_commands = ['python3.11', 'python3.10', 'python3.9', 'python3.8', 'python3', 'python']
    
    for cmd in python_commands:
        try:
            result = subprocess.run([cmd, "--version"], 
                                  capture_output=True, text=True, check=True)
            version = result.stdout.strip()
            
            # Check if version is 3.8 or higher
            if "3.8" in version or "3.9" in version or "3.10" in version or "3.11" in version or "3.12" in version or "3.13" in version:
                print_status(f"Found Python: {version}")
                # Update sys.executable to use found Python
                global python_cmd
                python_cmd = cmd
                return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    print_error("Python 3.8+ is not found. Please install Python 3.8+ first.")
    print_warning("You can install it using:")
    print_warning("  macOS: brew install python@3.11")
    print_warning("  Ubuntu: sudo apt install python3.11")
    print_warning("  Or download from: https://www.python.org/downloads/")
    return False

def create_virtual_environment(venv_name: str = "venv") -> bool:
    """Create a virtual environment."""
    venv_path = Path(venv_name)
    
    if venv_path.exists():
        print_status(f"Virtual environment '{venv_name}' already exists")
        return True
    
    try:
        print_status(f"Creating virtual environment '{venv_name}' using {python_cmd}...")
        subprocess.run([python_cmd, "-m", "venv", venv_name], check=True)
        print_status(f"Virtual environment '{venv_name}' created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False

def get_venv_python(venv_name: str = "venv") -> Path:
    """Get the Python executable path for a virtual environment."""
    venv_path = Path(venv_name)
    
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"

def install_requirements(venv_name: str, requirements_file: str) -> bool:
    """Install requirements in a virtual environment."""
    venv_path = Path(venv_name)
    python_path = get_venv_python(venv_name)
    
    if not venv_path.exists():
        print_error(f"Virtual environment '{venv_name}' does not exist")
        return False
    
    try:
        print_status(f"Installing requirements from {requirements_file}...")
        
        # Upgrade pip
        subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Install requirements with verbose output for debugging
        result = subprocess.run([str(python_path), "-m", "pip", "install", "-r", requirements_file], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print_error(f"Failed to install requirements from {requirements_file}")
            print_error(f"Error output: {result.stderr}")
            return False
        
        print_status(f"Requirements installed successfully in '{venv_name}'")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install requirements: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print_error(f"Error details: {e.stderr}")
        return False

def install_additional_dependencies(venv_name: str = "venv") -> bool:
    """Install additional dependencies for FLIM-FRET."""
    python_path = get_venv_python(venv_name)
    
    try:
        print_status("Installing additional dependencies...")
        
        additional_deps = [
            "tifffile", "scikit-image", "opencv-python", "scikit-learn"
        ]
        
        result = subprocess.run([str(python_path), "-m", "pip", "install"] + additional_deps, 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print_error(f"Failed to install additional dependencies")
            print_error(f"Error output: {result.stderr}")
            return False
        
        print_status("Additional dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install additional dependencies: {e}")
        return False

def install_package_development_mode(venv_name: str = "venv") -> bool:
    """Install the package in development mode."""
    python_path = get_venv_python(venv_name)
    
    try:
        print_status("Installing package in development mode...")
        subprocess.run([str(python_path), "-m", "pip", "install", "-e", "."], 
                      check=True, capture_output=True)
        print_status("Package installed in development mode successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install package: {e}")
        return False

def detect_software_paths() -> Dict[str, str]:
    """Detect software paths for configuration."""
    paths = {}
    
    # Common ImageJ/Fiji locations
    imagej_locations = [
        "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",
        "/Applications/ImageJ.app/Contents/MacOS/ImageJ-macosx",
        "/usr/local/Fiji.app/Contents/MacOS/ImageJ-macosx",
        "C:\\Program Files\\Fiji.app\\ImageJ-win64.exe",
        "C:\\Program Files\\ImageJ\\ImageJ.exe"
    ]
    
    for location in imagej_locations:
        if Path(location).exists():
            paths["imagej_path"] = location
            print_status(f"Found ImageJ/Fiji at: {location}")
            break
    else:
        print_warning("ImageJ/Fiji not found in common locations")
        print_warning("Please install Fiji from: https://fiji.sc/")
        paths["imagej_path"] = ""
    
    # Get Python path from main virtual environment
    main_python = get_venv_python("venv")
    if main_python.exists():
        paths["python_path"] = str(main_python)
        print_status(f"Found Python at: {main_python}")
    else:
        print_warning("Main Python virtual environment not found")
        paths["python_path"] = ""
    
    return paths

def create_config_file() -> bool:
    """Create or update the configuration file with detected paths."""
    config_path = Path("config/config.json")
    template_path = Path("config/config.template.json")
    
    try:
        # Load base config from existing file or template
        if config_path.exists():
            with open(config_path, 'r') as f:
                base_config = json.load(f)
        elif template_path.exists():
            with open(template_path, 'r') as f:
                base_config = json.load(f)
        else:
            print_error("No configuration template found to create config file")
            return False

        # Detect software paths
        detected_paths = detect_software_paths()
        
        # Update configuration with detected paths
        base_config["imagej_path"] = detected_paths.get("imagej_path", "")
        base_config["python_path"] = detected_paths.get("python_path", "")
        
        # Ensure config directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write configuration file
        with open(config_path, 'w') as f:
            json.dump(base_config, f, indent=2)
        print_status(f"Configuration file written to: {config_path}")

        return True
    except Exception as e:
        print_error(f"Failed to create configuration file: {e}")
        return False

def run_setup_script(venv_name: str = "venv") -> bool:
    """Run the setup script if it exists."""
    python_path = get_venv_python(venv_name)
    setup_script = Path("src/python/setup.py")
    
    if setup_script.exists():
        try:
            print_status("Running setup script...")
            result = subprocess.run([str(python_path), str(setup_script)], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                print_warning(f"Setup script completed with warnings: {result.stderr}")
            else:
                print_status("Setup script completed successfully")
            
            return True
        except Exception as e:
            print_warning(f"Setup script failed: {e}")
            return False
    else:
        print_status("No setup script found, skipping")
        return True

def verify_installation(venv_name: str = "venv") -> bool:
    """Verify the installation by testing imports."""
    python_path = get_venv_python(venv_name)
    
    try:
        print_status("Verifying installation...")
        
        # Test basic imports
        test_imports = [
            "import numpy",
            "import pandas", 
            "import scipy",
            "import matplotlib",
            "import skimage",
            "import tifffile",
            "import cv2",
            "import sklearn",
            "import dtcwt"
        ]
        
        failed_imports = []
        for import_statement in test_imports:
            result = subprocess.run([str(python_path), "-c", import_statement], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print_error(f"Failed to import: {import_statement}")
                failed_imports.append(import_statement)
        
        if failed_imports:
            print_error(f"Failed imports: {failed_imports}")
            return False
        
        print_status("All critical packages imported successfully")
        
        # Test main script
        result = subprocess.run([str(python_path), "main.py", "--help"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print_status("Main script verified successfully")
        else:
            print_warning("Main script test failed")
        
        # Test global command using the virtual environment's flimfret
        venv_bin = Path(venv_name) / "bin"
        flimfret_cmd = venv_bin / "flimfret"
        
        if flimfret_cmd.exists():
            result = subprocess.run([str(flimfret_cmd), "--help"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print_status("Global flimfret command verified successfully")
            else:
                print_warning("Global flimfret command test failed")
        else:
            print_warning("Global flimfret command not found in virtual environment")
        
        return True
    except Exception as e:
        print_error(f"Verification failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "="*60)
    print_status("Installation completed successfully!", Colors.BLUE)
    print("="*60)
    print("\nTo use FLIM-FRET:")
    print("\n1. Activate the virtual environment:")
    print("   source venv/bin/activate")
    print("\n2. Run the analysis tool (GLOBAL COMMAND):")
    print("   flimfret --help                    # Global command")
    print("   flimfret --interactive             # Interactive mode")
    print("\n3. Or run directly:")
    print("   python main.py --help              # Direct execution")
    print("\n4. Available options:")
    print("   --preprocessing       # Convert .bin to .tif files")
    print("   --processing         # Process .tif to .npz files")
    print("   --visualize          # Interactive phasor visualization")
    print("   --segment            # Interactive phasor segmentation")
    print("   --data-exploration   # Interactive ROI visualization")
    print("   --lifetime-images    # Generate lifetime images")
    print("   --average-lifetime   # Calculate average lifetime")
    print("\nFor more information, see the README.md file.")
    print("="*60)

def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="Install FLIM-FRET")
    parser.add_argument("--skip-config", action="store_true", 
                       help="Skip configuration file creation")
    parser.add_argument("--force", action="store_true", 
                       help="Force reinstallation even if environments exist")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("FLIM-FRET Installation Script")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create main virtual environment
    if args.force and Path("venv").exists():
        print_status("Removing existing venv (--force specified)")
        import shutil
        shutil.rmtree("venv")
    
    if not create_virtual_environment("venv"):
        sys.exit(1)
    
    # Install main requirements
    requirements_file = "src/scripts/requirements.txt"
    if not install_requirements("venv", requirements_file):
        sys.exit(1)
    
    # Install additional dependencies
    if not install_additional_dependencies("venv"):
        sys.exit(1)
    
    # Install package in development mode
    if not install_package_development_mode("venv"):
        sys.exit(1)
    
    # Create configuration file
    if not args.skip_config:
        if not create_config_file():
            print_warning("Configuration file creation failed, but installation continues")
    
    # Run setup script
    if not run_setup_script("venv"):
        print_warning("Setup script failed, but installation continues")
    
    # Verify installation
    if not verify_installation("venv"):
        print_warning("Installation verification failed, but installation may still work")
    
    # Print usage instructions
    print_usage_instructions()

if __name__ == "__main__":
    main()
