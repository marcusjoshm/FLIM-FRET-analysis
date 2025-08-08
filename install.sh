#!/bin/bash

# FLIM-FRET Analysis Pipeline Installation Script
# Updated for new modular architecture
# This script creates a virtual environment and installs all required dependencies

set -e  # Exit on any error

echo "============================================================"
echo " FLIM-FRET Analysis Pipeline Installation"
echo "============================================================"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

echo "Project directory: $PROJECT_DIR"

# Check if we're in the correct directory
if [ ! -f "$PROJECT_DIR/main.py" ]; then
    echo "❌ Error: Please run this script from the flimfret directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check for new architecture files
echo ""
echo "--- Checking New Architecture Files ---"
ARCHITECTURE_FILES=(
    "main.py"
    "src/python/modules/preprocessing.py"
    "src/python/modules/phasor_visualization.py"
    "src/python/modules/phasor_segmentation.py"
    "src/python/modules/data_exploration.py"
    "src/python/modules/calculate_average_lifetime.py"
    "src/python/modules/lifetime_images.py"
    "src/scripts/imagej/FLIM_processing_macro.ijm"
)

for file in "${ARCHITECTURE_FILES[@]}"; do
    if [ -f "$PROJECT_DIR/$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (missing)"
    fi
done

# Check Python version
echo ""
echo "--- Python Version Check ---"
python3 --version
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Error: Python 3.8 or higher is required"
    exit 1
fi
echo "✅ Python version is compatible"

# Create virtual environment
echo ""
echo "--- Creating Virtual Environment ---"
VENV_DIR="$PROJECT_DIR/venv"

if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment already exists at: $VENV_DIR"
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment..."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtual environment created successfully"
else
    echo "✅ Using existing virtual environment"
fi

# Activate virtual environment
echo ""
echo "--- Activating Virtual Environment ---"
source "$VENV_DIR/bin/activate"

if [ "$VIRTUAL_ENV" != "$VENV_DIR" ]; then
    echo "❌ Error: Failed to activate virtual environment"
    exit 1
fi
echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip
echo ""
echo "--- Upgrading pip ---"
python -m pip install --upgrade pip
echo "✅ pip upgraded successfully"

# Install dependencies
echo ""
echo "--- Installing Dependencies ---"
REQUIREMENTS_FILE="$PROJECT_DIR/src/scripts/requirements.txt"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "❌ Error: Requirements file not found at: $REQUIREMENTS_FILE"
    exit 1
fi

echo "Installing packages from: $REQUIREMENTS_FILE"
python -m pip install -r "$REQUIREMENTS_FILE"
echo "✅ All dependencies installed successfully"

# Install additional dependencies for new architecture
echo ""
echo "--- Installing Additional Dependencies ---"
python -m pip install tifffile scikit-image opencv-python scikit-learn
echo "✅ Additional dependencies installed successfully"

# Verify installation
echo ""
echo "--- Verifying Installation ---"
python -c "
import sys
print(f'Python executable: {sys.executable}')
print(f'Python version: {sys.version}')
print(f'Virtual environment: {sys.prefix}')
print()

# Test critical packages
packages_to_test = [
    'numpy', 'pandas', 'scipy', 'matplotlib',
    'skimage', 'tifffile', 'PIL', 'cv2',
    'sklearn', 'dtcwt', 'tifffile', 'skimage'
]

print('Testing package imports:')
failed_packages = []
for package in packages_to_test:
    try:
        __import__(package)
        print(f'  ✅ {package}')
    except ImportError as e:
        print(f'  ❌ {package}: {e}')
        failed_packages.append(package)

if failed_packages:
    print(f'\\n❌ Failed to import: {failed_packages}')
    sys.exit(1)
else:
    print('\\n✅ All critical packages imported successfully')
"

if [ $? -eq 0 ]; then
    echo "✅ Package verification successful"
else
    echo "❌ Package verification failed"
    exit 1
fi

# Run setup script
echo ""
echo "--- Running Setup Script ---"
SETUP_SCRIPT="$PROJECT_DIR/src/python/setup.py"
if [ -f "$SETUP_SCRIPT" ]; then
    echo "Running setup.py to generate configuration..."
    python "$SETUP_SCRIPT"
    if [ $? -eq 0 ]; then
        echo "✅ Setup script completed successfully"
    else
        echo "⚠️  Setup script completed with warnings (this is usually fine)"
    fi
else
    echo "⚠️  No setup.py found at $SETUP_SCRIPT, skipping configuration generation"
fi

# Check for ImageJ/Fiji installation
echo ""
echo "--- Checking ImageJ/Fiji Installation ---"
IMAGEJ_PATH="/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"
if [ -f "$IMAGEJ_PATH" ]; then
    echo "✅ ImageJ/Fiji found at: $IMAGEJ_PATH"
else
    echo "⚠️  ImageJ/Fiji not found at: $IMAGEJ_PATH"
    echo "   Please install Fiji from: https://fiji.sc/"
    echo "   This is required for .bin file processing"
fi

# Check configuration files
echo ""
echo "--- Checking Configuration Files ---"
CONFIG_DIR="$PROJECT_DIR/config"
if [ -d "$CONFIG_DIR" ]; then
    echo "✅ Configuration directory found"
    if [ -f "$CONFIG_DIR/config.template.json" ]; then
        echo "✅ Configuration template found"
    else
        echo "⚠️  Configuration template missing"
    fi
else
    echo "⚠️  Configuration directory missing"
fi

# Check data directory
echo ""
echo "--- Checking Data Directory ---"
DATA_DIR="$PROJECT_DIR/data"
if [ -d "$DATA_DIR" ]; then
    echo "✅ Data directory found"
    mkdir -p "$DATA_DIR/logs"
    echo "✅ Logs directory created/verified"
else
    echo "⚠️  Data directory missing, creating..."
    mkdir -p "$DATA_DIR/logs"
    echo "✅ Data and logs directories created"
fi

# Final instructions
echo ""
echo "============================================================"
echo " Installation Complete!"
echo "============================================================"
echo ""
echo "✅ Virtual environment created at: $VENV_DIR"
echo "✅ All dependencies installed"
echo "✅ Package verification passed"
echo ""
echo "📁 Project Structure:"
echo "  main.py                    - Main entry point"
echo "  src/python/modules/        - Core processing modules"
echo "  src/scripts/imagej/        - ImageJ macros"
echo "  config/                    - Configuration files"
echo "  data/                      - Data and logs"
echo ""
echo "🚀 Quick Start:"
echo "  source venv/bin/activate   # Activate virtual environment"
echo "  python main.py             # Run the pipeline"
echo ""
echo "📋 Available Options:"
echo "  1. Set Input/Output Directories"
echo "  2. Preprocessing (.bin to .tif)"
echo "  3. Preprocessing + Processing (.bin to .npz)"
echo "  4. Visualization (interactive phasor plots)"
echo "  5. Segmentation (interactive phasor segmentation)"
echo "  6. Data Exploration (interactive ROI visualization)"
echo "  7. Average Lifetime (calculate average lifetime)"
echo "  8. Lifetime Images (generate lifetime images)"
echo ""
echo "🔧 Troubleshooting:"
echo "  - If ImageJ fails, check Fiji installation"
echo "  - For large files (>500MB), use regular mode (automatic)"
echo "  - Check logs in data/logs/ for detailed error messages"
echo ""
echo "🎉 FLIM-FRET Analysis Pipeline is ready to use!" 