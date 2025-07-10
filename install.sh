#!/bin/bash

# FLIM-FRET Analysis Pipeline Installation Script
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
if [ ! -f "$PROJECT_DIR/run_pipeline.py" ]; then
    echo "❌ Error: Please run this script from the FLIM-FRET-analysis directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

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
    'sklearn', 'dtcwt'
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
echo "To activate the virtual environment manually:"
echo "  source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python run_pipeline.py --help"
echo ""
echo "To deactivate the virtual environment:"
echo "  deactivate"
echo ""
echo "🎉 FLIM-FRET Analysis Pipeline is ready to use!" 