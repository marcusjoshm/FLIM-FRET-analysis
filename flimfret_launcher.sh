#!/bin/bash

# FLIM-FRET Analysis Pipeline Launcher
# This script allows running flimfret from any directory without manually activating the virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Check if we're in the correct directory
if [ ! -f "$PROJECT_DIR/main.py" ]; then
    echo "❌ Error: This launcher script must be in the flimfret directory"
    echo "Current directory: $(pwd)"
    echo "Expected project directory: $PROJECT_DIR"
    exit 1
fi

# Check if virtual environment exists
VENV_DIR="$PROJECT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Error: Virtual environment not found at: $VENV_DIR"
    echo "Please run './install' first to set up the environment"
    exit 1
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Check if activation was successful
if [ "$VIRTUAL_ENV" != "$VENV_DIR" ]; then
    echo "❌ Error: Failed to activate virtual environment"
    exit 1
fi

# Run the flimfret command with all arguments
python -m python.main "$@"
