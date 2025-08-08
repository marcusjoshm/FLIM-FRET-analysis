#!/bin/bash

# FLIM-FRET Analysis Pipeline Global Launcher
# Place this script in ~/bin or another directory in your PATH
# Then you can run 'flimfret' from any directory

# Define the project directory (update this path to match your installation)
PROJECT_DIR="$HOME/FLIM-FRET-analysis"

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ Error: FLIM-FRET project not found at: $PROJECT_DIR"
    echo "Please update the PROJECT_DIR variable in this script to point to your installation"
    exit 1
fi

# Check if virtual environment exists
VENV_DIR="$PROJECT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Error: Virtual environment not found at: $VENV_DIR"
    echo "Please run './install' in the project directory first"
    exit 1
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Check if activation was successful
if [ "$VIRTUAL_ENV" != "$VENV_DIR" ]; then
    echo "❌ Error: Failed to activate virtual environment"
    exit 1
fi

# Change to project directory to ensure proper working directory
cd "$PROJECT_DIR"

# Run the flimfret command with all arguments
python -m python.main "$@"
