#!/bin/bash

# Script to copy directory structure and only stitched TIFF files
# Usage: ./copy_stitched.sh <input_directory>

set -e  # Exit on any error

# Function to display usage
usage() {
    echo "Usage: $0 <input_directory>"
    echo "Creates a new directory with '_stitched' suffix containing only stitched TIFF files"
    exit 1
}

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if directory argument is provided
if [ $# -ne 1 ]; then
    echo "Error: Please provide exactly one directory argument"
    usage
fi

INPUT_DIR="$1"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory '$INPUT_DIR' does not exist"
    exit 1
fi

# Remove trailing slash from input directory if present
INPUT_DIR="${INPUT_DIR%/}"

# Create output directory name
OUTPUT_DIR="${INPUT_DIR}_stitched"

# Check if output directory already exists
if [ -d "$OUTPUT_DIR" ]; then
    echo "Warning: Output directory '$OUTPUT_DIR' already exists"
    read -p "Do you want to continue? This may overwrite existing files. (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operation cancelled"
        exit 0
    fi
fi

log "Starting copy operation..."
log "Input directory: $INPUT_DIR"
log "Output directory: $OUTPUT_DIR"

# Create the output directory
mkdir -p "$OUTPUT_DIR"

# Function to recreate directory structure
recreate_structure() {
    local source_dir="$1"
    local target_dir="$2"
    
    # Find all directories in source and create them in target
    find "$source_dir" -type d | while read -r dir; do
        # Calculate relative path from source
        rel_path="${dir#$source_dir/}"
        
        # Skip the root directory itself
        if [ "$rel_path" != "$source_dir" ]; then
            target_path="$target_dir/$rel_path"
            mkdir -p "$target_path"
        fi
    done
}

# Function to copy stitched files
copy_stitched_files() {
    local source_dir="$1"
    local target_dir="$2"
    
    local copied_count=0
    
    # Find all files ending with "stitched.tiff"
    find "$source_dir" -name "*stitched*" -type f | while read -r file; do
        # Calculate relative path from source
        rel_path="${file#$source_dir/}"
        target_path="$target_dir/$rel_path"
        
        # Create parent directory if it doesn't exist
        target_parent=$(dirname "$target_path")
        mkdir -p "$target_parent"
        
        # Copy the file
        cp "$file" "$target_path"
        log "Copied: $rel_path"
        ((copied_count++))
    done
    
    return $copied_count
}

# Recreate directory structure
log "Recreating directory structure..."
recreate_structure "$INPUT_DIR" "$OUTPUT_DIR"

# Copy stitched files
log "Copying stitched TIFF files..."
copy_stitched_files "$INPUT_DIR" "$OUTPUT_DIR"

# Count the results
total_stitched=$(find "$INPUT_DIR" -name "*stitched.tiff" -type f | wc -l)
copied_stitched=$(find "$OUTPUT_DIR" -name "*stitched.tiff" -type f | wc -l)

log "Operation completed successfully!"
log "Directory structure recreated in: $OUTPUT_DIR"
log "Stitched files found: $total_stitched"
log "Stitched files copied: $copied_stitched"

# Verify the operation
if [ "$total_stitched" -eq "$copied_stitched" ]; then
    log "✓ All stitched files copied successfully"
else
    log "⚠ Warning: Expected $total_stitched files but copied $copied_stitched"
fi

# Display summary of what was copied
log "Summary of copied files:"
find "$OUTPUT_DIR" -name "*stitched.tiff" -type f | while read -r file; do
    rel_path="${file#$OUTPUT_DIR/}"
    echo "  $rel_path"
done

log "Done!"