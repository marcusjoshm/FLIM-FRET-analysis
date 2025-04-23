#!/usr/bin/env python3
"""
Cleanup script for FLIM-FRET-analysis repository.
This script identifies and moves unused files to a backup directory.
"""

import os
import shutil
import datetime

# Current directory
repo_dir = os.path.dirname(os.path.abspath(__file__))

# Create a backup directory with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = os.path.join(repo_dir, f"backup_{timestamp}")
os.makedirs(backup_dir, exist_ok=True)

# List of files to be backed up and removed
files_to_remove = [
    # Unused ImageJ Macros
    "FLIM_processing_macro_2.ijm",
    "FLIM_processing_macro_3.ijm",
    "FLIM_processing_macro_4.ijm",
    "FLIM_processing_macro_5.ijm",
    "simple_bioformats_test.ijm",
    "test_bioformats.ijm",
    
    # Test Files
    "test_flute.py",
    "test_flute_import.py",
    "test_flute_integration.py",
    "test_flute_with_venv.py",
    "test_imagej.py",
    "test_preprocessing.py",
    "run_preprocessing_test.py",
    
    # Potentially Obsolete Files
    "run_pipeline.py.new",
    "flute_logger.py",
    "flute_processing_script.py",
    "flute_commands.json",
    "flute_commands.log",
    "flute_log.txt",
    "saved_dict.pkl",
    "calibration_log.xlsx"
]

# Files that are actively used and should be kept
files_to_keep = [
    "run_pipeline.py",
    "TCSPC_preprocessing_AUTOcal_v2_0.py",
    "ComplexWaveletFilter_v2_0.py",
    "ComplexWaveletFilter_v1_6.py",
    "phasor_transform.py",
    "phasor_visualization.py",
    "GMMSegmentation_v2_6.py",
    "simplify_filenames.py",
    "generate_intensity_images.py",
    "organize_output_files.py",
    "bin_to_tiff_converter.py",
    "FLIM_processing_macro_1.ijm",
    "requirements.txt",
    "config.json",
    "README.md",
    "calibration.csv",
    ".gitignore",
    "cleanup_repo.py"  # Include this script in the keep list
]

# Count statistics
moved_count = 0
skipped_count = 0
not_found_count = 0

print(f"FLIM-FRET Repository Cleanup")
print(f"============================")
print(f"Backing up unused files to: {backup_dir}")
print()

# Process each file in the removal list
for filename in files_to_remove:
    src_path = os.path.join(repo_dir, filename)
    if os.path.exists(src_path):
        print(f"Moving: {filename}")
        dst_path = os.path.join(backup_dir, filename)
        shutil.move(src_path, dst_path)
        moved_count += 1
    else:
        print(f"Not found: {filename}")
        not_found_count += 1

print()
print(f"Cleanup Summary:")
print(f"- {moved_count} files moved to backup")
print(f"- {not_found_count} files not found")
print()
print(f"Files preserved in repository:")
for filename in files_to_keep:
    if os.path.exists(os.path.join(repo_dir, filename)):
        print(f"- {filename}")

print()
print("To restore any files, copy them from the backup directory.")
print(f"Backup directory: {backup_dir}")
