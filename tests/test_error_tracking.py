#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to demonstrate the new error tracking system.

This script shows how the error tracking works and generates sample
error reports for testing purposes.
"""

import os
import sys
import time
import traceback

# Add the src/python/modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python', 'modules'))

from error_tracker import create_error_tracker

def test_error_tracking():
    """Test the error tracking system with various scenarios."""
    
    # Create output directory for logs
    output_dir = os.path.join(os.path.dirname(__file__), "test_error_output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Testing FLIM-FRET Error Tracking System")
    print("=" * 50)
    
    # Test 1: Basic error tracking
    print("\n1. Testing basic error tracking...")
    tracker1 = create_error_tracker("TestModule1", output_dir)
    
    try:
        # Simulate a file processing error
        tracker1.log_info("Starting file processing...")
        tracker1.log_warning("Low memory detected", "Memory check")
        
        # Simulate an error
        raise ValueError("Invalid file format detected")
        
    except Exception as e:
        tracker1.log_error(e, "File processing", "test_file.tif")
    
    tracker1.print_summary()
    
    # Test 2: Context manager usage
    print("\n2. Testing context manager...")
    tracker2 = create_error_tracker("TestModule2", output_dir)
    
    with tracker2.error_context("Data validation", "data.csv"):
        # This will succeed
        tracker2.log_info("Data validation completed successfully")
    
    with tracker2.error_context("Complex calculation", "matrix.npy"):
        # This will fail and be logged
        try:
            result = 1 / 0
        except ZeroDivisionError:
            # Expected error - already logged by context manager
            pass
    
    tracker2.print_summary()
    
    # Test 3: Multiple errors and warnings
    print("\n3. Testing multiple errors and warnings...")
    tracker3 = create_error_tracker("TestModule3", output_dir)
    
    # Simulate processing multiple files
    files = ["file1.tif", "file2.tif", "file3.tif", "file4.tif"]
    
    for i, file in enumerate(files):
        tracker3.log_info(f"Processing {file}...")
        
        if i == 1:
            # Simulate a warning
            tracker3.log_warning("Low signal-to-noise ratio", f"Processing {file}")
        
        if i == 2:
            # Simulate an error
            tracker3.log_error(RuntimeError(f"Corrupted data in {file}"), f"Processing file {file}", file)
            continue
            
        time.sleep(0.1)  # Simulate processing time
    
    tracker3.print_summary()
    
    # Test 4: Integration with main pipeline logger
    print("\n4. Testing integration with main pipeline...")
    
    # Import the main pipeline logger
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from run_pipeline import PipelineLogger
    
    # Create a main pipeline logger
    main_logger = PipelineLogger(output_dir)
    main_logger.log_stage_start("Test Stage", "Testing error tracking integration")
    
    # Simulate some errors
    try:
        raise FileNotFoundError("Configuration file not found")
    except Exception as e:
        main_logger.log_error(e, "Loading configuration", "Test Stage")
    
    main_logger.log_warning("Using default parameters", "Configuration", "Test Stage")
    
    main_logger.log_stage_end("Test Stage", False, "Completed with errors")
    
    # Generate error report
    report_path = main_logger.save_error_report()
    print(f"\nError report saved to: {report_path}")
    
    print("\n" + "=" * 50)
    print("Error tracking test completed!")
    print(f"Check the 'tests/test_error_output' directory for log files and error reports.")

def demonstrate_usage():
    """Demonstrate how to use the error tracking in your own modules."""
    
    print("\n" + "=" * 60)
    print("HOW TO USE ERROR TRACKING IN YOUR MODULES")
    print("=" * 60)
    
    print("""
1. Import the error tracker:
   from error_tracker import create_error_tracker

2. Create a tracker for your module:
   tracker = create_error_tracker("YourModuleName", "logs")

3. Use it in your functions:
   def process_file(file_path):
       with tracker.error_context("File processing", file_path):
           # Your processing code here
           result = some_processing_function(file_path)
           return result

4. Log errors manually if needed:
   try:
       # Some risky operation
       result = risky_function()
   except Exception as e:
       tracker.log_error(e, "Risky operation", file_path)

5. Log warnings:
   if some_condition:
       tracker.log_warning("Something to watch out for", "Context")

6. Get summary at the end:
   tracker.print_summary()
""")

if __name__ == "__main__":
    test_error_tracking()
    demonstrate_usage() 