#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error Tracking Utility for FLIM-FRET Analysis Pipeline

This module provides a simple error tracking interface that can be used
by individual pipeline modules to log errors and warnings consistently.
"""

import os
import sys
import time
import traceback
import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

class ModuleErrorTracker:
    """
    Simple error tracker for individual pipeline modules.
    Can be used independently or integrated with the main PipelineLogger.
    """
    
    def __init__(self, module_name: str, log_dir: Optional[str] = None):
        self.module_name = module_name
        self.errors = []
        self.warnings = []
        self.start_time = time.time()
        
        # Setup logging if log_dir is provided
        self.logger = None
        if log_dir:
            self.setup_logging(log_dir)
    
    def setup_logging(self, log_dir: str):
        """Setup basic logging for the module."""
        try:
            import logging
            
            # Create log directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)
            
            # Create timestamp for log file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f'{self.module_name}_{timestamp}.log')
            
            # Setup logger
            self.logger = logging.getLogger(f'FLIM_FRET_{self.module_name}')
            self.logger.setLevel(logging.INFO)
            self.logger.handlers = []  # Clear existing handlers
            
            # File handler
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
        except Exception as e:
            print(f"Warning: Could not setup logging for {self.module_name}: {e}")
    
    def log_error(self, error: Exception, context: str = "", file_path: str = ""):
        """Log an error with context."""
        error_info = {
            'timestamp': datetime.datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'file_path': file_path,
            'traceback': traceback.format_exc()
        }
        
        self.errors.append(error_info)
        
        # Log to logger if available
        if self.logger:
            self.logger.error(f"ERROR in {self.module_name}: {context}")
            self.logger.error(f"Error type: {error_info['error_type']}")
            self.logger.error(f"Error message: {error_info['error_message']}")
            if file_path:
                self.logger.error(f"File: {file_path}")
            self.logger.error(f"Traceback:\n{error_info['traceback']}")
        else:
            # Fallback to print
            print(f"ERROR in {self.module_name}: {context}")
            print(f"Error type: {error_info['error_type']}")
            print(f"Error message: {error_info['error_message']}")
            if file_path:
                print(f"File: {file_path}")
            print(f"Traceback:\n{error_info['traceback']}")
    
    def log_warning(self, message: str, context: str = ""):
        """Log a warning."""
        warning_info = {
            'timestamp': datetime.datetime.now().isoformat(),
            'message': message,
            'context': context
        }
        
        self.warnings.append(warning_info)
        
        if self.logger:
            self.logger.warning(f"WARNING in {self.module_name}: {context} - {message}")
        else:
            print(f"WARNING in {self.module_name}: {context} - {message}")
    
    def log_info(self, message: str):
        """Log an info message."""
        if self.logger:
            self.logger.info(f"{self.module_name}: {message}")
        else:
            print(f"{self.module_name}: {message}")
    
    @contextmanager
    def error_context(self, context: str = "", file_path: str = ""):
        """Context manager for error handling with automatic logging."""
        try:
            yield
        except Exception as e:
            self.log_error(e, context, file_path)
            # Don't re-raise by default - let the caller decide
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of errors and warnings."""
        return {
            'module_name': self.module_name,
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'runtime_seconds': time.time() - self.start_time,
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    def print_summary(self):
        """Print a summary of errors and warnings."""
        summary = self.get_summary()
        
        print(f"\n=== {self.module_name} Error Summary ===")
        print(f"Total errors: {summary['total_errors']}")
        print(f"Total warnings: {summary['total_warnings']}")
        print(f"Runtime: {summary['runtime_seconds']:.2f} seconds")
        
        if summary['errors']:
            print("\nErrors:")
            for i, error in enumerate(summary['errors'], 1):
                print(f"  {i}. {error['error_type']}: {error['error_message']}")
                if error['context']:
                    print(f"     Context: {error['context']}")
                if error['file_path']:
                    print(f"     File: {error['file_path']}")
        
        if summary['warnings']:
            print("\nWarnings:")
            for i, warning in enumerate(summary['warnings'], 1):
                print(f"  {i}. {warning['message']}")
                if warning['context']:
                    print(f"     Context: {warning['context']}")

# Convenience function to create a tracker
def create_error_tracker(module_name: str, log_dir: Optional[str] = None) -> ModuleErrorTracker:
    """Create a new error tracker for a module."""
    return ModuleErrorTracker(module_name, log_dir)

# Example usage:
if __name__ == "__main__":
    # Example of how to use the error tracker
    tracker = create_error_tracker("TestModule", "logs")
    
    with tracker.error_context("Testing error handling"):
        # This will raise an error and be logged
        result = 1 / 0
    
    tracker.print_summary() 