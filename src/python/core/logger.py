"""
Logging System for FLIM-FRET Analysis Pipeline

Provides comprehensive logging with error tracking, performance monitoring,
and detailed reporting capabilities.
"""

import logging
import os
import sys
import time
import datetime
import traceback
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from pathlib import Path


class PipelineLogger:
    """
    Comprehensive logging system for the FLIM-FRET analysis pipeline.
    Tracks errors, performance metrics, and provides detailed reporting.
    """
    
    def __init__(self, output_base_dir: str, log_level: str = "INFO"):
        """
        Initialize the pipeline logger.
        
        Args:
            output_base_dir: Base directory for pipeline output
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.output_base_dir = output_base_dir
        self.log_dir = Path(output_base_dir) / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.setup_logging(log_level)
        
        # Error tracking
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.stage_performance: Dict[str, Any] = {}
        self.file_processing_stats: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.start_time = time.time()
        self.stage_timings: Dict[str, Dict[str, Any]] = {}
        
    def setup_logging(self, log_level: str = "INFO") -> None:
        """Setup comprehensive logging with multiple handlers."""
        # Create timestamp for log files
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define log file paths
        self.pipeline_log_path = self.log_dir / f'pipeline_{timestamp}.log'
        self.error_log_path = self.log_dir / f'errors_{timestamp}.log'
        self.performance_log_path = self.log_dir / f'performance_{timestamp}.log'
        
        # Setup main logger
        self.logger = logging.getLogger('FLIM_FRET_Pipeline')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler (INFO level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (DEBUG level)
        file_handler = logging.FileHandler(self.pipeline_log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error handler (ERROR level only)
        error_handler = logging.FileHandler(self.error_log_path)
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n')
        error_handler.setFormatter(error_formatter)
        self.logger.addHandler(error_handler)
        
        # Performance handler
        self.performance_logger = logging.getLogger('FLIM_FRET_Performance')
        self.performance_logger.setLevel(logging.INFO)
        self.performance_logger.handlers = []
        
        perf_handler = logging.FileHandler(self.performance_log_path)
        perf_handler.setLevel(logging.INFO)
        perf_formatter = logging.Formatter('%(asctime)s - %(message)s')
        perf_handler.setFormatter(perf_formatter)
        self.performance_logger.addHandler(perf_handler)
        
    def log_stage_start(self, stage_name: str, stage_description: str = "") -> None:
        """Log the start of a pipeline stage."""
        self.stage_timings[stage_name] = {'start': time.time()}
        self.logger.info(f"=== STAGE START: {stage_name} ===")
        if stage_description:
            self.logger.info(f"Description: {stage_description}")
        self.performance_logger.info(f"STAGE_START: {stage_name}")
        
    def log_stage_end(self, stage_name: str, success: bool, additional_info: str = "") -> None:
        """Log the end of a pipeline stage with performance metrics."""
        if stage_name in self.stage_timings:
            end_time = time.time()
            duration = end_time - self.stage_timings[stage_name]['start']
            self.stage_timings[stage_name]['end'] = end_time
            self.stage_timings[stage_name]['duration'] = duration
            self.stage_timings[stage_name]['success'] = success
            
            status = "SUCCESS" if success else "FAILED"
            self.logger.info(f"=== STAGE END: {stage_name} - {status} ({duration:.2f}s) ===")
            if additional_info:
                self.logger.info(f"Additional info: {additional_info}")
            
            self.performance_logger.info(f"STAGE_END: {stage_name} - {status} - {duration:.2f}s")
            
    def log_error(self, error: Exception, context: str = "", stage: str = "", file_path: str = "") -> None:
        """Log an error with full context and traceback."""
        error_info = {
            'timestamp': datetime.datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'stage': stage,
            'file_path': file_path,
            'traceback': traceback.format_exc()
        }
        
        self.errors.append(error_info)
        
        # Log to error log
        self.logger.error(f"ERROR in {stage}: {context}")
        self.logger.error(f"Error type: {error_info['error_type']}")
        self.logger.error(f"Error message: {error_info['error_message']}")
        if file_path:
            self.logger.error(f"File: {file_path}")
        self.logger.error(f"Traceback:\n{error_info['traceback']}")
        
    def log_warning(self, message: str, context: str = "", stage: str = "") -> None:
        """Log a warning."""
        warning_info = {
            'timestamp': datetime.datetime.now().isoformat(),
            'message': message,
            'context': context,
            'stage': stage
        }
        
        self.warnings.append(warning_info)
        self.logger.warning(f"WARNING in {stage}: {context} - {message}")
        
    def log_file_processing(self, file_path: str, success: bool, processing_time: float, stage: str = "") -> None:
        """Log file processing statistics."""
        if stage not in self.file_processing_stats:
            self.file_processing_stats[stage] = {'success': 0, 'failed': 0, 'total_time': 0}
        
        if success:
            self.file_processing_stats[stage]['success'] += 1
        else:
            self.file_processing_stats[stage]['failed'] += 1
            
        self.file_processing_stats[stage]['total_time'] += processing_time
        
    @contextmanager
    def error_context(self, context: str, stage: str = "", file_path: str = ""):
        """Context manager for error handling with automatic logging."""
        try:
            yield
        except Exception as e:
            self.log_error(e, context, stage, file_path)
            raise
    
    def generate_error_report(self) -> str:
        """Generate a comprehensive error report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FLIM-FRET ANALYSIS PIPELINE - ERROR REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total runtime: {time.time() - self.start_time:.2f} seconds")
        report_lines.append("")
        
        # Summary
        report_lines.append("SUMMARY:")
        report_lines.append(f"  Total errors: {len(self.errors)}")
        report_lines.append(f"  Total warnings: {len(self.warnings)}")
        report_lines.append("")
        
        # Stage performance
        report_lines.append("STAGE PERFORMANCE:")
        for stage_name, timing in self.stage_timings.items():
            status = "SUCCESS" if timing.get('success', False) else "FAILED"
            duration = timing.get('duration', 0)
            report_lines.append(f"  {stage_name}: {status} ({duration:.2f}s)")
        report_lines.append("")
        
        # File processing statistics
        if self.file_processing_stats:
            report_lines.append("FILE PROCESSING STATISTICS:")
            for stage, stats in self.file_processing_stats.items():
                total = stats['success'] + stats['failed']
                success_rate = (stats['success'] / total * 100) if total > 0 else 0
                avg_time = stats['total_time'] / total if total > 0 else 0
                report_lines.append(f"  {stage}: {stats['success']}/{total} files ({success_rate:.1f}% success, avg {avg_time:.2f}s)")
            report_lines.append("")
        
        # Detailed errors
        if self.errors:
            report_lines.append("DETAILED ERRORS:")
            for i, error in enumerate(self.errors, 1):
                report_lines.append(f"  Error {i}:")
                report_lines.append(f"    Time: {error['timestamp']}")
                report_lines.append(f"    Stage: {error['stage']}")
                report_lines.append(f"    Context: {error['context']}")
                report_lines.append(f"    Type: {error['error_type']}")
                report_lines.append(f"    Message: {error['error_message']}")
                if error['file_path']:
                    report_lines.append(f"    File: {error['file_path']}")
                report_lines.append("")
        
        # Warnings
        if self.warnings:
            report_lines.append("WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                report_lines.append(f"  Warning {i}:")
                report_lines.append(f"    Time: {warning['timestamp']}")
                report_lines.append(f"    Stage: {warning['stage']}")
                report_lines.append(f"    Context: {warning['context']}")
                report_lines.append(f"    Message: {warning['message']}")
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_error_report(self) -> str:
        """Save the error report to a file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.log_dir / f'error_report_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write(self.generate_error_report())
        
        self.logger.info(f"Error report saved to: {report_path}")
        return str(report_path)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)


class ModuleLogger:
    """
    Simplified logger for individual modules that connects to the main pipeline logger.
    """
    
    def __init__(self, module_name: str, pipeline_logger: Optional[PipelineLogger] = None):
        """
        Initialize module logger.
        
        Args:
            module_name: Name of the module
            pipeline_logger: Main pipeline logger to connect to
        """
        self.module_name = module_name
        self.pipeline_logger = pipeline_logger
        self.logger = logging.getLogger(f'FLIM_FRET.{module_name}')
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(f"[{self.module_name}] {message}")
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(f"[{self.module_name}] {message}")
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(f"[{self.module_name}] {message}")
        if self.pipeline_logger:
            self.pipeline_logger.log_warning(message, stage=self.module_name)
    
    def error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log an error message."""
        self.logger.error(f"[{self.module_name}] {message}")
        if self.pipeline_logger and exception:
            self.pipeline_logger.log_error(exception, message, self.module_name)
    
    @contextmanager
    def error_context(self, context: str, file_path: str = ""):
        """Context manager for error handling."""
        try:
            yield
        except Exception as e:
            self.error(f"Error in {context}: {str(e)}", e)
            if self.pipeline_logger:
                self.pipeline_logger.log_error(e, context, self.module_name, file_path)
            raise 