# FLIM-FRET Analysis Pipeline - Error Tracking System

## Overview

The FLIM-FRET analysis pipeline now includes a comprehensive error tracking and logging system that helps you identify, debug, and resolve issues that occur during data processing. This system provides detailed error reports, performance metrics, and logging capabilities.

## Features

### 1. Comprehensive Logging
- **Multiple log files**: Separate logs for pipeline execution, errors only, and performance metrics
- **Timestamped logs**: Each run creates timestamped log files for easy tracking
- **Console and file output**: See logs in real-time and save them for later analysis

### 2. Error Tracking
- **Detailed error context**: Each error includes timestamp, error type, message, context, and full traceback
- **File-specific errors**: Track which specific files caused errors
- **Stage-specific tracking**: Know exactly which pipeline stage failed

### 3. Performance Monitoring
- **Stage timing**: Track how long each pipeline stage takes
- **File processing statistics**: Monitor success rates and processing times per file
- **Overall performance metrics**: Total runtime and efficiency analysis

### 4. Error Reports
- **Comprehensive summaries**: Detailed reports with error counts, warnings, and performance data
- **Structured format**: Easy-to-read reports for debugging and analysis
- **Automatic generation**: Reports are automatically created at the end of each pipeline run

## How It Works

### Main Pipeline Logger

The main pipeline uses the `PipelineLogger` class which automatically:

1. **Creates log directories** in your output folder
2. **Tracks all pipeline stages** with start/end times and success/failure status
3. **Logs all errors and warnings** with full context
4. **Generates error reports** at the end of execution

### Module-Level Error Tracking

Individual modules can use the `ModuleErrorTracker` for more granular error tracking:

```python
from error_tracker import create_error_tracker

# Create a tracker for your module
tracker = create_error_tracker("YourModuleName", "logs")

# Use context manager for automatic error logging
with tracker.error_context("File processing", file_path):
    # Your processing code here
    result = process_file(file_path)

# Log warnings
if some_condition:
    tracker.log_warning("Low signal detected", "Signal analysis")

# Get summary
tracker.print_summary()
```

## Log Files Generated

When you run the pipeline, the following log files are created in `{output_dir}/logs/`:

### 1. Pipeline Log (`pipeline_YYYYMMDD_HHMMSS.log`)
- Complete pipeline execution log
- All INFO, WARNING, and ERROR messages
- Stage start/end information
- File processing details

### 2. Error Log (`errors_YYYYMMDD_HHMMSS.log`)
- Only ERROR level messages
- Full error tracebacks
- File paths and context information

### 3. Performance Log (`performance_YYYYMMDD_HHMMSS.log`)
- Stage timing information
- Performance metrics
- Processing statistics

### 4. Error Report (`error_report_YYYYMMDD_HHMMSS.txt`)
- Comprehensive summary of the entire run
- Error and warning counts
- Stage performance summary
- File processing statistics
- Detailed error descriptions

## Using the Error Tracking System

### 1. Running the Pipeline

The error tracking is automatically enabled when you run the pipeline:

```bash
python run_pipeline.py --input-dir /path/to/data --output-base-dir /path/to/output --all
```

### 2. Checking Logs

After running the pipeline, check the logs directory:

```bash
ls -la /path/to/output/logs/
```

### 3. Reading Error Reports

The error report provides a comprehensive summary:

```bash
cat /path/to/output/logs/error_report_YYYYMMDD_HHMMSS.txt
```

### 4. Testing the System

You can test the error tracking system independently:

```bash
python tests/test_error_tracking.py
```

## Error Report Format

The error report includes:

```
================================================================================
FLIM-FRET ANALYSIS PIPELINE - ERROR REPORT
================================================================================
Generated: 2024-01-15 14:30:25
Total runtime: 125.67 seconds

SUMMARY:
  Total errors: 3
  Total warnings: 2

STAGE PERFORMANCE:
  Stage 1: Preprocessing: SUCCESS (45.23s)
  Stage 2B: Wavelet Filtering: FAILED (12.34s)
  Stage 4: GMM Segmentation: SUCCESS (68.10s)

FILE PROCESSING STATISTICS:
  Stage 1: Preprocessing: 15/15 files (100.0% success, avg 3.01s)
  Stage 2B: Wavelet Filtering: 12/15 files (80.0% success, avg 0.82s)
  Stage 4: GMM Segmentation: 12/12 files (100.0% success, avg 5.68s)

DETAILED ERRORS:
  Error 1:
    Time: 2024-01-15T14:28:15.123456
    Stage: Stage 2B: Wavelet Filtering
    Context: Processing file dataset_3
    Type: ValueError
    Message: Invalid data format in file
    File: /path/to/data/dataset_3.tif

  Error 2:
    Time: 2024-01-15T14:28:20.456789
    Stage: Stage 2B: Wavelet Filtering
    Context: Processing file dataset_7
    Type: MemoryError
    Message: Insufficient memory for processing
    File: /path/to/data/dataset_7.tif

WARNINGS:
  Warning 1:
    Time: 2024-01-15T14:25:10.789012
    Stage: Stage 1: Preprocessing
    Context: Calibration loading
    Message: Using default calibration values
```

## Best Practices

### 1. Regular Monitoring
- Check error reports after each pipeline run
- Monitor for patterns in errors (same files, same stages)
- Track performance trends over time

### 2. Debugging Errors
- Use the detailed error information to identify root causes
- Check file paths and permissions
- Verify input data format and quality

### 3. Performance Optimization
- Use stage timing information to identify bottlenecks
- Monitor file processing success rates
- Optimize parameters based on performance data

### 4. Integration with Your Modules
- Use the `ModuleErrorTracker` in your custom modules
- Provide meaningful context for errors
- Log warnings for potential issues

## Troubleshooting

### Common Issues

1. **No log files generated**: Check that the output directory is writable
2. **Missing error details**: Ensure exceptions are properly caught and logged
3. **Performance issues**: Use the performance logs to identify slow stages

### Getting Help

If you encounter issues with the error tracking system:

1. Check the log files for detailed error information
2. Run the test script to verify the system is working
3. Review the error report for patterns and root causes
4. Check file permissions and disk space

## Advanced Usage

### Custom Error Tracking

You can extend the error tracking system for your specific needs:

```python
# Create a custom error tracker
tracker = create_error_tracker("CustomModule", "logs")

# Add custom error handling
def custom_error_handler(error, context):
    tracker.log_error(error, context)
    # Add your custom error handling logic here

# Use in your processing functions
def process_data(data):
    with tracker.error_context("Data processing"):
        # Your processing logic
        result = complex_processing(data)
        return result
```

### Integration with External Systems

The error tracking system can be integrated with external monitoring systems:

```python
# Get error summary for external reporting
summary = tracker.get_summary()

# Send to external monitoring system
send_to_monitoring_system(summary)
```

This comprehensive error tracking system will help you identify and resolve issues quickly, improving the reliability and maintainability of your FLIM-FRET analysis pipeline. 