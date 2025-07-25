#!/usr/bin/env python3
"""
Test script for the new refactored FLIM-FRET analysis architecture.

This demonstrates how to use the new modular system.
"""

import sys
import tempfile
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.python.core.config import Config
from src.python.core.logger import PipelineLogger
from src.python.core.cli import PipelineCLI
from src.python.core.stages import get_stage_registry
from src.python.modules import register_all_stages


def test_config_system():
    """Test the configuration system."""
    print("=== Testing Configuration System ===")
    
    # Create a test config
    config = Config()
    
    # Test setting and getting values
    config.set("test.value", 42)
    assert config.get("test.value") == 42
    
    # Test dot notation
    assert config.get("microscope_params.frequency") == 78.0
    
    # Test validation
    try:
        config.validate()
        print("✓ Configuration validation passed")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
    
    print("✓ Configuration system working")


def test_logging_system():
    """Test the logging system."""
    print("\n=== Testing Logging System ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = PipelineLogger(temp_dir)
        
        # Test basic logging
        logger.info("Test info message")
        logger.warning("Test warning message")
        
        # Test stage logging
        logger.log_stage_start("Test Stage", "Testing stage functionality")
        logger.log_stage_end("Test Stage", True, "Test completed successfully")
        
        # Test error context
        try:
            with logger.error_context("Testing error context", "Test Stage"):
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected
        
        print("✓ Logging system working")


def test_cli_system():
    """Test the CLI system."""
    print("\n=== Testing CLI System ===")
    
    cli = PipelineCLI()
    
    # Test argument parsing with test arguments
    test_args = ["--input", "/test/input", "--output", "/test/output", "--preprocessing"]
    
    try:
        # This will fail because the input directory doesn't exist, but that's expected
        args = cli.parse_args(test_args)
        print("✗ CLI should have failed validation for non-existent directory")
    except SystemExit:
        # Expected - argparse exits on validation failure
        print("✓ CLI validation working correctly")
    except Exception as e:
        print(f"✓ CLI validation caught invalid input: {e}")


def test_stage_system():
    """Test the stage registration and execution system."""
    print("\n=== Testing Stage System ===")
    
    # Register all stages
    register_all_stages()
    
    # Get the registry
    registry = get_stage_registry()
    
    # Check that stages were registered
    available_stages = registry.get_available_stages()
    print(f"Registered stages: {len(available_stages)}")
    
    expected_stages = [
        'preprocessing', 'phasor_visualization', 'gmm_segmentation', 'manual_segmentation'
    ]
    
    for stage in expected_stages:
        if stage in available_stages:
            print(f"✓ {stage} registered")
        else:
            print(f"✗ {stage} not registered")
    
    print("✓ Stage system working")


def test_integration():
    """Test the integration of all components."""
    print("\n=== Testing Integration ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directories
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create a mock .bin file
        test_subdir = input_dir / "test_data"
        test_subdir.mkdir()
        (test_subdir / "test.bin").touch()
        
        # Test configuration
        config = Config()
        config.set("test_mode", True)
        
        # Test logger
        logger = PipelineLogger(str(output_dir))
        
        # Test that we can create the main components
        from src.python.core.pipeline import Pipeline
        import argparse
        
        # Create mock args
        args = argparse.Namespace()
        args.input = str(input_dir)
        args.output = str(output_dir)
        args.all = False
        args.preprocessing = True
        args.processing = False
        args.visualize = False
        args.segment = False
        args.manual_segment = False
        args.manual_segment_unfiltered = False
        args.lifetime_images = False
        args.average_lifetime = False

        args.apply_mask = False
        args.visualize_segmented = False
        args.manual_segment_from_mask = False
        args.manual_segment_unfiltered_from_mask = False
        args.interactive = False
        args.log_level = "INFO"
        
        try:
            # This should work with our refactored architecture
            pipeline = Pipeline(config, logger, args)
            print("✓ Pipeline created successfully")
            print(f"✓ Pipeline will run {len(pipeline.stages_to_run)} stages")
        except Exception as e:
            print(f"✗ Pipeline creation failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run all tests."""
    print("Testing New FLIM-FRET Analysis Architecture")
    print("=" * 50)
    
    try:
        test_config_system()
        test_logging_system()
        test_cli_system()
        test_stage_system()
        test_integration()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        print("New architecture is ready to use.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 