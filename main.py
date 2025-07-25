#!/usr/bin/env python3
"""
FLIM-FRET Analysis Pipeline - Main Entry Point

Streamlined main script using the refactored architecture.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.python.core.config import Config, ConfigError
from src.python.core.logger import PipelineLogger
from src.python.core.cli import parse_arguments, CLIError
from src.python.core.pipeline import Pipeline


def main():
    """Main entry point for the FLIM-FRET analysis pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Load configuration
        config_path = args.config if hasattr(args, 'config') and args.config else None
        config = Config(config_path)
        config.validate()
        
        # Initialize logging
        logger = PipelineLogger(args.output, args.log_level)
        logger.info("FLIM-FRET Analysis Pipeline Starting")
        logger.info(f"Input directory: {args.input}")
        logger.info(f"Output directory: {args.output}")
        
        # Create and run pipeline
        pipeline = Pipeline(config, logger, args)
        success = pipeline.run()
        
        # Generate final report
        logger.save_error_report()
        
        if success:
            logger.info("Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("Pipeline completed with errors. Check error report for details.")
            sys.exit(1)
            
    except CLIError as e:
        print(f"CLI Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ConfigError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 