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
from src.python.modules.directory_setup import get_paths_interactively, save_config, validate_directory_path, add_recent_directory
from src.python.modules.set_directories import set_default_directories, check_default_directories, get_default_directories


def main():
    """Main entry point for the FLIM-FRET analysis pipeline."""
    try:
        # Load configuration
        config_path = "config/config.json"
        config = Config(config_path)
        config.validate()
        
        while True:
            # Parse command line arguments
            args = parse_arguments()
            
            # Check if user chose to exit
            if args is None:
                print("Goodbye!")
                sys.exit(0)
            
            # Handle directory setup
            input_path = getattr(args, 'input', None)
            output_path = getattr(args, 'output', None)
            
            # Check if user wants to set directories
            if hasattr(args, 'set_directories') and args.set_directories:
                # Set default directories
                input_path, output_path, updated_config = set_default_directories(config.to_dict(), config_path)
                # Update the config object with the new values
                config.update(updated_config)
                print("Directory setup complete.")
                continue  # Return to menu
            
            # Check if default directories are set and valid
            are_valid, default_input, default_output = check_default_directories(config.to_dict())
            
            # Track if config needs to be updated
            config_modified = False
            
            if are_valid and not input_path and not output_path:
                # Use default directories
                input_path = default_input
                output_path = default_output
                print(f"Using default directories:")
                print(f"  Input: {input_path}")
                print(f"  Output: {output_path}")
            elif not input_path or not output_path:
                # Get paths interactively
                input_path, output_path, config_modified = get_paths_interactively(
                    config.to_dict(), 
                    input_path, 
                    output_path
                )
            else:
                # Both input_path and output_path are provided via command line flags
                # Validate the provided paths
                if not validate_directory_path(input_path, create_if_missing=False):
                    print(f"Warning: Input directory '{input_path}' does not exist or is not accessible")
                    # Fall back to interactive selection
                    input_path, output_path, config_modified = get_paths_interactively(
                        config.to_dict(), 
                        input_path, 
                        output_path
                    )
                elif not validate_directory_path(output_path, create_if_missing=True):
                    print(f"Warning: Cannot create or access output directory '{output_path}'")
                    # Fall back to interactive selection
                    input_path, output_path, config_modified = get_paths_interactively(
                        config.to_dict(), 
                        input_path, 
                        output_path
                    )
                else:
                    # Both paths are valid, update config with new defaults
                    print(f"Using command line directories:")
                    print(f"  Input: {input_path}")
                    print(f"  Output: {output_path}")
                    
                    # Ask if user wants to save as defaults
                    save_defaults = input("\nSave these paths as defaults? (y/n): ").strip().lower()
                    if save_defaults in ['y', 'yes']:
                        config_dict = config.to_dict()
                        if 'directories' not in config_dict:
                            config_dict['directories'] = {}
                        config_dict['directories']['input'] = input_path
                        config_dict['directories']['output'] = output_path
                        
                        # Add to recent directories
                        add_recent_directory(config_dict, 'input', input_path)
                        add_recent_directory(config_dict, 'output', output_path)
                        
                        # Update the config object
                        config.update(config_dict)
                        config_modified = True
                        print("Paths saved as defaults!")
            
            # Save config if it was modified
            if config_modified:
                save_config(config.to_dict(), config_path)
            
            # Update args with the selected paths
            args.input = input_path
            args.output = output_path
            
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
            else:
                logger.error("Pipeline completed with errors. Check error report for details.")
            
            # Check if this was an interactive module that blocks
            interactive_modules = ['segment', 'visualize', 'data_exploration']
            is_interactive = any(getattr(args, module, False) for module in interactive_modules)
            
            if is_interactive:
                # For interactive modules, the module handles its own completion
                # The matplotlib plt.show() call will block until the user closes the plot
                print("\n" + "="*60)
                print("Interactive module completed. Returning to main menu...")
                print("="*60 + "\n")
                
                # Small delay to let user read the completion message
                import time
                time.sleep(1)
            else:
                # For non-interactive modules, show completion message
                print("\n" + "="*60)
                print("Pipeline completed. Returning to main menu...")
                print("="*60 + "\n")
                
                # Small delay to let user read the output
                import time
                time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
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