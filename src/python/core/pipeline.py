"""
Main Pipeline Orchestrator for FLIM-FRET Analysis

Coordinates the execution of all pipeline stages based on user arguments.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import argparse

from .config import Config
from .logger import PipelineLogger
from .stages import get_stage_registry, StageExecutor
from ..modules import register_all_stages


class Pipeline:
    """
    Main pipeline orchestrator for FLIM-FRET analysis.
    
    Manages the overall workflow, directory setup, and stage execution.
    """
    
    def __init__(self, config: Config, logger: PipelineLogger, args: argparse.Namespace):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
            logger: Pipeline logger
            args: Command line arguments
        """
        self.config = config
        self.logger = logger
        self.args = args
        
        # Setup directories
        self.setup_directories()
        
        # Register all available stages
        register_all_stages()
        
        # Create stage executor
        self.registry = get_stage_registry()
        self.executor = StageExecutor(config, logger, self.registry)
        
        # Determine which stages to run
        self.stages_to_run = self._determine_stages()
        
    def setup_directories(self) -> None:
        """Set up only required output directories for the selected stages."""
        # Always create logs directory
        logs_dir = Path(self.args.output) / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Created directory: {logs_dir}")

        # Determine which directories are needed based on selected stages
        needed_dirs = {
            'output': Path(self.args.output) / 'output',
            'preprocessed': Path(self.args.output) / 'preprocessed',
            'npz_datasets': Path(self.args.output) / 'npz_datasets',
            'logs': logs_dir
        }
        # Only create these at startup
        for name, path in needed_dirs.items():
            if name != 'logs':  # logs already created
                path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {path}")
        self.directories = needed_dirs
    
    def _determine_stages(self) -> List[str]:
        """
        Determine which stages to run based on command line arguments.
        
        Returns:
            List of stage names to execute
        """
        stages = []
        
        # Add stages based on flags
        if self.args.preprocessing or self.args.processing:
            stages.append('preprocessing')
        

            
        if self.args.processing:
            stages.append('processing')
            
        if self.args.visualize:
            stages.append('phasor_visualization')
            
        if self.args.segment:
            stages.append('phasor_segmentation')
            
        if getattr(self.args, 'data_exploration', False):
            stages.append('data_exploration')
            
        if self.args.lifetime_images:
            stages.append('lifetime_images')
            
        if self.args.average_lifetime:
            stages.append('average_lifetime')
            

    
        return stages
    
    def get_calibration_file_path(self) -> str:
        """
        Get the path to the calibration file.
        
        Returns:
            Path to calibration file
        """
        # Look for calibration file in input directory first
        input_calibration = Path(self.args.input) / "calibration.csv"
        if input_calibration.exists():
            self.logger.info(f"Using calibration file from input directory: {input_calibration}")
            return str(input_calibration)
        
        # Fall back to project directory
        project_calibration = Path("data/calibration.csv")
        if project_calibration.exists():
            self.logger.info(f"Using calibration file from project directory: {project_calibration}")
            return str(project_calibration)
        
        # If neither exists, create path for input directory (may be created later)
        self.logger.warning("No calibration file found, using input directory path")
        return str(input_calibration)
    
    def run(self) -> bool:
        """
        Run the complete pipeline.
        
        Returns:
            True if all stages completed successfully, False otherwise
        """
        start_time = time.time()
        
        self.logger.info("=" * 50)
        self.logger.info("FLIM-FRET Analysis Pipeline Starting")
        self.logger.info(f"Input Directory: {self.args.input}")
        self.logger.info(f"Output Directory: {self.args.output}")
        self.logger.info(f"Stages to run: {', '.join(self.stages_to_run)}")
        self.logger.info("=" * 50)
        
        # Prepare common arguments for all stages
        stage_kwargs = {
            'input_dir': self.args.input,
            'output_dir': self.args.output,
            'directories': self.directories,
            'calibration_file': self.get_calibration_file_path(),
            'interactive': getattr(self.args, 'interactive', False)
        }
        
        # Execute stages
        success = self.executor.execute_stages(self.stages_to_run, **stage_kwargs)
        
        # Log final summary
        end_time = time.time()
        duration = end_time - start_time
        
        self.logger.info("=" * 50)
        self.logger.info("FLIM-FRET Analysis Pipeline Complete")
        self.logger.info(f"Total Duration: {duration:.2f} seconds")
        self.logger.info(f"Overall Success: {'YES' if success else 'NO'}")
        
        # Log execution summary
        summary = self.executor.get_execution_summary()
        self.logger.info(f"Stages Executed: {summary['executed_stages']}/{summary['total_stages']}")
        if summary['failed_stages'] > 0:
            self.logger.error(f"Failed Stages: {', '.join(summary['failed_stage_list'])}")
        
        self.logger.info("=" * 50)
        
        return success 