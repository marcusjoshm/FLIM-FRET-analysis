"""
Stage Execution Framework for FLIM-FRET Analysis Pipeline

Provides base classes and execution framework for pipeline stages.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path

from .config import Config
from .logger import PipelineLogger, ModuleLogger


class StageError(Exception):
    """Custom exception for stage execution errors."""
    pass


class StageBase(ABC):
    """
    Base class for all pipeline stages.
    
    Provides common functionality for stage execution, logging, and error handling.
    """
    
    def __init__(self, config: Config, logger: PipelineLogger, stage_name: str):
        """
        Initialize the stage.
        
        Args:
            config: Pipeline configuration
            logger: Pipeline logger
            stage_name: Name of the stage
        """
        self.config = config
        self.pipeline_logger = logger
        self.stage_name = stage_name
        self.logger = ModuleLogger(stage_name, logger)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
    @abstractmethod
    def run(self, **kwargs) -> bool:
        """
        Execute the stage.
        
        Args:
            **kwargs: Stage-specific arguments
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate stage inputs.
        
        Args:
            **kwargs: Stage-specific arguments
            
        Returns:
            True if inputs are valid, False otherwise
        """
        pass
    
    def get_description(self) -> str:
        """
        Get stage description.
        
        Returns:
            Stage description string
        """
        return f"Executing {self.stage_name}"
    
    def setup(self, **kwargs) -> bool:
        """
        Setup stage (called before run).
        
        Args:
            **kwargs: Stage-specific arguments
            
        Returns:
            True if setup successful, False otherwise
        """
        return True
    
    def cleanup(self, **kwargs) -> bool:
        """
        Cleanup stage (called after run).
        
        Args:
            **kwargs: Stage-specific arguments
            
        Returns:
            True if cleanup successful, False otherwise
        """
        return True
    
    def execute(self, **kwargs) -> bool:
        """
        Execute the complete stage lifecycle.
        
        Args:
            **kwargs: Stage-specific arguments
            
        Returns:
            True if successful, False otherwise
        """
        self.start_time = time.time()
        success = False
        
        try:
            # Log stage start
            self.pipeline_logger.log_stage_start(self.stage_name, self.get_description())
            
            # Validate inputs
            if not self.validate_inputs(**kwargs):
                raise StageError("Input validation failed")
            
            # Setup
            if not self.setup(**kwargs):
                raise StageError("Stage setup failed")
            
            # Run the stage
            success = self.run(**kwargs)
            
            if not success:
                raise StageError("Stage execution failed")
            
            # Cleanup
            if not self.cleanup(**kwargs):
                self.logger.warning("Stage cleanup failed, but continuing")
            
        except Exception as e:
            self.logger.error(f"Stage execution failed: {str(e)}", e)
            success = False
        
        finally:
            # Log stage end
            self.end_time = time.time()
            duration = self.end_time - self.start_time if self.start_time else 0
            additional_info = f"Duration: {duration:.2f}s"
            self.pipeline_logger.log_stage_end(self.stage_name, success, additional_info)
        
        return success
    
    def get_duration(self) -> Optional[float]:
        """
        Get stage execution duration.
        
        Returns:
            Duration in seconds, or None if stage hasn't completed
        """
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class FileProcessingStage(StageBase):
    """
    Base class for stages that process files.
    
    Provides common functionality for file processing stages.
    """
    
    def __init__(self, config: Config, logger: PipelineLogger, stage_name: str):
        """Initialize the file processing stage."""
        super().__init__(config, logger, stage_name)
        self.processed_files: List[str] = []
        self.failed_files: List[str] = []
    
    def process_file(self, file_path: str, **kwargs) -> bool:
        """
        Process a single file.
        
        Args:
            file_path: Path to file to process
            **kwargs: Additional arguments
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            self.logger.debug(f"Processing file: {file_path}")
            success = self._process_file_impl(file_path, **kwargs)
            
            if success:
                self.processed_files.append(file_path)
                self.logger.debug(f"Successfully processed: {file_path}")
            else:
                self.failed_files.append(file_path)
                self.logger.warning(f"Failed to process: {file_path}")
            
            # Log file processing stats
            processing_time = time.time() - start_time
            self.pipeline_logger.log_file_processing(
                file_path, success, processing_time, self.stage_name
            )
            
            return success
            
        except Exception as e:
            self.failed_files.append(file_path)
            self.logger.error(f"Error processing file {file_path}: {str(e)}", e)
            
            processing_time = time.time() - start_time
            self.pipeline_logger.log_file_processing(
                file_path, False, processing_time, self.stage_name
            )
            
            return False
    
    @abstractmethod
    def _process_file_impl(self, file_path: str, **kwargs) -> bool:
        """
        Implementation of file processing logic.
        
        Args:
            file_path: Path to file to process
            **kwargs: Additional arguments
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get processing summary.
        
        Returns:
            Dictionary with processing statistics
        """
        total_files = len(self.processed_files) + len(self.failed_files)
        success_rate = len(self.processed_files) / total_files * 100 if total_files > 0 else 0
        
        return {
            "total_files": total_files,
            "processed_files": len(self.processed_files),
            "failed_files": len(self.failed_files),
            "success_rate": success_rate,
            "processed_file_list": self.processed_files.copy(),
            "failed_file_list": self.failed_files.copy()
        }


class StageRegistry:
    """
    Registry for pipeline stages.
    
    Manages available stages and their execution order.
    """
    
    def __init__(self):
        """Initialize the stage registry."""
        self._stages: Dict[str, type] = {}
        self._stage_order: List[str] = []
    
    def register(self, stage_name: str, stage_class: type, order: Optional[int] = None) -> None:
        """
        Register a stage.
        
        Args:
            stage_name: Name of the stage
            stage_class: Stage class (must inherit from StageBase)
            order: Optional order for stage execution
        """
        if not issubclass(stage_class, StageBase):
            raise ValueError(f"Stage class must inherit from StageBase: {stage_class}")
        
        self._stages[stage_name] = stage_class
        
        if order is not None:
            if len(self._stage_order) <= order:
                self._stage_order.extend([None] * (order + 1 - len(self._stage_order)))
            self._stage_order[order] = stage_name
        elif stage_name not in self._stage_order:
            self._stage_order.append(stage_name)
    
    def get_stage_class(self, stage_name: str) -> Optional[type]:
        """
        Get stage class by name.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Stage class or None if not found
        """
        return self._stages.get(stage_name)
    
    def get_available_stages(self) -> List[str]:
        """
        Get list of available stage names.
        
        Returns:
            List of stage names
        """
        return list(self._stages.keys())
    
    def get_stage_order(self) -> List[str]:
        """
        Get ordered list of stage names.
        
        Returns:
            List of stage names in execution order
        """
        return [stage for stage in self._stage_order if stage is not None]


class StageExecutor:
    """
    Executes pipeline stages in order.
    """
    
    def __init__(self, config: Config, logger: PipelineLogger, registry: StageRegistry):
        """
        Initialize the stage executor.
        
        Args:
            config: Pipeline configuration
            logger: Pipeline logger
            registry: Stage registry
        """
        self.config = config
        self.logger = logger
        self.registry = registry
        self.executed_stages: List[str] = []
        self.failed_stages: List[str] = []
    
    def execute_stage(self, stage_name: str, **kwargs) -> bool:
        """
        Execute a single stage.
        
        Args:
            stage_name: Name of the stage to execute
            **kwargs: Stage-specific arguments
            
        Returns:
            True if successful, False otherwise
        """
        stage_class = self.registry.get_stage_class(stage_name)
        if not stage_class:
            self.logger.error(f"Unknown stage: {stage_name}")
            return False
        
        # Create stage instance
        stage = stage_class(self.config, self.logger, stage_name)
        
        # Execute stage
        success = stage.execute(**kwargs)
        
        if success:
            self.executed_stages.append(stage_name)
        else:
            self.failed_stages.append(stage_name)
        
        return success
    
    def execute_stages(self, stage_names: List[str], **kwargs) -> bool:
        """
        Execute multiple stages in order.
        
        Args:
            stage_names: List of stage names to execute
            **kwargs: Arguments passed to all stages
            
        Returns:
            True if all stages successful, False otherwise
        """
        overall_success = True
        
        for stage_name in stage_names:
            self.logger.info(f"Executing stage: {stage_name}")
            
            success = self.execute_stage(stage_name, **kwargs)
            if not success:
                self.logger.error(f"Stage failed: {stage_name}")
                overall_success = False
                
                # Optionally stop on first failure
                if self.config.get('pipeline.stop_on_failure', True):
                    break
        
        return overall_success
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get execution summary.
        
        Returns:
            Dictionary with execution statistics
        """
        total_stages = len(self.executed_stages) + len(self.failed_stages)
        success_rate = len(self.executed_stages) / total_stages * 100 if total_stages > 0 else 0
        
        return {
            "total_stages": total_stages,
            "executed_stages": len(self.executed_stages),
            "failed_stages": len(self.failed_stages),
            "success_rate": success_rate,
            "executed_stage_list": self.executed_stages.copy(),
            "failed_stage_list": self.failed_stages.copy()
        }


# Global stage registry
_stage_registry = StageRegistry()


def register_stage(stage_name: str, order: Optional[int] = None):
    """
    Decorator to register a stage.
    
    Args:
        stage_name: Name of the stage
        order: Optional order for stage execution
    """
    def decorator(stage_class):
        _stage_registry.register(stage_name, stage_class, order)
        return stage_class
    return decorator


def get_stage_registry() -> StageRegistry:
    """
    Get the global stage registry.
    
    Returns:
        Stage registry instance
    """
    return _stage_registry 