"""
Configuration Management Module

Handles all configuration loading, validation, and default settings
for the FLIM-FRET analysis pipeline.
"""

import json
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging


class ConfigError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class Config:
    """
    Centralized configuration management for the FLIM-FRET pipeline.
    
    This class handles loading configuration from files, environment variables,
    and provides default values with validation.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default locations.
        """
        self.logger = logging.getLogger(__name__)
        self._config: Dict[str, Any] = {}
        self._config_path: Optional[Path] = None
        
        # Load configuration
        self._load_defaults()
        if config_path:
            self.load_from_file(config_path)
        else:
            self._load_from_default_locations()
    
    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self._config = {
            "imagej_path": "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",
            "macro_path": "src/scripts/imagej/",
            "microscope_params": {
                "frequency": 78.0,
                "harmonic": 1,
                "bin_width_ns": 0.2208
            },
            "wavelet_params": {
                "filter_level": 9,
                "reference_g": 0.30227996721890404,
                "reference_s": 0.4592458920992018
            },
            "processing_params": {
                "apply_filter": 1,
                "threshold_min": 0,
                "threshold_max": None,
                "median_filter_size": 3
            },
            "output_params": {
                "save_plots": True,
                "plot_format": "png",
                "save_intermediate": False
            },
            "gmm_segmentation_params": {
                "n_components": 2,
                "covariance_type": "full",
                "max_iter": 100,
                "random_state": 42
            },
            "logging": {
                "level": "INFO",
                "console_level": "INFO",
                "file_level": "DEBUG"
            }
        }
    
    def _load_from_default_locations(self) -> None:
        """Try to load configuration from default locations."""
        default_paths = [
            Path("config/config.json"),
            Path("config.json"),
            Path.home() / ".flim_fret_config.json"
        ]
        
        for path in default_paths:
            if path.exists():
                self.logger.info(f"Loading configuration from: {path}")
                self.load_from_file(path)
                break
        else:
            self.logger.info("No configuration file found, using defaults")
    
    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            ConfigError: If file cannot be loaded or parsed
        """
        config_path = Path(config_path)
        self._config_path = config_path
        
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Recursively update configuration
            self._deep_update(self._config, file_config)
            self.logger.info(f"Configuration loaded from: {config_path}")
            
        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading configuration: {e}")
    
    def save_to_file(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration. If None, uses loaded path.
        """
        if config_path is None:
            config_path = self._config_path
        
        if config_path is None:
            raise ConfigError("No configuration path specified")
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
            self.logger.info(f"Configuration saved to: {config_path}")
        except Exception as e:
            raise ConfigError(f"Error saving configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'microscope_params.frequency')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'microscope_params.frequency')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigError: If configuration is invalid
        """
        # Validate ImageJ path
        imagej_path = self.get('imagej_path')
        if imagej_path and not os.path.exists(imagej_path):
            raise ConfigError(f"ImageJ path does not exist: {imagej_path}")
        
        # Validate macro path
        macro_path = self.get('macro_path')
        if macro_path and not os.path.exists(macro_path):
            raise ConfigError(f"Macro path does not exist: {macro_path}")
        
        # Validate numeric parameters
        frequency = self.get('microscope_params.frequency')
        if frequency is not None and frequency <= 0:
            raise ConfigError(f"Invalid frequency value: {frequency}")
        
        harmonic = self.get('microscope_params.harmonic')
        if harmonic is not None and harmonic < 1:
            raise ConfigError(f"Invalid harmonic value: {harmonic}")
        
        filter_level = self.get('wavelet_params.filter_level')
        if filter_level is not None and filter_level < 1:
            raise ConfigError(f"Invalid filter level: {filter_level}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        self._deep_update(self._config, updates)
    
    @staticmethod
    def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Recursively update nested dictionary.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Updates to apply
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                Config._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Support dictionary-style assignment."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return self.get(key) is not None


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration object
    """
    return Config(config_path) 