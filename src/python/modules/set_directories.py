#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set Default Directories Module
=============================

This module provides functionality to set and manage default input and output directories.

Part of FLIM-FRET Analysis Pipeline
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .directory_setup import (
    validate_directory_path, 
    get_recent_directories, 
    add_recent_directory,
    prompt_for_directory,
    save_config
)


def set_default_directories(config: Dict, config_path: str) -> Tuple[str, str]:
    """
    Set default input and output directories interactively.
    
    Args:
        config (Dict): Configuration dictionary
        config_path (str): Path to config file
        
    Returns:
        Tuple[str, str]: (input_path, output_path)
    """
    print("\n=== Set Default Directories ===")
    print("This will set the default input and output directories for future runs.")
    print("These directories will be used automatically unless overridden with --input and --output arguments.")
    
    # Get recent directories
    recent_inputs = get_recent_directories(config, 'input')
    recent_outputs = get_recent_directories(config, 'output')
    
    # Get current default paths
    current_input = config.get('directories', {}).get('input', '')
    current_output = config.get('directories', {}).get('output', '')
    
    print(f"\nCurrent defaults:")
    print(f"  Input: {current_input if current_input else 'Not set'}")
    print(f"  Output: {current_output if current_output else 'Not set'}")
    
    # Get new input path
    print(f"\nSetting input directory:")
    input_path = prompt_for_directory('input', recent_inputs, current_input)
    
    # Get new output path
    print(f"\nSetting output directory:")
    output_path = prompt_for_directory('output', recent_outputs, current_output)
    
    # Update config
    if 'directories' not in config:
        config['directories'] = {}
    
    config['directories']['input'] = input_path
    config['directories']['output'] = output_path
    
    # Add to recent directories
    add_recent_directory(config, 'input', input_path)
    add_recent_directory(config, 'output', output_path)
    
    # Save config
    save_config(config, config_path)
    
    print(f"\n✅ Default directories set successfully!")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Config saved to: {config_path}")
    
    return input_path, output_path


def check_default_directories(config: Dict) -> Tuple[bool, str, str]:
    """
    Check if default directories are set and valid.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        Tuple[bool, str, str]: (are_valid, input_path, output_path)
    """
    if 'directories' not in config:
        return False, "", ""
    
    input_path = config['directories'].get('input', '')
    output_path = config['directories'].get('output', '')
    
    if not input_path or not output_path:
        return False, input_path, output_path
    
    # Validate paths
    input_valid = validate_directory_path(input_path, create_if_missing=False)
    output_valid = validate_directory_path(output_path, create_if_missing=True)
    
    return input_valid and output_valid, input_path, output_path


def get_default_directories(config: Dict) -> Tuple[str, str]:
    """
    Get default directories from config.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        Tuple[str, str]: (input_path, output_path)
    """
    if 'directories' not in config:
        return "", ""
    
    input_path = config['directories'].get('input', '')
    output_path = config['directories'].get('output', '')
    
    return input_path, output_path 