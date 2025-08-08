#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Directory Setup Module
================================

This module handles interactive setup of input and output directories
with recent path memory functionality.

Part of FLIM-FRET Analysis Pipeline
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def validate_directory_path(path: str, create_if_missing: bool = False) -> bool:
    """
    Validate if a directory path exists and is accessible.
    
    Args:
        path (str): Directory path to validate
        create_if_missing (bool): Whether to create directory if it doesn't exist
        
    Returns:
        bool: True if path is valid, False otherwise
    """
    try:
        path_obj = Path(path)
        if path_obj.exists():
            return path_obj.is_dir()
        elif create_if_missing:
            path_obj.mkdir(parents=True, exist_ok=True)
            return True
        else:
            return False
    except Exception:
        return False


def get_recent_directories(config: Dict, directory_type: str) -> List[str]:
    """
    Get recent directories from config.
    
    Args:
        config (Dict): Configuration dictionary
        directory_type (str): Either 'input' or 'output'
        
    Returns:
        List[str]: List of recent directory paths
    """
    if 'directories' not in config:
        return []
    
    recent_key = f"recent_{directory_type}s"
    return config['directories'].get(recent_key, [])


def add_recent_directory(config: Dict, directory_type: str, path: str, max_recent: int = 5) -> None:
    """
    Add a directory to recent directories list.
    
    Args:
        config (Dict): Configuration dictionary
        directory_type (str): Either 'input' or 'output'
        path (str): Directory path to add
        max_recent (int): Maximum number of recent directories to keep
    """
    if 'directories' not in config:
        config['directories'] = {}
    
    recent_key = f"recent_{directory_type}s"
    if recent_key not in config['directories']:
        config['directories'][recent_key] = []
    
    # Remove if already exists
    if path in config['directories'][recent_key]:
        config['directories'][recent_key].remove(path)
    
    # Add to beginning
    config['directories'][recent_key].insert(0, path)
    
    # Keep only the most recent ones
    config['directories'][recent_key] = config['directories'][recent_key][:max_recent]


def prompt_for_directory(directory_type: str, recent_dirs: List[str], default_path: str = "") -> str:
    """
    Prompt user for directory path with recent options.
    
    Args:
        directory_type (str): Type of directory ('input' or 'output')
        recent_dirs (List[str]): List of recent directories
        default_path (str): Default path if provided
        
    Returns:
        str: Selected directory path
    """
    print(f"\n=== {directory_type.title()} Directory Selection ===")
    
    if recent_dirs:
        print(f"Recent {directory_type} directories:")
        for i, path in enumerate(recent_dirs, 1):
            print(f"  {i}. {path}")
        print(f"  {len(recent_dirs) + 1}. Enter new path")
        print()
        
        while True:
            choice = input(f"Select {directory_type} directory (1-{len(recent_dirs) + 1}) or enter path: ").strip()
            
            # Check if it's a number
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(recent_dirs):
                    selected_path = recent_dirs[choice_num - 1]
                    print(f"Selected: {selected_path}")
                    return selected_path
                elif choice_num == len(recent_dirs) + 1:
                    break
                else:
                    print(f"Please enter a number between 1 and {len(recent_dirs) + 1}")
                    continue
            else:
                # Treat as a path
                if choice:
                    return choice
                else:
                    print("Please enter a valid path or number")
                    continue
    
    # No recent directories or user chose to enter new path
    while True:
        if default_path:
            path = input(f"Enter {directory_type} directory path (default: {default_path}): ").strip()
            if not path:
                path = default_path
        else:
            path = input(f"Enter {directory_type} directory path: ").strip()
        
        if not path:
            print("Please enter a valid path")
            continue
        
        # Remove quotes if present
        path = path.strip("'\"")
        
        # Validate path
        if directory_type == "input":
            if not validate_directory_path(path, create_if_missing=False):
                print(f"Error: Input directory '{path}' does not exist or is not accessible")
                continue
        else:  # output
            if not validate_directory_path(path, create_if_missing=True):
                print(f"Error: Cannot create or access output directory '{path}'")
                continue
        
        return path


def get_paths_interactively(config, args_input: Optional[str] = None, args_output: Optional[str] = None) -> Tuple[str, str, bool]:
    """
    Get input and output paths interactively if not provided.
    
    Args:
        config: Configuration dictionary or Config object
        args_input (Optional[str]): Input path from command line args
        args_output (Optional[str]): Output path from command line args
        
    Returns:
        Tuple[str, str, bool]: (input_path, output_path, config_modified)
    """
    # Convert Config object to dict if needed
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = config
    
    # Get recent directories
    recent_inputs = get_recent_directories(config_dict, 'input')
    recent_outputs = get_recent_directories(config_dict, 'output')
    
    # Get default paths from config
    default_input = config_dict.get('directories', {}).get('input', '')
    default_output = config_dict.get('directories', {}).get('output', '')
    
    # Get input path
    if not args_input:
        input_path = prompt_for_directory('input', recent_inputs, default_input)
    else:
        input_path = args_input
        if not validate_directory_path(input_path, create_if_missing=False):
            print(f"Warning: Input directory '{input_path}' does not exist or is not accessible")
            input_path = prompt_for_directory('input', recent_inputs, default_input)
    
    # Get output path
    if not args_output:
        output_path = prompt_for_directory('output', recent_outputs, default_output)
    else:
        output_path = args_output
        if not validate_directory_path(output_path, create_if_missing=True):
            print(f"Warning: Cannot create or access output directory '{output_path}'")
            output_path = prompt_for_directory('output', recent_outputs, default_output)
    
    # Ask if user wants to save as defaults
    print(f"\nSelected paths:")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    
    save_defaults = input("\nSave these paths as defaults? (y/n): ").strip().lower()
    config_modified = False
    if save_defaults in ['y', 'yes']:
        # Handle both Config objects and dictionaries
        if hasattr(config, 'to_dict'):
            # Config object
            if 'directories' not in config:
                config['directories'] = {}
            config['directories']['input'] = input_path
            config['directories']['output'] = output_path
            
            # Add to recent directories
            add_recent_directory(config.to_dict(), 'input', input_path)
            add_recent_directory(config.to_dict(), 'output', output_path)
        else:
            # Dictionary
            if 'directories' not in config:
                config['directories'] = {}
            config['directories']['input'] = input_path
            config['directories']['output'] = output_path
            
            # Add to recent directories
            add_recent_directory(config, 'input', input_path)
            add_recent_directory(config, 'output', output_path)
        
        config_modified = True
        print("Paths saved as defaults!")
    
    return input_path, output_path, config_modified


def save_config(config: Dict, config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config (Dict): Configuration dictionary
        config_path (str): Path to config file
    """
    try:
        # Load existing config if it exists
        existing_config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    existing_config = json.load(f)
            except Exception:
                # If we can't read the existing config, start fresh
                existing_config = {}
        
        # Merge the new config with existing config
        merged_config = existing_config.copy()
        merged_config.update(config)
        
        # Ensure the config directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save the merged config
        with open(config_path, 'w') as f:
            json.dump(merged_config, f, indent=4)
            
        print(f"✅ Configuration saved to: {config_path}")
    except Exception as e:
        print(f"❌ Error saving config to {config_path}: {e}")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from file.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        Dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {} 