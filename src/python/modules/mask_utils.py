#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask Utilities for FLIM-FRET Analysis
=====================================

This module provides utilities for working with masks stored in NPZ files.

Created by Joshua Marcus
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple

def list_available_masks(npz_file_path: str) -> Dict[str, Dict]:
    """
    List all available masks in an NPZ file.
    
    Args:
        npz_file_path: Path to NPZ file
        
    Returns:
        Dictionary mapping mask names to their metadata
    """
    try:
        data = np.load(npz_file_path, allow_pickle=True)
        npz_data = dict(data)
        
        available_masks = {}
        
        # Method 1: Check mask_registry if it exists
        if 'mask_registry' in npz_data:
            available_masks.update(npz_data['mask_registry'])
        
        # Method 2: Find all keys ending with '_mask'
        for key in npz_data.keys():
            if key.endswith('_mask'):
                if key not in available_masks:
                    # Create basic metadata for masks without registry entry
                    available_masks[key] = {
                        'type': 'binary',  # Assume binary for now
                        'description': f'Mask: {key}',
                        'created_by': 'Unknown',
                        'created_timestamp': 'Unknown'
                    }
        
        return available_masks
        
    except Exception as e:
        print(f"Error reading masks from {npz_file_path}: {e}")
        return {}

def get_mask_data(npz_file_path: str, mask_name: str) -> Optional[np.ndarray]:
    """
    Get mask data from an NPZ file.
    
    Args:
        npz_file_path: Path to NPZ file
        mask_name: Name of the mask to retrieve
        
    Returns:
        Mask data as numpy array, or None if not found
    """
    try:
        data = np.load(npz_file_path, allow_pickle=True)
        npz_data = dict(data)
        
        if mask_name in npz_data:
            return npz_data[mask_name]
        else:
            print(f"Mask '{mask_name}' not found in {npz_file_path}")
            return None
            
    except Exception as e:
        print(f"Error reading mask '{mask_name}' from {npz_file_path}: {e}")
        return None

def apply_mask_to_data(data: np.ndarray, mask: np.ndarray, 
                      background_value: float = 0.0) -> np.ndarray:
    """
    Apply a binary mask to data.
    
    Args:
        data: Input data array
        mask: Binary mask array (same shape as data)
        background_value: Value to set for masked-out regions
        
    Returns:
        Masked data array
    """
    if data.shape != mask.shape:
        raise ValueError(f"Data shape {data.shape} does not match mask shape {mask.shape}")
    
    masked_data = data.copy()
    masked_data[mask == 0] = background_value
    return masked_data

def print_mask_info(npz_file_path: str) -> None:
    """
    Print information about all available masks in an NPZ file.
    
    Args:
        npz_file_path: Path to NPZ file
    """
    masks = list_available_masks(npz_file_path)
    
    if not masks:
        print(f"No masks found in {npz_file_path}")
        return
    
    print(f"\nAvailable masks in {os.path.basename(npz_file_path)}:")
    print("=" * 60)
    
    for mask_name, metadata in masks.items():
        print(f"\nMask: {mask_name}")
        print(f"  Type: {metadata.get('type', 'Unknown')}")
        print(f"  Description: {metadata.get('description', 'No description')}")
        print(f"  Created by: {metadata.get('created_by', 'Unknown')}")
        print(f"  Created: {metadata.get('created_timestamp', 'Unknown')}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mask_utils.py <npz_file_path>")
        sys.exit(1)
    
    npz_file = sys.argv[1]
    print_mask_info(npz_file) 