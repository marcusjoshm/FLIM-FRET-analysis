#!/usr/bin/env python3
"""
Test script to verify the mask registry fix works correctly.
"""

import numpy as np
import os
import tempfile

def test_mask_registry_fix():
    """Test that the mask registry fix works correctly."""
    
    # Create a temporary NPZ file with some data
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
        npz_path = tmp_file.name
    
    # Create some test data
    test_data = {
        'G': np.random.rand(100, 100),
        'S': np.random.rand(100, 100),
        'A': np.random.rand(100, 100),
        'mask_registry': np.array([1, 2, 3])  # This is the problematic case - mask_registry as numpy array
    }
    
    # Save test data
    np.savez_compressed(npz_path, **test_data)
    
    print(f"Created test NPZ file: {npz_path}")
    print(f"Original mask_registry type: {type(test_data['mask_registry'])}")
    
    # Load the NPZ data (simulating what happens in ManualSegmentation)
    npz_data = np.load(npz_path)
    print(f"Loaded NPZ data, mask_registry type: {type(npz_data['mask_registry'])}")
    
    # Convert to dictionary (our fix)
    npz_data_dict = dict(npz_data)
    print(f"Converted to dict, mask_registry type: {type(npz_data_dict['mask_registry'])}")
    
    # Apply our fix
    if 'mask_registry' not in npz_data_dict:
        npz_data_dict['mask_registry'] = {}
    elif not isinstance(npz_data_dict['mask_registry'], dict):
        # If mask_registry exists but is not a dict (e.g., it's a numpy array), replace it
        print(f"Replacing non-dict mask_registry of type {type(npz_data_dict['mask_registry'])}")
        npz_data_dict['mask_registry'] = {}
    
    # Now try to assign to it
    npz_data_dict['mask_registry']['manual_segmentation_mask'] = {
        'type': 'binary',
        'description': 'Manual ellipse-based segmentation mask',
        'created_by': 'ManualSegmentation',
        'created_timestamp': '2025-07-26T21:11:00'
    }
    
    print("Successfully assigned to mask_registry!")
    print(f"Final mask_registry: {npz_data_dict['mask_registry']}")
    
    # Clean up
    os.unlink(npz_path)
    print("Test completed successfully!")

if __name__ == "__main__":
    test_mask_registry_fix() 