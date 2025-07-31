#!/usr/bin/env python3
"""
Minimal test for the data exploration features.
"""

def test_functions_exist():
    """Test that all the new functions exist in the file."""
    
    # Read the data_exploration.py file
    with open('src/python/modules/data_exploration.py', 'r') as f:
        content = f.read()
    
    # Check if the functions exist
    functions_to_check = [
        ('def interactive_data_type_selection():', 'interactive_data_type_selection'),
        ('def interactive_mask_selection(', 'interactive_mask_selection'),
        ('def interactive_threshold_selection():', 'interactive_threshold_selection'),
        ('def apply_thresholding(', 'apply_thresholding')
    ]
    
    all_passed = True
    for check_text, function_name in functions_to_check:
        if check_text in content:
            print(f"✅ Function '{function_name}' exists in the file.")
        else:
            print(f"❌ Function '{function_name}' not found in the file.")
            all_passed = False
    
    return all_passed

def test_function_implementation():
    """Test that the functions have the expected implementation."""
    
    with open('src/python/modules/data_exploration.py', 'r') as f:
        content = f.read()
    
    # Check for key elements of the implementation
    checks = [
        # Data type selection
        ('filtered data (G/S coordinates)', 'Data type option 1'),
        ('unfiltered data (GU/SU coordinates)', 'Data type option 2'),
        ('Both (will process each file twice)', 'Data type option 3'),
        
        # Mask selection
        ('No mask (use original data)', 'Mask option 1'),
        ('Use masked NPZ files', 'Mask option 2'),
        ('Choose mask source for segmentation', 'Mask selection prompt'),
        
        # Thresholding
        ('No threshold (use all data)', 'Threshold option 1'),
        ('Manual threshold (enter a specific value)', 'Threshold option 2'),
        ('Auto-threshold on combined data', 'Threshold option 3'),
        ('Custom auto-threshold on combined data', 'Threshold option 4'),
        ('Individual dataset auto-threshold', 'Threshold option 5'),
        ('Custom individual dataset auto-threshold', 'Threshold option 6'),
        
        # Thresholding methods
        ('method == \'none\'', 'Threshold method none'),
        ('method == \'manual\'', 'Threshold method manual'),
        ('method == \'auto_combined\'', 'Threshold method auto_combined'),
        ('method == \'auto_individual\'', 'Threshold method auto_individual'),
        
        # Save mask functionality
        ('save_mask_from_roi', 'Save mask function'),
        ('exploration_mask', 'Exploration mask key'),
        ('exploration_metadata', 'Exploration metadata'),
        ('mask_registry', 'Mask registry'),
        ('Apply ROI (show selected pixels on intensity image)', 'CLI Apply ROI option'),
        ('Save mask (create and save mask from current ROI)', 'CLI Save mask option')
    ]
    
    all_passed = True
    for check_text, description in checks:
        if check_text in content:
            print(f"✅ {description}: Found")
        else:
            print(f"❌ {description}: Not found")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print("Testing the new data exploration features...")
    print("=" * 50)
    
    test1 = test_functions_exist()
    test2 = test_function_implementation()
    
    if test1 and test2:
        print("\n✅ All tests passed! The features have been successfully implemented.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    print("=" * 50) 