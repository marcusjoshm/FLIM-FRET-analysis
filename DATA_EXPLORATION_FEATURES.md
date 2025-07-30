# Data Exploration Module Features

## Overview
The data_exploration module has been enhanced with new interactive features that allow users to select between filtered and unfiltered phasor plots, choose mask sources, and apply various thresholding methods.

## New Features

### 1. Data Type Selection
Users can now choose between different data types for phasor plots:

- **Filtered data (G/S coordinates)** - Default option using processed G and S coordinates
- **Unfiltered data (GU/SU coordinates)** - Using unprocessed GU and SU coordinates  
- **Both** - Process each file twice, showing both filtered and unfiltered data

### 2. Mask Source Selection
Users can select how to handle masks:

- **No mask** - Use original data without any masking
- **Use masked NPZ files** - Apply masks from NPZ files (with automatic mask detection)

### 3. Thresholding Options
Users can apply various thresholding methods to the intensity data:

- **No threshold** - Use all data without filtering
- **Manual threshold** - Enter a specific threshold value
- **Auto-threshold on combined data** - Remove bottom 90% of intensity values across all data
- **Custom auto-threshold on combined data** - Specify percentile to remove from combined data
- **Individual dataset auto-threshold** - Remove bottom 90% from each dataset separately
- **Custom individual dataset auto-threshold** - Specify percentile to remove from each dataset

## Implementation Details

### New Functions Added

1. `interactive_data_type_selection()` - Handles data type selection
2. `interactive_mask_selection(npz_dir)` - Handles mask source selection
3. `interactive_threshold_selection()` - Handles thresholding method selection
4. `apply_thresholding(intensity_data, threshold_config)` - Applies thresholding to data

### Integration

The main function has been updated to:
- Call all interactive selection functions when in interactive mode
- Apply thresholding to intensity data before creating plots
- Handle mask selection and application
- Support processing both filtered and unfiltered data for the same files

### Stage Integration

The `DataExplorationStage` class has been updated to import and make available all new functions.

## Usage

To use these features, run the data exploration module:

```bash
python main.py --data-exploration
```

The module will prompt for:
1. Data type selection (filtered/unfiltered/both)
2. Mask source selection (none/masked files)
3. Thresholding method selection (6 options)

## Technical Notes

- All thresholding is applied to intensity data before creating phasor plots
- Mask selection automatically detects available masks in NPZ files
- The 'both' data type option processes each file twice (filtered then unfiltered)
- Threshold descriptions are displayed in plot titles for reference
- All features are backward compatible with existing functionality 