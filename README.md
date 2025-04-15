# FLIM-FRET Analysis Automation

This repository contains tools for automating Fluorescence Lifetime Imaging Microscopy (FLIM) and Förster Resonance Energy Transfer (FRET) analysis without requiring a GUI. It provides an end-to-end workflow from raw .bin files to complete FLIM-FRET analysis.

## Key Features

- Complete end-to-end workflow from raw .bin files to processed analysis
- Extracts and automates the Fast Fourier Transform (FFT) functionality from FLUTE
- Works with calibration files to properly process FLIM data
- Replicates all output files and naming conventions from the FLUTE GUI
- Performs median filtering, thresholding, and angle/circle range filtering
- Smart calibration matching between different file structures
- Python-based file organization for robust file handling
- Phasor transformation and visualization

## Main Components

- `run_pipeline.py`: The main pipeline orchestrator for end-to-end workflow
- `TCSPC_preprocessing_AUTOcal_v2_0.py`: Handles the preprocessing stage (ImageJ + FLUTE)
- `ComplexWaveletFilter_v2_0.py`: Performs advanced wavelet filtering with DTCWT and creates NPZ datasets
- `simplify_filenames.py`: Optional tool to convert complex filenames to simpler format
- `GMMSegmentation_v2_6.py`: Performs GMM-based segmentation and analysis
- `phasor_transform.py`: Performs phasor transformation without GUI dependencies
- `flim_fft_automated.py`: The main script that processes FLIM data using FFT
- `organize_output_files.py`: Organizes processed files into the required directory structure
- `calibration.csv`: Contains phi_cal and m_cal calibration values for data files
- ImageJ macros (`.ijm` files): Used for converting .bin files to .tif files

## Installation

Follow these steps to set up the FLIM-FRET analysis pipeline on your system:

### Prerequisites

1. **Python 3.8+** - We recommend using Python 3.8 or newer. [Download Python](https://www.python.org/downloads/)
2. **ImageJ/Fiji** - Required for .bin file conversion. [Download Fiji](https://fiji.sc/)
3. **Git** - For cloning the repository. [Download Git](https://git-scm.com/downloads)

### Step 1: Clone the Repository

```bash
git clone https://github.com/marcusjoshm/FLIM-FRET-analysis.git
cd FLIM-FRET-analysis
```

### Step 2: Create a Python Virtual Environment

Creating a virtual environment keeps the dependencies for this project separate from other Python projects.

#### On macOS and Linux:

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

#### On Windows:

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

After activation, your command prompt should show `(venv)` at the beginning of the line, indicating the virtual environment is active.

### Step 3: Install Dependencies

```bash
# Install all required packages from requirements.txt
pip install -r requirements.txt
```

This will install the following dependencies:
- numpy, pandas, scipy, matplotlib (data handling and visualization)
- scikit-image, tifffile, pillow (image processing)
- scikit-learn (machine learning for GMM segmentation)
- dtcwt (dual-tree complex wavelet transform for noise reduction)
- other utility packages

### Step 4: Configure the Pipeline

Create a `config.json` file in the root directory with the following structure:

```json
{
  "imagej_path": "/path/to/ImageJ/or/Fiji",
  "flute_path": "/path/to/FLUTE/executable",
  "flute_python_path": "/path/to/FLUTE/python/interpreter",
  "macro_files": [
    "/path/to/FLIM_processing_macro_1.ijm",
    "/path/to/FLIM_processing_macro_2.ijm"
  ],
  "microscope_params": {
    "bin_width": 0.097,
    "frequency": 78,
    "harmonic": 1
  }
}
```

Replace the paths with the actual paths on your system.

### Step 5: Verify Installation

Run the test mode to verify that all components are working properly:

```bash
python run_pipeline.py --test
```

### Troubleshooting

- **Import Errors**: Make sure your virtual environment is activated (`source venv/bin/activate` or `venv\Scripts\activate`)
- **ImageJ/FLUTE Path Errors**: Check that your `config.json` has the correct paths
- **Missing Dependencies**: If you encounter errors about missing packages, try installing them individually with `pip install package_name`
- **Permission Issues**: Make sure ImageJ/Fiji and macro files have execution permissions on Unix systems (`chmod +x /path/to/file`)

## Data Preparation Guide

To successfully run the FLIM-FRET analysis pipeline, your input data must be structured correctly. Follow these guidelines to ensure proper processing:

### Required Files

1. **FLIM .bin Files**: Your microscope acquisition data files
2. **FITC.bin**: Calibration reference file that must be present in your input directory
3. **calibration.csv**: Contains calibration values for each .bin file

### Directory Structure

The pipeline supports two different directory structures for organizing your .bin files:

#### 1. Hierarchical Structure (Recommended)

```
Input-Directory/
├── FITC.bin
├── calibration.csv
└── Dish_1_Post-Rapa/
    ├── R1/
    │   ├── R_1_s1.bin
    │   ├── R_1_s2.bin
    │   ├── R_1_s3.bin
    │   └── R_1_s4.bin
    ├── R2/
    │   ├── R_2_s1.bin
    │   └── ...
    └── R3/
        ├── R_3_s1.bin
        └── ...
```

In this structure, .bin files are organized into region folders (R1, R2, R3), which helps keep files organized when you have many samples.

#### 2. Flat Structure

```
Input-Directory/
├── FITC.bin
├── calibration.csv
└── Dish_1_Post-Rapa/
    ├── R_1_s1.bin
    ├── R_1_s2.bin
    ├── R_1_s3.bin
    ├── R_1_s4.bin
    ├── R_2_s1.bin
    └── ...
```

In this structure, all .bin files are in the same directory without region-specific folders.

### Calibration File Format

The `calibration.csv` file should contain the following columns:
- `file_path`: Path to the .bin file (can be relative to input directory)
- `phi_cal`: Phase calibration value
- `m_cal`: Modulation calibration value

Example:
```csv
file_path,phi_cal,m_cal
/Dish_1_Post-Rapa/R1/R_1_s1.bin,0.0135,0.98
/Dish_1_Post-Rapa/R1/R_1_s2.bin,0.0135,0.98
```

### Notes

- The pipeline now supports placing the `calibration.csv` file in either the project directory or the input directory, with input directory taking precedence if both exist.
- At least one level of subdirectory is required inside the input directory (e.g., "Dish_1_Post-Rapa").
- File naming should follow a consistent pattern (e.g., R_1_s2.bin where 1 is the region number and 2 is the sample number).
- When using the `--simplify-filenames` option, the script automatically detects your directory structure and handles naming accordingly.

## Usage

### Currently Implemented Workflow Options

The following workflow flags are fully implemented and tested:

```bash
# OPTION 1: Run preprocessing only (Stages 1-2A) - Convert files and organize them
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --preprocessing

# OPTION 2: Run complete processing pipeline (Stages 1-2B) - Preprocessing + wavelet filtering and lifetime calculation
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --processing

# OPTION 3: Run LF-specific workflow - Preprocessing with automatic filename simplification 
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --LF-preprocessing

# OPTION 4: Run only wavelet filtering (Stage 2B) - For already preprocessed data
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --filter
```

### Additional Options

You can still run all stages or use the legacy flags:

```bash
# Run complete workflow (all implemented stages)
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --all

# Legacy preprocessing flag (deprecated, use --preprocessing instead)
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --preprocess

# Future stages (not yet fully implemented)
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --segment
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --phasor
```

### Using flim_fft_automated.py Directly

```bash
# Process a single file
python flim_fft_automated.py --file /path/to/input.tif /path/to/output/dir --calibration calibration.csv

# Process a folder of files
python flim_fft_automated.py --folder /path/to/input/folder /path/to/output/dir --calibration calibration.csv
```

### Advanced Options

The script accepts several parameters that control how the FFT is processed:

```
--phi <value>           # Manual phase calibration value
--mod <value>           # Manual modulation calibration value
--bin <value>           # Bin width in nanoseconds (default: 0.097)
--freq <value>          # Laser frequency in MHz (default: 78)
--harmonic <value>      # Harmonic to use (default: 1)
--filter <size>         # Median filter size (0 for none)
--threshold <min> <max> # Intensity threshold range (default: 0 1000000)
--angle <min> <max>     # Angle range in degrees (default: 0 90)
--circle <min> <max>    # Modulation lifetime range in ns (default: 0 120)
```

## Calibration Matching

The script intelligently matches calibration values to input files using several strategies:

1. Direct path matching
2. Directory matching
3. Filename matching (with or without extensions)
4. Smart pattern matching for different directory structures
   - Specifically handles cases where calibration refers to `.bin` files in a different path than the `.tif` input files
   - Example: Can match `/Volumes/NX-01-A/FLIM_workflow_test_data/Dish_1_Post-Rapa/R1/R_1_s1.bin` calibration to `/Volumes/NX-01-A/FLIM_workflow_test_data_analysis/output/Dish_1_Post-Rapa/R1/R_1_s1.tif` input file

## Advanced Complex Wavelet Filtering

The pipeline incorporates an advanced noise reduction technique using the Dual-Tree Complex Wavelet Transform (DTCWT) to denoise FLIM-FRET data. This approach produces significantly improved lifetime measurements while preserving important structural details.

### Key Features of Wavelet Filtering

- **Anscombe Transform**: Stabilizes variance for better signal processing
- **Dual-Tree Complex Wavelet Transform**: Multi-resolution decomposition with directional selectivity
- **Local Noise Variance Estimation**: Adapts to varying noise levels across the image
- **Adaptive Coefficient Modification**: Uses sophisticated phi_prime function to preserve edges
- **Both Filtered and Unfiltered Results**: Enables comparison between denoised and original data

### NPZ File Structure

The NPZ files created by the wavelet filtering stage contain the following data arrays:

- `G`: Wavelet-filtered G coordinates (real part of phasor)
- `S`: Wavelet-filtered S coordinates (imaginary part of phasor)
- `A`: Intensity values
- `T`: Lifetime calculated from filtered G/S coordinates
- `GU`: Unfiltered G coordinates
- `SU`: Unfiltered S coordinates
- `TU`: Lifetime calculation from unfiltered G/S

### Controlling Wavelet Filtering

You can customize the wavelet filtering behavior in the `config.json` file:

```json
{
  "wavelet_params": {
    "filter_level": 9,
    "reference_g": 0.30227996721890404,
    "reference_s": 0.4592458920992018
  },
  "microscope_params": {
    "frequency": 78.0,
    "harmonic": 1
  }
}
```

- `filter_level`: Controls the depth of wavelet decomposition (higher values = more aggressive filtering)
- `reference_g` and `reference_s`: Reference fluorophore coordinates (typically from a fluorescent standard)
- `frequency`: Laser frequency in MHz
- `harmonic`: Harmonic used in the FLIM acquisition

## Pipeline Structure

The pipeline creates the following directory structure:

- `output/`: Contains the direct output from the preprocessing step (TIF conversions + phasor results)
- `preprocessed/`: Contains organized files for further processing with subdirectories:
  - `G_unfiltered/`: G (real) phasor coordinates
  - `S_unfiltered/`: S (imaginary) phasor coordinates
  - `Intensity/`: Intensity maps
- `npz_datasets/`: Contains processed NPZ datasets from wavelet filtering
- `segmented/`: Contains segmentation masks and outputs
- `plots/`: Contains visualization plots
- `lifetime_images/`: Contains extracted lifetime maps
- `phasor_output/`: Contains results from the phasor transformation

## Output Files

### Preprocessing Output

The preprocessing stage creates the following files for each processed input file:

- `*_g.tiff`: G (real) phasor coordinates
- `*_s.tiff`: S (imaginary) phasor coordinates
- `*_intensity.tiff`: Intensity map
- `*_mask.tiff`: Binary mask of valid pixels
- `*_taup.tiff`: Phase lifetime map
- `*_taum.tiff`: Modulation lifetime map
- `*_phasor.png`: Phasor plot visualization

### Wavelet Filtering Output

The wavelet filtering stage processes the G, S, and intensity files and creates NPZ files that contain:

- Filtered G and S coordinates for improved signal-to-noise ratio
- Unfiltered G and S coordinates for comparison
- Calculated lifetime values from both filtered and unfiltered data
- Intensity values from the input images
- Metadata about processing parameters

## Requirements

- Python 3.x
- Dependencies in `requirements.txt`
- ImageJ/Fiji with Bio-Formats plugin
- Configuration in `config.json` with paths to:
  - ImageJ executable
  - FLUTE path (if using original FLUTE functionality)
  - ImageJ macro files
