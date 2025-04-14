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
- `ComplexWaveletFilter_v1_6.py`: Performs wavelet filtering and creates NPZ datasets
- `GMMSegmentation_v2_6.py`: Performs GMM-based segmentation and analysis
- `phasor_transform.py`: Performs phasor transformation without GUI dependencies
- `flim_fft_automated.py`: The main script that processes FLIM data using FFT
- `organize_output_files.py`: Organizes processed files into the required directory structure
- `calibration.csv`: Contains phi_cal and m_cal calibration values for data files
- ImageJ macros (`.ijm` files): Used for converting .bin files to .tif files

## Usage

### Complete End-to-End Workflow

To run the complete workflow from raw .bin files to processed analysis:

```bash
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --all
```

### Run Specific Stages

You can run specific stages of the pipeline:

```bash
# Run just the preprocessing stage
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --preprocess

# Run just the wavelet filtering stage
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --filter

# Run just the GMM segmentation stage
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --segment

# Run just the phasor transformation stage
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

## Pipeline Structure

The pipeline creates the following directory structure:

- `output/`: Contains the direct output from the preprocessing step (TIF conversions + phasor results)
- `preprocessed/`: Contains organized files for further processing with subdirectories:
  - `G_unfiltered/`: G (real) phasor coordinates
  - `S_unfiltered/`: S (imaginary) phasor coordinates
  - `Intensity/`: Intensity maps
- `npz_datasets/`: Contains processed NPZ datasets for analysis
- `segmented/`: Contains segmentation masks and outputs
- `plots/`: Contains visualization plots
- `lifetime_images/`: Contains extracted lifetime maps
- `phasor_output/`: Contains results from the phasor transformation

## Output Files

The script creates the following files for each processed input file:

- `*_g.tiff`: G (real) phasor coordinates
- `*_s.tiff`: S (imaginary) phasor coordinates
- `*_intensity.tiff`: Intensity map
- `*_mask.tiff`: Binary mask of valid pixels
- `*_taup.tiff`: Phase lifetime map
- `*_taum.tiff`: Modulation lifetime map
- `*_phasor.png`: Phasor plot visualization

## Requirements

- Python 3.x
- Dependencies in `requirements.txt`
- ImageJ/Fiji with Bio-Formats plugin
- Configuration in `config.json` with paths to:
  - ImageJ executable
  - FLUTE path (if using original FLUTE functionality)
  - ImageJ macro files
