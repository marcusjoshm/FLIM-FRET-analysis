# FLIM-FRET Analysis Automation

This repository contains tools for automating Fluorescence Lifetime Imaging Microscopy (FLIM) and Förster Resonance Energy Transfer (FRET) analysis without requiring a GUI.

## Key Features

- Extracts and automates the Fast Fourier Transform (FFT) functionality from FLUTE
- Works with calibration files to properly process FLIM data
- Replicates all output files and naming conventions from the FLUTE GUI
- Performs median filtering, thresholding, and angle/circle range filtering
- Smart calibration matching between different file structures

## Main Components

- `flim_fft_automated.py`: The main script that processes FLIM data using FFT
- `calibration.csv`: Contains phi_cal and m_cal calibration values for data files

## Usage

### Process a Single File

```bash
python flim_fft_automated.py --file /path/to/input.tif /path/to/output/dir --calibration calibration.csv
```

### Process a Folder of Files

```bash
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

## Output Files

The script creates the following files for each processed input file:

- `*_g.tif`: G (real) phasor coordinates
- `*_s.tif`: S (imaginary) phasor coordinates
- `*_intensity.tif`: Intensity map
- `*_mask.tif`: Binary mask of valid pixels
- `*_taup.tif`: Phase lifetime map
- `*_taum.tif`: Modulation lifetime map
- `*_phasor.png`: Phasor plot visualization

All output files are saved in a `FLUTE_output` directory inside the specified output directory.
