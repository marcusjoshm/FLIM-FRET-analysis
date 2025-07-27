# FLIM-FRET Analysis Automation Protocol

## Introduction
This document provides step-by-step instructions for running our FLIM-FRET analysis workflow. Each section includes detailed explanations and commands you can copy and paste directly into your Terminal.

**Note:** Terminal is Mac's command-line interface where you can type commands to interact with your computer.

## Preview

When you run the FLIM-FRET analysis tool, you'll see a colorful interactive menu like this:

```

      ███████╗██╗     ██╗███╗   ███╗    ███████╗██████╗ ███████╗████████╗ 
      ██╔════╝██║     ██║████╗ ████║    ██╔════╝██╔══██╗██╔════╝╚══██╔══╝ 
      █████╗  ██║     ██║██╔████╔██║    █████╗  ██████╔╝█████╗     ██║    
      ██╔══╝  ██║     ██║██║╚██╔╝██║    ██╔══╝  ██╔══██╗██╔══╝     ██║    
      ██║     ███████╗██║██║ ╚═╝ ██║    ██║     ██║  ██║███████╗   ██║    
      ╚═╝     ╚══════╝╚═╝╚═╝     ╚═╝    ╚═╝     ╚═╝  ╚═╝╚══════╝   ╚═╝    

  🔬 Welcome to my FLIM-FRET analysis tool! 🔬

MENU:
1. Set Input/Output Directories
2. Preprocessing (.bin to .tif)
3. Preprocessing + Processing (.bin to .npz)
4. Visualization (interactive phasor plots)
5. Segmentation (interactive phasor segmentation - GMM or manual)
6. Average Lifetime (calculate average lifetime from segmented data)
7. Lifetime Images (generate lifetime images from NPZ files)
8. Exit

Select an option (1-8):
```

*Note: In the actual terminal, the FLIM text appears in green and the FRET text appears in red, with all menu options displayed in yellow for better visibility. The menu now includes 8 options instead of 7.*

## Table of Contents
1. [Getting Started](#getting-started)
2. [Step 1: Remove Spaces from File Names](#step-1-remove-spaces-from-file-names)
3. [Step 2: Set Up Input Directory Structure](#step-2-set-up-input-directory-structure)
4. [Step 3: Create Calibration File](#step-3-create-calibration-file)
5. [Step 4: Navigate and Activate Environment](#step-4-navigate-and-activate-environment)
6. [Step 5: Run FLIM-FRET Analysis](#step-5-run-flim-fret-analysis)
7. [Tips and Troubleshooting](#tips-and-troubleshooting)
8. [Technical Documentation](#technical-documentation)

## Getting Started

### Export Data from Your Microscope
This workflow is designed to work with .bin files exported from time-correlated single photon counting (TCSPC) FLIM microscopes. Make sure you have:
- Raw .bin files from your FLIM acquisition
- Knowledge of the phi and modulation calibration values for your data

### Opening Terminal
1. Press **Command + Space** to open Spotlight Search
2. Type "Terminal"
3. Click on the Terminal application

When Terminal opens, you'll see a prompt that looks something like `(base) ➜  ~`

## Step 1: Remove Spaces from File Names

Our analysis workflow requires file paths without spaces. Follow these steps to convert spaces in your file names to underscores:

1. Copy and paste the following command into Terminal:
   ```bash
   cd ~/bash_scripts/
   ```
   Press **Enter**.

2. Next, copy and paste this command:
   ```bash
   ./replace_spaces.sh
   ```

3. Drag the folder containing your .bin files from Finder into the Terminal window. The file path will appear automatically.

4. Press **Enter**.

5. You will see a prompt asking for confirmation. Type `y` and press **Enter** to run the program.

6. When the process completes successfully, you will see a success message indicating that spaces have been removed from all file names.

## Step 2: Set Up Input Directory Structure

**⚠️ IMPORTANT:** Your .bin files **CANNOT** be placed directly in the root of your input directory. They **MUST** be organized in at least one level of subdirectories.

### Valid Directory Structure:
```
Your-Input-Directory/
├── calibration.csv             # Must be in root directory
└── Experiment_Data/            # At least one subdirectory required
    ├── sample1.bin
    ├── sample2.bin
    └── sample3.bin
```

### Organization Options:
- **By Experiment:** Create folders like `Control_Group/`, `Treatment_A/`, `Treatment_B/`
- **By Sample:** Create folders like `Sample_1/`, `Sample_2/`, `Sample_3/`
- **By Region:** Create folders like `Region_1/`, `Region_2/`, `Region_3/`

Choose whatever organization makes sense for your experiment!

## Step 3: Create Calibration File

You need to create a `calibration.csv` file in the root of your input directory:

1. Open a spreadsheet application (Excel, Numbers, or Google Sheets)

2. Create a CSV file with three columns:
   ```
   file_path,phi_cal,m_cal
   ```

3. For each .bin file, add a row with:
   - **file_path:** The absolute path to your .bin file
   - **phi_cal:** Phase calibration value (in radians, usually negative)
   - **m_cal:** Modulation calibration value (usually close to 1.0)

4. **To get the absolute file path:** Drag a .bin file from Finder into a text editor or Terminal - the full path will appear

### Example calibration.csv:
```csv
file_path,phi_cal,m_cal
/Volumes/Data/My_Experiment/Experiment_Data/sample1.bin,-0.621860812,0.9995
/Volumes/Data/My_Experiment/Experiment_Data/sample2.bin,-0.621860812,0.9995
/Volumes/Data/My_Experiment/Experiment_Data/sample3.bin,-0.621860812,0.9995
```

5. Save the file as `calibration.csv` in the root of your input directory

## Step 4: Navigate and Activate Environment

1. In Terminal, navigate to the FLIM-FRET analysis directory:
   ```bash
   cd ~/FLIM-FRET-analysis
   ```
   Press **Enter**.

2. Activate the Python virtual environment:
   ```bash
   source venv/bin/activate
   ```
   Press **Enter**.

3. You should now see `(venv)` at the beginning of your command prompt, indicating the environment is activated:
   ```
   (venv) ➜  FLIM-FRET-analysis
   ```

## Step 5: Run FLIM-FRET Analysis

Now you'll run the FLIM-FRET analysis using the interactive menu system:

1. Simply run the main script:
   ```bash
   python main.py
   ```

2. You'll see the colorful FLIM-FRET menu with options:
   ```
   MENU:
   1. Set Input/Output Directories
   2. Preprocessing (.bin to .tif)
   3. Preprocessing + Processing (.bin to .npz)
   4. Visualization (interactive phasor plots)
   5. Segmentation (interactive phasor segmentation - GMM or manual)
   6. Average Lifetime (calculate average lifetime from segmented data)
   7. Lifetime Images (generate lifetime images from NPZ files)
   8. Exit
   ```

3. **For first-time setup**, start with option 1 to set your input and output directories.

4. **For complete processing**, use option 3 which will:
   - Convert .bin files to .tif files using ImageJ
   - Perform phasor transformation to generate G, S, and intensity maps
   - Apply complex wavelet filtering for noise reduction
   - Create NPZ datasets with both filtered and unfiltered lifetime data

5. **For analysis**, use options 4-7 to:
   - Visualize your data with interactive phasor plots
   - Perform segmentation (GMM or manual)
   - Calculate average lifetime from segmented data
   - Generate lifetime images from NPZ files

### What Each Option Does:

- **Option 1:** Set input/output directories (required for first-time setup)
- **Option 2:** Preprocessing only (.bin to .tif conversion and organization)
- **Option 3:** Complete processing pipeline (.bin to .npz with wavelet filtering)
- **Option 4:** Interactive phasor visualization
- **Option 5:** Interactive segmentation (GMM clustering or manual ellipse selection)
- **Option 6:** Calculate average lifetime from segmented NPZ files
- **Option 7:** Generate lifetime images from NPZ files
- **Option 8:** Exit the program

The analysis will create an output folder with the following structure:
```
Your_Output_Directory/
├── output/                     # Raw converted files
├── preprocessed/              # Organized G, S, intensity files
├── npz_datasets/             # Final processed datasets
├── segmented/                 # Segmentation results
├── segmented_npz_datasets/   # NPZ files with segmentation masks
├── average_lifetime_results/  # Average lifetime calculations
├── lifetime_images/           # Generated lifetime images
└── logs/                     # Analysis logs and reports
```

## Tips and Troubleshooting

### Common Issues:
- **"File not found" errors:** Make sure all paths in calibration.csv are correct
- **"Directory structure" errors:** Ensure .bin files are in subdirectories, not the root
- **"Permission denied":** Make sure the remove_spaces.sh script is executable

### Success Indicators:
- You see processing messages for each file
- No error messages in the terminal
- Output directories are created with files inside

### If Something Goes Wrong:
1. Check the log files in the `logs/` directory of your output folder
2. Verify your calibration.csv file format
3. Make sure your directory structure follows the requirements
4. Ensure all file paths are correct and accessible

---

# Technical Documentation

This repository contains tools for automating Fluorescence Lifetime Imaging Microscopy (FLIM) and Förster Resonance Energy Transfer (FRET) analysis without requiring a GUI. It provides an end-to-end workflow from raw .bin files to complete FLIM-FRET analysis.




## --LF-preprocessed Workflow

For Noah and Leyla, if you would like to process raw .bin files from LASX without having to use FLUTE, here is a workflow that will work seemlessly with your FFF python scripts. All you need to do is set up your input directory following the instructions below and run the script from ther Terminal app.

### Setting up the Input Directory

- **All `.bin` files must be in subdirectories** within the input directory - they cannot be placed directly in the root directory
- You can organize these subdirectories however you want - by experimental condition, sample type, region, etc.
- **`FITC.bin`** can be placed anywhere in the directory tree (root or subdirectories)
- **`calibration.csv`** must be in the root input directory with the phi and modulation values entered for every `.bin` file

## Creating the Calibration File

Enter the file absolute file path, phi, and modulation values for every file that needs to be preprocessed. You can find the absolute file path by dragging a file from a finder window into an open terminal, then copy and paste it into the spreadsheet. Use the same phi and modulation values for every file from the same tile-scan. Convert the phi angles from degrees to radians (must be negative).

## Running the Script

Open a new terminal and enter:
```bash
cd ~/FLIM-FRET-analysis
source venv/bin/activate
```

To run the script enter the following in the terminal:
```bash
python run_pipeline.py --input-dir {/path/to/input/dir} --output-base-dir {/path/to/output/dir} --preprocessing
```

Change `{/path/to/input/dir}` to the actual path of your input directory that contains the raw data `.bin` files, `FITC.bin` file, and `calibration.csv` file. Change `{/path/to/output/dir}` to a location you want the output to go. It doesn't have to exist, the script will create the directory and save the output files to that location.

When the script finishes running, your preprocessed files will be in a folder called "preprocessed" in your output folder. You can then use those files with your current FFF python scripts. Simply copy preprocessed from your output directory to `~/FLIM_processing_dir/`




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
  - Uses the **dtcwt** Python package for sophisticated noise reduction
  - Implements Anscombe transform for variance stabilization
  - Produces both filtered and unfiltered lifetime calculations

- `GMMSegmentation_v2_6.py`: Performs GMM-based segmentation and analysis
- `ManualSegmentation.py`: Interactive manual ellipse-based segmentation
- `lifetime_images.py`: Interactive lifetime image generation from NPZ files with file selection
- `generate_lifetime_images.py`: Extracts lifetime data from NPZ files and saves as TIFF images *(DEPRECATED: Use lifetime_images.py instead)*
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
4. **Python Libraries** - Several specialized Python packages are required:
   - **dtcwt** (≥0.12.0) - Required for advanced wavelet filtering
   - See requirements.txt for complete dependency list

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
- **dtcwt** (Dual-Tree Complex Wavelet Transform) - **CRITICAL** for the advanced wavelet filtering in Stage 2B
- other utility packages

> **Note:** The dtcwt package is essential for the Complex Wavelet Filter functionality. This advanced filtering uses dual-tree complex wavelets to significantly improve noise reduction compared to traditional methods while preserving important signal features.

### Step 4: Configure the Pipeline

Run the automated setup script to configure the pipeline:

```bash
python setup.py
```

This script will:
- ✅ Check your Python version and required packages
- 🔍 Automatically detect ImageJ/Fiji and FLUTE installations
- 📝 Guide you through configuration parameters
- 🧪 Test the ImageJ connection
- 💾 Generate a `config.json` file with your settings

Alternatively, you can manually create a `config.json` file using the template:

```bash
cp config/config.template.json config/config.json
```

Then edit `config/config.json` with your specific paths and parameters.

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

### Directory Structure Requirements

**⚠️ IMPORTANT:** .bin files **CANNOT** be placed directly in the root of the input directory. They **MUST** be organized in at least one level of subdirectories within the input folder.

The pipeline supports flexible directory structures for organizing your .bin files, as long as they follow this basic rule:

#### ✅ Supported Directory Organizations

**1. Hierarchical Structure (Recommended)**
```
Input-Directory/
├── FITC.bin                    # Can be in root or subdirectories
├── calibration.csv             # Must be in root directory
└── Dish_1_Post-Rapa/           # At least one subdirectory required
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

**2. Flat Structure**
```
Input-Directory/
├── FITC.bin                    # Can be in root or subdirectories
├── calibration.csv             # Must be in root directory
└── Experiment_Data/            # At least one subdirectory required
    ├── R_1_s1.bin
    ├── R_1_s2.bin
    ├── R_1_s3.bin
    ├── R_1_s4.bin
    ├── R_2_s1.bin
    └── ...
```

**3. Multiple Experiment Folders**
```
Input-Directory/
├── FITC.bin
├── calibration.csv
├── Experiment_A/
│   ├── sample1.bin
│   └── sample2.bin
├── Experiment_B/
│   ├── sample3.bin
│   └── sample4.bin
└── Control_Group/
    ├── control1.bin
    └── control2.bin
```

#### ❌ Invalid Directory Structure

```
Input-Directory/
├── FITC.bin
├── calibration.csv
├── R_1_s1.bin              # ❌ WRONG: .bin files cannot be in root
├── R_1_s2.bin              # ❌ WRONG: .bin files cannot be in root
└── R_1_s3.bin              # ❌ WRONG: .bin files cannot be in root
```

#### Key Rules

1. **Subdirectory Requirement**: All .bin files (except FITC.bin) must be in at least one subdirectory
2. **Flexible Organization**: You can organize subdirectories however you want - by experiment, by sample, by region, etc.
3. **Multiple Levels**: You can have multiple levels of subdirectories (e.g., `Experiment/Region/Sample/`)
4. **FITC.bin Placement**: FITC.bin can be placed anywhere in the directory tree (root or subdirectories)
5. **Calibration File**: calibration.csv must be in the root input directory

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

### Important Notes

- **Directory Structure**: .bin files **MUST** be organized in subdirectories within the input folder - they cannot be placed directly in the root directory
- **Calibration File**: The `calibration.csv` file can be placed in either the project directory or the input directory, with input directory taking precedence if both exist
- **Flexible Organization**: You can organize your subdirectories however makes sense for your workflow - by experiment, condition, sample, region, etc.
- **File Naming**: While there's no strict naming requirement, consistent patterns help (e.g., R_1_s2.bin where 1 is the region number and 2 is the sample number)
- **Filename Simplification**: When using the `--simplify-filenames` option, the script automatically detects your directory structure and handles naming accordingly

## Usage

### Interactive Menu System (Recommended)

The easiest way to use the FLIM-FRET analysis pipeline is through the interactive menu system:

```bash
python main.py
```

This will launch the interactive menu with 8 options:

1. **Set Input/Output Directories** - Configure paths for your data
2. **Preprocessing (.bin to .tif)** - Convert and organize files
3. **Preprocessing + Processing (.bin to .npz)** - Complete pipeline with wavelet filtering
4. **Visualization (interactive phasor plots)** - Interactive data visualization
5. **Segmentation (interactive phasor segmentation)** - GMM or manual segmentation
6. **Average Lifetime** - Calculate average lifetime from segmented data
7. **Lifetime Images** - Generate lifetime images from NPZ files
8. **Exit** - Close the program

### Command Line Options (Legacy)

For advanced users, the following command-line options are still available:

```bash
# OPTION 1: Run preprocessing only (Stages 1-2A) - Convert files and organize them
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --preprocessing

# OPTION 2: Run complete processing pipeline (Stages 1-2B) - Preprocessing + wavelet filtering and lifetime calculation
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --processing

# OPTION 3: Run only wavelet filtering (Stage 2) - For already preprocessed data
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --filter

# OPTION 4: Generate lifetime images from NPZ files (Stage 4C) - Extract lifetime data as TIFF images
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --lifetime-images

# OPTION 5: Calculate average lifetime from segmented data (Stage 4D) - Calculate average lifetime from segmented NPZ files
python run_pipeline.py --input-dir /path/to/raw/bin/files --output-base-dir /path/to/output/directory --average-lifetime
```

### Additional Legacy Options

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
- `segmented_npz_datasets/`: Contains NPZ files with original data plus segmentation masks
- `plots/`: Contains visualization plots
- `lifetime_images/`: Contains extracted lifetime maps
- `average_lifetime_results/`: Contains CSV files with average lifetime statistics

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

### Lifetime Image Generation

The lifetime image generation stage extracts lifetime data from NPZ files and saves them as TIFF images for visualization and further analysis. This stage can process both individual NPZ files and entire directories.

#### Features

- **Multiple Lifetime Data Types**: Supports various lifetime data keys including `lifetime`, `tau_p`, `tau_m`, and `lifetime_map`
- **Automatic Data Handling**: Converts object arrays to float, handles NaN/infinite values, and reshapes data appropriately
- **TIFF Output Only**: Saves lifetime data as TIFF images without preview plots
- **Directory Structure Preservation**: Maintains the original directory structure when processing multiple files
- **Robust Error Handling**: Continues processing even if individual files fail

#### Output Structure

The lifetime image generation saves TIFF files directly to the `lifetime_images` directory:

```
lifetime_images/
├── file1_TU.tiff               # Lifetime image as TIFF
├── subdirectory/
│   ├── file2_TU.tiff          # Lifetime image as TIFF
└── ...
```

#### Usage

```bash
# Generate lifetime images from NPZ files
python run_pipeline.py --input /path/to/input --output /path/to/output --lifetime-images

# Run as standalone script
python src/python/modules/generate_lifetime_images.py /path/to/npz/files /path/to/output/directory

# Generate with preview plots (standalone only)
python src/python/modules/generate_lifetime_images.py /path/to/npz/files /path/to/output/directory
```

### Average Lifetime Calculation

The average lifetime calculation stage extracts TU (unfiltered lifetime) data from segmented NPZ files, applies the segmentation mask, and calculates average lifetime statistics for each segmented region.

#### Features

- **Masked Lifetime Calculation**: Multiplies TU data by the full_mask to get only selected region lifetime values
- **Comprehensive Statistics**: Calculates mean, standard deviation, min, max, and pixel counts
- **Data Validation**: Filters out NaN and infinite values for accurate statistics
- **CSV Output**: Saves results to a structured CSV file with all statistics and metadata
- **Summary Statistics**: Provides overall statistics across all processed files

#### Output Structure

The average lifetime calculation creates:

```
average_lifetime_results/
└── average_lifetime_results.csv  # CSV file with all statistics
```

The CSV file contains columns:
- `filename`: Base filename of the segmented NPZ file
- `average_lifetime_ns`: Mean lifetime in nanoseconds
- `std_lifetime_ns`: Standard deviation of lifetime
- `min_lifetime_ns`: Minimum lifetime value
- `max_lifetime_ns`: Maximum lifetime value
- `pixel_count`: Number of pixels in masked region
- `valid_pixel_count`: Number of valid lifetime pixels
- `total_pixels`: Total pixels in the dataset
- `pixels_selected`: Metadata from segmentation
- `pixels_thresholded`: Metadata from segmentation
- `mask_type`: Type of mask used
- `source_file`: Path to original segmented NPZ file

#### Usage

```bash
# Calculate average lifetime from segmented data
python run_pipeline.py --input /path/to/input --output /path/to/output --average-lifetime

# Run as standalone script
python src/python/modules/calculate_average_lifetime.py /path/to/segmented_npz_datasets /path/to/output/directory
```

## Requirements

- Python 3.x
- Dependencies in `requirements.txt`
- ImageJ/Fiji with Bio-Formats plugin
- Configuration in `config.json` with paths to:
  - ImageJ executable
  - FLUTE path (if using original FLUTE functionality)
  - ImageJ macro files
