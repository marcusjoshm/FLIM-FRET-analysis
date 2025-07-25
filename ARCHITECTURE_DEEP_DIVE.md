# FLIM-FRET Pipeline Architecture Deep Dive
## Step-by-Step Execution Flow Analysis

This document provides a detailed trace through the new refactored architecture, showing exactly what happens when you run the pipeline, which files are loaded, and how the execution flows through the codebase.

---

## 🚀 **EXECUTION START: `python main.py --input ... --output ...`**

When you execute the pipeline, here's exactly what happens:

### **Step 1: Entry Point - `main.py`**
```python
# File: main.py
# Lines executed first:

#!/usr/bin/env python3
"""
FLIM-FRET Analysis Pipeline - Main Entry Point
...
"""

# Line 13-20: Import statements execute immediately
import sys
import os
from pathlib import Path
from src.python.core.config import Config, ConfigError
from src.python.core.logger import PipelineLogger
from src.python.core.cli import parse_arguments, CLIError
from src.python.core.pipeline import Pipeline
```

**🔍 DEBUGGING BREAK: Import Cascade**
When these imports execute, they trigger a cascade of imports:

#### **Import 1: `src.python.core.config`**
```python
# File: src/python/core/config.py
# Lines 1-10: Module initialization
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
```

#### **Import 2: `src.python.core.logger`**
```python
# File: src/python/core/logger.py
# Lines 1-15: Module initialization
import logging
import sys
import os
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
```

#### **Import 3: `src.python.core.cli`**
```python
# File: src/python/core/cli.py
# Lines 1-10: Module initialization
import argparse
import sys
from typing import Dict, Any, List
```

#### **Import 4: `src.python.core.pipeline`** ⚠️ **CRITICAL IMPORT CASCADE**
```python
# File: src/python/core/pipeline.py
# Lines 1-16: This triggers the BIGGEST import cascade

from pathlib import Path
from typing import List, Dict, Any
from .config import Config
from .logger import PipelineLogger  
from .stages import StageRegistry, StageExecutor, StageError
from ..modules import register_all_stages  # ← THIS IS THE BIG ONE!
```

**🔥 MASSIVE IMPORT CASCADE: `..modules import register_all_stages`**

This single import triggers loading of ALL pipeline modules:

```python
# File: src/python/modules/__init__.py
# Line 12: 
from .stage_registry import register_all_stages

# File: src/python/modules/stage_registry.py
# Lines 7-23: ALL stage imports happen here
from .preprocessing import PreprocessingStage
from .wavelet_filter import WaveletFilterStage  
from .phasor_transform import PhasorTransformStage
from .phasor_visualization import PhasorVisualizationStage
# ... and 15+ more stage imports
```

Each stage import loads its dependencies:

```python
# File: src/python/modules/preprocessing.py
# Lines 11-13:
from ..core.stages import StageBase
from ..core.config import Config
from ..core.logger import PipelineLogger

# Lines 25-30: Critical import of original preprocessing
try:
    from .TCSPC_preprocessing_AUTOcal_v2_0 import run_preprocessing
    self.run_preprocessing = run_preprocessing
```

---

### **Step 2: Function Definitions Load**
```python
# File: main.py
# Lines 22-91: Function definitions are loaded into memory

def main():
    try:
        # This is where execution will jump to later
        args = parse_arguments()  # ← Will call CLI module
        # ...
```

### **Step 3: Module-Level Execution**
```python
# File: main.py  
# Lines 93-95: This executes immediately after imports
if __name__ == "__main__":
    main()  # ← Execution jumps to main() function
```

---

## 🎯 **MAIN EXECUTION FLOW**

### **Step 4: `main()` Function Execution**

```python
# File: main.py, Line 23
def main():
    try:
        # Line 25: Parse command line arguments
        args = parse_arguments()  # ← JUMPS TO CLI MODULE
```

**🔍 DEBUGGING BREAK: CLI Parsing**
```python
# File: src/python/core/cli.py
# Line 45: parse_arguments() function executes

def parse_arguments():
    # Line 50: Create argument parser
    parser = argparse.ArgumentParser(...)
    
    # Lines 52-85: Add all argument definitions
    parser.add_argument('--input', required=True, ...)
    parser.add_argument('--output', required=True, ...)
    parser.add_argument('--processing', action='store_true', ...)
    # ... 15+ more arguments
    
    # Line 87: Parse the actual command line
    args = parser.parse_args()
    
    # Lines 89-95: Interactive menu if no specific action
    if not any([args.preprocessing, args.filter, args.processing, ...]):
        cli = PipelineCLI()  # ← Create CLI instance
        selection = cli.show_menu()  # ← Show interactive menu
        args = cli.apply_menu_selection(args, selection)
    
    return args  # ← RETURNS TO main.py
```

**🔍 DEBUGGING BREAK: Interactive Menu** (if triggered)
```python
# File: src/python/core/cli.py
# Line 15: PipelineCLI class instantiation

class PipelineCLI:
    def show_menu(self):
        # Lines 25-60: Display the beautiful menu
        print("==============================")
        print("      FLIM-FRET Analysis")
        print("==============================")
        # ... menu options
        
        # Line 70: Get user input
        choice = input("Select an option (1-16): ")
        return int(choice)
```

### **Step 5: Configuration Loading**
```python
# File: main.py, Line 26-28
config_path = args.config if hasattr(args, 'config') and args.config else None
config = Config(config_path)  # ← JUMPS TO CONFIG MODULE
config.validate()
```

**🔍 DEBUGGING BREAK: Config Initialization**
```python
# File: src/python/core/config.py
# Line 20: Config.__init__()

def __init__(self, config_path: Optional[str] = None):
    # Line 25: Load default configuration
    self.config = self._load_defaults()
    
    # Lines 27-32: Load from file if provided
    if config_path and os.path.exists(config_path):
        file_config = self._load_from_file(config_path)
        self.config.update(file_config)
    
    # Line 75: _load_defaults() method
    def _load_defaults(self) -> Dict[str, Any]:
        return {
            "microscope_params": {
                "bin_width_ns": 0.097,
                "freq_mhz": 78.0,
                "harmonic": 1
            },
            "wavelet_params": {
                "filter_level": 9,
                "reference_g": 0.30227996721890404,
                "reference_s": 0.4592458920992018
            },
            # ... more defaults
        }
```

### **Step 6: Logger Initialization**
```python
# File: main.py, Line 30
logger = PipelineLogger(args.output, args.log_level)  # ← JUMPS TO LOGGER
```

**🔍 DEBUGGING BREAK: Logger Setup**
```python
# File: src/python/core/logger.py
# Line 20: PipelineLogger.__init__()

def __init__(self, output_dir: str, log_level: str = "INFO"):
    # Line 25: Setup directory structure  
    self.output_dir = Path(output_dir)
    self.log_dir = self.output_dir / "logs"
    self.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Lines 30-45: Configure Python logging
    log_file = self.log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    self.logger = logging.getLogger(__name__)
```

### **Step 7: Pipeline Creation and Execution**
```python
# File: main.py, Lines 35-36
pipeline = Pipeline(config, logger, args)  # ← JUMPS TO PIPELINE MODULE
success = pipeline.run()  # ← MAIN EXECUTION STARTS HERE
```

---

## 🏗️ **PIPELINE ARCHITECTURE EXECUTION**

### **Step 8: Pipeline Initialization**
```python
# File: src/python/core/pipeline.py
# Line 25: Pipeline.__init__()

def __init__(self, config: Config, logger: PipelineLogger, args):
    # Line 30: Store references
    self.config = config
    self.logger = logger  
    self.args = args
    
    # Line 35: Setup stage system
    self.registry = StageRegistry()
    register_all_stages()  # ← REGISTER ALL AVAILABLE STAGES
    self.executor = StageExecutor(config, logger)
    
    # Line 40: Setup directories
    self.directories = self._setup_directories()
```

**🔍 DEBUGGING BREAK: Stage Registration**
```python
# File: src/python/modules/stage_registry.py
# Line 25: register_all_stages() function

def register_all_stages():
    # Lines 28-51: Register each stage with the registry
    register_stage('preprocessing', order=1)(PreprocessingStage)
    register_stage('phasor_transform', order=2)(PhasorTransformStage)  
    register_stage('wavelet_filter', order=3)(WaveletFilterStage)
    # ... 15+ more stage registrations
```

### **Step 9: Pipeline Execution - `pipeline.run()`**
```python
# File: src/python/core/pipeline.py
# Line 55: Pipeline.run()

def run(self) -> bool:
    # Line 60: Determine which stages to run
    stages = self._determine_stages()  # ← Based on CLI args
    
    # Line 65: Log pipeline start
    self.logger.info("FLIM-FRET Analysis Pipeline Starting")
    self.logger.info(f"Stages to run: {', '.join(stages)}")
    
    # Line 70: Execute each stage
    for stage_name in stages:
        self.logger.info(f"Executing stage: {stage_name}")
        success = self.executor.execute(stage_name, **stage_kwargs)
        if not success:
            self.logger.error(f"Stage failed: {stage_name}")
            return False
    
    return True
```

**🔍 DEBUGGING BREAK: Stage Determination**
```python
# File: src/python/core/pipeline.py  
# Line 80: _determine_stages()

def _determine_stages(self) -> List[str]:
    stages = []
    
    # Lines 85-120: Logic based on CLI arguments
    if self.args.all:
        stages = self.registry.get_stage_order()  # All stages in order
    else:
        if self.args.preprocessing or self.args.processing:
            stages.append('preprocessing')
        if self.args.filter or self.args.processing:  
            stages.append('wavelet_filter')
        # ... more conditional logic
    
    return stages  # e.g., ['preprocessing', 'wavelet_filter']
```

---

## ⚙️ **STAGE EXECUTION DEEP DIVE**

### **Step 10: Stage Executor**
```python
# File: src/python/core/stages.py
# Line 140: StageExecutor.execute()

def execute(self, stage_name: str, **kwargs) -> bool:
    # Line 145: Get stage class from registry
    stage_class = self.registry.get_stage(stage_name)
    
    # Line 150: Create stage instance
    stage = stage_class(self.config, self.logger, stage_name)
    
    # Line 155: Validate inputs
    if not stage.validate_inputs(**kwargs):
        return False
    
    # Line 160: Execute the stage
    try:
        success = stage.run(**kwargs)  # ← JUMPS TO SPECIFIC STAGE
        return success
    except Exception as e:
        self.logger.error(f"ERROR in {stage_name}: {e}")
        return False
```

### **Step 11: Example - Preprocessing Stage Execution**
```python
# File: src/python/modules/preprocessing.py
# Line 45: PreprocessingStage.run()

def run(self, **kwargs) -> bool:
    # Lines 50-70: Interactive file selection prompt
    print("\n=== Preprocessing File Selection ===")
    print("Choose how to select files for preprocessing:")
    print("  [1] Process all .bin files (default)")
    print("  [2] Select specific .bin files interactively")
    
    # Line 75: Get user choice
    choice = input("Select option (1 or 2, default: 1): ").strip()
    
    # Lines 80-95: Branch based on choice
    if choice == "2":
        success = self._run_interactive_preprocessing(...)
    else:
        success = self._run_batch_preprocessing(...)
```

**🔍 DEBUGGING BREAK: Interactive Preprocessing**
```python
# File: src/python/modules/preprocessing.py
# Line 110: _run_interactive_preprocessing()

def _run_interactive_preprocessing(self, ...):
    # Line 115: Call original preprocessing function
    success = self.run_preprocessing(
        self.config.to_dict(),
        data_dir,
        str(directories['output']),
        str(directories['preprocessed']),
        calibration_file,
        raw_data_root,
        interactive_file_selection=True  # ← Key parameter
    )
```

**🔍 DEBUGGING BREAK: Original Preprocessing Call**
```python
# File: src/python/modules/TCSPC_preprocessing_AUTOcal_v2_0.py
# Line 628: run_preprocessing() function

def run_preprocessing(config, input_dir, output_dir, preprocessed_dir, 
                     calibration_file, raw_data_root, interactive_file_selection=False):
    
    # Lines 650-700: File selection logic
    if interactive_file_selection:
        # Lines 660-680: Interactive file selection
        selected_files = prompt_file_selection(input_dir)
        # Create filtered calibration file
        # Run ImageJ macros on selected files only
    else:
        # Lines 690-700: Process all files
        # Run ImageJ macros on all .bin files
    
    # Lines 750-780: Phasor transformation
    phasor_success = process_tiffs_with_phasor_transform(
        calibration_file, 
        output_dir, 
        raw_data_root, 
        microscope_params
    )
```

**🔍 DEBUGGING BREAK: Phasor Transform Import**
```python
# File: src/python/modules/TCSPC_preprocessing_AUTOcal_v2_0.py
# Lines 444-460: Dynamic import resolution

try:
    # Try new package structure first
    try:
        from . import phasor_transform  # ← NEW STRUCTURE
        print("Successfully imported phasor_transform module (new structure)")
    except ImportError:
        # Fallback to legacy import
        modules_dir = os.path.dirname(os.path.abspath(__file__))
        if modules_dir not in sys.path:
            sys.path.insert(0, modules_dir)
        import phasor_transform  # ← LEGACY FALLBACK
```

**🔍 DEBUGGING BREAK: Phasor Processing**
```python
# File: src/python/modules/phasor_transform.py
# Line 20: process_flim_file() function

def process_flim_file(input_file: str, output_dir: str, phi_cal: float, 
                     m_cal: float, bin_width_ns: float, freq_mhz: float, ...):
    
    # Lines 35-55: File loading
    if input_file.endswith(('.tif', '.tiff')):
        with Image.open(input_file) as img:
            if hasattr(img, 'n_frames'):
                # Multi-frame TIFF processing
                frames = []
                for i in range(img.n_frames):
                    img.seek(i)
                    frames.append(np.array(img))
                flim_data = np.stack(frames, axis=-1)
    
    # Lines 70-120: Phasor calculation
    omega = 2 * np.pi * freq_mhz * 1e6 * bin_width_ns * 1e-9 * harmonic
    
    for y in range(height):
        for x in range(width):
            pixel_data = flim_data[y, x, :]
            pixel_intensity = np.sum(pixel_data)
            
            # Calculate G and S coordinates
            cos_component = np.sum(pixel_data * np.cos(omega * time_indices))
            sin_component = np.sum(pixel_data * np.sin(omega * time_indices))
            
            # Apply calibration
            g_corrected = (g_raw * m_cal * np.cos(phi_cal) - s_raw * m_cal * np.sin(phi_cal))
            s_corrected = (g_raw * m_cal * np.sin(phi_cal) + s_raw * m_cal * np.cos(phi_cal))
```

### **Step 12: Wavelet Filter Stage**
```python
# File: src/python/modules/wavelet_filter.py
# Line 25: WaveletFilterStage.run()

def run(self, **kwargs) -> bool:
    # Line 30: Call original wavelet filter
    from .ComplexWaveletFilter_v2_0 import main as run_wavelet_filter
    
    # Line 35: Execute with parameters
    success = run_wavelet_filter(
        preprocessed_dir=str(directories['preprocessed']),
        output_dir=str(directories['npz_datasets']),
        flevel=self.config.get('wavelet_params.filter_level', 9),
        ref_g=self.config.get('wavelet_params.reference_g'),
        ref_s=self.config.get('wavelet_params.reference_s'),
        omega=calculated_omega
    )
```

**🔍 DEBUGGING BREAK: Complex Wavelet Processing**
```python
# File: src/python/modules/ComplexWaveletFilter_v2_0.py
# Line 651: main() function

def main(preprocessed_dir, output_dir, flevel=9, ref_g=REF_G, ref_s=REF_S, omega=None):
    
    # Lines 670-690: Directory scanning
    for root, dirs, files in os.walk(preprocessed_dir):
        g_files = [f for f in files if f.endswith('_g.tiff')]
        s_files = [f for f in files if f.endswith('_s.tiff')]
        intensity_files = [f for f in files if f.endswith('_intensity.tiff')]
    
    # Lines 700-750: Dataset processing
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        
        # Line 720: Core processing function
        result = process_dataset(g_file, s_file, int_file, flevel, omega, ref_g, ref_s)
        
        # Lines 740-750: Save NPZ file
        npz_path = os.path.join(output_dir, f"{dataset_name}.npz")
        np.savez_compressed(npz_path, **result)
```

---

## 🔄 **EXECUTION FLOW SUMMARY**

### **Complete Call Stack for `python main.py --processing`:**

1. **`main.py`** → Entry point
2. **Import cascade** → Loads entire module system
3. **`main()`** → Main execution function
4. **`parse_arguments()`** → CLI parsing (shows menu if needed)
5. **`Config()`** → Configuration loading
6. **`PipelineLogger()`** → Logging setup
7. **`Pipeline()`** → Pipeline initialization
8. **`register_all_stages()`** → Stage registration
9. **`pipeline.run()`** → Main execution
10. **`_determine_stages()`** → Stage selection
11. **`executor.execute()`** → Stage execution loop
12. **`PreprocessingStage.run()`** → First stage
13. **`run_preprocessing()`** → Original preprocessing
14. **`phasor_transform.process_flim_file()`** → Phasor calculation
15. **`WaveletFilterStage.run()`** → Second stage  
16. **`ComplexWaveletFilter_v2_0.main()`** → Wavelet processing
17. **`process_dataset()`** → Core wavelet algorithm

### **Key Import Dependencies:**
```
main.py
├── src.python.core.config
├── src.python.core.logger  
├── src.python.core.cli
└── src.python.core.pipeline
    ├── src.python.core.stages
    └── src.python.modules
        ├── preprocessing → TCSPC_preprocessing_AUTOcal_v2_0
        ├── wavelet_filter → ComplexWaveletFilter_v2_0
        ├── phasor_transform → process_flim_file
        └── 15+ other stage modules
```

### **Configuration Flow:**
1. **Defaults loaded** in `Config._load_defaults()`
2. **File config** merged if provided
3. **CLI arguments** override configuration
4. **Interactive menu** modifies CLI arguments
5. **Stage execution** uses final merged configuration

### **Data Flow:**
1. **Raw .bin files** → Input directory
2. **ImageJ conversion** → .tif files in output directory
3. **Phasor transformation** → _g.tiff, _s.tiff, _intensity.tiff files
4. **File organization** → Sorted into preprocessed directory structure
5. **Wavelet filtering** → .npz files with filtered and unfiltered data

This architecture provides:
- **Modularity**: Each component has a single responsibility
- **Extensibility**: Easy to add new stages or modify existing ones
- **Maintainability**: Clear separation of concerns
- **Testability**: Each component can be tested independently
- **Configuration**: Centralized and hierarchical configuration management
- **Logging**: Comprehensive tracking and error reporting

The modular design makes it easy to understand, debug, and extend the pipeline while maintaining backward compatibility with the original processing algorithms. 