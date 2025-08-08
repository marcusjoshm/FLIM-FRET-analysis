# FLIM-FRET Analysis Pipeline - New Refactored Architecture

## Overview

The FLIM-FRET analysis pipeline has been completely refactored to improve readability, maintainability, and shareability. The massive `run_pipeline.py` (1400+ lines) has been broken down into smaller, focused modules with clear separation of concerns.

## Key Improvements

### ✅ **Modular Design**
- **Separation of concerns**: Configuration, logging, CLI, and stages are now in separate modules
- **Clear inheritance hierarchy**: Base classes provide common functionality  
- **Consistent patterns**: All stages follow the same execution pattern

### ✅ **Improved Maintainability**
- **Single responsibility**: Each class has one clear purpose
- **Easy to extend**: Adding new stages is now trivial with the decorator pattern
- **Better error handling**: Consistent error handling across all components

### ✅ **Enhanced Shareability**
- **Proper package structure**: Can be installed as a Python package
- **Clear API boundaries**: Well-defined interfaces between components
- **Configuration-driven**: Easy to customize without code changes

## Architecture Overview

```
flimfret/
├── main.py                    # New streamlined entry point (50 lines vs 1400+)
├── src/
│   ├── __init__.py           # Package initialization
│   └── python/
│       ├── __init__.py
│       ├── core/             # Core framework components
│       │   ├── __init__.py
│       │   ├── config.py     # Configuration management
│       │   ├── logger.py     # Logging system
│       │   ├── cli.py        # Command line interface
│       │   ├── stages.py     # Stage execution framework
│       │   └── pipeline.py   # Main pipeline orchestrator
│       └── modules/          # Processing modules
│           ├── __init__.py
│           ├── stage_registry.py      # Stage registration
│           ├── preprocessing.py       # Preprocessing stage
│           ├── wavelet_filter.py     # Wavelet filtering stage
│           ├── phasor_transform.py   # Phasor transformation
│           ├── segmentation.py       # Segmentation stages
│           ├── visualization.py      # Visualization stages
│           ├── lifetime_processing.py # Lifetime processing
│           └── mask_processing.py    # Mask processing
└── run_pipeline.py           # Legacy script (still works)
```

## Core Components

### 1. Configuration Management (`src/python/core/config.py`)
- **Centralized configuration** with dot notation access
- **Validation and defaults** built-in
- **Multiple config sources** (files, environment, defaults)

```python
from src.python.core.config import Config

config = Config()
config.set("microscope_params.frequency", 80.0)
frequency = config.get("microscope_params.frequency")
```

### 2. Logging System (`src/python/core/logger.py`)
- **Comprehensive error tracking** and performance monitoring
- **Multiple log levels** and outputs
- **Automatic report generation**

```python
from src.python.core.logger import PipelineLogger

logger = PipelineLogger("/path/to/output")
logger.log_stage_start("Processing", "Starting data processing")
```

### 3. CLI Module (`src/python/core/cli.py`)
- **Clean argument parsing** with validation
- **Interactive menu system** 
- **Reusable user interaction methods**

### 4. Stage Framework (`src/python/core/stages.py`)
- **StageBase** abstract class for all pipeline stages
- **FileProcessingStage** specialized for file operations
- **StageRegistry** for managing available stages
- **Decorator-based stage registration**

```python
from src.python.core.stages import register_stage, StageBase

@register_stage('my_stage', order=1)
class MyStage(StageBase):
    def run(self, **kwargs) -> bool:
        # Your stage implementation
        return True
```

### 5. Pipeline Orchestrator (`src/python/core/pipeline.py`)
- **Main workflow coordination**
- **Directory setup and management**
- **Stage execution with error handling**

## Usage

### Using the New Architecture

```bash
# Run with the new streamlined main.py
python main.py --input /path/to/data --output /path/to/output --processing

# Or use the legacy run_pipeline.py (still works)
python run_pipeline.py --input /path/to/data --output /path/to/output --processing
```

### Testing the New Architecture

```bash
# Test all components
python test_new_architecture.py
```

### Adding New Stages

1. **Create your stage class**:
```python
# In src/python/modules/my_new_module.py
from ..core.stages import StageBase

class MyNewStage(StageBase):
    def get_description(self) -> str:
        return "My new processing stage"
    
    def validate_inputs(self, **kwargs) -> bool:
        # Validate inputs
        return True
    
    def run(self, **kwargs) -> bool:
        # Your processing logic
        self.logger.info("Running my new stage")
        return True
```

2. **Register the stage**:
```python
# In src/python/modules/stage_registry.py
from .my_new_module import MyNewStage

def register_all_stages():
    # ... existing registrations ...
    register_stage('my_new_stage', order=14)(MyNewStage)
```

3. **Update the CLI and pipeline** to handle your new stage flag.

## Migration Guide

### From Old Architecture to New

The new architecture is **backward compatible**. Your existing workflows will continue to work with `run_pipeline.py`.

**To migrate to the new system:**

1. **Use `main.py` instead of `run_pipeline.py`**
2. **Update any custom scripts** to use the new modular imports
3. **Customize via configuration files** instead of code modifications

### Key Differences

| Old Architecture | New Architecture |
|------------------|------------------|
| Single 1400+ line file | Modular design with focused components |
| Mixed concerns | Clean separation of concerns |
| Hard to extend | Easy to add new stages |
| Limited error handling | Comprehensive error tracking |
| No configuration management | Centralized config system |

## Benefits

### For Development
- **Easier debugging**: Smaller, focused modules
- **Faster iteration**: Change one component without affecting others
- **Better testing**: Each component can be tested independently

### For Users
- **Better error messages**: More detailed error tracking and reporting
- **Easier customization**: Configuration-driven behavior
- **More reliable**: Better error handling and validation

### For Collaboration
- **Clear interfaces**: Well-defined APIs between components
- **Easy onboarding**: Clear structure and documentation
- **Maintainable**: Code is easier to understand and modify

## Current Status

### ✅ Completed
- ✅ **Package structure** with proper `__init__.py` files
- ✅ **Configuration management** module
- ✅ **Logging system** extraction and enhancement
- ✅ **CLI module** for argument parsing and user interaction
- ✅ **Stage execution framework** with base classes
- ✅ **Pipeline orchestrator** for workflow coordination
- ✅ **Stage registration system** for modular stage management
- ✅ **Working preprocessing and wavelet filter stages**

### 🚧 Remaining Work
The existing modules can be easily migrated to the new framework by:

1. **Implementing the placeholder stages** with actual functionality
2. **Adding comprehensive unit tests**
3. **Creating proper documentation**
4. **Adding data models with type hints**
5. **Creating a proper setup.py** for package installation

## Getting Started

1. **Test the new architecture**:
   ```bash
   python test_new_architecture.py
   ```

2. **Run a simple pipeline**:
   ```bash
   python main.py --input /path/to/data --output /path/to/output --preprocessing
   ```

3. **Explore the modular structure** in the `src/python/` directory

4. **Customize configuration** by creating a `config/config.json` file

The new architecture provides a solid foundation for continued development while maintaining compatibility with existing workflows. 