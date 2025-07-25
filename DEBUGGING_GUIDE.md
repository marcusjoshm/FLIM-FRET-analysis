# FLIM-FRET Pipeline Debugging Guide
## Quick Reference for Troubleshooting

This guide helps you quickly identify where issues occur and how to debug them.

---

## 🔧 **Common Debugging Scenarios**

### **1. Import Errors**
**Symptoms**: `ModuleNotFoundError`, `ImportError`
**Where to look**:
```
main.py:15-20          → Core imports
src/python/modules/stage_registry.py:7-23  → Stage imports  
src/python/modules/TCSPC_preprocessing_AUTOcal_v2_0.py:444-460 → Dynamic imports
```

**Debugging steps**:
1. Check if virtual environment is activated
2. Verify `src/python/modules/__init__.py` exists
3. Look for circular import issues in stage modules

### **2. Configuration Issues**
**Symptoms**: Wrong parameters, missing settings
**Where to look**:
```
src/python/core/config.py:75-95    → Default values
src/python/core/config.py:25-32    → File loading logic
main.py:26-28                      → Config initialization
```

**Debugging steps**:
1. Check if `config.json` exists and is valid JSON
2. Verify default values in `Config._load_defaults()`
3. Use `config.get('key', default)` for safe access

### **3. Interactive Menu Not Appearing** 
**Symptoms**: Pipeline runs without showing menu
**Where to look**:
```
src/python/core/cli.py:89-95       → Menu trigger logic
src/python/core/cli.py:25-60       → Menu display
main.py:25                         → Argument parsing
```

**Debugging steps**:
1. Check if any CLI flags are set (they bypass menu)
2. Verify `any([args.preprocessing, args.filter, ...])` logic
3. Test with no arguments: `python main.py --input X --output Y`

### **4. Stage Execution Failures**
**Symptoms**: Stage shows as failed, pipeline stops
**Where to look**:
```
src/python/core/stages.py:140-170  → Stage executor
src/python/core/pipeline.py:70-85  → Stage loop
src/python/modules/[stage].py      → Specific stage implementation
```

**Debugging steps**:
1. Check log files in `output/logs/` directory
2. Look for exception traces in stage execution
3. Verify stage dependencies (e.g., input files exist)

### **5. File Not Found Errors**
**Symptoms**: Cannot find .bin files, .tif files, etc.
**Where to look**:
```
src/python/modules/TCSPC_preprocessing_AUTOcal_v2_0.py:660-680 → File selection
src/python/modules/ComplexWaveletFilter_v2_0.py:670-690 → File scanning
src/python/core/pipeline.py:40-50 → Directory setup
```

**Debugging steps**:
1. Verify input directory structure
2. Check file naming conventions (e.g., `_g.tiff` vs `_g.tif`)
3. Ensure ImageJ conversion completed successfully

### **6. Phasor Transform Errors**
**Symptoms**: "Error importing phasor_transform module"
**Where to look**:
```
src/python/modules/TCSPC_preprocessing_AUTOcal_v2_0.py:444-460 → Import logic
src/python/modules/phasor_transform.py:20-170 → Function implementation
```

**Debugging steps**:
1. Verify `phasor_transform.py` has `process_flim_file` function
2. Check import path resolution
3. Test relative vs absolute imports

---

## 🕵️ **Debugging Techniques**

### **Add Debug Prints**
Insert debug prints at key points:

```python
# In any stage file
print(f"DEBUG: Entering {self.__class__.__name__}.run()")
print(f"DEBUG: kwargs = {kwargs}")
print(f"DEBUG: Config = {self.config.to_dict()}")
```

### **Check Execution Flow**
Add execution markers:

```python
# In main.py
print("DEBUG: Starting main()")
print(f"DEBUG: Args parsed: {vars(args)}")

# In pipeline.py  
print(f"DEBUG: Stages to run: {stages}")
print(f"DEBUG: Executing stage: {stage_name}")
```

### **Inspect Variables**
Use Python debugger:

```python
import pdb; pdb.set_trace()  # Breakpoint
# Or use modern debugger
import ipdb; ipdb.set_trace()
```

### **Log File Analysis**
Check these log locations:
- `output/logs/pipeline_YYYYMMDD_HHMMSS.log` - Main pipeline log
- `output/logs/error_report_YYYYMMDD_HHMMSS.txt` - Error summary

---

## 📋 **Pre-Execution Checklist**

Before running the pipeline, verify:

1. **Environment Setup**:
   - [ ] Virtual environment activated
   - [ ] All dependencies installed (`pip install -r src/scripts/requirements.txt`)
   - [ ] Python 3.8+ being used

2. **File Structure**:
   - [ ] Input directory contains `.bin` files
   - [ ] `calibration.csv` exists in input directory
   - [ ] Output directory is writable
   - [ ] ImageJ/Fiji is installed and accessible

3. **Configuration**:
   - [ ] Config file (if used) is valid JSON
   - [ ] Microscope parameters are correct
   - [ ] File paths use correct separators for OS

4. **Permissions**:
   - [ ] Read access to input directory
   - [ ] Write access to output directory
   - [ ] Execute permissions on ImageJ/Fiji

---

## 🚨 **Emergency Debugging**

If the pipeline completely fails to start:

1. **Test minimal import**:
   ```bash
   python -c "from src.python.core.config import Config; print('Config OK')"
   python -c "from src.python.core.pipeline import Pipeline; print('Pipeline OK')"
   ```

2. **Check Python path**:
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```

3. **Verify package structure**:
   ```bash
   find src -name "__init__.py" | head -10
   ```

4. **Test with minimal arguments**:
   ```bash
   python main.py --help  # Should show help without errors
   ```

---

## 📝 **Common Error Messages and Solutions**

| Error | Location | Solution |
|-------|----------|----------|
| `ModuleNotFoundError: No module named 'src'` | `main.py:15` | Check current directory, ensure you're in project root |
| `attempted relative import with no known parent package` | `TCSPC_preprocessing_AUTOcal_v2_0.py:454` | Fixed in new architecture, update import statement |
| `name 'args' is not defined` | `organize_output_files.py` | Known issue, pipeline uses fallback method |
| `Command returned non-zero exit status 1` | ImageJ macro execution | Check ImageJ installation and file paths |
| `No module named 'numpy'` | Any stage | Install dependencies: `pip install numpy` |

---

## 🔍 **Advanced Debugging**

### **Trace Execution with Line Numbers**
```python
import traceback

def trace_calls(frame, event, arg):
    if event == 'call':
        filename = frame.f_code.co_filename
        if 'FLIM-FRET-analysis' in filename:  # Only our code
            line_no = frame.f_lineno
            func_name = frame.f_code.co_name
            print(f"TRACE: {filename}:{line_no} in {func_name}()")
    return trace_calls

# Add to main.py before pipeline.run()
import sys
sys.settrace(trace_calls)
```

### **Memory and Performance Monitoring**
```python
import psutil
import time

# Add to stage execution
start_time = time.time()
start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

# ... stage execution ...

end_time = time.time()
end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
print(f"Stage duration: {end_time - start_time:.2f}s")
print(f"Memory change: {end_memory - start_memory:.2f} MB")
```

This debugging guide provides practical tools for identifying and resolving issues in the new architecture. 