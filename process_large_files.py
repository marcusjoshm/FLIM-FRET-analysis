#!/usr/bin/env python3
"""
Process Large .bin Files
=======================

Alternative approaches for processing large .bin files when command-line ImageJ fails.
"""

import os
import sys
import shutil
from pathlib import Path

def create_processing_instructions():
    """Create instructions for processing large .bin files"""
    
    print("=" * 80)
    print("LARGE .BIN FILE PROCESSING INSTRUCTIONS")
    print("=" * 80)
    
    print("\nISSUE IDENTIFIED:")
    print("- Command-line ImageJ is failing due to Java runtime issues")
    print("- Files are 6.5x larger (856 MB vs 132 MB)")
    print("- Dimensions: 1304×1304×132 vs 512×512×132 pixels")
    print("- Both file types work in ImageJ GUI")
    
    print("\nSOLUTION OPTIONS:")
    print("\n1. FIX JAVA RUNTIME (Recommended):")
    print("   - Install Java 8 or 11 (ImageJ/Fiji compatible)")
    print("   - Set JAVA_HOME environment variable")
    print("   - Restart terminal and try again")
    
    print("\n2. MANUAL GUI PROCESSING:")
    print("   - Open ImageJ/Fiji GUI")
    print("   - File > Import > Bio-Formats")
    print("   - Select .bin files")
    print("   - Save as .tif files")
    print("   - Process multiple files using ImageJ macro recorder")
    
    print("\n3. ALTERNATIVE PROCESSING:")
    print("   - Use Python libraries (tifffile, bioformats)")
    print("   - Convert files in smaller batches")
    print("   - Use different ImageJ installation")
    
    print("\n4. MEMORY OPTIMIZATION:")
    print("   - Increase ImageJ memory allocation")
    print("   - Process files one at a time")
    print("   - Use 64-bit Java if available")
    
    print("\nFILE COMPARISON:")
    print("- Working files: 512×512×132 pixels (132 MB)")
    print("- Problematic files: 1304×1304×132 pixels (856 MB)")
    print("- Both contain valid FLIM data (0.0-0.097 range)")
    
    print("\nNEXT STEPS:")
    print("1. Try fixing Java runtime first")
    print("2. If that fails, use GUI processing")
    print("3. Consider alternative processing methods")
    print("4. The macro fix (args[2] issue) is already applied")

def check_java_installation():
    """Check Java installation"""
    print("\nCHECKING JAVA INSTALLATION:")
    
    # Check if java is available
    import subprocess
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        print(f"Java version: {result.stderr.split('version')[1].split()[0] if 'version' in result.stderr else 'Unknown'}")
    except FileNotFoundError:
        print("Java not found in PATH")
    
    # Check JAVA_HOME
    java_home = os.environ.get('JAVA_HOME')
    if java_home:
        print(f"JAVA_HOME: {java_home}")
    else:
        print("JAVA_HOME not set")
    
    # Check ImageJ Java
    imagej_java = "/Applications/Fiji.app/java/macosx/adoptopenjdk-8.jdk/Contents/Home/bin/java"
    if os.path.exists(imagej_java):
        print(f"ImageJ bundled Java: {imagej_java}")
    else:
        print("ImageJ bundled Java not found")

def create_gui_processing_script():
    """Create a script for GUI processing"""
    
    script_content = """
// ImageJ Macro for GUI Processing of Large .bin Files
// Save this as "process_large_bin_files.ijm" and run in ImageJ GUI

// Set memory to 4GB for large files
setOption("Memory", "4GB");

// Function to process a single .bin file
function processBinFile(inputPath, outputPath) {
    print("Processing: " + inputPath);
    
    // Open with Bio-Formats
    run("Bio-Formats Importer", "open=[" + inputPath + "]");
    
    if (nImages > 0) {
        // Get image info
        width = getWidth();
        height = getHeight();
        print("Image dimensions: " + width + "x" + height);
        
        // Save as TIFF
        saveAs("Tiff", outputPath);
        print("Saved: " + outputPath);
        
        // Close image
        close();
        return true;
    } else {
        print("Failed to open: " + inputPath);
        return false;
    }
}

// Example usage:
// processBinFile("/path/to/file.bin", "/path/to/output.tif");

print("Large .bin file processing macro loaded.");
print("Use: processBinFile(inputPath, outputPath)");
"""
    
    with open("process_large_bin_files.ijm", "w") as f:
        f.write(script_content)
    
    print("\nCreated GUI processing macro: process_large_bin_files.ijm")
    print("Load this macro in ImageJ GUI and use the processBinFile function")

def main():
    print("Large .bin File Processing Helper")
    print("=" * 40)
    
    create_processing_instructions()
    check_java_installation()
    create_gui_processing_script()
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("- The macro argument issue has been fixed")
    print("- The problem is Java runtime, not file format")
    print("- Both file types are valid and can be processed")
    print("- Try fixing Java first, then use GUI if needed")
    print("=" * 80)

if __name__ == "__main__":
    main() 