// Get the input and output directories from macro arguments
macroArgs = getArgument();
args = split(macroArgs, ",");
input_dir = args[0];
output_dir = args[1];

// Normalize input/output paths (remove trailing slashes)
if (endsWith(input_dir, "/")) {
    input_dir = substring(input_dir, 0, lengthOf(input_dir) - 1);
}
if (endsWith(output_dir, "/")) {
    output_dir = substring(output_dir, 0, lengthOf(output_dir) - 1);
}

// Print debug info
print("FLIM_processing_macro_2.ijm starting");
print("Input directory: " + input_dir);
print("Output directory: " + output_dir);

// Track progress
processed = 0;
failures = 0;

// Function to normalize file path (remove double slashes)
function normalizePath(path) {
    while (indexOf(path, "//") >= 0) {
        path = replace(path, "//", "/");
    }
    return path;
}

// Function to create directory and all parent directories
function makeDirectoryRecursive(dir) {
    dir = normalizePath(dir);
    if (File.exists(dir)) {
        return true;
    }
    
    // Get parent directory
    parent = File.getParent(dir);
    
    // Create parent directory if it doesn't exist
    if (parent != "" && !File.exists(parent)) {
        makeDirectoryRecursive(parent);
    }
    
    // Create this directory
    return File.makeDirectory(dir);
}

// Function to get relative path from input directory
function getRelativePath(fullPath, basePath) {
    // Normalize both paths
    fullPath = normalizePath(fullPath);
    basePath = normalizePath(basePath);
    
    // Check if the full path starts with the base path
    if (startsWith(fullPath, basePath)) {
        // Remove the base path and any leading separator
        relativePath = substring(fullPath, lengthOf(basePath));
        if (startsWith(relativePath, File.separator)) {
            relativePath = substring(relativePath, 1);
        }
        return relativePath;
    }
    
    // If not a subpath, return the full path
    return fullPath;
}

// Recursively process .bin files except FITC.bin which is handled by macro 1
function scanDirectory(dir) {
    // Normalize directory path
    dir = normalizePath(dir);
    
    list = getFileList(dir);
    
    for (i = 0; i < list.length; i++) {
        if (endsWith(list[i], "/") || File.isDirectory(dir + File.separator + list[i])) {
            // It's a directory, scan it recursively
            scanDirectory(dir + File.separator + list[i]);
        } 
        else if (endsWith(list[i], ".bin") && !endsWith(list[i], "FITC.bin")) {
            // Found .bin file, process it
            processBinFile(dir, list[i]);
        }
    }
}

// Process a .bin file
function processBinFile(dir, filename) {
    binFilePath = normalizePath(dir + File.separator + filename);
    print("Processing BIN file: " + binFilePath);
    
    // Get relative path from input directory
    relativePath = getRelativePath(dir, input_dir);
    
    // Create output directory mirroring the input directory structure
    targetDir = output_dir;
    if (relativePath != "") {
        targetDir = normalizePath(output_dir + File.separator + relativePath);
    } else {
        // If relativePath is empty, preserve the basename of the input directory
        inputBasename = File.getName(input_dir);
        if (inputBasename != "") {
            targetDir = normalizePath(output_dir + File.separator + inputBasename);
        }
    }
    
    // Always create the target directory
    makeDirectoryRecursive(targetDir);
    print("Created directory: " + targetDir);
    
    print("Output directory: " + targetDir);
    
    // Check if output file already exists
    outFilename = replace(filename, ".bin", ".tif");
    outPath = normalizePath(targetDir + File.separator + outFilename);
    
    if (File.exists(outPath)) {
        print("Output file already exists, skipping: " + outPath);
        return;
    } else {
        print("Output file not found, will create: " + outPath);
    }
    
    // Try to open the file with Bio-Formats Importer
    run("Bio-Formats Importer", "open=[" + binFilePath + "]");
    
    // Check if any window was opened
    if (nImages == 0) {
        print("Warning: Failed to open " + binFilePath);
        failures++;
        return;
    }
    
    // Get the title of the current image
    title = getTitle();
    print("Image title: " + title);
    
    // Save the image as a TIFF
    print("Saving to: " + outPath);
    saveAs("Tiff", outPath);
    
    // Close the current image
    close();
    processed++;
    print("Successfully processed: " + filename);
}

// Create output directory recursively
makeDirectoryRecursive(output_dir);
print("Created output directory: " + output_dir);

// Start the scan
scanDirectory(input_dir);

print("FLIM_processing_macro_2.ijm finished.");
print("Successfully processed: " + processed + " BIN files");
print("Failed: " + failures + " BIN files");

// Exit ImageJ when done
run("Quit");
