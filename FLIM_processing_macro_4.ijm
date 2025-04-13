// Get the input and preprocessed directories from macro arguments
macroArgs = getArgument();
args = split(macroArgs, ",");
input_dir = args[0];
preprocessed_dir = args[1];

// Normalize input/output paths (remove trailing slashes)
if (endsWith(input_dir, "/")) {
    input_dir = substring(input_dir, 0, lengthOf(input_dir) - 1);
}
if (endsWith(preprocessed_dir, "/")) {
    preprocessed_dir = substring(preprocessed_dir, 0, lengthOf(preprocessed_dir) - 1);
}

// Print debug info
print("FLIM_processing_macro_4.ijm starting");
print("Input directory: " + input_dir);
print("Preprocessed directory: " + preprocessed_dir);

// Track progress
intensity_files_processed = 0;
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

// Process intensity files recursively
function scanDirectory(dir) {
    // Normalize directory path
    dir = normalizePath(dir);
    
    list = getFileList(dir);
    
    for (i = 0; i < list.length; i++) {
        path = normalizePath(dir + File.separator + list[i]);
        
        if (endsWith(list[i], "/") || File.isDirectory(path)) {
            // Skip directories named FLUTE_* (these are created by FLUTE and should be ignored)
            if (!startsWith(list[i], "FLUTE_")) {
                scanDirectory(path);
            }
        } 
        else if (endsWith(list[i], "_intensity.tiff")) {
            processIntensityFile(dir, list[i]);
            intensity_files_processed++;
        }
    }
}

// Process an intensity file
function processIntensityFile(dir, filename) {
    filePath = normalizePath(dir + File.separator + filename);
    print("Processing intensity file: " + filePath);
    
    // Create relative output path
    relativePath = replace(dir, input_dir, "");
    if (startsWith(relativePath, File.separator)) {
        relativePath = substring(relativePath, 1);
    }
    
    // Create output directory with intensity subdirectory
    targetDir = preprocessed_dir;
    if (relativePath != "") {
        targetDir = normalizePath(preprocessed_dir + File.separator + relativePath);
    }
    
    targetDir = normalizePath(targetDir + File.separator + "intensity");
    makeDirectoryRecursive(targetDir);
    print("Target directory: " + targetDir);
    
    // Check if output file already exists
    outPath = normalizePath(targetDir + File.separator + filename);
    if (File.exists(outPath)) {
        print("Output file already exists, skipping: " + outPath);
        return;
    } else {
        print("Output file not found, will create: " + outPath);
    }
    
    try {
        // Open the file
        open(filePath);
        
        // Check if file was opened successfully
        if (nImages == 0) {
            print("Failed to open: " + filePath);
            failures++;
            return;
        }
        
        // Get original title
        originalTitle = getTitle();
        
        // If the image is a stack, perform Z projection
        if (nSlices > 1) {
            print("Performing Z-projection on stack with " + nSlices + " slices");
            run("Z Project...", "projection=[Sum Slices]");
            
            // Close the original image
            selectWindow(originalTitle);
            close();
            
            // Get the Z-projected image title
            zTitle = getTitle();
        }
        
        // Save to target location
        saveAs("Tiff", outPath);
        
        // Close the image
        close();
        print("Successfully processed: " + filename);
    } catch (exception) {
        print("Error processing " + filename + ": " + exception);
        failures++;
        
        // Close any open images
        while (nImages > 0) {
            close();
        }
    }
}

// Create the main preprocessed directory
makeDirectoryRecursive(preprocessed_dir);
print("Created preprocessed directory: " + preprocessed_dir);

// Start scanning
scanDirectory(input_dir);

print("FLIM_processing_macro_4.ijm finished");
print("Processed " + intensity_files_processed + " intensity files");
print("Failed: " + failures + " files");
