// Get the input and output files or directories from macro arguments
macroArgs = getArgument();
args = split(macroArgs, ",");
bin_path = args[0];
tif_path = args[1];

// Function to normalize file path (remove double slashes)
function normalizePath(path) {
    while (indexOf(path, "//") >= 0) {
        path = replace(path, "//", "/");
    }
    return path;
}

// Normalize paths
bin_path = normalizePath(bin_path);
tif_path = normalizePath(tif_path);

// Print debug info
print("FLIM_processing_macro_1.ijm starting");
print("Input file/dir: " + bin_path);
print("Output file/dir: " + tif_path);

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

// Create output directory recursively
outputDir = File.getParent(tif_path);
makeDirectoryRecursive(outputDir);
print("Ensured output directory exists: " + outputDir);

// Process the bin file directly
print("Processing bin file: " + bin_path);

// Check if output file already exists
if (File.exists(tif_path)) {
    print("Output file already exists, will overwrite: " + tif_path);
}

// Open the file with Bio-Formats Importer
run("Bio-Formats Importer", "open=[" + bin_path + "]");

// Check if any window was opened
if (nImages == 0) {
    print("Error: Failed to open " + bin_path);
} else {
    // Get the title of the current image
    title = getTitle();
    print("Image opened with title: " + title);
    
    // Save the image as a TIFF
    print("Saving to: " + tif_path);
    saveAs("Tiff", tif_path);
    
    // Close the current image
    close();
    print("Successfully processed: " + bin_path);
}

print("FLIM_processing_macro_1.ijm finished.");

// Exit ImageJ when done
run("Quit");
