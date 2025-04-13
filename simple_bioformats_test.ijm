// Simple Bio-Formats test
inputFile = "/Volumes/NX-01-A/FLIM_workflow_test_data/Dish_1_Post-Rapa/R1/R_1_s1.bin";
outputFile = "/Volumes/NX-01-A/FLIM_workflow_test_data_analysis/output/Dish_1_Post-Rapa/R1/R_1_s1_test.tif";

print("Starting simple Bio-Formats test");
print("Input file: " + inputFile);
print("Output file: " + outputFile);

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

// Make sure the output directory exists recursively
outputDir = File.getParent(outputFile);
print("Creating directory recursively: " + outputDir);
makeDirectoryRecursive(outputDir);

// Try to open with Bio-Formats Importer
print("Opening file with Bio-Formats Importer");
run("Bio-Formats Importer", "open=[" + inputFile + "]");

// Check if it worked
if (nImages > 0) {
    print("Success! Opened file with Bio-Formats.");
    print("Saving to: " + outputFile);
    saveAs("Tiff", outputFile);
    close();
    print("Test completed successfully.");
} else {
    print("Failed to open the file with Bio-Formats.");
}

print("Test finished.");
run("Quit"); 