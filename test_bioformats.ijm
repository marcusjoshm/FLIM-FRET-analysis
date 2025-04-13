// Test script for Bio-Formats
inputFile = "/Volumes/NX-01-A/FLIM_workflow_test_data/Dish_1_Post-Rapa/R1/R_1_s1.bin";
outputFile = "/Volumes/NX-01-A/FLIM_workflow_test_data_analysis/output/test_output.tif";

print("Starting Bio-Formats test");
print("Input file: " + inputFile);
print("Output file: " + outputFile);

// Make sure input file exists
if (!File.exists(inputFile)) {
    print("Error: Input file does not exist: " + inputFile);
    exit();
}

// Create output directory if it doesn't exist
outputDir = File.getParent(outputFile);
File.makeDirectory(outputDir);

// Try different methods of opening the file
print("Method 1: Using Bio-Formats Importer plugin");
run("Bio-Formats Importer", "open=[" + inputFile + "] color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");

if (nImages > 0) {
    print("Success! Image opened with " + nImages + " images and " + nSlices + " slices");
    print("Saving to: " + outputFile);
    saveAs("Tiff", outputFile);
    close();
} else {
    print("Method 1 failed. No images were opened.");
}

print("Method 2: Using direct open");
open(inputFile);

if (nImages > 0) {
    print("Success! Image opened with " + nImages + " images and " + nSlices + " slices");
    print("Saving to: " + outputFile + "_method2.tif");
    saveAs("Tiff", outputFile + "_method2.tif");
    close();
} else {
    print("Method 2 failed. No images were opened.");
}

// Try File > Import > Bio-Formats
print("Method 3: Using File > Import > Bio-Formats");
run("Bio-Formats", "open=[" + inputFile + "] color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");

if (nImages > 0) {
    print("Success! Image opened with " + nImages + " images and " + nSlices + " slices");
    print("Saving to: " + outputFile + "_method3.tif");
    saveAs("Tiff", outputFile + "_method3.tif");
    close();
} else {
    print("Method 3 failed. No images were opened.");
}

print("Bio-Formats test completed"); 