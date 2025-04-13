// Get the preprocessed directory from macro arguments
macroArgs = getArgument();
preprocessed_dir = macroArgs;

// Normalize input path (remove trailing slashes)
if (endsWith(preprocessed_dir, "/")) {
    preprocessed_dir = substring(preprocessed_dir, 0, lengthOf(preprocessed_dir) - 1);
}

// Print debug info
print("FLIM_processing_macro_5.ijm starting");
print("Preprocessed directory: " + preprocessed_dir);

// Track renamed files
g_renamed = 0;
s_renamed = 0;
intensity_renamed = 0;

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

// Function to rename files in a given directory
function rename_files_in_directory(directory, prefix) {
    directory = normalizePath(directory);
    
    if (!File.exists(directory)) {
        print("Directory does not exist, skipping: " + directory);
        return 0;
    }
    
    print("Processing directory: " + directory);
    list = getFileList(directory);
    
    // Filter and sort tiff files
    tiff_files = newArray();
    for (i = 0; i < list.length; i++) {
        if (endsWith(list[i], ".tiff")) {
            tiff_files = Array.concat(tiff_files, list[i]);
        }
    }
    
    if (tiff_files.length == 0) {
        print("No .tiff files found in: " + directory);
        return 0;
    }
    
    print("Found " + tiff_files.length + " TIFF files");
    Array.sort(tiff_files);
    
    // Calculate the number of digits needed for padded numbers
    maxDigits = lengthOf("" + tiff_files.length);
    renamed_count = 0;
    
    for (i = 0; i < tiff_files.length; i++) {
        // Format the new filename with leading zeros
        formattedNumber = padWithZeros(i + 1, maxDigits);
        oldFilePath = normalizePath(directory + File.separator + tiff_files[i]);
        newFilePath = normalizePath(directory + File.separator + prefix + "_" + formattedNumber + ".tiff");
        
        print("Renaming: " + tiff_files[i] + " -> " + prefix + "_" + formattedNumber + ".tiff");
        
        // Rename the file
        File.rename(oldFilePath, newFilePath);
        renamed_count++;
    }
    
    return renamed_count;
}

// Function to pad a number with leading zeros
function padWithZeros(number, totalDigits) {
    numberStr = "" + number;
    while (lengthOf(numberStr) < totalDigits) {
        numberStr = "0" + numberStr;
    }
    return numberStr;
}

// Process all subdirectories in the preprocessed directory recursively
function processSubdirectories(baseDir) {
    baseDir = normalizePath(baseDir);
    dirs = getFileList(baseDir);
    
    for (i = 0; i < dirs.length; i++) {
        if (endsWith(dirs[i], "/") || File.isDirectory(baseDir + File.separator + dirs[i])) {
            subdir = normalizePath(baseDir + File.separator + dirs[i]);
            
            // Process G_unfiltered, S_unfiltered, and intensity subdirectories if they exist
            g_dir = normalizePath(subdir + File.separator + "G_unfiltered");
            s_dir = normalizePath(subdir + File.separator + "S_unfiltered");
            intensity_dir = normalizePath(subdir + File.separator + "intensity");
            
            // Rename files in each directory with appropriate prefixes
            g_renamed += rename_files_in_directory(g_dir, "G");
            s_renamed += rename_files_in_directory(s_dir, "S");
            intensity_renamed += rename_files_in_directory(intensity_dir, "I");
            
            // Continue recursively
            processSubdirectories(subdir);
        }
    }
}

// Create the main preprocessed directory if it doesn't exist
makeDirectoryRecursive(preprocessed_dir);
print("Created preprocessed directory: " + preprocessed_dir);

// Start processing
processSubdirectories(preprocessed_dir);

print("FLIM_processing_macro_5.ijm finished");
print("Renamed " + g_renamed + " G files");
print("Renamed " + s_renamed + " S files");
print("Renamed " + intensity_renamed + " intensity files");
print("Total renamed: " + (g_renamed + s_renamed + intensity_renamed) + " files");

run("Quit");
