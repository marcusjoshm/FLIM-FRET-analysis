// Get the preprocessed directory from macro arguments
macroArgs = getArgument();
preprocessed_dir = macroArgs;

// Function to rename files in a given directory
function rename_files_in_directory(directory) {
    list = getFileList(directory);
    fileCount = list.length;

    // Calculate the number of digits needed
    maxDigits = lengthOf(""+fileCount);
    
    for (i = 0; i < fileCount; i++) {
        if (endsWith(list[i], ".tiff")) {
            oldFilePath = directory + File.separator + list[i];
            
            // Format the new filename with leading zeros
            formattedNumber = padWithZeros(i + 1, maxDigits);
            newFilePath = directory + File.separator + formattedNumber + ".tiff";
            
            File.rename(oldFilePath, newFilePath);
        }
    }
}

// Function to pad a number with leading zeros
function padWithZeros(number, totalDigits) {
    numberStr = "" + number;
    while (lengthOf(numberStr) < totalDigits) {
        numberStr = "0" + numberStr;
    }
    return numberStr;
}

// Get the list of all subdirectories in the preprocessed directory
preprocessed_list = getFileList(preprocessed_dir);
for (i = 0; i < preprocessed_list.length; i++) {
    preprocessed_subdir = preprocessed_dir + File.separator + preprocessed_list[i];
    
    // Check if the item is a directory
    if (File.isDirectory(preprocessed_subdir)) {
        // Define the specific subdirectories
        g_unfiltered_dir = preprocessed_subdir + File.separator + "G_unfiltered";
        s_unfiltered_dir = preprocessed_subdir + File.separator + "S_unfiltered";
        intensity_dir = preprocessed_subdir + File.separator + "intensity";
        
        // Rename files in the specified subdirectories
        rename_files_in_directory(g_unfiltered_dir);
        rename_files_in_directory(s_unfiltered_dir);
        rename_files_in_directory(intensity_dir);
    }
}

run("Quit");
