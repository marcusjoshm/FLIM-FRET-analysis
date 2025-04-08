// Get the input and preprocessed directories from macro arguments
macroArgs = getArgument();
args = split(macroArgs, ",");
input_dir = args[0];
preprocessed_dir = args[1];

// Function to perform Z projection and save the summed stack
function create_intensity_image(input_subdir, output_subdir) {
    list = getFileList(input_subdir);
    for (i = 0; i < list.length; i++) {
        if (endsWith(list[i], ".tiff")) {
            filePath = input_subdir + File.separator + list[i];
            
            // Open the .tif file
            open(filePath);

            // Perform Z projection (sum of slices)
            run("Z Project...", "projection=[Sum Slices]");

            // Get the title of the current image (Z projection)
            zTitle = getTitle();

            // Construct the full path with the modified title
            path = output_subdir + File.separator + zTitle + ".tiff";
            // Save the Z projection as a TIFF at the constructed path
            saveAs("Tiff", path);

            // Close the Z projection image
            close();

            // Close the original image
            selectWindow(list[i]);
            close();
        }
    }
}

// Get the list of all subdirectories in the input directory
input_list = getFileList(input_dir);
for (i = 0; i < input_list.length; i++) {
    input_subdir = input_dir + File.separator + input_list[i];
    
    // Check if the item is a directory
    if (File.isDirectory(input_subdir)) {
        // Define the output intensity directory
        output_intensity_dir = preprocessed_dir + File.separator + input_list[i] + File.separator + "intensity";
        
        // Create output intensity directory
        File.makeDirectory(preprocessed_dir + File.separator + input_list[i]);
        File.makeDirectory(output_intensity_dir);
        
        // Process files in the input subdirectory
        create_intensity_image(input_subdir, output_intensity_dir);
    }
}

run("Quit");
