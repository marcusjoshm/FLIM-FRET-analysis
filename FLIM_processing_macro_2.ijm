// Get the input and output directories from macro arguments
macroArgs = getArgument();
args = split(macroArgs, ",");
input_dir = args[0];
output_dir = args[1];

// Get the list of all files and directories in the input directory
input_list = getFileList(input_dir);

// Iterate through each item in the input directory
for (i = 0; i < input_list.length; i++) {
    input_subdir = input_dir + File.separator + input_list[i];
    
    // Check if the item is a directory
    if (File.isDirectory(input_subdir)) {
        // Create the corresponding output subdirectory
        output_subdir = output_dir + File.separator + input_list[i];
        File.makeDirectory(output_subdir);
        
        // Get the list of all .bin files in the input subdirectory
        file_list = getFileList(input_subdir);
        for (j = 0; j < file_list.length; j++) {
            if (endsWith(file_list[j], ".bin")) {
                openFile = input_subdir + File.separator + file_list[j];
                
                // Set options for Bio-Formats Importer
                options = "open=[" + openFile + "] autoscale color_mode=Default open_files_individually open_all_series concatenate_series_when_compatible";
                run("Bio-Formats Importer", options);
                
                // Get the title of the current image
                title = getTitle();
                // Replace slashes ("/") with underscores ("_") in the title
                title = replace(title, "/", "_");
                // Construct the full path with the modified title
                path = output_subdir + File.separator + replace(file_list[j], ".bin", ".tiff");
                // Save the image as a TIFF at the constructed path
                saveAs("Tiff", path);
                // Close the current image
                close();
            }
        }
    }
}
run("Quit");
