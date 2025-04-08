// Get the input and output directories from macro arguments
macroArgs = getArgument();
args = split(macroArgs, ",");
input_dir = args[0];
preprocessed_dir = args[1];

// Function to move and process files with a given suffix
function process_flute_files(input_subdir, output_subdir, suffix) {
    list = getFileList(input_subdir);
    for (i = 0; i < list.length; i++) {
        if (endsWith(list[i], suffix + ".tiff")) {
            openFile = input_subdir + File.separator + list[i];
            
            // Set options for Bio-Formats Importer
            options = "open=[" + openFile + "] autoscale color_mode=Default open_files_individually open_all_series concatenate_series_when_compatible";
            run("Bio-Formats Importer", options);
            
            // Get the title of the current image
            title = getTitle();
            // Replace slashes ("/") with underscores ("_") in the title
            title = replace(title, "/", "_");
            // Construct the full path with the modified title
            path = output_subdir + File.separator + title + ".tiff";
            // Save the image as a TIFF at the constructed path
            saveAs("Tiff", path);
            // Close the current image
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
        // Define the FLUTE_median and FLUTE_unfiltered directories
        flute_median_dir = input_subdir + File.separator + "FLUTE_median";
        flute_unfiltered_dir = input_subdir + File.separator + "FLUTE_unfiltered";
        
        // Define the output directories for G and S files
        output_g_unfiltered_dir = preprocessed_dir + File.separator + input_list[i] + File.separator + "G_unfiltered";
        output_s_unfiltered_dir = preprocessed_dir + File.separator + input_list[i] + File.separator + "S_unfiltered";
        
        // Create output directories
        File.makeDirectory(preprocessed_dir + File.separator + input_list[i]);
        File.makeDirectory(output_g_unfiltered_dir);
        File.makeDirectory(output_s_unfiltered_dir);
        
        // Process files in FLUTE_median and FLUTE_unfiltered directories
        process_flute_files(flute_unfiltered_dir, output_g_unfiltered_dir, "g");
        process_flute_files(flute_unfiltered_dir, output_s_unfiltered_dir, "s");
    }
}

run("Quit");
