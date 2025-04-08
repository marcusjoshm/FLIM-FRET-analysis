// Get the input and output directories from macro arguments
macroArgs = getArgument();
args = split(macroArgs, ",");
input_dir = args[0];
output_dir = args[1];

// Find the FITC.bin file in the input directory
list = getFileList(input_dir);
openFile = "";
for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], "FITC.bin")) {
        openFile = input_dir + File.separator + list[i];
        break;
    }
}
if (openFile == "") {
    print("FITC.bin file not found in the input directory.");
    run("Quit");
}

// Set options for Bio-Formats Importer
options = "open=[" + openFile + "] autoscale color_mode=Default open_files_individually open_all_series concatenate_series_when_compatible";
run("Bio-Formats Importer", options);

// Get the title of the current image
title = getTitle();
// Replace slashes ("/") with underscores ("_") in the title
title = replace(title, "/", "_");
// Construct the full path with the modified title
path = output_dir + File.separator + title + ".tif";
// Save the image as a TIFF at the constructed path
saveAs("Tiff", path);
// Close the current image
close();
run("Quit");
