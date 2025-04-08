#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 09:54:22 2024

@author: joshuamarcus
"""

import os
import numpy as np
from PIL import Image
import dtcwt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import LogFormatter
from matplotlib import colors
import math
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse, Circle
import tifffile as tiff
import time
import json
import sys

# GAUSSIAN MIXTURE MODEL SEGMENTATION v2.6
#
# Will propigate multiple .npz files and combine into a single phasor
# depending on how the input and output directories are set

# Record the start time
start_time = time.time()

# Specify first_level
first_level = "U2OS_mNG11-EIF4G1_As_SG-ONLY"

# Define base directories
main_preprocessed_directory = '/Users/leelab/FLIM_processing_dir/processed'
output_base_directory = '/Users/leelab/FLIM_processing_dir/processed'

# Construct the path to the first level directory
first_level_directory = os.path.join(main_preprocessed_directory, first_level)

# Initialize empty lists to store the arrays and dimensions from all .npz files
G_list = []
S_list = []
I_list = []
T_list = []
dimensions = []  # To store dimensions of each .npz file
file_names = []  # To store the corresponding .npz file names

# Iterate over all second level directories
for second_level in os.listdir(first_level_directory):
    second_level_directory = os.path.join(first_level_directory, second_level)
    datasets_directory = os.path.join(second_level_directory, "datasets")
    
    # Skip if datasets directory doesn't exist
    if not os.path.exists(datasets_directory):
        continue

    # Define the expected .npz file name pattern
    npz_file_name = f"{second_level}_AUTOcal_NP130_CWFlevels=9_dataset.npz"
    npz_file_path = os.path.join(datasets_directory, npz_file_name)
    
    # Skip if the .npz file doesn't exist
    if not os.path.isfile(npz_file_path):
        continue

    # Load the .npz file and append the data to the lists
    npz_file = np.load(npz_file_path)
    G_list.append(npz_file['G'])
    S_list.append(npz_file['S'])
    I_list.append(npz_file['A'])
    T_list.append(npz_file['T'])
  
    # Record the dimensions and file name
    dimensions.append(npz_file['G'].shape)
    file_names.append(npz_file_name)

# Combine the arrays from all .npz files along the first dimension
G_array = np.concatenate(G_list, axis=0)
S_array = np.concatenate(S_list, axis=0)
I_array = np.concatenate(I_list, axis=0)
T = np.concatenate(T_list, axis=0)

# Define output directory
output_directory = os.path.join(output_base_directory, first_level)

# Ensure the output directories exist
masks_dir = os.path.join(output_directory, "masks")
lifetime_dir = os.path.join(output_directory, "lifetime_images")
os.makedirs(masks_dir, exist_ok=True)
os.makedirs(lifetime_dir, exist_ok=True)

# Continue with the rest of your processing as usual
threshold = 0
G001_array = G_array * (I_array > threshold)
S001_array = S_array * (I_array > threshold)
G001_array = np.nan_to_num(G001_array)
S001_array = np.nan_to_num(S001_array)
G001_array = np.clip(G001_array, -0.1, 1.1)
S001_array = np.clip(S001_array, -0.1, 1.1)

# Define file naming conventions
filter_method = "CWF-9levels"
mask1 = f'{second_level}_{filter_method}_cond_ellipse_mask2.tiff'
mask3 = f'{second_level}_{filter_method}_cond_circle_mask2.tiff'
T_cond_ellipse_file_name = f'{second_level}_{filter_method}_cond_ellipse_mask_segmented_lifetime2.tiff'
T_cond_circle_file_name = f'{second_level}_{filter_method}_cond_circle_mask_segmented_lifetime2.tiff'
threshold = 0
reference_fluor = "mNeonGreen"
harmonic = 1
Gc = 0.30227996691280035
Sc = 0.459245891966028
cov_f = 4
cov_f_circle = 1
shift = 0
center_cond = (Gc, Sc)
center_dilu = (0.5, 0)
radius_ref = 0.1
radius_dilu = 0.45

def check_either_value_greater_than_zero(list1, list2):
    results = [x > 0 or y > 0 for x, y in zip(list1, list2)]
    return results

def is_point_inside_circle(point, center, radius):
    """
    Check if a point is inside a circle.

    Args:
        point: Tuple (x, y) representing the coordinates of the point.
        center: Tuple (x, y) representing the coordinates of the circle's center.
        radius: Float representing the radius of the circle.

    Returns:
        True if the point is inside the circle, False otherwise.
    """
    distance = math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
    if distance <= radius:
        return True
    else:
        return False
    
def are_points_inside_circle(points, center, radius):
    """
    Check if a list of points are inside a circle.

    Args:
        points: List of tuples representing the coordinates of the points.
        center: Tuple (x, y) representing the coordinates of the circle's center.
        radius: Float representing the radius of the circle.

    Returns:
        List of booleans indicating whether each point is inside the circle.
    """
    results = []
    for point in points:
        results.append(is_point_inside_circle(point, center, radius))
    return results

def is_points_inside_rotated_ellipse(center_x, center_y, semi_major_axis, semi_minor_axis, angle_degrees, points):
    """
    Args:
        Takes the output from function #3
    Returns:
        True if the point is inside the ellipse, False otherwise.
    """
    # Calculate the distance between each point and the center of the ellipse
    distances = [(point[0] - center_x)**2 + (point[1] - center_y)**2 for point in points]

    # Check if the ellipse is a circle (semi-major and semi-minor axes are equal)
    is_circle = math.isclose(semi_major_axis, semi_minor_axis)

    results = []

    if is_circle:
        # If it's a circle, check if each point is inside the circle
        for distance in distances:
            results.append(distance <= semi_major_axis**2)
    else:
        # Calculate the rotation angle of the ellipse
        angle_radians = math.radians(angle_degrees)
        cos_a = math.cos(angle_radians)
        sin_a = math.sin(angle_radians)

        for i, point in enumerate(points):
            point_x, point_y = point

            # Translate the point to the ellipse's coordinate system
            translated_x = point_x - center_x
            translated_y = point_y - center_y

            # Apply the rotation transformation
            rotated_x = cos_a * translated_x + sin_a * translated_y
            rotated_y = -sin_a * translated_x + cos_a * translated_y

            # Calculate the normalized coordinates
            normalized_x = rotated_x / semi_major_axis
            normalized_y = rotated_y / semi_minor_axis

            # Check if the transformed point is inside the unrotated ellipse
            results.append(normalized_x ** 2 + normalized_y ** 2 <= 1)

    return results

def is_points_inside_circle(center_x, center_y, radius, points):
    """
    Args:
        center_x (float): The x-coordinate of the center of the circle.
        center_y (float): The y-coordinate of the center of the circle.
        radius (float): The radius of the circle.
        points (list of tuples): A list of (x, y) points to check.
    Returns:
        list of bool: A list of booleans where True indicates the point is inside the circle, False otherwise.
    """
    results = []
    for point in points:
        point_x, point_y = point
        # Calculate the distance between the point and the center of the circle
        distance = (point_x - center_x)**2 + (point_y - center_y)**2
        # Check if the point is inside the circle
        results.append(distance <= radius**2)
    
    return results

def convert_list_to_array_with_dimensions(lst, rows, columns):
    array = np.array(lst)
    array_with_dimensions = array.reshape(rows, columns)
    return array_with_dimensions

def reassign_good_center(cluster_center_1, cluster_center_2, eigenvalues_1, eigenvalues_2, eigenvectors_1, eigenvectors_2):
    # Extract x values
    cluster_center_1_x = cluster_center_1[0]
    cluster_center_2_x = cluster_center_2[0]
    
    # Compare x values and assign the good center and its corresponding eigenvalues and eigenvectors
    if cluster_center_1_x < cluster_center_2_x:
        good_center = cluster_center_1
        good_eigenvalues = eigenvalues_1
        good_eigenvectors = eigenvectors_1
        bad_center = cluster_center_2
        bad_eigenvalues = eigenvalues_2
        bad_eigenvectors = eigenvectors_2
    else:
        good_center = cluster_center_2
        good_eigenvalues = eigenvalues_2
        good_eigenvectors = eigenvectors_2
        bad_center = cluster_center_1
        bad_eigenvalues = eigenvalues_1
        bad_eigenvectors = eigenvectors_1
    
    return good_center, good_eigenvalues, good_eigenvectors, bad_center, bad_eigenvalues, bad_eigenvectors

# Threshold and G and S arrays
G001_array = G_array * (I_array > threshold)
S001_array = S_array * (I_array > threshold)
G001_array = np.nan_to_num(G001_array)
S001_array = np.nan_to_num(S001_array)
G001_array = np.clip(G001_array, -0.1, 1.1)
S001_array = np.clip(S001_array, -0.1, 1.1)

# SINGLE COLUMN VARIABLES FOR PHASOR HISTOGRAM PLOT
# thresholded G and S phasor data
G001 = G001_array.ravel()
S001 = S001_array.ravel()

# intensity array for weighting
I = I_array.ravel().astype(int)

# x and y dimensions of the datasets
x_dim = I_array.shape[0]
y_dim = I_array.shape[1]

# Weight G and S coordinates by intensity for barycenter analysis
G001_weighted = np.repeat(G001, I)
S001_weighted = np.repeat(S001, I)
G001_weighted = np.nan_to_num(G001_weighted)
S001_weighted = np.nan_to_num(S001_weighted)

# POINTS
points001 = list(zip(G001, S001))

# Calculate G and S coordinates inside and outside the circle for cond phase
results_cond = are_points_inside_circle(points001, center_cond, radius_ref)
matrix_for_cond_GMM = np.reshape(results_cond, (x_dim, y_dim))

# Arrays of G and S coordinates inside and outside the universal circle
G_for_cond_GMM = matrix_for_cond_GMM * G001_array
S_for_cond_GMM = matrix_for_cond_GMM * S001_array

# Flattened arrays
G_GMMcond = G_for_cond_GMM.ravel()
S_GMMcond = S_for_cond_GMM.ravel()

# Weight G and S coordinates by intensity for barycenter analysis
G_GMMcond_weighted = np.repeat(G_GMMcond, I)
S_GMMcond_weighted = np.repeat(S_GMMcond, I)

# Combine G and S weighted arrays into a single array
data_GMMcond = np.column_stack((G_GMMcond_weighted[G_GMMcond_weighted != 0], S_GMMcond_weighted[S_GMMcond_weighted != 0]))

# Fit GMM with 2 components for cond phase
num_clusters = 2
gmm_cond = GaussianMixture(n_components=num_clusters)
gmm_cond.fit(data_GMMcond)

# Get cluster centers, covariances, and weights
cluster_centers_cond = gmm_cond.means_
cov_matrices_cond = gmm_cond.covariances_

# Plot cluster's ellipse boundaries for cond phase
cond_cluster_center_1 = cluster_centers_cond[0]
cond_cluster_center_2 = cluster_centers_cond[1]
cov_matrix_1_cond = cov_matrices_cond[0]
cov_matrix_2_cond = cov_matrices_cond[1]
eigenvalues_1_cond, eigenvectors_1_cond = np.linalg.eigh(cov_matrix_1_cond)
eigenvalues_2_cond, eigenvectors_2_cond = np.linalg.eigh(cov_matrix_2_cond)
print("cond center 1:", cond_cluster_center_1)
print("cond center 2:", cond_cluster_center_2)

# Reassign cluster centers and corresponding eigenvalues/eigenvectors for condensed and dilute
real_cond_cluster_center, real_cond_eigenvalues, real_cond_eigenvectors, real_dilu_cluster_center, real_dilu_eigenvalues, real_dilu_eigenvectors = reassign_good_center(
    cond_cluster_center_1, 
    cond_cluster_center_2, 
    eigenvalues_1_cond, 
    eigenvalues_2_cond, 
    eigenvectors_1_cond, 
    eigenvectors_2_cond
)

# Calculate distance between centers
centers_distance = np.sqrt(((real_cond_cluster_center[0] - real_dilu_cluster_center[0])**2) + ((real_cond_cluster_center[1] - real_dilu_cluster_center[1])**2))

# Compute the correct angle for the ROIs from the correct eigenvector for condition
if real_cond_eigenvectors[0, 1] > 0 and real_cond_eigenvectors[1, 1] > 0:
    angle_cond = np.arctan2(-real_cond_eigenvectors[1, 1], -real_cond_eigenvectors[0, 1])
else:
    angle_cond = np.arctan2(real_cond_eigenvectors[1, 1], real_cond_eigenvectors[1, 1])
angle_degrees_cond = np.degrees(angle_cond)

# Calculate ellipse width and height for condition
width_cond = cov_f * np.sqrt(real_cond_eigenvalues[1])
height_cond = cov_f * np.sqrt(real_cond_eigenvalues[0])

# Calculate the shifts for condition
dx_cond = shift * width_cond * np.cos(angle_cond)
dy_cond = shift * width_cond * np.sin(angle_cond)

# Calculate the new center coordinates for condition
center_cond_x = real_cond_cluster_center[0] + dx_cond
center_cond_y = real_cond_cluster_center[1] + dy_cond

# Create a new center point for condition
ROI_center_cond = np.array([center_cond_x, center_cond_y])

# Calculating G and S coordinates inside cluster ellipses for condition
results_cluster_cond = is_points_inside_rotated_ellipse(ROI_center_cond[0], ROI_center_cond[1], width_cond, height_cond, angle_degrees_cond, points001)
matrix_cluster_cond = np.reshape(results_cluster_cond, (x_dim, y_dim))
circle_radius_cond = cov_f_circle * np.sqrt(real_cond_eigenvalues[0])
results_cluster_cond_circle = is_points_inside_circle(real_cond_cluster_center[0], real_cond_cluster_center[1], circle_radius_cond, points001)
matrix_cluster_cond_circle = np.reshape(results_cluster_cond_circle, (x_dim, y_dim))

# Arrays of G and S coordinates for condition
G_cluster_cond = matrix_cluster_cond * G001_array
S_cluster_cond = matrix_cluster_cond * S001_array
G_cluster_cond_circle = matrix_cluster_cond_circle * G001_array
S_cluster_cond_circle = matrix_cluster_cond_circle * S001_array

# Flattened arrays for condition
G_cond = G_cluster_cond.ravel()
S_cond = S_cluster_cond.ravel()
G_cond_circle = G_cluster_cond_circle.ravel()
S_cond_circle = S_cluster_cond_circle.ravel()

# Debugging
print("ROI_center_cond:", ROI_center_cond)
print("width_cond, height_cond:", width_cond, height_cond)
print("angle_degrees_cond:", angle_degrees_cond)

x = np.linspace(0, 1.0, 100)
y = np.linspace(0, 1.0, 100)
X, Y = np.meshgrid(x,y)
F = (X**2 + Y**2 - X)
# x-axis limits
x_scale = [-0.005, 1.005]
# y-axis limits
y_scale = [0, 0.9]

# 1. Calculate bin width based on the G001 dataset
iqr_x = np.percentile(G001_weighted, 75) - np.percentile(G001_weighted, 25)
# Check if G001_weighted has sufficient variance
print(f"G001_weighted IQR: {iqr_x}")
if iqr_x == 0:
    print("Warning: Zero interquartile range in G001_weighted.")
    # Set a default bin width or skip the operation
    bin_width_x = 0.1  # Example default value
else:
    bin_width_x = 2 * iqr_x * (len(G001_weighted) ** (-1/3))

bin_width_x = 2 * iqr_x * (len(G001_weighted) ** (-1/3))
bin_width_x = np.nan_to_num(bin_width_x)

iqr_y = np.percentile(S001_weighted, 75) - np.percentile(S001_weighted, 25)
# Similar check for S001_weighted
print(f"S001_weighted IQR: {iqr_y}")
if iqr_y == 0:
    print("Warning: Zero interquartile range in S001_weighted.")
    bin_width_y = 0.1  # Example default value
else:
    bin_width_y = 2 * iqr_y * (len(S001_weighted) ** (-1/3))

# Check bin widths
print(f"Calculated bin_width_x: {bin_width_x}, bin_width_y: {bin_width_y}")

bin_width_y = 2 * iqr_y * (len(S001_weighted) ** (-1/3))
bin_width_y = np.nan_to_num(bin_width_y)

# 2. Use this bin width to determine the number of bins for G001
num_bins_x_G001 = int(np.ceil((np.max(G001_weighted) + np.min(G001_weighted)) / bin_width_x))
num_bins_y_G001 = int(np.ceil((np.max(S001_weighted) + np.min(S001_weighted)) / bin_width_y))

num_bins_x_G001 = num_bins_x_G001 // 4
num_bins_y_G001 = num_bins_y_G001 // 4

# Now, you can use these calculated bin numbers in your hist2d plots:
hist_vals, _, _ = np.histogram2d(G001, S001, bins=(num_bins_x_G001, num_bins_y_G001), weights=I)

vmax = hist_vals.max()
vmin = hist_vals.min()
print("num_bins_x_G001:", num_bins_x_G001)
print("num_bins_y_G001:", num_bins_y_G001)
print("vmax:", vmax)
print("vmin:", vmin)

gray_cmap = ListedColormap(['#c8c8c8'])

# Create the figure and axis only once
fig, ax = plt.subplots(figsize=(8, 6))
h = ax.hist2d(G001, S001, bins = (num_bins_x_G001, num_bins_y_G001), weights=I, cmap = 'nipy_spectral', norm = colors.SymLogNorm(linthresh=100, linscale=1, vmax=vmax, vmin=vmin), zorder=1, cmin=0.01)
ax.set_facecolor('white')
ax.set_xlabel('\n$G$')
ax.set_ylabel('$S$\n')
ax.set_xlim(x_scale)
ax.set_ylim(y_scale)
ax.contour(X,Y,F,[0],colors='black',linewidths=1, zorder=2)
ell_cond = Ellipse(ROI_center_cond, 2 * width_cond, 2 * height_cond, angle=angle_degrees_cond, color='blue', fill=False, linewidth=1)
ax.add_patch(ell_cond)

near_zero = 0.1

# After the hist2d call, create the colorbar
cbar = fig.colorbar(h[3], ax=ax, format=LogFormatter(10, labelOnlyBase=True))

# Define the ticks you want to use, starting with 'near_zero' to represent zero
ticks = [near_zero] + [10**i for i in range(1, int(np.log10(vmax))+1)]

# Set the ticks on the colorbar
cbar.set_ticks(ticks)

# Manually set the labels to ensure they are in the format you want
# Include a label for the 'near_zero' tick, representing zero
tick_labels = ['0'] + [f'$10^{i}$' for i in range(1, int(np.log10(vmax))+1)]
cbar.set_ticklabels(tick_labels)

cbar.set_label('Frequency')

fig.tight_layout()
#fig.savefig('/Users/leelab/FLIM_processing_dir/processed/AUTOcal/phasors/phasor.png', format='png')
plt.show()

# Create the figure and axis only once
fig, ax = plt.subplots(figsize=(8, 6))
h = ax.hist2d(G001, S001, bins = (num_bins_x_G001, num_bins_y_G001), weights=I, cmap = 'nipy_spectral', norm = colors.SymLogNorm(linthresh=100, linscale=1, vmax=vmax, vmin=vmin), zorder=1, cmin=0.01)
ax.set_facecolor('white')
ax.set_xlabel('\n$G$')
ax.set_ylabel('$S$\n')
ax.set_xlim(x_scale)
ax.set_ylim(y_scale)
ax.contour(X,Y,F,[0],colors='black',linewidths=1, zorder=2)
circ_cond = Circle(real_cond_cluster_center, circle_radius_cond, color='blue', fill=False, linewidth=1)
ax.add_patch(circ_cond)
ax.scatter(real_cond_cluster_center[0], real_cond_cluster_center[1], s=50, c='k')
ax.scatter(0.30227996721890404, 0.4592458920992018, s=20, c='b')
# Coordinates for the line
line_x = [0.30227996721890404, 0.30227996721890404]
line_y = [0.4592458920992018, 0.4592458920992018 - radius_ref]

# Plot the line
ax.plot(line_x, line_y, 'b-')


near_zero = 0.1

# After the hist2d call, create the colorbar
cbar = fig.colorbar(h[3], ax=ax, format=LogFormatter(10, labelOnlyBase=True))

# Define the ticks you want to use, starting with 'near_zero' to represent zero
ticks = [near_zero] + [10**i for i in range(1, int(np.log10(vmax))+1)]

# Set the ticks on the colorbar
cbar.set_ticks(ticks)

# Manually set the labels to ensure they are in the format you want
# Include a label for the 'near_zero' tick, representing zero
tick_labels = ['0'] + [f'$10^{i}$' for i in range(1, int(np.log10(vmax))+1)]
cbar.set_ticklabels(tick_labels)

cbar.set_label('Frequency')

fig.tight_layout()
plt.show()

num_rows = x_dim
num_columns = y_dim

# CLUSTER1
list_cond_G = G_cond
list_cond_S = S_cond
new_results_cond = check_either_value_greater_than_zero(list_cond_G, list_cond_S)
cond_list = new_results_cond
cluster_1_list = convert_list_to_array_with_dimensions(cond_list, num_rows, num_columns)

# Scale the values in matrix_dilute_45min to the range of 0-255
rgb_cluster_cond = (cluster_1_list * 255).astype(np.uint8)

# Convert to PIL Image
CL1 = Image.fromarray(rgb_cluster_cond)

# CLUSTER3
list_cond_G_circle = G_cond_circle
list_cond_S_circle = S_cond_circle
new_results_cond_circle = check_either_value_greater_than_zero(list_cond_G_circle, list_cond_S_circle)
cond_list_circle = new_results_cond_circle
cluster_3_list = convert_list_to_array_with_dimensions(cond_list_circle, num_rows, num_columns)

# Scale the values in matrix_dilute_45min to the range of 0-255
rgb_cluster_cond_circle = (cluster_3_list * 255).astype(np.uint8)

# Convert to PIL Image
CL3 = Image.fromarray(rgb_cluster_cond_circle)

T_cond_ellipse = T * cluster_1_list
T_cond_circle = T * cluster_3_list

# Split the final mask based on the original dimensions and save them with corresponding filenames
start_x = 0
for i, (dim, file_name) in enumerate(zip(dimensions, file_names)):
    end_x = start_x + dim[0]

    # Slice the mask
    sliced_mask = cluster_1_list[start_x:end_x, :]

    # Convert the sliced mask to a PIL Image
    mask_image = Image.fromarray((sliced_mask * 255).astype(np.uint8))

    # Save the mask with the corresponding .npz filename
    mask_file_name = os.path.splitext(file_name)[0] + f"_mask_{time.strftime('%Y%m%d_%H%M%S')}.tiff"
    mask_image.save(os.path.join(masks_dir, mask_file_name))

    # Update start_x for the next slice
    start_x = end_x

if __name__ == "__main__":
    # Create the metadata directory
    metadata_dir = os.path.join(output_directory, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    # Create a log file with the current date and time in the metadata directory
    log_file_name = f"parameters_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    log_file_path = os.path.join(metadata_dir, log_file_name)

    # Write the parameters to the log file
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"main_preprocessed_directory = {main_preprocessed_directory}\n")
        log_file.write(f"output_base_directory = {output_base_directory}\n")
        log_file.write(f"Processed second_level directories: {[os.path.basename(d) for d in os.listdir(first_level_directory)]}\n\n")
        log_file.write(f"threshold = {threshold}\n")
        log_file.write(f"G_array shape = {G_array.shape}\n")
        log_file.write(f"S_array shape = {S_array.shape}\n")
        log_file.write(f"I_array shape = {I_array.shape}\n")
        log_file.write(f"T_array shape = {T.shape}\n")
        log_file.write(f"cov_f = {cov_f}\n")
        log_file.write(f"shift = {shift}\n")
        log_file.write(f"radius_ref = {radius_ref}\n")
        log_file.write(f"angle_degrees_cond = {angle_degrees_cond}\n")
        log_file.write(f"width_cond = {width_cond}\n")
        log_file.write(f"height_cond = {height_cond}\n")
        log_file.write(f"ROI_center_cond = {ROI_center_cond}\n")
        log_file.write(f"cond_cluster_center_1 = {cond_cluster_center_1}\n")
        log_file.write(f"cond_cluster_center_2 = {cond_cluster_center_2}\n")
        # Add other relevant parameters...

    print(f"Parameter log saved to {log_file_path}")

    # Save the final masks
    CL1.save(os.path.join(masks_dir, f"mask1_{time.strftime('%Y%m%d_%H%M%S')}.tiff"))
    CL3.save(os.path.join(masks_dir, "circle_unfiltered_mask.tiff"))
    tiff_path_2 = os.path.join(lifetime_dir, "segmented_lifetime_unifiltered_circle.tiff")
    tiff.imwrite(tiff_path_2, T)

    print("Masks saved in directory:", masks_dir)

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Script execution time: {elapsed_time:.2f} seconds")

# --- Configuration Loading ---
def load_config(config_path="config.json"):
    """Loads configuration from a JSON file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        # Basic validation - add more checks as needed
        required_keys = ["npz_dir", "segmented_dir", "gmm_segmentation_params"]
        if not all(key in config for key in required_keys):
            raise ValueError("Config file missing required keys: npz_dir, segmented_dir, gmm_segmentation_params")
        # Example check within params
        if "n_components" not in config["gmm_segmentation_params"]:
             raise ValueError("Config missing gmm_segmentation_params.n_components")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Configuration file {config_path} is not valid JSON.", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error in configuration file: {e}", file=sys.stderr)
        sys.exit(1)

# --- File Handling ---
def load_npz_data(file_path):
    """Loads G, S, A arrays from an NPZ file."""
    try:
        with np.load(file_path) as data:
            # Use keys consistent with wavelet script output
            g = data['G'] 
            s = data['S']
            intensity = data['A']
            if not (g.shape == s.shape == intensity.shape):
                print(f"Warning: Array shapes mismatch in {file_path}. Skipping file.", file=sys.stderr)
                return None, None, None
            return g, s, intensity
    except FileNotFoundError:
        # print(f"Warning: NPZ file not found: {file_path}") # Handled by os.path.exists check
        return None, None, None
    except KeyError as e:
        print(f"Warning: Missing expected key {e} in NPZ file {file_path}. Skipping file.", file=sys.stderr)
        return None, None, None
    except Exception as e:
        print(f"Error loading NPZ file {file_path}: {e}", file=sys.stderr)
        return None, None, None

def save_mask(file_path, mask_array):
    """Saves a boolean or integer mask array as a TIFF image."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Save masks as integer types (e.g., uint8) for compatibility
        if mask_array.dtype == bool:
             img_data = mask_array.astype(np.uint8) * 255 # Convert boolean True to 255
        else:
             img_data = mask_array.astype(np.uint8) # Assume integer mask
        tiff.imsave(file_path, img_data)
    except Exception as e:
        print(f"Error saving mask {file_path}: {e}", file=sys.stderr)

# --- Geometric Masking Functions (from original script) ---
def is_point_inside_circle(point, center, radius):
    distance_sq = (point[0] - center[0])**2 + (point[1] - center[1])**2
    return distance_sq <= radius**2

def are_points_inside_circle(points, center, radius):
    return [is_point_inside_circle(p, center, radius) for p in points]

# Note: This ellipse function from the original script assumes points is a list of tuples.
# We will adapt the GMM part to work with flattened arrays first.
# This function might need modification if used directly on image arrays.
def is_points_inside_rotated_ellipse(center_x, center_y, semi_major_axis, semi_minor_axis, angle_degrees, points):
    if semi_major_axis <= 0 or semi_minor_axis <= 0:
        return [False] * len(points)
        
    angle_radians = math.radians(angle_degrees)
    cos_a = math.cos(angle_radians)
    sin_a = math.sin(angle_radians)
    a_sq = semi_major_axis**2
    b_sq = semi_minor_axis**2

    results = []
    for point in points:
        tx = point[0] - center_x
        ty = point[1] - center_y
        rx = cos_a * tx + sin_a * ty
        ry = -sin_a * tx + cos_a * ty
        # Check if inside ellipse defined by a_sq and b_sq
        # Avoid division by zero if axes are zero
        if a_sq == 0 or b_sq == 0:
             results.append(False)
        else:
            normalized_dist_sq = (rx**2 / a_sq) + (ry**2 / b_sq)
            results.append(normalized_dist_sq <= 1)
    return results

# --- GMM Helper Functions ---
def reassign_gmm_centers(centers, covariances, eigenvalues, eigenvectors):
    """Sorts GMM components based on G-coordinate (centers[:, 0]).

       Assumes 2 components and aims to consistently label the one with lower G.
    """
    if centers.shape[0] != 2:
        # Cannot apply this specific reassignment logic
        print("Warning: GMM center reassignment logic assumes 2 components.")
        return centers, covariances, eigenvalues, eigenvectors

    idx0, idx1 = 0, 1
    if centers[0, 0] > centers[1, 0]: # If center 0 has larger G than center 1
        idx0, idx1 = 1, 0 # Swap indices
        
    sorted_centers = centers[[idx0, idx1]]
    sorted_covariances = covariances[[idx0, idx1]]
    sorted_eigenvalues = eigenvalues[[idx0, idx1]]
    sorted_eigenvectors = eigenvectors[[idx0, idx1]]
    
    return sorted_centers, sorted_covariances, sorted_eigenvalues, sorted_eigenvectors

def get_ellipse_params(covariance, factor=1.0):
    """Calculates ellipse parameters (axes lengths, angle) from a 2x2 covariance matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance) # Use eigh for symmetric matrices
    
    # Eigenvalues are variances along principal axes. Convert to std dev, scale by factor.
    # Ensure eigenvalues are non-negative before sqrt
    eigenvalues = np.maximum(eigenvalues, 0)
    std_devs = np.sqrt(eigenvalues)
    semi_major_axis = factor * std_devs[1] # Larger eigenvalue corresponds to major axis
    semi_minor_axis = factor * std_devs[0]
    
    # Angle of the first eigenvector (corresponding to eigenvalue[0]) with x-axis
    angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle_deg = np.degrees(angle_rad)
    
    # Return eigenvalues and vectors too, as they might be useful
    return semi_major_axis, semi_minor_axis, angle_deg, eigenvalues, eigenvectors

# --- Main Processing Function ---
def perform_gmm_segmentation(g_map, s_map, intensity_map, params):
    """
    Applies GMM segmentation to a single set of G, S, Intensity maps.

    Args:
        g_map (np.array): Filtered G-coordinate map.
        s_map (np.array): Filtered S-coordinate map.
        intensity_map (np.array): Intensity map.
        params (dict): Dictionary containing GMM parameters from config.

    Returns:
        dict: Dictionary containing output masks (or None if error).
              Keys might include 'component_mask', 'ellipse_mask', 'circle_mask'.
    """
    
    # --- 1. Data Preparation ---
    print("  Preparing data for GMM...")
    intensity_threshold = params.get("intensity_threshold", 0.0)
    
    # Create a mask for pixels above the intensity threshold
    valid_intensity_mask = intensity_map > intensity_threshold
    
    # Flatten the G, S maps and filter by intensity
    g_flat = g_map[valid_intensity_mask]
    s_flat = s_map[valid_intensity_mask]
    
    if g_flat.size == 0 or s_flat.size == 0:
        print("  Warning: No pixels above intensity threshold. Cannot perform GMM.", file=sys.stderr)
        return None
        
    # Combine G and S into a 2D array for GMM input (N_pixels, 2)
    phasor_data = np.vstack((g_flat, s_flat)).T
    
    # --- 2. GMM Fitting ---
    n_components = params.get("n_components", 2)
    print(f"  Fitting GMM with {n_components} components...")
    try:
        gmm = GaussianMixture(n_components=n_components, 
                              covariance_type='full', 
                              random_state=0, # for reproducibility
                              reg_covar=1e-6) # Regularization for stability
        gmm.fit(phasor_data)
    except Exception as e:
        print(f"  Error during GMM fitting: {e}", file=sys.stderr)
        return None
        
    print("  GMM fitting complete.")
    
    # --- 3. Generate Component Mask ---
    print("  Generating component assignment mask...")
    # Predict component for all pixels (even those below threshold - assign them a default label like -1 or 0?)
    # Let's create a full-size mask initialized to 0 (background/below threshold)
    component_mask = np.zeros(g_map.shape, dtype=np.int8)
    # Predict labels only for pixels used in fitting
    labels = gmm.predict(phasor_data)
    # Assign labels (+1 to avoid 0 being background) to the valid pixels
    component_mask[valid_intensity_mask] = labels + 1 
    
    # --- 4. Generate Geometric Masks (Optional, based on params) ---
    # This part replicates the logic suggested by the original script's functions
    # It uses GMM results (means/covariances) to define ellipses/circles.
    
    # Get GMM parameters (means, covariances)
    means = gmm.means_
    covariances = gmm.covariances_
    
    # Calculate ellipse params for each component (needed for reassign and ellipse mask)
    ellipse_params_list = [get_ellipse_params(cov, 1.0) for cov in covariances]
    all_eigenvalues = np.array([p[3] for p in ellipse_params_list])
    all_eigenvectors = np.array([p[4] for p in ellipse_params_list])
    
    # Reassign/Sort components (assuming n_components=2, based on original script logic)
    if n_components == 2:
        means, covariances, all_eigenvalues, all_eigenvectors = reassign_gmm_centers(
            means, covariances, all_eigenvalues, all_eigenvectors
        )
        # Update ellipse params list after sorting
        ellipse_params_list = [
            get_ellipse_params(cov, 1.0) for cov in covariances
        ] 
        print("  Reassigned GMM components based on G-coordinate (comp 0 expected lower G).")
        # Now index 0 should correspond to the component with lower G-mean ('good' cluster in original?)
        # Index 1 corresponds to the component with higher G-mean ('bad' cluster in original?)

    # Initialize empty masks
    ellipse_mask = np.zeros(g_map.shape, dtype=bool)
    circle_mask = np.zeros(g_map.shape, dtype=bool)

    # --- Ellipse Mask (based on first component after potential sorting) ---
    print("  Generating ellipse mask (based on component 0)...")
    cov_factor_ellipse = params.get("covariance_factor_ellipse", 1.0)
    center_ellipse = means[0] # Use mean of the first component
    # Recalculate ellipse params with factor
    major_axis_e, minor_axis_e, angle_e, _, _ = get_ellipse_params(covariances[0], cov_factor_ellipse)
    
    # Apply ellipse check to all pixels (using their G/S values)
    # Create list of points (G, S) for all pixels
    all_points = np.stack((g_map.ravel(), s_map.ravel()), axis=-1)
    inside_ellipse_flat = is_points_inside_rotated_ellipse(
        center_ellipse[0], center_ellipse[1],
        major_axis_e, minor_axis_e, angle_e,
        all_points
    )
    ellipse_mask = np.array(inside_ellipse_flat).reshape(g_map.shape)
    # Combine with intensity threshold? Original script logic unclear here.
    # Let's only keep pixels inside ellipse AND above intensity threshold.
    ellipse_mask = ellipse_mask & valid_intensity_mask

    # --- Circle Mask (based on reference center from config) ---
    print("  Generating circle mask (based on reference center)...")
    ref_center_g = params.get("reference_center_G", 0.0)
    ref_center_s = params.get("reference_center_S", 0.0)
    cov_factor_circle = params.get("covariance_factor_circle", 1.0)
    # The original script used cov_f_circle but didn't explicitly define the radius.
    # It might have used the std dev from the GMM? Let's assume it uses the std dev
    # of the first component scaled by cov_factor_circle as the radius.
    # Calculate average std dev for the first component
    std_devs_c = np.sqrt(np.maximum(all_eigenvalues[0], 0))
    radius_circle = cov_factor_circle * np.mean(std_devs_c) # Use mean std dev as radius base

    inside_circle_flat = are_points_inside_circle(
        all_points,
        (ref_center_g, ref_center_s),
        radius_circle
    )
    circle_mask = np.array(inside_circle_flat).reshape(g_map.shape)
    circle_mask = circle_mask & valid_intensity_mask
    
    # --- 5. Collect Output Masks ---
    output_masks = {
        "component": component_mask, # Mask showing component assignment (0=bg, 1=comp0, 2=comp1, ...)
        "ellipse": ellipse_mask,     # Boolean mask for pixels within ellipse & above threshold
        "circle": circle_mask       # Boolean mask for pixels within circle & above threshold
    }

    print("  Segmentation complete.")
    return output_masks


def main():
    """Main execution: Load config, find NPZ files, run GMM segmentation, save masks."""
    config = load_config()
    
    npz_dir = config["npz_dir"]
    segmented_dir = config["segmented_dir"]
    gmm_params = config["gmm_segmentation_params"]

    print(f"Starting GMM Segmentation")
    print(f"Input (NPZ datasets) directory: {npz_dir}")
    print(f"Output (segmented masks) directory: {segmented_dir}")
    print(f"GMM Parameters: {gmm_params}")

    if not os.path.isdir(npz_dir):
        print(f"Error: NPZ directory not found: {npz_dir}", file=sys.stderr)
        sys.exit(1)
        
    processed_count = 0
    skipped_count = 0

    # Iterate through condition subdirectories in the NPZ directory
    for condition_subdir_name in os.listdir(npz_dir):
        condition_subdir_path = os.path.join(npz_dir, condition_subdir_name)
        if not os.path.isdir(condition_subdir_path):
            continue 

        print(f"\nProcessing condition: {condition_subdir_name}")

        # Find NPZ files in the condition subdirectory
        try:
            npz_files = sorted([
                f for f in os.listdir(condition_subdir_path) 
                if f.lower().endswith("_processed.npz") # Match wavelet output naming
            ])
            if not npz_files:
                 print(f" Warning: No *_processed.npz files found in {condition_subdir_name}. Skipping.")
                 skipped_count +=1
                 continue
        except OSError as e:
             print(f" Error listing files in {condition_subdir_path}: {e}. Skipping condition.", file=sys.stderr)
             skipped_count += 1
             continue
             
        print(f" Found {len(npz_files)} NPZ files to process.")

        # Process each NPZ file independently
        for npz_filename in npz_files:
            npz_file_path = os.path.join(condition_subdir_path, npz_filename)
            base_name = os.path.splitext(npz_filename)[0].replace("_processed", "") # Get e.g., "1"
            
            print(f" Processing file: {npz_filename}")
            
            try:
                # Load data
                g_map, s_map, intensity_map = load_npz_data(npz_file_path)
                
                if g_map is None: # Check if loading failed
                    skipped_count += 1
                    continue 

                # Perform segmentation
                output_masks = perform_gmm_segmentation(g_map, s_map, intensity_map, gmm_params)

                if output_masks is not None: # Check if segmentation was successful
                    # Define output directory for this condition
                    output_condition_dir = os.path.join(segmented_dir, condition_subdir_name)
                    
                    print(f"  Saving output masks to: {output_condition_dir}")
                    # Save each generated mask
                    for mask_type, mask_data in output_masks.items():
                        mask_filename = f"{base_name}_{mask_type}_mask.tiff"
                        mask_out_path = os.path.join(output_condition_dir, mask_filename)
                        save_mask(mask_out_path, mask_data)
                    
                    processed_count += 1
                else:
                    skipped_count += 1 # Segmentation failed 

            except Exception as e:
                print(f" Error processing file {npz_filename}: {e}", file=sys.stderr)
                # import traceback
                # print(traceback.format_exc(), file=sys.stderr)
                skipped_count += 1

    print(f"\nGMM Segmentation finished.")
    print(f"Successfully processed files: {processed_count}")
    print(f"Skipped/failed files: {skipped_count}")


if __name__ == "__main__":
    main()