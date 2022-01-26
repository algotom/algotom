# ===========================================================================
# ===========================================================================
# Copyright (c) 2021 Nghia T. Vo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================
# Author: Nghia T. Vo
# E-mail:  
# Description: Examples of how to use the Algotom package to fully reconstruct
# a grid scan.
# ===========================================================================

"""
The following examples show how to use Algotom to calculate overlaps between 
cells of a grid scan which will be used for fully reconstructing the data.

Raw data is at: https://zenodo.org/record/4614789 (searching on Zenodo to
download other parts: *Part01, ..., *Part24 ).
There're 24 scans (8 rows x 3 columns) of projection images (hdf files) under
folders named from "scan_00052" to "scan_00075". Dark-field and flat-field 
images are under the folder named "scan_00051".

Referring to "example_01_*.py" to know how to find key-paths and datasets
in a hdf/nxs file.

Referring to "example_06_*.py" to know how to include distortion correction.

Output of the script are:
- Overlap values (overlap-area, overlap-sides) between rows of the grid scan.
- Overlap values between columns in each row of the grid scan.
- Overlap values (and center-of-rotations) to convert 380-sinograms to 
  180-sinograms for each row of the grid scan.
"""

import timeit
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.removal as remo
import algotom.prep.calculation as calc
import algotom.prep.conversion as conv
import algotom.prep.filtering as filt
import algotom.util.utility as util
import algotom.rec.reconstruction as reco


input_base = "D:/data/"
key_path = "/entry/data/data"
# Where to save the outputs
output_base = "D:/overlap_calculation/"

# Get scan names.
proj_scan = np.arange(52,76)
df_scan = 51
prefix = "0000" + str(df_scan)
df_name = "scan_" + prefix[-5:]
proj_name = []
for i in proj_scan:
    prefix = "0000" + str(i)
    proj_name.append("scan_" + prefix[-5:])

# Separate scans to 8 rows x 3 columns
num_scan_total = len(proj_scan)
num_scan_col = 3
num_scan_row = num_scan_total//num_scan_col
scan_grid = np.reshape(proj_scan, (num_scan_row, num_scan_col))
overlap_window = 100 # Used to calculate the overlap-area and overlap-side.

# Load dark-field and flat-field images, average each of them.
flat_path = losa.find_file(input_base + "/" + df_name + "/*flat*")[0]
flat_field = np.mean(losa.load_hdf(flat_path, key_path)[:], axis = 0)
dark_path = losa.find_file(input_base + "/" + df_name + "/*dark*")[0]
dark_field = np.mean(losa.load_hdf(dark_path, key_path)[:], axis = 0)
# Load projection images of each scan as hdf objects
data_objects = []
list_depth = []
list_height = []
list_width = []
for i in range(num_scan_total):
    file_path = losa.find_file(input_base + "/" + proj_name[i] + "/*proj*")[0]
    hdf_object = losa.load_hdf(file_path, key_path)
    (depth1, height1, width1) = hdf_object.shape
    list_depth.append(depth1)
    list_height.append(height1)
    list_width.append(width1)
    data_objects.append(hdf_object)

# To crop data if the beam size smaller than the FOV
crop_top = 10 # To remove time-stamp
crop_bottom = 0
crop_left = 0
crop_right = 0

# Number of projections may be different around 1 frame between scans caused
# by the synchronizer in the fly-scan mode.
depth = min(list_depth)
height0 = min(list_height)
width0 = min(list_width)
height = height0 -crop_top - crop_bottom
width = width0 - crop_left - crop_right

top = crop_top
bot = height0 - crop_bottom
left = crop_left
right = width0 - crop_right


print("!!! Start !!!")
time_start = timeit.default_timer()

# Step 1: Generating projection-images used to calculate overlaps between
# rows of the grid scan. Ideally, slices along axis 2 of each 3D tomographic
# data should be used to improve the reliability of overlapping calculation.
# However, extracting slices along axis 2 of a hdf file is very slow. 
pro_idx = depth // 2 # Index of projection-images
blob_mask = remo.generate_blob_mask(flat_field[top:bot, left:right], 71, 3.0) # Generate a blob mask
proj_list = [[] for i in range(num_scan_total)]
print("|-> Extract projection-images between grid-rows for determining overlaps")
for row_idx in range(num_scan_row):
    scan_name = "Ver_scan"
    for i in range(num_scan_col):
        tmp_name = "_" + str(scan_grid[row_idx, i])
        scan_name = scan_name + tmp_name
    output_base1 = output_base + "/" + scan_name + "/"
    losa.make_folder(output_base1)
    t_start = timeit.default_timer()
    for j in range(num_scan_col):
        grid_idx = j + row_idx*num_scan_col
        image_pro = corr.flat_field_correction(data_objects[grid_idx][pro_idx, top:bot, left:right],
                                               flat_field[top:bot, left:right], dark_field[top:bot, left:right],
                                               ratio=1.0, use_dark=False)
        image_pro = remo.remove_blob(image_pro, blob_mask)
        proj_list[grid_idx] = image_pro
        out1 = "0000" + str(j)
        out2 = "0000" + str(row_idx)
        out3 = "0000" + str(pro_idx)
        name = "row_" + out2[-4:] + "_col_" + out1[-4:] + "_pro_" +out3[-4:] + ".tif"
        losa.save_image(output_base1 + "/" + name, image_pro)
    t_stop = timeit.default_timer()
    print("   Done row: {0}. Time :{1}".format(row_idx, t_stop - t_start))
print("|-> Find overlap-area and overlap-side between projection-images")
# Calculate overlaps between grid-rows
ver_overlap_mat = np.zeros((num_scan_row-1, num_scan_col, 2), dtype = np.int16)
col_use = num_scan_col//2 # Use grid-columns having the sample in their FOV 
for i in range(num_scan_row - 1):
    grid_idx = col_use + i * num_scan_col
    grid_idx1 = col_use + (i + 1) * num_scan_col
    overlap_area, overlap_side, _ = calc.find_overlap(np.transpose(proj_list[grid_idx]),
                                                      np.transpose(proj_list[grid_idx1]),
                                                      overlap_window, norm=True,
                                                      use_overlap=True)
    if overlap_side == 1:
        ver_overlap_mat[i, :, 1] = 1  # Use the same across grid-row
    ver_overlap_mat[i, :, 0] = np.int16(overlap_area) # Use the same across grid-row
    print("   Row {0}-{1}: Overlap {3}. Side {2}".format(i, i+1, overlap_side, overlap_area))

side_check = np.max(np.abs(np.diff(ver_overlap_mat[:, 0, 1])))
if side_check > 0:
    print("------!!! Warning !!!-------")
    print("Overlap-side results are not consistent."
          " You may want to use some of the following options:")
    print(" - Select other projection-indices to make sure there're samples in the FOV")
    print(" - Change the window-size in the overlap-determination method")
    print(" - Select norm=True  in the overlap-determination method")
    print(" - Select overlap_area=True in the overlap-determination method")

# Save the results
np.save(output_base + "/Vertical_overlaps.npy", ver_overlap_mat)

# Step 2: Generating sinogram-images used to calculate overlaps between
# columns of the grid scan. 
sino_list = [[] for i in range(num_scan_total)]
sino_idx = height // 2 # Index of sinogram-images
print("|-> Extract sinogram-images between grid-columns for determining overlaps")
for row_idx in np.arange(num_scan_row):
    scan_name = "Hor_scan"
    for i in range(num_scan_col):
        tmp_name = "_" + str(scan_grid[row_idx, i])
        scan_name = scan_name + tmp_name
    output_base1 = output_base + "/" + scan_name + "/"
    losa.make_folder(output_base1)
    t_start = timeit.default_timer()
    for j in range(num_scan_col):
        grid_idx = j + row_idx * num_scan_col
        sino_list[grid_idx] = corr.flat_field_correction(data_objects[grid_idx][0:depth,sino_idx + top, left:right],
                                                         flat_field[sino_idx + top, left:right],
                                                         dark_field[sino_idx + top, left:right])
        out1 = "0000" + str(j)
        out2 = "0000" + str(row_idx)
        out3 = "0000" + str(sino_idx)
        name = "row_" + out2[-4:] + "_col_" + out1[-4:] + "_sino_" +out3[-4:] + ".tif"
        losa.save_image(output_base1 + "/" + name, sino_list[grid_idx])
    t_stop = timeit.default_timer()
    print("   Done row {0} . Time {1}".format(row_idx,  t_stop - t_start))
# Calculate overlaps between grid-columns.
print("|-> Find overlap-area and overlap-side between sinogram-images")
hor_overlap_mat = np.zeros((num_scan_row, num_scan_col-1, 2), dtype = np.float32)
for i in range(num_scan_row):
    print("   Results of row {}:".format(i))
    for j in np.arange(num_scan_col-1):
        grid_idx = j + i * num_scan_col
        sino1 = sino_list[grid_idx]
        sino2 = sino_list[grid_idx + 1]
        check1 = util.detect_sample(sino1) # Detect sample
        check2 = util.detect_sample(sino2) # Detect sample
        if check1 and check2:
            overlap_area, overlap_side, _ = calc.find_overlap(sino1, sino2, overlap_window)
        else:
            overlap_area = 0
            overlap_side = -1
        if overlap_side == 1:
            hor_overlap_mat[i, j, 1] = 1
        hor_overlap_mat[i, j, 0] = overlap_area
        print("      Col {0}-{1}: Overlap {3}. Side {2}".format(j, j + 1, overlap_side, overlap_area))
# If there're areas without sample, use calculation results of adjacent cells.
print(" -> Use nearby results for non-sample areas...")
hor_overlap_mat = util.fix_non_sample_areas(hor_overlap_mat)
print(" -> Done !!!")
# Save the results
np.save(output_base + "/Horizontal_overlaps.npy", hor_overlap_mat)

# Step 3: Find the center-of-rotation (COR) at each grid-row,
# to convert a 360-degree offset-COR sinogram to a 180-degree.
# We do that by finding the overlap between two halves of 360-degree sinograms
print("|-> Find the center-of-rotation of each grid-row")
two_halves_sino_overlaps = np.zeros((num_scan_row, 3), dtype = np.float32)
for i in range(num_scan_row):
    sinos = sino_list[i * num_scan_col:i * num_scan_col + num_scan_col]
    overlaps = hor_overlap_mat[i]
    sino_360 = conv.stitch_image_multiple(sinos, overlaps, norm=True)
    (center0, overlap, side,_) = calc.find_center_360(sino_360, overlap_window)
    two_halves_sino_overlaps[i] = np.asarray([center0, overlap, side])
    print("   Grid-row: {0} ->  COR: {1}. Overlap: {2}. Side: {3}".format(i, center0, overlap, side))
    sino_180, center1 = conv.convert_sinogram_360_to_180(sino_360, center0)
    out1 = "0000" + str(i)
    name = "stitched_sinogram_row_" + out1[-4:] + ".tif"
    losa.save_image(output_base + "/sitched_sinograms/" + name, sino_180)
# Save the results
np.save(output_base + "/Two_halves_sinogram_overlaps.npy", two_halves_sino_overlaps)

# Step 4 (optional): Stitching all projection-images together.
print("|-> Demonstrate by stitching all projection-images")
# Stitch images along the horizontal direction.
list_stitched_pro = []
for i in range(num_scan_row):
    proj_0 = np.copy(proj_list[i * num_scan_col])
    for j in range(1, num_scan_col):
        side =  hor_overlap_mat[i, j-1, 1]
        overlap = hor_overlap_mat[i, j-1, 0]
        proj_0 = conv.stitch_image(proj_0, np.asarray(proj_list[j + i * num_scan_col]), overlap, side)
    out1 = "0000" + str(i)
    name = "row_" + out1[-4:] + ".tif"
    losa.save_image(output_base + "/stitched_projections/" + name, proj_0)
    list_stitched_pro.append(proj_0)
# Stitch images along the vertical direction.
width_total = min([mat.shape[1] for mat in list_stitched_pro])
proj_0 = np.transpose(np.copy(list_stitched_pro[0]))[0:width_total]
for i in range(1, num_scan_row):
    side =  ver_overlap_mat[i-1, 0, 1]
    overlap = ver_overlap_mat[i-1, 0, 0]
    mat_tmp = np.transpose(np.asarray(list_stitched_pro[i]))[0:width_total]
    proj_0 = conv.stitch_image(proj_0, mat_tmp, overlap, side)
proj_0 = np.transpose(proj_0)
name = "fully_stitched_projection.tif"
losa.save_image(output_base + "/" + name, proj_0)

time_stop = timeit.default_timer()
print("All done!!! Total time cost: {}".format(time_stop - time_start))
print("----------------------------------------------------------------------\n")
print("Metadata files of overlap-values to be used for full reconstruction of the grid scan:")
print("For stitching vertically: " + output_base + "/Vertical_overlaps.npy")
print("For stitching horizontally: " + output_base + "/Horizontal_overlaps.npy")
print("For converting 360-sinograms to 180-sinograms: " + output_base + "/Two_halves_sinogram_overlaps.npy")
print("----------------------------------------------------------------------\n")
