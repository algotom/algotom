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
The following examples show how to use Algotom to fully reconstruct a grid
scan as a whole.
Running "example_07_*_step_01.py to get overlap values before trying this
script.

Raw data is at: https://zenodo.org/record/4614789 (searching on Zenodo to
download other parts: *Part01, ..., *Part24 ).
There're 24 scans (8 rows x 3 columns) of projection images (hdf files) under
folders named from "scan_00052" to "scan_00075". Dark-field and flat-field 
images are under the folder named "scan_00051".

Referring to "example_01_*.py" to know how to find key-paths and datasets
in a hdf/nxs file.

Referring to "example_06_*.py" to know how to include distortion correction.

In the following example, whole sample is reconstructed and stitched without
generating intermediate files. To do that, we have to map requested
slice-indices of whole grid to slice-indices of each tomographic dataset.
This adds complexity to the implementation below. To avoid that, users can
reconstruct each grid-row independently, then perform vertical stitching in
the reconstruction space. The cons of this approach is that slices in the
vertical overlapping areas are duplicated on disk.
"""

import timeit
import numpy as np
import scipy.ndimage as ndi
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.removal as remo
import algotom.prep.calculation as calc
import algotom.prep.conversion as conv
import algotom.prep.filtering as filt
import algotom.util.utility as util
import algotom.rec.reconstruction as reco


# Input data
input_base = "D:/data/"
key_path = "/entry/data/data"

# Overlap values determined from *_step_01.py 
hor_overlap_mat = np.load("D:/overlap_calculation/Horizontal_overlaps.npy")
ver_overlap_mat = np.load("D:/overlap_calculation/Vertical_overlaps.npy")
half_sino_overlap_mat = np.load("D:/overlap_calculation/Two_halves_sinogram_overlaps.npy")
# Convert these values to integers.
# Algotom allows to stitch images with sub-pixel accuracy but there's not much different
# between using float values and integer values for this data.
hor_overlap_mat = np.int16(hor_overlap_mat)
ver_overlap_mat = np.int16(ver_overlap_mat)
half_sino_overlap_mat = np.int16(half_sino_overlap_mat)

# Where to save the outputs
output_base = "D:/Full_reconstruction/"

# Because the full size of reconstructed grid-scan is huge (~ 11.3 TB for this data), 
# we should reconstruct chunk-by-chunk of the data. Each of the chunk can be
# reconstructed independently and in parallel.
num_output = 12 # Separate reconstruction output to 12 files.
dataset_idx = 1 # Index of the file to be reconstructed.
# Can be passed by another script to run in parallel as below
# dataset_idx = int(sys.argv[1])
dataset_idx = np.clip(dataset_idx, 0, num_output - 1)

# Define the range of slices for reconstruction.
# Note that for slices in the overlapping area between two grid-rows,
# we need to combine slices from two areas using a linear ramp
# as shown in Fig. 24 of the paper:https://doi.org/10.1364/OE.418448
start_grid_slice = 100
stop_grid_slice = -1

# Select a number of sinograms to be processed in one go.
# This is to reduce the IO overhead of loading a hdf file.
# Making sure that data can fit to the RAM memory as the size of a slice is big. 
slice_chunk = 8

# Crop images, note that it must be the same as used in *_step_01.py 
crop_top = 10 # To remove time-stamp
crop_bottom = 0
crop_left = 0
crop_right = 0


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

print("!!! Start !!!")
time_start = timeit.default_timer()

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

# Get the full size of reconstructed data
list_width = np.zeros(num_scan_row, dtype = np.int16)
for i in range(num_scan_row):
    sino_width = num_scan_col * width - np.sum(hor_overlap_mat[i, :, 0])
    list_width[i] = 2 * sino_width - half_sino_overlap_mat[i, 1]
total_width = np.max(list_width)
total_height = num_scan_row * height - np.sum(ver_overlap_mat[:, 0, 0])
if stop_grid_slice < 0:
    stop_grid_slice = total_height + stop_grid_slice + 1

total_height = stop_grid_slice - start_grid_slice
ver_side = ver_overlap_mat[0, 0, 1]
print("\n -> Total width of a reconstructed slice: {}".format(total_width))
print(" -> Total number of slices: {}".format(total_height))
print(" -> Stitching side between two next grid-rows: {} (0->'top'; 1->'bottom)".format(ver_side))


# Options to remove artfacts
opt1 = {"method": "remove_zinger", "para1": 0.08, "para2": 1}
opt2 = {"method": "remove_all_stripe", "para1": 3.0, "para2": 61, "para3": 21}

# Denoising using the low-pass filter.
denoise_ratio = 300

# Define functions for generating a chunk of sinograms which are pre-processed,
# stitched and converted to a 180-degree sinogram ready for reconstruction.
def stitch_sinogram_chunk(sino_list, grid_row_idx, hor_overlap_mat, half_sino_overlap_mat, total_width):
    """
    Stitch a chunk of sinograms and convert them to 180-degree sinograms 
    """
    num_scan_col = len(hor_overlap_mat[0]) + 1
    num_sino = sino_list[0].shape[1]
    sino_chunk = []
    for j in np.arange(num_sino):
        sino_list1 = [sino_list[k][:, j, :] for k in range(num_scan_col)]
        sino_360 = conv.stitch_image_multiple(sino_list1, hor_overlap_mat[grid_row_idx], norm=True)
        cor = half_sino_overlap_mat[grid_row_idx][0]
        (sino_180, _) = conv.convert_sinogram_360_to_180(sino_360, cor, norm=True)
        ncol_180 = sino_180.shape[1]
        if total_width > ncol_180:
            sino_180 = np.pad(sino_180,((0, 0), (0, total_width - ncol_180)),mode = 'edge')
            sino_180 = ndi.shift(sino_180, (0, (total_width - ncol_180-1)/2.0), 
                                 order=3, prefilter=True, mode = "nearest")
        sino_chunk.append(sino_180)
    return np.asarray(sino_chunk, dtype=np.float32)


def get_sino_list(idx_start, idx_stop, grid_row_idx, num_scan_col, data_objects, flat_field, dark_field, 
                  crop, **options):
    """
    Load sinograms and apply pre-processing methods
    """
    (top, bot, left, right) = crop
    sino_list = [[] for j in range(num_scan_col)]
    for j in range(num_scan_col):
        grid_idx = j + grid_row_idx * num_scan_col
        if idx_stop >= idx_start:
            sino_list[j] = corr.flat_field_correction(data_objects[grid_idx][:, idx_start: idx_stop + 1, left:right],
                                                      flat_field[idx_start: idx_stop + 1, left:right], 
                                                      dark_field[idx_start: idx_stop + 1, left:right], **options)
        else:
            sino_list[j] = np.flip(corr.flat_field_correction(data_objects[grid_idx][:, idx_stop: idx_start + 1, left:right],
                                                              flat_field[idx_start: idx_stop + 1, left:right],
                                                              dark_field[idx_start: idx_stop + 1, left:right],
                                                              **options), axis=1)
    return sino_list


def generate_sinogram_chunk(start_sino, stop_sino, height, ver_overlap_mat, hor_overlap_mat, 
                            half_sino_overlap_mat, data_objects, flat_field, dark_field, 
                            total_width, crop, **options):
    """
    It may look complicated but the idea is simple. The requested singorams may stay completely inside, 
    partly inside, or completely outside overlapping areas. If sinograms stay inside overlapping areas
    we have to combine them (from adjacent grid-rows) using a linear ramp.
    """
    (top, bot, left, right) = crop
    locations = util.locate_slice_chunk(start_sino, stop_sino, height, ver_overlap_mat)
    num_scan_col = len(hor_overlap_mat[0]) + 1
    if len(locations) == 0:
        raise ValueError("Can't locate the selected range of indices {0} -> {1}".format(start_sino, stop_sino))
    if len(locations) == 1:
        # If slices stay completely outside an overlapping area
        grid_row_idx = locations[0][0][0]
        idx_start = locations[0][0][1] + top
        idx_stop = locations[0][-1][1] + top
        sino_list = get_sino_list(idx_start, idx_stop, grid_row_idx, num_scan_col, data_objects, 
                                  flat_field, dark_field, crop, **options)
        sino_chunk = stitch_sinogram_chunk(sino_list, grid_row_idx, hor_overlap_mat, 
                                           half_sino_overlap_mat, total_width)
    else:
        result1 = locations[0]
        result2 = locations[1]
        if len(result1) == len(result2):
            # If slices stay completely inside an overlapping area
            grid_row_idx1 = result1[0][0]
            grid_row_idx2 = result2[0][0]
            idx_start = result1[0][1] + top
            idx_stop = result1[-1][1] + top
            sino_list = get_sino_list(idx_start, idx_stop, grid_row_idx1, num_scan_col, data_objects, 
                                      flat_field, dark_field, crop, **options)
            sino_chunk1 = stitch_sinogram_chunk(sino_list, grid_row_idx1, hor_overlap_mat, 
                                                half_sino_overlap_mat, total_width)

            idx_start = result2[0][1] + top
            idx_stop = result2[-1][1] + top
            sino_list = get_sino_list(idx_start, idx_stop, grid_row_idx2, num_scan_col, data_objects,
                                      flat_field, dark_field, crop, **options)
            sino_chunk2 = stitch_sinogram_chunk(sino_list, grid_row_idx2, hor_overlap_mat,
                                                half_sino_overlap_mat, total_width)
            # Combine sinograms of two grid-rows
            sino_chunk = []
            for j in range(len(result1)):
                fact1 = result1[j][2]
                fact2 = result2[j][2]
                sino_sum = sino_chunk1[j]*fact1 + sino_chunk2[j]*fact2
                sino_chunk.append(sino_sum)
        elif len(result1) > len(result2):
            # If there're more slices of grid-row 1 in the overlapping area than the ones of grid-row 2
            len1 = len(result1)
            len2 = len(result2)
            result1a = result1[:len1-len2]
            result1b = result1[len1-len2:]

            grid_row_idx = result1a[0][0]
            idx_start = result1a[0][1] + top
            idx_stop = result1a[-1][1] + top
            sino_list = get_sino_list(idx_start, idx_stop, grid_row_idx, num_scan_col, data_objects,
                                      flat_field, dark_field, crop, **options)
            sino_chunk_a = stitch_sinogram_chunk(sino_list, grid_row_idx, hor_overlap_mat,
                                                 half_sino_overlap_mat, total_width)

            grid_row_idx1 = result1b[0][0]
            grid_row_idx2 = result2[0][0]
            idx_start = result1b[0][1] + top
            idx_stop = result1b[-1][1] + top
            sino_list = get_sino_list(idx_start, idx_stop, grid_row_idx1, num_scan_col, data_objects,
                                      flat_field, dark_field, crop, **options)
            sino_chunk1 = stitch_sinogram_chunk(sino_list, grid_row_idx1, hor_overlap_mat,
                                                half_sino_overlap_mat, total_width)

            idx_start = result2[0][1] + top
            idx_stop = result2[-1][1] + top
            sino_list = get_sino_list(idx_start, idx_stop, grid_row_idx2, num_scan_col, data_objects,
                                      flat_field, dark_field, crop, **options)
            sino_chunk2 = stitch_sinogram_chunk(sino_list, grid_row_idx2, hor_overlap_mat,
                                                half_sino_overlap_mat, total_width)
            sino_chunk_b = []
            for j in range(len2):
                fact1 = result1b[j][2]
                fact2 = result2[j][2]
                sino_sum = sino_chunk1[j]*fact1 + sino_chunk2[j]*fact2
                sino_chunk_b.append(sino_sum)
            sino_chunk_b = np.asarray(sino_chunk_b, dtype=np.float32)
            sino_chunk = np.concatenate((sino_chunk_a, sino_chunk_b), axis = 0)
        else:
            # If there're less slices of grid-row 1 in the overlapping area than grid-row 2
            len1 = len(result1)
            len2 = len(result2)
            result2a = result2[:len1]
            result2b = result2[len1:]

            grid_row_idx = result2b[0][0]
            idx_start = result2b[0][1] + top
            idx_stop = result2b[-1][1] + top
            sino_list = get_sino_list(idx_start, idx_stop, grid_row_idx, num_scan_col, data_objects,
                                      flat_field, dark_field, crop, **options)
            sino_chunk_a = stitch_sinogram_chunk(sino_list, grid_row_idx, hor_overlap_mat,
                                                 half_sino_overlap_mat, total_width)

            grid_row_idx1 = result1[0][0]
            grid_row_idx2 = result2a[0][0]
            idx_start = result1[0][1] + top
            idx_stop = result1[-1][1] + top
            sino_list = get_sino_list(idx_start, idx_stop, grid_row_idx1, num_scan_col, data_objects,
                                      flat_field, dark_field, crop, **options)
            sino_chunk1 = stitch_sinogram_chunk(sino_list, grid_row_idx1, hor_overlap_mat,
                                                half_sino_overlap_mat, total_width)

            idx_start = result2a[0][1] + top
            idx_stop = result2a[-1][1] + top
            sino_list = get_sino_list(idx_start, idx_stop, grid_row_idx2, num_scan_col, data_objects,
                                      flat_field, dark_field, crop, **options)
            sino_chunk2 = stitch_sinogram_chunk(sino_list, grid_row_idx2, hor_overlap_mat,
                                                half_sino_overlap_mat, total_width)
            sino_chunk_b = []
            for j in range(len1):
                fact1 = result1[j][2]
                fact2 = result2a[j][2]
                sino_sum = sino_chunk1[j]*fact1 + sino_chunk2[j]*fact2
                sino_chunk_b.append(sino_sum)
            sino_chunk_b = np.asarray(sino_chunk_b, dtype=np.float32)
            sino_chunk = np.concatenate((sino_chunk_a, sino_chunk_b), axis = 0)
    return np.asarray(sino_chunk, dtype=np.float32)


# The following section is to determine slice-indices to be reconstructed
# and how to run through a loop to reconstruct chunk-by-chunk of them.

list_grid_slice = np.array_split(np.arange(start_grid_slice, stop_grid_slice), num_output)
start_slice = list_grid_slice[dataset_idx][0]
stop_slice = list_grid_slice[dataset_idx][-1] + 1
total_slice = stop_slice - start_slice
offset = start_slice
if slice_chunk > total_slice:
    slice_chunk = total_slice
num_iter = total_slice // slice_chunk
num_rest = total_slice - num_iter * slice_chunk
print("To check. Start: {0}, Stop {1}".format(start_slice, stop_slice))

# -------------------------------------------------
# -------------------------------------------------
# Perform reconstruction
# -------------------------------------------------
# Open hdf stream for saving results.
name_part = "000" + str(dataset_idx)
file_name = "recon_part_" + name_part[-3:] + ".hdf"
output_hdf = losa.open_hdf_stream(output_base + "/" + file_name ,
                                  (total_slice, total_width, total_width),
                                  key_path="entry/data", data_type="float32")
for i in range(num_iter):
#     t_start = timeit.default_timer()
    start_sino = i * slice_chunk + offset
    stop_sino = start_sino + slice_chunk
    sino_chunk = generate_sinogram_chunk(start_sino, stop_sino, height, ver_overlap_mat, hor_overlap_mat,
                                         half_sino_overlap_mat, data_objects, flat_field, dark_field,
                                         total_width, crop=(top, bot, left, right),
                                         option1=opt1, option2=opt2)
    (num_sino, _, ncol_180) = sino_chunk.shape
    center_rot = (ncol_180 - 1.0) / 2.0
    for sino_idx in np.arange(start_sino, stop_sino):
        idx = "0000" + str(sino_idx)
        sino_180 = sino_chunk[sino_idx - start_sino]
        sino_180 = filt.fresnel_filter(sino_180, denoise_ratio, 1)
        img_rec = reco.fbp_reconstruction(sino_180, center_rot)
#         img_rec = reco.astra_reconstruction(sino_180, center_rot)
        output_hdf[sino_idx - offset] = img_rec
        if (sino_idx - offset) % 100 == 0:
            losa.save_image(output_base + "/rec_" + idx[-5:] + ".tif", ndi.zoom(img_rec, 1/8.0))
    t_stop = timeit.default_timer()
    print("Done slice: {0} - {1} . Time {2}".format(start_sino, stop_sino, t_stop - time_start))
if num_rest != 0:
#     t_start = timeit.default_timer()
    start_sino = num_iter * slice_chunk + offset
    stop_sino = start_sino + num_rest
    sino_chunk = generate_sinogram_chunk(start_sino, stop_sino, height, ver_overlap_mat, hor_overlap_mat,
                                         half_sino_overlap_mat, data_objects, flat_field, dark_field,
                                         total_width, crop=(top, bot, left, right),
                                         option1=opt1, option2=opt2)
    (num_sino, _, ncol_180) = sino_chunk.shape
    center_rot = (ncol_180 - 1.0) / 2.0
    for sino_idx in np.arange(start_sino, stop_sino):
        idx = "0000"  + str(sino_idx)
        sino_180 = sino_chunk[sino_idx - start_sino]
        sino_180 = filt.fresnel_filter(sino_180, denoise_ratio, 1)
        img_rec = reco.fbp_reconstruction(sino_180, center_rot)
#         img_rec = reco.astra_reconstruction(sino_180, center_rot)
        output_hdf[sino_idx - offset] = img_rec
        if (sino_idx- offset) % 100 == 0:
            losa.save_image(output_base + "/rec_" + idx[-5:] + ".tif", ndi.zoom(img_rec, 1/8.0))
    t_stop = timeit.default_timer()
    print("Done slice: {0} - {1} . Time {2}".format(start_sino, stop_sino, t_stop - time_start))

time_stop = timeit.default_timer()
print("All done!!! Total time cost: {}".format(time_stop - time_start))
