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
# E-mail: algotomography@gmail.com
# Description: Examples of how to downsample a full reconstruction of a grid
# scan.
# ===========================================================================

"""
The following examples show how to downsample a full reconstruction of a grid
scan.

Running "example_07_*_step_02.py" before trying this script.

Reconstruction data from "example_07_*_step_02.py" are separated into 12 hdf
files with a size of ~944 GB/file. The whole volume will be downsampled by a 
factor of 8 without an intermediate step of combining 12 files to 1 huge 
file (11.3 TB in total).
"""

import timeit
import numpy as np
import algotom.io.loadersaver as losa


input_base = "D:/Full_reconstruction/"

# Where to save the outputs
output_base = "D:/Dsp_grid_scan/"
output_file = "full_size_dsp_8_8_8.hdf"
cube = (8, 8, 8) # Downsampling factor

list_file = losa.find_file(input_base + "*.hdf")
key_path = "entry/data"

list_hdf_object = []
num_file = len(list_file)
list_nslice = []
for i in range(num_file):
    hdf_object = losa.load_hdf(list_file[i], key_path)
    list_hdf_object.append(hdf_object)
    (nslice, height, width) = hdf_object.shape 
    list_nslice.append(nslice)
total_slice = np.sum(np.asarray(list_nslice))

total_slice_r = (total_slice // cube[0]) * cube[0]
height_r = (height // cube[1]) * cube[1]
width_r = (width // cube[2]) * cube[2]

# Calculate the size of downsampled data.
dsp_slice = total_slice_r // cube[0]
dsp_height = height_r // cube[1]
dsp_width = width_r // cube[2]

next_slice = 0
list_slices = []
for i in range(num_file):
    list1 = next_slice + np.arange(list_nslice[i])
    list_slices.append(list1)
    next_slice = list1[-1] + 1
# Locate slices in hdf_files given a range of requested slices
def locate_slice_chunk(slice_start, slice_stop, list_slices):
    """
    Map requested slices to slice-indices in each hdf file.
    Return: [[file_index0, slice_start0, slice_stop0]]
    or [[file_index0, slice_start0, slice_stop0], [file_index1, slice_start1, slice_stop1]]
    """
    results = []
    for i, list1 in enumerate(list_slices):
        result_tmp = []
        for slice_idx in range(slice_start, slice_stop):
            pos = np.squeeze(np.where(list1 == slice_idx)[0])
            if pos.size == 1:
                result_tmp.append(pos)
        if len(result_tmp) > 0:
            result_tmp = np.asarray(result_tmp)
            results.append([i, result_tmp[0], result_tmp[-1]])
    return results

print("!!! Start !!!")
time_start = timeit.default_timer()

# Open hdf_stream for saving data.
hdf_stream = losa.open_hdf_stream(output_base + "/" + output_file, 
                                  (dsp_slice, dsp_height, dsp_width), key_path)

list_idx_nslice = np.reshape(np.arange(total_slice_r), (dsp_slice, cube[0]))
dsp_method = np.mean # Use mean for downsampling
for idx in np.arange(dsp_slice):
    slice_start = list_idx_nslice[idx, 0]
    slice_stop = list_idx_nslice[idx, -1] + 1
    slices = locate_slice_chunk(slice_start, slice_stop, list_slices)
    if len(slices) == 1:
        data_chunk = list_hdf_object[slices[0][0]][slices[0][1]:slices[0][2] + 1, :height_r, :width_r]
    else:
        data_chunk1 = list_hdf_object[slices[0][0]][slices[0][1]:slices[0][2] + 1, :height_r, :width_r]
        data_chunk2 = list_hdf_object[slices[1][0]][slices[1][1]:slices[1][2] + 1, :height_r, :width_r]
        data_chunk = np.concatenate((data_chunk1, data_chunk2), axis=0)
    mat_dsp = data_chunk.reshape(1, cube[0], dsp_height, cube[1], dsp_width, cube[2])
    mat_dsp = dsp_method(dsp_method(dsp_method(mat_dsp, axis=-1), axis=1), axis=2)
    hdf_stream[idx] = mat_dsp
    if idx % 200 == 0:
        out_name = "0000" + str(idx)
        losa.save_image(output_base +"/some_tif_files/dsp_" + out_name[-5:] + ".tif", mat_dsp[0])
        time_now = timeit.default_timer()
        print("Done slices up to {0}. Time cost  {1}".format(slice_stop, time_now - time_start))
time_stop = timeit.default_timer()
print("All done!!! Total time cost: {}".format(time_stop - time_start))
