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
# Description: Examples of how to use the Algotom package.
# ===========================================================================

"""
The following examples show how to use Algotom to apply pre-processing methods
in the projection space and save the results as a hdf file.

Raw data (hdf format) was collected at Beamline I13-DLS where the paths to
datasets are similar to data used in example_05*.py.
"""

import timeit
import numpy as np
import multiprocessing as mp
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
from joblib import Parallel, delayed

scan_num = 129441
input_base = "/dls/i13/data/2021/mg26241-2/raw/"
output_base = "/dls/i13/data/2021/mg26241-2/processing/preprocessed/"
file_path = input_base + str(scan_num) + ".nxs"
ofile_name = str(scan_num) + "_processed.nxs"

# Load an MTF window determined outside Algotom.
mtf_win = losa.load_image("/dls/i12/data/2020/cm26476-4/processing/mtf_window.tif")
mtf_pad = 150
# Load distortion coefficients determined using the Discorpy package.
xcenter, ycenter, list_fact = losa.load_distortion_coefficient(
    "/dls/i12/data/2020/cm26476-4/processing/coefficients_bw.txt")

ncore = mp.cpu_count() - 1 # To process data in parallel.
chunk = 32  # Number of images to be loaded and processed in one go.

# Crop images after the distortion correction (pincushion type) to remove
# unwanted values around the edges.
crop_top = 20
crop_bottom = 20
crop_left = 20
crop_right = 20

# Keys to datasets in the nxs file
data_key = "/entry1/tomo_entry/data/data"
image_key = "/entry1/tomo_entry/instrument/detector/image_key"
angle_key = "/entry1/tomo_entry/data/rotation_angle"

print("----------------------------------------------------------")
print("---------------------------Start--------------------------")

ikey = np.squeeze(np.asarray(losa.load_hdf(file_path, image_key)))
angles = np.squeeze(np.asarray(losa.load_hdf(file_path, angle_key)))
data = losa.load_hdf(file_path, data_key) # This is an object not ndarray.
proj_idx = np.squeeze(np.where(ikey == 0)) # Indices of projection images.

(depth, height0, width0) = data.shape
height = height0 - crop_top - crop_bottom
width = width0 - crop_left - crop_right
top = crop_top
bot = height0 - crop_bottom
left = crop_left
right = width0 - crop_right
time_start = timeit.default_timer()

# Load dark-field images and flat-field images, averaging each result.
print("1 -> Load dark-field and flat-field images, average each result")
dark_field = np.mean(np.asarray(data[np.squeeze(np.where(ikey == 2.0)), :, :]), axis=0)
flat_field = np.mean(np.asarray(data[np.squeeze(np.where(ikey == 1.0)), :, :]), axis=0)

# Apply distortion correction and MTF deconvolution.
flat_field = corr.unwarp_projection(corr.mtf_deconvolution(flat_field, mtf_win, mtf_pad),
                                   xcenter, ycenter, list_fact)
# MTF deconvolution no applicable to dark-noise images.
dark_field = corr.unwarp_projection(dark_field, xcenter, ycenter, list_fact)

# Calculate parameters for looping.
start_image = proj_idx[0]
stop_image = proj_idx[-1] + 1
total_image = stop_image - start_image
offset = start_image
if chunk > total_image:
    chunk = total_image
num_iter = total_image // chunk
num_rest = total_image - num_iter * chunk

# Open hdf stream for saving data.
output_hdf = losa.open_hdf_stream(output_base + "/" + ofile_name ,
                                  (total_image, height, width),
                                  key_path="entry/data", data_type="float32")
for i in range(num_iter):
    start_proj = i * chunk + offset
    stop_proj = start_proj + chunk
    # Load projections
    proj_chunk = data[start_proj:stop_proj]
    # Apply MTF deconvolution
    proj_chunk = np.asarray(Parallel(n_jobs=ncore, backend="threading")(
        delayed(corr.mtf_deconvolution)(proj_chunk[i], mtf_win, mtf_pad) for i in range(chunk)))
    # Apply distortion correction
    proj_chunk = np.asarray(Parallel(n_jobs=ncore, backend="threading")(
        delayed(corr.unwarp_projection)(proj_chunk[i], xcenter, ycenter, list_fact) for i in range(chunk)))
    # Apply flat-field correction
    proj_corr = corr.flat_field_correction(proj_chunk[:, top:bot, left:right], flat_field[top:bot, left:right],
                                           dark_field[top:bot, left:right])
    output_hdf[start_proj - offset: stop_proj - offset] = proj_corr
    t_stop = timeit.default_timer()
    print("Done slice: {0} - {1} . Time {2}".format(start_proj, stop_proj, t_stop - time_start))
if num_rest != 0:
    start_proj = num_iter * chunk + offset
    stop_proj = start_proj + num_rest
    # Load projections
    proj_chunk = data[start_proj:stop_proj]
    # Apply MTF deconvolution
    proj_chunk = np.asarray(Parallel(n_jobs=ncore, backend="threading")(
        delayed(corr.mtf_deconvolution)(proj_chunk[i], mtf_win, mtf_pad) for i in range(num_rest)))
    # Apply distortion correction
    proj_chunk = np.asarray(Parallel(n_jobs=ncore, backend="threading")(
        delayed(corr.unwarp_projection)(proj_chunk[i], xcenter, ycenter, list_fact) for i in range(num_rest)))
    # Apply flat-field correction
    proj_corr = corr.flat_field_correction(proj_chunk[:, top:bot, left:right], flat_field[top:bot, left:right],
                                           dark_field[top:bot, left:right])
    output_hdf[start_proj - offset: stop_proj - offset] = proj_corr
    t_stop = timeit.default_timer()
    print("Done slice: {0} - {1} . Time {2}".format(start_proj, stop_proj, t_stop - time_start))

time_stop = timeit.default_timer()
print("All done!!! Total time cost: {}".format(time_stop - time_start))
