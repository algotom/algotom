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
# Description: Examples of how to use the Algotom package.
# ===========================================================================

"""
The following examples show how to reconstruct a full size of each frame
in a time-series tomography data.

The code written based on datasets collected at the beamline I12-DLS which
often have 4 files for each time-series scan:
- a hdf-file contains projection-images.
- a nxs-file contains metadata of an experiment: energy, number-of-projections
  per tomo, number-of-tomographs, detector-sample distance, detector
  pixel-size, exposure time,...
- a hdf-file contains flat-field images.
- a hdf-file contains dark-field images.

Referring to "example_01_*.py" to know how to find key-paths and datasets
in a hdf/nxs file.
"""

import timeit
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.calculation as calc
import algotom.rec.reconstruction as reco
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.util.utility as util

start_slice = 10
stop_slice = -1
slice_chunk = 10  # Number of slices to be reconstructed in one go to reduce
                  # IO overhead (in loading a hdf file) and process in
                  # parallel (for CPU-based methods).

output_base = "/home/user_id/reconstruction/"

proj_path = "/i12/data/projections.hdf"
flat_path = "/i12/data/flats.hdf"
dark_path = "/i12/data/darks.hdf"
metadata_path = "/i12/data/metadata.nxs"
metadata_path = "/dls/i12/data/2021/cm28139-4/processing/metadata/scan_00024/scan_00024.nxs"

scan_type = "continuous"  # stage is freely rotated
# scan_type = "swinging" # stage is restricted to rotate back-and-forward between 0 and 180 degree.

# Provide paths (keys) to datasets in the hdf/nxs files.
hdf_key = "/entry/data/data"
angle_key = "/entry1/tomo_entry/data/rotation_angle"
num_proj_key = "/entry1/information/number_projections"

# Crop images if need to.
crop_left = 0
crop_right = 0

# Options to include artifact-removal methods in the flat-field correction method.
opt1 = {"method": "remove_zinger", "para1": 0.08, "para2": 1}
opt2 = {"method": "remove_all_stripe", "para1": 3.0, "para2": 51, "para3": 17}
opt3 = None
# opt3 = {"method": "fresnel_filter", "para1": 100, "para2": 1} # Denoising


data = losa.load_hdf(proj_path, hdf_key)  # This is an hdf-object not ndarray.
(depth, height, width) = data.shape
left = crop_left
right = width - crop_right
if (stop_slice == -1) or (stop_slice > height - 1):
    stop_slice = height - 1

# Load metatdata
num_proj = int(np.asarray(losa.load_hdf(metadata_path, num_proj_key)))
num_tomo = depth // num_proj

angles = np.squeeze(np.asarray(losa.load_hdf(metadata_path, angle_key)))
# Sometime there's a mismatch between the number of acquired projections
# and number of angles due to technical reasons or early terminated scan.
# In such cases, we have to provide calculated angles.
if len(angles) < depth:
    if scan_type == "continuous":
        list_tmp = np.linspace(0, 180.0, num_proj)
        angles = np.ndarray.flatten(np.asarray([list_tmp for i in range(num_tomo)]))
    else:
        list_tmp1 = np.linspace(0, 180.0, num_proj)
        list_tmp2 = np.linspace(180.0, 0, num_proj)
        angles = np.ndarray.flatten(np.asarray(
            [list_tmp1 if i % 2 == 0 else list_tmp2 for i in range(num_tomo)]))
else:
    angles = angles[0:depth]

time_start = timeit.default_timer()
# Load flat-field images and dark-field images, average each of them
print("1 -> Load dark-field and flat-field images, average each result")
flat_field = np.mean(losa.load_hdf(flat_path, hdf_key)[:], axis=0)
dark_field = np.mean(losa.load_hdf(dark_path, hdf_key)[:], axis=0)

# Find the center of rotation using the sinogram of the first tomograph
mid_slice = height // 2
print("2 -> Calculate the center-of-rotation...")
sinogram = corr.flat_field_correction(data[0: num_proj, mid_slice, left:right],
                                      flat_field[mid_slice, left:right],
                                      dark_field[mid_slice, left:right])
center = calc.find_center_vo(sinogram)
print("Center-of-rotation = {0}".format(center))

if (stop_slice == -1) or (stop_slice > height):
    stop_slice = height
total_slice = stop_slice - start_slice
offset = start_slice
if slice_chunk > total_slice:
    slice_chunk = total_slice
num_iter = total_slice // slice_chunk
num_rest = total_slice - num_iter * slice_chunk

for ii in range(num_tomo):
    folder_name = "tomo_" + ("0000" + str(ii))[-5:]
    thetas = np.deg2rad(angles[ii * num_proj: (ii + 1) * num_proj])
    for i in range(num_iter):
        start_sino = i * slice_chunk + offset
        stop_sino = start_sino + slice_chunk
        sinograms = corr.flat_field_correction(
            data[ii * num_proj: (ii + 1) * num_proj, start_sino:stop_sino, left:right],
            flat_field[start_sino:stop_sino, left:right],
            dark_field[start_sino:stop_sino, left:right],
            option1=opt1, option2=opt2, option3=opt3)
        for j in range(start_sino, stop_sino):
            # img_rec = reco.dfi_reconstruction(sinograms[:, j - start_sino, :], center, angles=thetas, apply_log=True)
            # img_rec = reco.gridrec_reconstruction(sinograms[:, j - start_sino, :], center, angles=thetas,
            #                                       apply_log=True)
            img_rec = reco.fbp_reconstruction(sinograms[:, j - start_sino, :], center, angles=thetas, apply_log=True)
            name = "0000" + str(j)
            losa.save_image(output_base + "/" + folder_name + "/rec_" + name[-5:] + ".tif", img_rec)

        t_stop = timeit.default_timer()
        print("Tomograph {0} -> Done slice: {1} - {2} . Time {3}".format(ii, start_sino, stop_sino,
                                                                         t_stop - time_start))
    if num_rest != 0:
        for i in range(num_rest):
            start_sino = num_iter * slice_chunk + offset
            stop_sino = start_sino + num_rest
            sinograms = corr.flat_field_correction(
                data[ii * num_proj: (ii + 1) * num_proj, start_sino:stop_sino, left:right],
                flat_field[start_sino:stop_sino, left:right],
                dark_field[start_sino:stop_sino, left:right],
                option1=opt1, option2=opt2, option3=opt3)
            for j in range(start_sino, stop_sino):
                # img_rec = reco.dfi_reconstruction(sinograms[:, j - start_sino, :], center, angles=thetas,
                #                                   apply_log=True)
                # img_rec = reco.gridrec_reconstruction(sinograms[:, j - start_sino, :], center, angles=thetas,
                #                                       apply_log=True)
                img_rec = reco.fbp_reconstruction(sinograms[:, j - start_sino, :], center, angles=thetas,
                                                  apply_log=True)
                name = "0000" + str(j)
                losa.save_image(output_base + "/" + folder_name + "/rec_" + name[-5:] + ".tif", img_rec)

            t_stop = timeit.default_timer()
            print("Tomograph {0} -> Done slice: {1} - {2} . Time {3}".format(ii, start_sino, stop_sino,
                                                                             t_stop - time_start))
    print("-----------------------------")
    print("Done tomograph {0}".format(ii))
    print("-----------------------------")

time_stop = timeit.default_timer()
print("All done!!! Total time cost: {}".format(time_stop - time_start))
