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
The following examples show how to use Algotom to reconstruct a few slices
from a tomographic data affected by the radial lens distortion problem.

Raw data is at: https://zenodo.org/record/3339629
There're 4 files to be used: "tomographic_projections.hdf"; "flats.hdf";
"darks.hdf"; and "coefficients_bw.txt" which contains the distortion
coefficients calculated using Vounwarp package:
https://github.com/nghia-vo/vounwarp

Referring to "example_01_*.py" to know how to find key-paths and datasets
in a hdf/nxs file.
"""

import timeit
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.removal as remo
import algotom.prep.calculation as calc
import algotom.prep.filtering as filt
import algotom.rec.reconstruction as reco

# Paths to data
proj_path = "D:/data/tomographic_projections.hdf"
flat_path = "D:/data/flats.hdf"
dark_path = "D:/data/darks.hdf"
coef_path = "D:/data/coefficients_bw.txt"
key_path = "/entry/data/data"
# Where to save the outputs
output_base = "D:/output/"
# Load data of projection images as an hdf object
proj_data = losa.load_hdf(proj_path, key_path)
(depth, height, width) = proj_data.shape
# Load flat-field images and dark-field images, average each of them
print("1 -> Load dark-field and flat-field images, average each result")
flat_field = np.mean(losa.load_hdf(flat_path, key_path)[:], axis=0)
dark_field = np.mean(losa.load_hdf(dark_path, key_path)[:], axis=0)
# Load distortion coefficients
print("2 -> Load distortion coefficients, apply correction to averaged flat-field and dark-field images")
xcenter, ycenter, list_fact = losa.load_distortion_coefficient(coef_path)
# Apply distortion correction
flat_discor = corr.unwarp_projection(flat_field, xcenter, ycenter, list_fact)
dark_discor = corr.unwarp_projection(dark_field, xcenter, ycenter, list_fact)

# Generate a sinogram without distortion correction and perform reconstruction to compare latter.
index = 800
print("3 -> Generate a sinogram without distortion correction")
sinogram = corr.flat_field_correction(proj_data[:, index, :], flat_field[index], dark_field[index])
sinogram = remo.remove_all_stripe(sinogram, 3.0, 51, 17)
sinogram = filt.fresnel_filter(sinogram, 10, 1)
t_start = timeit.default_timer()
print("4 -> Calculate the center-of-rotation...")
center = calc.find_center_vo(sinogram, width//2-50, width//2+50)
t_stop = timeit.default_timer()
print("Center-of-rotation = {0}. Take {1} second".format(center, t_stop-t_start))
t_start = timeit.default_timer()
print("5 -> Perform reconstruction")
img_rec = reco.dfi_reconstruction(sinogram, center,apply_log=True)
losa.save_image(output_base + "/rec_before_00800.tif", img_rec)
t_stop = timeit.default_timer()
print("Done reconstruction without distortion correction!!!")
# Generate a sinogram with distortion correction and perform reconstruction.
print("6 -> Generate a sinogram with distortion correction")
sinogram = corr.unwarp_sinogram(proj_data, index, xcenter, ycenter, list_fact)
sinogram = corr.flat_field_correction(sinogram, flat_discor[index], dark_discor[index])
sinogram = remo.remove_all_stripe(sinogram, 3.0, 51, 17)
sinogram = filt.fresnel_filter(sinogram, 10, 1)
t_start = timeit.default_timer()
print("7 -> Calculate the center-of-rotation...")
# Center-of-rotation can change due to the distortion effect.
center = calc.find_center_vo(sinogram, width//2-50, width//2+50)
t_stop = timeit.default_timer()
print("Center-of-rotation = {0}. Take {1} second".format(center, t_stop-t_start))
t_start = timeit.default_timer()
print("8 -> Perform reconstruction")
img_rec = reco.dfi_reconstruction(sinogram, center, apply_log=True)
losa.save_image(output_base + "/rec_after_00800.tif", img_rec)
print("!!! Done reconstruction with distortion correction !!!")

# For reconstructing full sample, we can process sinograms chunk-by-chunk to reduce the IO overhead.
print("9-> Generate a chunk of sinograms with distortion correction")
start_slice = 500
stop_slice = start_slice + 8
center = 1266.5
sinograms = corr.unwarp_sinogram_chunk(proj_data, start_slice, stop_slice, xcenter, ycenter, list_fact)
opt1 = {"method": "remove_all_stripe", "para1": 3.0, "para2": 51, "para3": 17}
opt2 = {"method": "fresnel_filter", "para1": 10.0, "para2": 1}
sinograms = corr.flat_field_correction(sinograms, flat_discor[start_slice:stop_slice],
                                       dark_discor[start_slice:stop_slice], option1=opt1, option2=opt2)
# Perform reconstruction
print("10 -> Perform reconstruction")
for i in range(start_slice, stop_slice):
    img_rec = reco.dfi_reconstruction(sinograms[:,i - start_slice, :],
                                      center, apply_log=True)
    name = "0000" + str(i)
    losa.save_image(output_base + "/reconstruction/rec_" + name[-5:] + ".tif", img_rec)
print("!!! Done !!!")

