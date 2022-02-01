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
from a tomographic data acquired by using a 360-degree scan with the offset
rotation-axis.

Raw data is at: https://doi.org/10.5281/zenodo.4386983
There're 4 files: "projections_00000.hdf", "flats_00000.hdf",
"darks_00000.hdf", and "scan_00008.nxs" containing projection images,
flat-field images, dark-field images, and meta-data, respectively.

Referring to "example_01_*.py" to know how to find key-paths and datasets
in a hdf/nxs file.

Referring to "example_06_*.py" to know how to include distortion correction.
"""

import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.removal as remo
import algotom.prep.calculation as calc
import algotom.prep.conversion as conv
import algotom.prep.filtering as filt
import algotom.rec.reconstruction as reco
import timeit



proj_path = "D:/data/scan_00008/projections_00000.hdf"
flat_path = "D:/data/scan_00009/flats_00000.hdf"
dark_path = "D:/data/scan_00009/darks_00000.hdf"
meta_path = "D:/data/scan_00008/scan_00008.nxs"
key_path = "/entry/data/data"
angle_key = "/entry1/tomo_entry/data/rotation_angle"

output_base = "D:/output/"

data = losa.load_hdf(proj_path, key_path) # This is an object not ndarray.
(depth, height, width) = data.shape
angles = np.squeeze(np.asarray(losa.load_hdf(meta_path, angle_key)[:]))
# Load dark-field images and flat-field images, averaging each result.
flat_field = np.mean(losa.load_hdf(flat_path, key_path)[:], axis = 0)
dark_field = np.mean(losa.load_hdf(dark_path, key_path)[:], axis = 0)
# Generate a sinogram and perform reconstruction.
index = height // 2
print("1 -> Extract a sinogram with flat-field correction")
sino_360 = corr.flat_field_correction(data[:, index, :], flat_field[index], dark_field[index])
t0 = timeit.default_timer()
print("2 -> Calculate the center-of-rotation, the overlap-side and overlap-area used for stitching")
(center0, overlap, side, _) = calc.find_center_360(sino_360, 100)
print("Center-of-rotation: {0}. Side: {1} (0->'left', 1->'right'). Overlap: {2}".format(center0, side, overlap))
t1 = timeit.default_timer()
print("Time cost {}".format(t1-t0))

# Remove artifacts. They also can be passed to flat_field_correction method above as parameters.
# Remove zingers
sino_360 = remo.remove_zinger(sino_360, 0.08)
# Remove ring artifacts
sino_360 = remo.remove_all_stripe(sino_360, 3, 51, 17)

# 1st way: Convert the 360-degree sinogram to the 180-degree sinogram.
sino_180, center1 = conv.convert_sinogram_360_to_180(sino_360, center0)
losa.save_image(output_base + "/reconstruction/sino_180.tif", sino_180)
## Denoising
sino_180 = filt.fresnel_filter(sino_180, 250, 1, apply_log=True)
# Perform reconstruction
img_rec = reco.dfi_reconstruction(sino_180, center1, apply_log=True)
## Using fbp with GPU for faster reconstruction
# img_rec = reco.fbp_reconstruction(sino_180, center1, apply_log=True, gpu=True)
losa.save_image(output_base + "/reconstruction/recon_image_1.tif", img_rec)

# 2nd way: Extending the 360-degree sinogram (by weighting and padding).
(sino_ext, center2) = conv.extend_sinogram(sino_360, center0)
losa.save_image(output_base + "/reconstruction/sino_360_extened.tif", sino_ext)
# Denoising
sino_ext = filt.fresnel_filter(sino_ext, 250, 1, apply_log=False)
# Perform reconstruction
# img_rec = reco.dfi_reconstruction(sino_ext, center2, angles=angles*np.pi/180.0,apply_log=False)
## Using fbp with GPU for faster reconstruction
img_rec = reco.fbp_reconstruction(sino_ext, center2,
                                  angles=angles * np.pi / 180.0,
                                  apply_log=False, gpu=True)
losa.save_image(output_base + "/reconstruction/recon_image_2.tif", img_rec)
print("!!! Done !!!")
