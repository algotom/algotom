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
The following examples show how to use Algotom to generate tilted sinograms.
This is useful for correcting the image rotation problem without saving
intermediate data in the projection space.

Raw data is at: https://zenodo.org/record/1443568
There're two files: "pco1-68067.hdf" contains flat-field, dark-field, and
projection images; "68067.nxs" contains metadata with a link to the
"pco1-68067.hdf" file.

Referring to "example_01_*.py" to know how to find key-paths and datasets
in a hdf/nxs file.

"""
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr

file_path = "E:/Tomo_data/68067.nxs"
output_base = "E:/tmp/output5/"

data_key = "/entry1/flyScanDetector/data"
image_key = "/entry1/flyScanDetector/image_key"
angle_key = "/entry1/tomo_entry/data/rotation_angle"

ikey = np.squeeze(np.asarray(losa.load_hdf(file_path, image_key)))
angles = np.squeeze(np.asarray(losa.load_hdf(file_path, angle_key)))
data = losa.load_hdf(file_path, data_key)  # This is an object not ndarray.
(depth, height, width) = data.shape

# Load dark-field images and flat-field images, averaging each result.
print("1 -> Load dark-field and flat-field images, average each result")
dark_field = np.mean(np.asarray(data[np.squeeze(np.where(ikey == 2.0)), :, :]),
                     axis=0)
flat_field = np.mean(np.asarray(data[np.squeeze(np.where(ikey == 1.0)), :, :]),
                     axis=0)
proj_idx = np.squeeze(np.where(ikey == 0))

# The following examples are for demonstration only because the raw data were
# acquired from a well-aligned tomography system.
tilted = -5.0  # Degree
# Generate a tilted sinogram and apply the flat-field correction.
index = height // 2  # Index of a sinogram.
# As there're dark-field images and flat-field images in the raw data, we use
# opt=(proj_idx[0], proj_idx[-1], 1) to apply correction only to
# projection images
sino_tilted = corr.generate_tilted_sinogram(data, index, tilted,
                                            opt=(proj_idx[0], proj_idx[-1], 1))

flat_line = corr.generate_tilted_profile_line(flat_field, index, tilted)
dark_line = corr.generate_tilted_profile_line(dark_field, index, tilted)
sinogram = corr.flat_field_correction(sino_tilted, flat_line, dark_line)
losa.save_image(output_base + "/sinogram1/sinogram_mid_tilted.tif", sinogram)

# Generate a chunk of tilted sinograms and apply the flat-field correction.
start_sino = index
stop_sino = start_sino + 20
sinos_tilted = corr.generate_tilted_sinogram_chunk(data, start_sino, stop_sino,
                                                   tilted,
                                                   opt=(proj_idx[0],
                                                        proj_idx[-1], 1))
flat_lines = corr.generate_tilted_profile_chunk(flat_field, start_sino,
                                                stop_sino, tilted)
dark_lines = corr.generate_tilted_profile_chunk(dark_field, start_sino,
                                                stop_sino, tilted)
sino_chunk = corr.flat_field_correction(sinos_tilted, flat_lines, dark_lines)
for i in range(start_sino, stop_sino):
    losa.save_image(output_base + "/sinogram2/sino_"+ ("00000" + str(i))[-5:]
                    + ".tif", sino_chunk[:, i - start_sino, :])
print("!!! Done !!!")
