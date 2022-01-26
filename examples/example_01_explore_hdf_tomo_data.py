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
The following example shows how to use Algotom to explore a tomographic data
in the hdf/nxs format.

Raw data is at: https://zenodo.org/record/1443568
There're two files: "pco1-68067.hdf" contains flat-field, dark-field, and
projection images; "68067.nxs" contains metadata with a link to the
"pco1-68067.hdf" file.

To load a dataset from a hdf/nxs file, one needs to know the key path. This
can be done using the Hdfview software or using Algotom functions as below.
"""

import numpy as np
import algotom.io.converter as conv
import algotom.io.loadersaver as losa


file_path = "D:/data/68067.nxs"
output_base = "D:/output/"


# The following function returns a full list of key-paths, data-shapes,
# and data-types to datasets in the hdf/nxs file.
print("1 -> Print list of keys to datasets: ")
keys, shapes, dtypes = losa.get_hdf_information(file_path)
for i, key in enumerate(keys):
    print("Key :-> {0} | Shape :-> {1} | Type :-> {2}".format(
        key, shapes[i], dtypes[i]))


# The following function returns a list of key-paths having a pattern,
# e.g "rotation", in the keys. Note that in this data there are lots
# of soft links pointing to the same dataset.
print("2 -> Find keys having the key-word 'rotation': ")
angle_key, _, _ = losa.find_hdf_key(file_path, "rotation")
for key in angle_key:
    print("Key having the 'rotation' pattern :-> {0}".format(key))


# For tomographic reconstruction, we need key-paths to dark-field images,
# flat-field images, projection images, and rotation-angles. In this data
# (collected at the I12 beamline, Diamond Light Source), all images were
# recorded into a single 3D array. To indicate the type of an image, they
# use a metadata named "image_key":
# image_key = 2 <-> a dark-field image
# image_key = 1 <-> a flat-field image
# image_key = 0 <-> a projection image
# Using the following commands we'll get the keys for loading the only
# necessary datasets.
list_key, list_shape, _ = losa.find_hdf_key(file_path, "data")
for i, key in enumerate(list_key):
    # There're datasets with keys containing "data", we only choose a 3D array.
    if len(list_shape[i])==3:
        data_key = key
        break
image_key = losa.find_hdf_key(file_path, "image_key")[0][0]
# Results are:
# data_key = "/entry1/flyScanDetector/data"
# image_key = "/entry1/flyScanDetector/image_key"

# Load flat-field images, average them, and save the result as a tif image.
print("3 -> Load flat-field images, average them, and save the result: ")
data = losa.load_hdf(file_path, data_key) # This is an object. No data are loaded to the memory yet.
ikey = np.asarray(losa.load_hdf(file_path, image_key))
flat_field = np.mean(np.asarray(data[np.squeeze(np.where(ikey==1.0)), :,:]), axis=0)
losa.save_image(output_base + "/flat/flat_field.tif", flat_field)

# Load few projection images and save them as tifs.
print("4 -> Load projection images in a step of 250 and save to tiff: ")
proj_idx = np.squeeze(np.where(ikey == 0))
for i in range(0, len(proj_idx), 250):
    mat = data[proj_idx[i], :, :]
    name = "0000" + str(proj_idx[i])
    losa.save_image(output_base + "/projection/img_" + name[-5:] + ".tif", mat)

print("5 -> Same as 2 but using a built-in function: ")
# The above example can be done using the "converter" module as follows:
conv.extract_tif_from_hdf(file_path, output_base + "/projection2/",
                          key_path=data_key, index=(proj_idx[0],proj_idx[-1],250), axis=0)

# We also can convert a folder of tif images to a hdf file.
# We use the output of the above example as an input of the following example.
print("6 -> Convert a list of tiffs to a hdf file: ")
angles = np.asarray(losa.load_hdf(file_path, angle_key[0]))
metadata = {"entry/angle": angles[proj_idx[0]:proj_idx[-1]:250]}
conv.convert_tif_to_hdf(output_base + "/projection2/", output_base + "/hdf/converted.hdf",
                         key_path="entry/data", option = metadata)
print("!!! Done !!!")
