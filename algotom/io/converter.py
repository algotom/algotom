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
# Description: Python codes for converting data format.
# Contributors:
# ===========================================================================

"""
Module for converting data type:
- Convert a list of tif files to a hdf/nxs file.
- Extract tif images from a hdf/nxs file.
"""

import os
import numpy as np
import algotom.io.loadersaver as losa


def convert_tif_to_hdf(input_path, output_path, key_path="entry/data",
                       crop=(0, 0, 0, 0), pattern=None, **options):
    """
    Convert a folder of tif files to a hdf/nxs file.

    Parameters
    ----------
    input_path : str
        Folder path to the tif files.
    output_path : str
        Path to the hdf/nxs file.
    key_path : str, optional
        Key path to the dataset.
    crop : tuple of int, optional
        Crop the images from the edges, i.e.
        crop = (crop_top, crop_bottom, crop_left, crop_right).
    pattern : str, optional
        Used to find tif files with names matching the pattern.
    options : dict, optional
        Add metadata. E.g. options={"entry/angles": angles, "entry/energy": 53}.

    Returns
    -------
    str
        Path to the hdf/nxs file.
    """
    if pattern is None:
        list_file = losa.find_file(input_path + "/*.tif*")
    else:
        list_file = losa.find_file(input_path + "/*" + pattern + "*.tif*")
    depth = len(list_file)
    if depth == 0:
        raise ValueError("No tif files in the folder: {}".format(input_path))
    (height, width) = np.shape(losa.load_image(list_file[0]))
    file_base, file_ext = os.path.splitext(output_path)
    if not (file_ext == '.hdf' or file_ext == '.h5' or file_ext == ".nxs"):
        file_ext = '.hdf'
    output_path = file_base + file_ext
    cr_top, cr_bottom, cr_left, cr_right = crop
    cr_height = height - cr_top - cr_bottom
    cr_width = width - cr_left - cr_right
    data_out = losa.open_hdf_stream(output_path, (depth, cr_height, cr_width),
                                    key_path=key_path, overwrite=True,
                                    **options)
    for i, fname in enumerate(list_file):
        data_out[i] = losa.load_image(fname)[cr_top:cr_height + cr_top,
                      cr_left:cr_width + cr_left]
    return output_path


def extract_tif_from_hdf(input_path, output_path, key_path, index=(0, -1, 1),
                         axis=0, crop=(0, 0, 0, 0), prefix="img"):
    """
    Extract tif images from a hdf/nxs file.

    Parameters
    ----------
    input_path : str
        Path to the hdf/nxs file.
    output_path : str
        Output folder.
    key_path : str
        Key path to the dataset in the hdf/nxs file.
    index : tuple of int or int.
        Indices of extracted images. A tuple corresponds to (start,stop,step).
    axis : int
        Axis which the images are extracted.
    crop : tuple of int, optional
        Crop the images from the edges, i.e.
        crop = (crop_top, crop_bottom, crop_left, crop_right).
    prefix : str, optional
        Prefix of names of tif files.

    Returns
    -------
    str
        Folder path to the tif files.
    """
    data = losa.load_hdf(input_path, key_path)
    (depth, height, width) = data.shape
    if isinstance(index, tuple):
        start, stop, step = index
    else:
        start, stop, step = index, index + 1, 1
    cr_top, cr_bottom, cr_left, cr_right = crop
    if axis == 1:
        if (stop == -1) or (stop > height):
            stop = height
        for i in range(start, stop, step):
            mat = data[cr_top:depth - cr_bottom, i, cr_left:width - cr_right]
            out_name = "0000" + str(i)
            losa.save_image(
                output_path + "/" + prefix + "_" + out_name[-5:] + ".tif", mat)
    elif axis == 2:
        if (stop == -1) or (stop > width):
            stop = width
        for i in range(start, stop, step):
            mat = data[cr_top:depth - cr_bottom, cr_left:height - cr_right, i]
            out_name = "0000" + str(i)
            losa.save_image(
                output_path + "/" + prefix + "_" + out_name[-5:] + ".tif", mat)
    else:
        if (stop == -1) or (stop > depth):
            stop = depth
        for i in range(start, stop, step):
            mat = data[i, cr_top:height - cr_bottom, cr_left:width - cr_right]
            out_name = "0000" + str(i)
            losa.save_image(
                output_path + "/" + prefix + "_" + out_name[-5:] + ".tif", mat)
    return output_path
