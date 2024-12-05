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
# Description: Python module for converting data format.
# Contributors:
# ===========================================================================

"""
Module for converting data type:

    -   Convert a list of tif files to a hdf/nxs file.
    -   Extract tif images from a hdf/nxs file.
    -   Emulate an HDF5-like interface for TIF files in a folder.
"""

from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
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
        Add metadata. E.g options={"entry/angles": angles, "entry/energy": 53}.

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
    (height, width) = np.shape(losa.load_image(list_file[0]))
    output_path = Path(output_path)
    output_path = output_path.resolve()
    if output_path.suffix.lower() not in {'.hdf', '.h5', '.nxs', '.hdf5'}:
        output_path = output_path.with_suffix('.hdf')
    cr_top, cr_bottom, cr_left, cr_right = crop
    cr_height = height - cr_top - cr_bottom
    cr_width = width - cr_left - cr_right
    if cr_height < 1 or cr_width < 1:
        raise ValueError("Can't crop images with the given parameters !!!")
    data_out = losa.open_hdf_stream(output_path, (depth, cr_height, cr_width),
                                    key_path=key_path, overwrite=True,
                                    **options)
    for i, file_path in enumerate(list_file):
        data_out[i] = losa.load_image(file_path)[cr_top:cr_height + cr_top,
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
        if stop < 1 or stop > height:
            stop = height
        start = np.clip(start, 0, stop - 1)
        for i in range(start, stop, step):
            mat = data[cr_top:depth - cr_bottom, i, cr_left:width - cr_right]
            out_name = f"{i:05}"
            losa.save_image(f"{output_path}/{prefix}_{out_name}.tif", mat)
    elif axis == 2:
        if stop < 1 or stop > width:
            stop = width
        start = np.clip(start, 0, stop - 1)
        for i in range(start, stop, step):
            mat = data[cr_top:depth - cr_bottom, cr_left:height - cr_right, i]
            out_name = f"{i:05}"
            losa.save_image(f"{output_path}/{prefix}_{out_name}.tif", mat)
    else:
        if stop < 1 or stop > depth:
            stop = depth
        start = np.clip(start, 0, stop - 1)
        for i in range(start, stop, step):
            mat = data[i, cr_top:height - cr_bottom, cr_left:width - cr_right]
            out_name = f"{i:05}"
            losa.save_image(f"{output_path}/{prefix}_{out_name}.tif", mat)
    return output_path


class HdfEmulatorFromTif:
    """
    Emulate an HDF5-like interface for TIF files in a folder, allowing
    indexed and sliced data access.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing TIFF files.
    ncore : int, optional
        Number of cores to use for parallel processing. The default is 1
        (sequential processing).

    Examples
    --------
    >>> hdf_emulator = HdfEmulatorFromTif('C:/path/to/tif/files', ncore=4)
    >>> print(hdf_emulator.shape)
    >>> last_image = hdf_emulator[-1]
    >>> image_stack = hdf_emulator[:, 0:4, :]
    """
    def __init__(self, folder_path, ncore=1):
        self.files = losa.find_file(folder_path + "/*tif*")
        if len(self.files) == 0:
            files = losa.find_file(folder_path + "/*TIF*")
        if len(self.files) == 0:
            raise ValueError(f"!!! No tif files found in: {folder_path}")
        self.n_jobs = ncore
        self._shape, self._dtype = self._get_shape_and_dtype()

    def _get_shape_and_dtype(self):
        img = losa.load_image(self.files[0])
        shape = (len(self.files), *img.shape)
        dtype = img.dtype
        return shape, dtype

    def __getitem__(self, index):
        if isinstance(index, int):
            return losa.load_image(self.files[index])
        elif isinstance(index, slice):
            indices = range(*index.indices(len(self.files)))
            return np.stack(Parallel(n_jobs=self.n_jobs)(
                delayed(losa.load_image)(self.files[i]) for i in indices))
        elif isinstance(index, tuple):
            z, y, x = index
            if isinstance(z, slice):
                images = Parallel(n_jobs=self.n_jobs)(
                    delayed(losa.load_image)(self.files[i]) for i in
                    range(*z.indices(self.shape[0])))
                return np.stack([img[y, x] for img in images])
            else:
                return losa.load_image(self.files[z])[y, x]
        else:
            raise TypeError("Invalid index type")

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return len(self.files)
