# ============================================================================
# ============================================================================
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
# ============================================================================
# Author: Nghia T. Vo
# E-mail:  
# Description: Python module for postprocessing techniques.
# Contributors:
# ============================================================================

"""
Module of methods in the postprocessing stage:

    -   Get statistical information of reconstructed images or a dataset.
    -   Downsample 2D, 3D array, or a dataset.
    -   Rescale 2D, 3D array or a dataset to 8-bit or 16-bit data-type.
    -   Reslice 3D array or a dataset (hdf/nxs file or tif images).
    -   Removing ring artifacts in a reconstructed image by transform back and
        forth between the polar coordinates and the Cartesian coordinates.
"""

import os
import sys
import shutil
import timeit
import glob
import h5py
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import algotom.util.utility as util
import algotom.io.loadersaver as losa
import algotom.prep.removal as remo


def __get_input_type(input_):
    """
    Supplementary method: to get input type
    """
    in_type = None
    if isinstance(input_, np.ndarray):
        in_type = "numpy_array"
    else:
        if isinstance(input_, str):
            file_ext = os.path.splitext(input_)[-1]
            if file_ext == "":
                list_file = losa.find_file(input_ + "/*.tif*")
                if list_file:
                    in_type = "tif"
                else:
                    raise ValueError(
                        "No tif files in the folder: {}".format(input_))
            else:
                if (file_ext == '.hdf' or file_ext == '.h5'
                        or file_ext == ".nxs"):
                    in_type = "hdf"
    return in_type


def __get_output_type(output):
    """
    Supplementary method: to get output type
    """
    out_type = None
    if isinstance(output, str):
        file_ext = os.path.splitext(output)[-1]
        if file_ext == "":
            out_type = "tif"
        else:
            if (file_ext == '.hdf' or file_ext == '.h5'
                    or file_ext == ".nxs"):
                out_type = "hdf"
            else:
                raise ValueError("File format must be hdf/h5/nxs !!!")
    return out_type


def __check_output(output):
    """
    Supplementary method: to check if output folder/file exists
    """
    if isinstance(output, str):
        file_base, file_ext = os.path.splitext(output)
        if file_ext == "":
            if os.path.exists(file_base):
                raise ValueError(
                    "Folder exists!!! Please choose another path!!!")
        else:
            if os.path.isfile(output):
                raise ValueError(
                    "File exists!!! Please choose another file path!!!")


def __get_shape(input_, key_path=None):
    """
    Supplementary method: to get the shape of a 3D data which can be given as
    a folder of tif files, a hdf file-path, or a 3D array.
    """
    in_type = __get_input_type(input_)
    if in_type == "numpy_array":
        (depth, height, width) = input_.shape
    elif in_type == "tif":
        list_file = losa.find_file(input_ + "/*.tif*")
        depth = len(list_file)
        (height, width) = np.shape(losa.load_image(list_file[0]))
    elif in_type == "hdf":
        if key_path is None:
            raise ValueError(
                "Please provide the key path to the dataset!!!")
        else:
            hdf_object = h5py.File(input_, 'r')
            check = key_path in hdf_object
            if not check:
                raise ValueError("!!! Wrong key !!!")
            data = hdf_object[key_path]
            (depth, height, width) = data.shape
            hdf_object.close()
    else:
        raise ValueError("Input must be a folder-path to tif files, a hdf "
                         "file-path, or a numpy array!!!")
    return depth, height, width


def __get_cropped_shape(input_, crop, key_path=None):
    """
    Supplementary method: to get the cropped information of a 3D data which
    can be given as a folder of tif files, a hdf file-path, or a 3D array.
    """
    if len(crop) != 6:
        raise ValueError("Crop must be a tuple/list with the length of 6")
    (cr_d1, cr_d2, cr_h1, cr_h2, cr_w1, cr_w2) = crop
    (depth, height, width) = __get_shape(input_, key_path=key_path)
    d1, d2 = cr_d1, depth - cr_d2
    depth1 = d2 - d1
    h1, h2 = cr_h1, height - cr_h2
    w1, w2 = cr_w1, width - cr_w2
    height1, width1 = h2 - h1, w2 - w1
    if (depth1 <= 0) or (height1 <= 0) or (width1 <= 0):
        raise ValueError("Check crop parameters!!! Can't crop the data having"
                         " shape: {}".format((depth, height, width)))
    return (d1, d2, h1, h2, w1, w2), (depth1, height1, width1)


def __get_dataset_size(input_):
    """
    To get the size of 3D array, folder of tif files, or a hdf file in MB.
    """
    in_type = __get_input_type(input_)
    b_unit = 1024.0 * 1024.0
    if in_type == "numpy_array":
        size_in_MB = input_.nbytes / b_unit
    elif in_type == "hdf":
        size_in_MB = os.path.getsize(input_) / b_unit
    else:
        list_file = losa.find_file(input_ + "/*.tif*")
        if list_file:
            size_1_file = np.asarray(Image.open(list_file[0])).nbytes / b_unit
        else:
            size_1_file = 0.0
        size_in_MB = len(list_file) * size_1_file
    return size_in_MB


def get_statistical_information(mat, percentile=(0, 100), denoise=False):
    """
    Get statistical information of an image.

    Parameters
    ----------
    mat : array_like
        2D array. Projection image, sinogram image, or reconstructed image.
    percentile : tuple of floats
        Tuple of (min_percentile, max_percentile) to compute.
        Must be between 0 and 100 inclusive.
    denoise: bool, optional
        Enable/disable denoising before extracting statistical information.

    Returns
    -------
    gmin : float
        The minimum value of the data array.
    gmax : float
        The maximum value of the data array.
    min_percent : float
        The first computed percentile of the data array.
    max_percent : tuple of floats
        The last computed percentile of the data array.
    mean : float
        The mean of the data array.
    median : float
        The median of the data array.
    variance : float
        The variance of the data array.
    """
    if denoise is True:
        mat = ndi.gaussian_filter(mat, 2)
    gmin = np.min(mat)
    gmax = np.max(mat)
    min_percent = np.percentile(mat, percentile[0])
    max_percent = np.percentile(mat, percentile[-1])
    median = np.median(mat)
    mean = np.mean(mat)
    variance = np.var(mat)
    return gmin, gmax, min_percent, max_percent, mean, median, variance


def get_statistical_information_dataset(input_, percentile=(0, 100), skip=5,
                                        denoise=False, key_path=None,
                                        crop=(0, 0, 0, 0, 0, 0)):
    """
    Get statical information of a dataset. This can be a folder of tif files,
    a hdf file, or a 3D array.

    Parameters
    ----------
    input_ : str, hdf file, or array_like
        It can be a folder path to tif files, a hdf file, or a 3D array.
    percentile : tuple of floats
        Tuple of (min_percentile, max_percentile) to compute.
        Must be between 0 and 100 inclusive.
    skip : int
        Skipping step of reading input.
    denoise: bool, optional
        Enable/disable denoising before extracting statistical information.
    key_path : str, optional
        Key path to the dataset if input is a hdf file.
    crop : tuple of int, optional
        Crop 3D data from the edges, i.e.
        crop = (crop_depth1, crop_depth2, crop_height1, crop_height2,
        crop_width1, crop_width2).

    Returns
    -------
    gmin : float
        The global minimum value of the data array.
    gmax : float
        The global maximum value of the data array.
    min_percent : float
        The global min of the first computed percentile of the data array.
    max_percent : tuple of floats
        The global min of the last computed percentile of the data array.
    mean : float
        The mean of the data array.
    median : float
        The median of the data array.
    variance : float
        The mean of the variance of the data array.
    """
    results = __get_cropped_shape(input_, crop=crop, key_path=key_path)
    (d1, d2, h1, h2, w1, w2) = results[0]
    depth1 = results[1][0]
    skip = int(np.clip(skip, 1, depth1 - 1))
    in_type = __get_input_type(input_)
    f_alias = get_statistical_information
    if in_type == "tif":
        list_file = losa.find_file(input_ + "/*.tif*")
        list_stat = []
        for i in range(d1, d2, skip):
            mat = losa.load_image(list_file[i])[h1:h2, w1:w2]
            if denoise is True:
                mat = ndi.gaussian_filter(mat, 2)
            list_stat.append(f_alias(mat, percentile, denoise))
    elif in_type == "hdf":
        data = losa.load_hdf(input_, key_path)
        list_stat = []
        for i in range(d1, d2, skip):
            mat = data[i, h1:h2, w1:w2]
            if denoise is True:
                mat = ndi.gaussian_filter(mat, 2)
            list_stat.append(f_alias(mat, percentile, denoise))
    else:
        list_stat = []
        for i in range(d1, d2, skip):
            mat = input_[i, h1:h2, w1:w2]
            if denoise is True:
                mat = ndi.gaussian_filter(mat, 2)
            list_stat.append(f_alias(mat, percentile, denoise))
    list_stat = np.asarray(list_stat)
    gmin, gmax = np.min(list_stat[:, 0]), np.max(list_stat[:, 1])
    min_percent, max_percent = np.min(list_stat[:, 2]), np.max(list_stat[:, 3])
    median, mean = np.median(list_stat[:, 4]), np.mean(list_stat[:, 5])
    variance = np.mean(list_stat[:, 6])
    return gmin, gmax, min_percent, max_percent, mean, median, variance


def downsample(mat, cell_size, method="mean"):
    """
    Downsample an image.

    Parameters
    ----------
    mat : array_like
        2D array.
    cell_size : int or tuple of int
        Window size along axes used for grouping pixels.
    method : {"mean", "median", "max", "min"}
        Downsampling method.

    Returns
    -------
    array_like
        Downsampled image.
    """
    if method == "median":
        dsp_method = np.median
    elif method == "max":
        dsp_method = np.max
    elif method == "min":
        dsp_method = np.amin
    else:
        dsp_method = np.mean
    (height, width) = mat.shape
    if isinstance(cell_size, int):
        cell_size = (cell_size, cell_size)
    height_dsp = height // cell_size[0]
    width_dsp = width // cell_size[1]
    mat = mat[:height_dsp * cell_size[0], :width_dsp * cell_size[1]]
    mat_dsp = mat.reshape(
        height_dsp, cell_size[0], width_dsp, cell_size[1])
    mat_dsp = dsp_method(dsp_method(mat_dsp, axis=-1), axis=1)
    return mat_dsp


def rescale(mat, nbit=16, minmax=None):
    """
    Rescale a 32-bit array to 16-bit/8-bit data.

    Parameters
    ----------
    mat : array_like
    nbit : {8,16}
        Rescaled data-type: 8-bit or 16-bit.
    minmax : tuple of float, or None
        Minimum and maximum values used for rescaling.

    Returns
    -------
    array_like
        Rescaled array.
    """
    if nbit != 8 and nbit != 16:
        raise ValueError("Only two options for nbit: 8 or 16 !!!")
    if minmax is None:
        gmin, gmax = np.min(mat), np.max(mat)
    else:
        (gmin, gmax) = minmax
    mat = np.clip(mat, gmin, gmax)
    mat = (mat - gmin) / (gmax - gmin)
    if nbit == 8:
        mat = np.uint8(np.clip(mat * 255, 0, 255))
    else:
        mat = np.uint16(np.clip(mat * 65535, 0, 65535))
    return mat


def downsample_dataset(input_, output, cell_size, method="mean", key_path=None,
                       rescaling=False, nbit=16, minmax=None, skip=None,
                       crop=(0, 0, 0, 0, 0, 0)):
    """
    Downsample a dataset. Input can be a folder of tif files, a hdf file,
    or a 3D array.

    Parameters
    ----------
    input_ : str, array_like
        It can be a folder path to tif files, a hdf file, or a 3D array.
    output : str, None
        It can be a folder path, a hdf file path, or None (memory consuming).
    cell_size : int or tuple of int
        Window size along axes used for grouping pixels.
    method : {"mean", "median", "max", "min"}
        Downsampling method.
    key_path : str, optional
        Key path to the dataset if the input is a hdf file.
    rescaling : bool
        Rescale dataset if True.
    nbit : {8,16}
        If rescaling is True, select data-type: 8-bit or 16-bit.
    minmax : tuple of float, or None
        Minimum and maximum values used for rescaling if True.
    skip : int or None
        Skipping step of images used for getting statistical information if
        rescaling is True and input is 32-bit data.
    crop : tuple of int, optional
        Crop 3D data from the edges, i.e.
        crop = (crop_depth1, crop_depth2, crop_height1, crop_height2,
        crop_width1, crop_width2).

    Returns
    -------
    array_like or None
        If output is None, returning a 3D array.
    """
    __check_output(output)
    results = __get_cropped_shape(input_, crop=crop, key_path=key_path)
    (d1, d2, h1, h2, w1, w2) = results[0]
    (depth1, height1, width1) = results[1]
    if method == "median":
        dsp_method = np.median
    elif method == "max":
        dsp_method = np.amax
    elif method == "min":
        dsp_method = np.amin
    else:
        dsp_method = np.mean
    if isinstance(cell_size, int):
        cell_size = (cell_size, cell_size, cell_size)
    depth_dsp = depth1 // cell_size[0]
    height_dsp = height1 // cell_size[1]
    width_dsp = width1 // cell_size[2]
    if (depth_dsp == 0) or (height_dsp == 0) or (width_dsp == 0):
        raise ValueError("Incorrect cell size {}".format(cell_size))
    in_type = __get_input_type(input_)
    if in_type == "tif":
        data = losa.find_file(input_ + "/*.tif*")
        data_type = np.asarray(Image.open(data[0])).dtype
    elif in_type == "hdf":
        data = losa.load_hdf(input_, key_path)
        data_type = data.dtype
    else:
        data = input_
        data_type = data.dtype
    res_type = str(data_type)
    if rescaling is True:
        if nbit == 16:
            res_type = "uint16"
        elif nbit == 8:
            res_type = "uint8"
        else:
            raise ValueError("Only two options for nbit: 8 or 16 !!!")
        if str(data_type) != res_type:
            if data_type == np.uint8:
                minmax = (0, 255)
            elif data_type == np.uint16:
                minmax = (0, 65535)
            else:
                if skip is None:
                    skip = int(np.clip(np.ceil(0.15 * depth1), 1, depth1 - 1))
                if minmax is None:
                    f_alias = get_statistical_information_dataset
                    minmax = f_alias(input_, percentile=(0, 100), skip=skip,
                                     crop=crop, key_path=key_path)[0:2]
        else:
            rescaling = False
    out_type = __get_output_type(output)
    if out_type == "hdf":
        data_dsp = losa.open_hdf_stream(
            output, (depth_dsp, height_dsp, width_dsp),
            data_type=res_type, key_path="entry/data", overwrite=True)
    elif out_type is None:
        data_dsp = []
    else:
        data_dsp = None
    num = 0
    for i in range(d1, d2, cell_size[0]):
        if (i + cell_size[0]) > (d1 + depth1):
            break
        else:
            if in_type == "tif":
                mat = np.asarray([losa.load_image(data[j])[h1:h2, w1:w2]
                                  for j in range(i, i + cell_size[0])])
                mat = mat[:, :height_dsp * cell_size[1],
                          :width_dsp * cell_size[2]]
                mat = mat.reshape(1, cell_size[0], height_dsp,
                                  cell_size[1], width_dsp, cell_size[2])
                mat_dsp = dsp_method(
                    dsp_method(dsp_method(mat, axis=-1), axis=1), axis=2)[0]
            else:
                mat = data[i:i + cell_size[0],
                           h1:h1 + height_dsp * cell_size[1],
                           w1:w1 + width_dsp * cell_size[2]]
                mat = mat.reshape(1, cell_size[0], height_dsp,
                                  cell_size[1], width_dsp,
                                  cell_size[2])
                mat_dsp = dsp_method(dsp_method(
                    dsp_method(mat, axis=-1), axis=1), axis=2)[0]
            if rescaling:
                mat_dsp = rescale(mat_dsp, nbit, minmax)
            if out_type is None:
                data_dsp.append(mat_dsp)
            elif out_type == "hdf":
                data_dsp[num] = mat_dsp.astype(res_type)
            else:
                out_name = "0000" + str(num)
                losa.save_image(output + "/img_" + out_name[-5:] + ".tif",
                                mat_dsp.astype(res_type))
            num += 1
    if out_type is None:
        data_dsp = np.asarray(data_dsp).astype(res_type)
    return data_dsp


def rescale_dataset(input_, output, nbit=16, minmax=None, skip=None,
                    key_path=None, crop=(0, 0, 0, 0, 0, 0)):
    """
    Rescale a dataset to 8-bit or 16-bit data-type. The dataset can be a
    folder of tif files, a hdf file, or a 3D array.

    Parameters
    ----------
    input_ : str, array_like
        It can be a folder path to tif files, a hdf file, or 3D array.
    output : str, None
        It can be a folder path, a hdf file path, or None (memory consuming).
    nbit : {8,16,32}
        Select rescaled data-type: 8-bit/16-bit. 32 is for cropping data only.
    minmax : tuple of float, or None
        Minimum and maximum values used for rescaling. They are calculated if
        None is given.
    skip : int or None
        Skipping step of images used for getting statistical information.
    key_path : str, optional
        Key path to the dataset if the input is a hdf file.
    crop : tuple of int, optional
        Crop 3D data from the edges, i.e.
        crop = (crop_depth1, crop_depth2, crop_height1, crop_height2,
        crop_width1, crop_width2).

    Returns
    -------
    array_like or None
        If output is None, returning an 3D array.
    """
    __check_output(output)
    results = __get_cropped_shape(input_, crop=crop, key_path=key_path)
    (d1, d2, h1, h2, w1, w2) = results[0]
    (depth1, height1, width1) = results[1]
    if minmax is None:
        if skip is None:
            skip = int(np.clip(np.ceil(0.15 * depth1), 1, None))
        minmax = get_statistical_information_dataset(input_, skip=skip,
                                                     crop=crop,
                                                     key_path=key_path)[0:2]
    if nbit == 8:
        data_type = "uint8"
    elif nbit == 16:
        data_type = "uint16"
    else:
        data_type = "float32"
    out_type = __get_output_type(output)
    if out_type == "hdf":
        data_res = losa.open_hdf_stream(output, (depth1, height1, width1),
                                        key_path="entry/data",
                                        data_type=data_type,
                                        overwrite=False)
    elif out_type is None:
        data_res = []
    else:
        data_res = None
    in_type = __get_input_type(input_)
    if in_type == "tif":
        data = losa.find_file(input_ + "/*.tif*")
    elif in_type == "hdf":
        data = losa.load_hdf(input_, key_path)
    else:
        data = input_
    for i in range(d1, d2):
        if in_type == "tif":
            mat = losa.load_image(data[i])[h1:h2, w1:w2]
            if nbit != 32:
                mat = rescale(mat, nbit=nbit, minmax=minmax)
        else:
            mat = data[i, h1:h2, w1:w2]
            if nbit != 32:
                mat = rescale(mat, nbit=nbit, minmax=minmax)
        if out_type is None:
            data_res.append(mat)
        elif out_type == "hdf":
            data_res[i - d1] = mat
        else:
            out_name = "0000" + str(i)
            losa.save_image(output + "/img_" + out_name[-5:] + ".tif", mat)
    if out_type is None:
        data_res = np.asarray(data_res)
    return data_res


def __save_intermediate_data(input_, output, axis, crop, key_path=None,
                             rotate=0.0, chunk=16, mode="constant",
                             ncore=None, show_progress=True):
    """
    Supplementary method: save data to an intermediate hdf-file for fast
    reslicing and reducing the RAM requirements.
    """

    in_type = __get_input_type(input_)
    results = __get_cropped_shape(input_, crop=crop, key_path=key_path)
    (d1, d2, h1, h2, w1, w2) = results[0]
    (depth1, height1, width1) = results[1]
    chunk = np.clip(chunk, 1, depth1 - 1)
    last_chunk = depth1 - chunk * (depth1 // chunk)
    folder_tmp = os.path.splitext(output)[0] + "/tmp_/"
    file_tmp = folder_tmp + "/file_tmp.hdf"
    losa.make_folder(folder_tmp)
    out_key = "entry/data"
    b_unit = 1024.0 * 1024.0
    with h5py.File(file_tmp, 'w') as ofile:
        t0 = timeit.default_timer()
        if in_type == "tif":
            data = losa.find_file(input_ + "/*.tif*")
            data_type = np.asarray(Image.open(data[0])).dtype
            if axis == 2:
                hdf_chunk = (chunk, min(100, width1), min(100, height1))
                data_tmp = ofile.create_dataset(out_key,
                                                (depth1, width1, height1),
                                                dtype=data_type,
                                                chunks=hdf_chunk)
                for i in range(0, depth1 - last_chunk, chunk):
                    if show_progress:
                        t1 = timeit.default_timer()
                        f_size = os.path.getsize(file_tmp) / b_unit
                        msg = "Writing to an intermediate hdf-file: {0:.2f} " \
                              "MB. Time: {1:0.2f}s".format(f_size, t1 - t0)
                        len_msg = len(msg)
                        sys.stdout.write(msg)
                        sys.stdout.flush()
                    mat_tmp = []
                    mat_chunk = losa.load_image_multiple(
                        data[i + d1: i + chunk + d1], ncore=ncore,
                        prefer="threads")
                    for j in range(chunk):
                        mat = mat_chunk[j]
                        if rotate != 0.0:
                            mat = ndi.rotate(mat, rotate, mode=mode,
                                             reshape=False, order=1)
                        mat_tmp.append(np.transpose(mat[h1:h2, w1:w2]))
                    data_tmp[i:i + chunk] = np.asarray(mat_tmp)
                    if show_progress:
                        sys.stdout.write("\r" + " " * len_msg + "\r")
                if last_chunk != 0:
                    mat_tmp = []
                    mat_chunk = losa.load_image_multiple(
                        data[depth1 - last_chunk + d1: depth1 + d1],
                        ncore=ncore, prefer="threads")
                    for j in range(last_chunk):
                        mat = mat_chunk[j]
                        if rotate != 0.0:
                            mat = ndi.rotate(mat, rotate, mode=mode,
                                             reshape=False, order=1)
                        mat_tmp.append(np.transpose(mat[h1:h2, w1:w2]))
                    data_tmp[depth1 - last_chunk: depth1] = np.asarray(mat_tmp)
            else:
                hdf_chunk = (chunk, min(100, height1), min(100, width1))
                data_tmp = ofile.create_dataset(out_key,
                                                (depth1, height1, width1),
                                                dtype=data_type,
                                                chunks = hdf_chunk)
                for i in np.arange(0, depth1 - last_chunk, chunk):
                    if show_progress:
                        t1 = timeit.default_timer()
                        f_size = os.path.getsize(file_tmp) / b_unit
                        msg = "Writing to an intermediate hdf-file: {0:0.2f}" \
                              "MB. Time: {1:0.2f}s".format(f_size, t1 - t0)
                        len_msg = len(msg)
                        sys.stdout.write(msg)
                        sys.stdout.flush()
                    mat_tmp = []
                    mat_chunk = losa.load_image_multiple(
                        data[i + d1: i + chunk + d1], ncore=ncore,
                        prefer="threads")
                    for j in range(chunk):
                        mat = mat_chunk[j]
                        if rotate != 0.0:
                            mat = ndi.rotate(mat, rotate, mode=mode,
                                             reshape=False, order=1)
                        mat_tmp.append(mat[h1:h2, w1:w2])
                    data_tmp[i:i + chunk] = np.asarray(mat_tmp)
                    if show_progress:
                        sys.stdout.write("\r" + " " * len_msg + "\r")
                if last_chunk != 0:
                    mat_tmp = []
                    mat_chunk = losa.load_image_multiple(
                        data[depth1 - last_chunk + d1: depth1 + d1],
                        ncore=ncore, prefer="threads")
                    for j in range(last_chunk):
                        mat = mat_chunk[j]
                        if rotate != 0.0:
                            mat = ndi.rotate(mat, rotate, mode=mode,
                                             reshape=False, order=1)
                        mat_tmp.append(mat[h1:h2, w1:w2])
                    data_tmp[depth1 - last_chunk: depth1] = np.asarray(mat_tmp)
        else:
            data = losa.load_hdf(input_, key_path)
            data_type = data.dtype
            if axis == 2:
                hdf_chunk = (chunk, min(100, width1), min(100, height1))
                data_tmp = ofile.create_dataset(out_key,
                                                (depth1, width1, height1),
                                                dtype=data_type,
                                                chunks=hdf_chunk)
                for i in np.arange(0, depth1 - last_chunk, chunk):
                    if show_progress:
                        t1 = timeit.default_timer()
                        f_size = os.path.getsize(file_tmp) / b_unit
                        msg = "Writing to an intermediate hdf-file: {0:0.2f}" \
                              "MB. Time: {1:0.2f}s".format(f_size, t1 - t0)
                        len_msg = len(msg)
                        sys.stdout.write(msg)
                        sys.stdout.flush()
                    mat_chunk = data[i: i + chunk]
                    mat_tmp = []
                    for j in np.arange(chunk):
                        mat = mat_chunk[j]
                        if rotate != 0.0:
                            mat = ndi.rotate(mat, rotate, mode=mode,
                                             reshape=False, order=1)
                        mat_tmp.append(np.transpose(mat[h1:h2, w1:w2]))
                    data_tmp[i:i + chunk] = np.asarray(mat_tmp)
                    if show_progress:
                        sys.stdout.write("\r" + " " * len_msg + "\r")
                if last_chunk != 0:
                    mat_chunk = data[depth1 - last_chunk: depth1]
                    mat_tmp = []
                    for j in np.arange(last_chunk):
                        mat = mat_chunk[j]
                        if rotate != 0.0:
                            mat = ndi.rotate(mat, rotate, mode=mode,
                                             reshape=False, order=1)
                        mat_tmp.append(np.transpose(mat[h1:h2, w1:w2]))
                    data_tmp[depth1 - last_chunk: depth1] = np.asarray(mat_tmp)
            else:
                hdf_chunk = (chunk, min(100, height1), min(100, width1))
                data_tmp = ofile.create_dataset(out_key,
                                                (depth1, height1, width1),
                                                dtype=data_type,
                                                chunks=hdf_chunk)
                for i in np.arange(0, depth1 - last_chunk, chunk):
                    if show_progress:
                        t1 = timeit.default_timer()
                        f_size = os.path.getsize(file_tmp) / b_unit
                        msg = "Writing to an intermediate hdf-file: {0:0.2f}" \
                              "MB. Time: {1:0.2f}s".format(f_size, t1 - t0)
                        len_msg = len(msg)
                        sys.stdout.write(msg)
                        sys.stdout.flush()
                    mat_chunk = data[i: i + chunk]
                    mat_tmp = []
                    for j in np.arange(chunk):
                        mat = mat_chunk[j]
                        if rotate != 0.0:
                            mat = ndi.rotate(mat, rotate, mode=mode,
                                             reshape=False, order=1)
                        mat_tmp.append(mat[h1:h2, w1:w2])
                    data_tmp[i:i + chunk] = np.asarray(mat_tmp)
                    if show_progress:
                        sys.stdout.write("\r" + " " * len_msg + "\r")
                if last_chunk != 0:
                    mat_chunk = data[depth1 - last_chunk: depth1]
                    mat_tmp = []
                    for j in np.arange(last_chunk):
                        mat = mat_chunk[j]
                        if rotate != 0.0:
                            mat = ndi.rotate(mat, rotate, mode=mode,
                                             reshape=False, order=1)
                        mat_tmp.append(mat[h1:h2, w1:w2])
                    data_tmp[depth1 - last_chunk: depth1] = np.asarray(mat_tmp)
    if show_progress:
        t1 = timeit.default_timer()
        f_size = os.path.getsize(file_tmp) / b_unit
        print("Finish saving intermediate file! File size: {0:0.2f}MB. Time: "
              "{1:0.2f}s. The file will be deleted at the end!"
              "".format(f_size, t1 - t0))
    return file_tmp, out_key, folder_tmp


def reslice_dataset(input_, output, axis=1, key_path=None, rescaling=False,
                    nbit=16, minmax=None, skip=None, rotate=0.0, chunk=16,
                    mode="constant", crop=(0, 0, 0, 0, 0, 0),
                    ncore=None, show_progress=True):
    """
    Reslice a 3d dataset. Input can be a folder of tif files or a hdf file.

    Parameters
    ----------
    input_ : str, array_like
        It can be a folder path to tif files or a hdf file.
    output : str
        It can be a folder path (for generated tif-files) or a hdf file-path.
    axis : {1,2}
        Slicing axis. This axis becomes the 0-axis of the output.
    key_path : str, optional
        Key path to the dataset if the input is a hdf file.
    rescaling : bool
        Rescale dataset if True.
    nbit : {8,16}
        If rescaling is True, select data-type: 8-bit or 16-bit.
    minmax : tuple of float, or None
        Minimum and maximum values used for rescaling if True.
    skip : int or None
        Skipping step of images used for getting statistical information if
        rescaling is True and input is 32-bit data.
    rotate : float
        Rotate image (degree). Positive direction is counterclockwise.
    chunk : int
        Number of images to be loaded/saved in one go to reduce IO overhead.
    mode : {'reflect', 'grid-mirror', 'constant', 'grid-constant', \
           'nearest', 'mirror', 'grid-wrap', 'wrap'}
        Select how the input array is extended beyond its boundaries.
    crop : tuple of int, optional
        Crop 3D data from the edges, i.e.
        crop = (crop_depth1, crop_depth2, crop_height1, crop_height2,
        crop_width1, crop_width2). Cropping is done before reslicing.
    ncore : int or None
        Number of cpu-cores. Automatically selected if None.
    show_progress : bool
        Show the progress of reslicing data if True.

    Returns
    -------
    array_like or None
        If output is None, returning a 3D array.
    """
    if output is None:
        raise ValueError("Wrong output type !!!")
    if axis != 1 and axis != 2:
        raise ValueError("Only two options for axis: 1 or 2")
    else:
        axis = int(axis)
    __check_output(output)
    in_type = __get_input_type(input_)
    if in_type != "tif" and in_type != "hdf":
        raise ValueError("Wrong input type !!!")
    results = __save_intermediate_data(input_, output, axis, crop, key_path,
                                       rotate, chunk, mode, ncore,
                                       show_progress)
    file_tmp, key_tmp, folder_tmp = results
    with h5py.File(file_tmp, 'r') as hdf_object:
        data = hdf_object[key_tmp]
        (depth1, height1, width1) = data.shape
        chunk = np.clip(chunk, 1, height1 - 1)
        last_chunk = height1 - chunk * (height1 // chunk)
        data_type = data.dtype
        res_type = str(data_type)
        if rescaling is True:
            if nbit == 16:
                res_type = "uint16"
            elif nbit == 8:
                res_type = "uint8"
            else:
                raise ValueError("Only two options for nbit: 8 or 16 !!!")
            if str(data_type) != res_type:
                if data_type == np.uint8:
                    minmax = (0, 255)
                elif data_type == np.uint16:
                    minmax = (0, 65535)
                else:
                    if skip is None:
                        skip = min(20, int(0.02 * depth1))
                    skip = int(np.clip(skip, 1, depth1 - 1))
                    if minmax is None:
                        f_alias = get_statistical_information_dataset
                        minmax = f_alias(input_, percentile=(0, 100),
                                         skip=skip, crop=crop,
                                         key_path=key_path)[0:2]
            else:
                rescaling = False
        out_type = __get_output_type(output)
        t0 = timeit.default_timer()
        if out_type == "hdf":
            key_path = "entry/data" if key_path is None else key_path
            data_slice = losa.open_hdf_stream(output,
                                              (height1, depth1, width1),
                                              data_type=res_type,
                                              key_path=key_path,
                                              overwrite=True)
            for i in np.arange(0, height1 - last_chunk, chunk):
                if show_progress:
                    t1 = timeit.default_timer()
                    f_size = __get_dataset_size(output)
                    msg = "Save resliced data to file: {0:0.2f}MB." \
                          " Time: {1:0.2f}s".format(f_size, t1 - t0)
                    len_msg = len(msg)
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                mat_chunk = data[:, i: i + chunk, :]
                if rescaling:
                    mat_tmp = []
                    for j in np.arange(chunk):
                        mat = rescale(mat_chunk[:, j, :], nbit, minmax)
                        mat_tmp.append(mat)
                    mat_tmp = np.asarray(mat_tmp)
                else:
                    mat_tmp = np.moveaxis(mat_chunk, 1, 0)
                data_slice[i:i + chunk] = mat_tmp
                if show_progress:
                    sys.stdout.write("\r" + " " * len_msg + "\r")
            if last_chunk != 0:
                mat_chunk = data[:, height1 - last_chunk: height1, :]
                if rescaling:
                    mat_tmp = []
                    for j in np.arange(last_chunk):
                        mat = rescale(mat_chunk[:, j, :], nbit, minmax)
                        mat_tmp.append(mat)
                    mat_tmp = np.asarray(mat_tmp)
                else:
                    mat_tmp = np.moveaxis(mat_chunk, 1, 0)
                data_slice[height1 - last_chunk: height1] = mat_tmp
        else:
            list_file, len_msg = None, None
            for i in np.arange(0, height1 - last_chunk, chunk):
                if show_progress:
                    t1 = timeit.default_timer()
                    list_file = glob.glob(output + "/*tif*")
                    if list_file:
                        f_size = __get_dataset_size(output)
                        msg = "Save resliced data to file: {0:0.2f}MB." \
                              " Time: {1:0.2f}s".format(f_size, t1 - t0)
                        len_msg = len(msg)
                        sys.stdout.write(msg)
                        sys.stdout.flush()
                mat_chunk = data[:, i: i + chunk, :]
                out_files = [output + "/img_" + ("0000" + str(
                    i + j))[-5:] + ".tif" for j in range(chunk)]
                if rescaling:
                    mat_chunk = rescale(mat_chunk, nbit, minmax)
                losa.save_image_multiple(out_files, mat_chunk.astype(res_type),
                                         axis=1, ncore=ncore, prefer="threads")
                if show_progress:
                    if list_file:
                        sys.stdout.write("\r" + " " * len_msg + "\r")
            if last_chunk != 0:
                idx = height1 - last_chunk
                mat_chunk = data[:, idx: height1, :]
                out_files = [output + "/img_" + ("0000" + str(
                    idx + j))[-5:] + ".tif" for j in range(last_chunk)]
                if rescaling:
                    mat_chunk = rescale(mat_chunk, nbit, minmax)
                losa.save_image_multiple(out_files, mat_chunk.astype(res_type),
                                         axis=1, ncore=ncore, prefer="threads")
    if os.path.isdir(folder_tmp):
        shutil.rmtree(folder_tmp)
        if out_type == "hdf":
            shutil.rmtree(os.path.splitext(output)[0])
    if show_progress:
        t1 = timeit.default_timer()
        f_size = __get_dataset_size(output)
        print("Finish reslicing data! File size: {0:0.2f}MB. Time: {1:0.2f}s"
              "".format(f_size, t1 - t0))


def remove_ring_based_fft(mat, u=20, n=8, v=1, sort=False):
    """
    Remove ring artifacts in the reconstructed image by combining the polar
    transform and the fft-based method.

    Parameters
    ----------
    mat : array_like
        Square array. Reconstructed image
    u : int
        Cutoff frequency.
    n : int
        Filter order.
    v : int
        Number of rows (* 2) to be applied the filter.
    sort : bool, optional
        Apply sorting (Ref. [2]) if True.

    Returns
    -------
    array_like
        Ring-removed image.

    References
    ----------
    [1] : https://doi.org/10.1063/1.1149043

    [2] : https://doi.org/10.1364/OE.26.028396
    """
    (nrow, ncol) = mat.shape
    if nrow != ncol:
        raise ValueError(
            "Width and height of the reconstructed image are not the same")
    mask = util.make_circle_mask(ncol, 1.0)
    (x_mat, y_mat) = util.rectangular_from_polar(ncol, ncol, ncol, ncol)
    (r_mat, theta_mat) = util.polar_from_rectangular(ncol, ncol, ncol, ncol)
    polar_mat = util.mapping(mat, x_mat, y_mat)
    polar_mat = remo.remove_stripe_based_fft(polar_mat, u, n, v, sort=sort)
    mat_rec = util.mapping(polar_mat, r_mat, theta_mat)
    return mat_rec * mask


def remove_ring_based_wavelet_fft(mat, level=5, size=1, wavelet_name="db9",
                                  sort=False):
    """
    Remove ring artifacts in a reconstructed image by combining the polar
    transform and the wavelet-fft-based method (Ref. [1]).

    Parameters
    ----------
    mat : array_like
        Square array. Reconstructed image
    level : int
        Wavelet decomposition level.
    size : int
        Damping parameter. Larger is stronger.
    wavelet_name : str
        Name of a wavelet. Search pywavelets API for a full list.
    sort : bool, optional
        Apply sorting (Ref. [2]) if True.

    Returns
    -------
    array_like
        Ring-removed image.

    References
    ----------
    [1] : https://doi.org/10.1364/OE.17.008567

    [2] : https://doi.org/10.1364/OE.26.028396
    """
    (nrow, ncol) = mat.shape
    if nrow != ncol:
        raise ValueError(
            "Width and height of the reconstructed image are not the same")
    mask = util.make_circle_mask(ncol, 1.0)
    (x_mat, y_mat) = util.rectangular_from_polar(ncol, ncol, ncol, ncol)
    (r_mat, theta_mat) = util.polar_from_rectangular(ncol, ncol, ncol, ncol)
    polar_mat = util.mapping(mat, x_mat, y_mat)
    polar_mat = remo.remove_stripe_based_wavelet_fft(polar_mat, level, size,
                                                     wavelet_name, sort=sort)
    mat_rec = util.mapping(polar_mat, r_mat, theta_mat)
    return mat_rec * mask
