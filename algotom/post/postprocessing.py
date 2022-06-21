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
# Description: Python implementations of postprocessing techniques.
# Contributors:
# ============================================================================

"""
Module of methods in the postprocessing stage:
    - Get statistical information of reconstructed images or a dataset.
    - Downsample 2D, 3D array, or a dataset.
    - Rescale 2D, 3D array or a dataset to 8-bit or 16-bit data-type.
    - Removing ring artifacts in a reconstructed image by transform back and
      forth between the polar coordinates and the Cartesian coordinates.
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter
import algotom.util.utility as util
import algotom.io.loadersaver as losa
import algotom.prep.removal as remo


def get_statical_information(mat, percentile=(1, 99), denoise=False):
    """
    Get statical information of an image.

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
        mat = gaussian_filter(mat, 2)
    gmin = np.min(mat)
    gmax = np.max(mat)
    min_percent = np.percentile(mat, percentile[0])
    max_percent = np.percentile(mat, percentile[-1])
    median = np.median(mat)
    mean = np.mean(mat)
    variance = np.var(mat)
    return gmin, gmax, min_percent, max_percent, mean, median, variance


def get_statical_information_dataset(input_, percentile=(1, 99), skip=5,
                                     denoise=False, key_path=None):
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
        Key path to the dataset if the input is the hdf file.

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
    if isinstance(input_, str) and (os.path.splitext(input_)[-1] == ""):
        list_file = losa.find_file(input_ + "/*.tif*")
        depth = len(list_file)
        if depth == 0:
            raise ValueError("No tif files in the folder: {}".format(input_))
        list_stat = []
        for i in range(0, depth, skip):
            mat = losa.load_image(list_file[i])
            if denoise is True:
                mat = gaussian_filter(mat, 2)
            list_stat.append(get_statical_information(mat, percentile, denoise))
    else:
        if isinstance(input_, str):
            file_ext = os.path.splitext(input_)[-1]
            if not (file_ext == '.hdf' or file_ext == '.h5'
                    or file_ext == ".nxs"):
                raise ValueError(
                    "Can't open this type of file format {}".format(file_ext))
            if key_path is None:
                raise ValueError(
                    "Please provide the key path to the dataset!!!")
            input_ = losa.load_hdf(input_, key_path)
        depth = len(input_)
        list_stat = []
        for i in range(0, depth, skip):
            mat = input_[i]
            if denoise is True:
                mat = gaussian_filter(mat, 2)
            list_stat.append(get_statical_information(mat, percentile, denoise))
    list_stat = np.asarray(list_stat)
    gmin = np.min(list_stat[:, 0])
    gmax = np.max(list_stat[:, 1])
    min_percent = np.min(list_stat[:, 2])
    max_percent = np.max(list_stat[:, 3])
    median = np.median(list_stat[:, 4])
    mean = np.mean(list_stat[:, 5])
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


def downsample_dataset(input_, output, cell_size, method="mean", key_path=None):
    """
    Downsample a dataset. This can be a folder of tif files, a hdf file,
    or a 3D array.

    Parameters
    ----------
    input_ : str, array_like
        It can be a folder path to tif files, a hdf file, or 3D array.
    output : str, None
        It can be a folder path, a hdf file path, or None (memory consuming).
    cell_size : int or tuple of int
        Window size along axes used for grouping pixels.
    method : {"mean", "median", "max", "min"}
        Downsampling method.
    key_path : str, optional
        Key path to the dataset if the input is the hdf file.

    Returns
    -------
    array_like or None
        If output is None, returning an 3D array.
    """
    if output is not None:
        file_base, file_ext = os.path.splitext(output)
        if file_ext != "":
            file_base = os.path.dirname(output)
        if os.path.exists(file_base):
            raise ValueError("Folder exists!!! Please choose another path!!!")
    if method == "median":
        dsp_method = np.median
    elif method == "max":
        dsp_method = np.max
    elif method == "min":
        dsp_method = np.amin
    else:
        dsp_method = np.mean
    if isinstance(cell_size, int):
        cell_size = (cell_size, cell_size, cell_size)
    if isinstance(input_, str) and (os.path.splitext(input_)[-1] == ""):
        list_file = losa.find_file(input_ + "/*.tif*")
        depth = len(list_file)
        if depth == 0:
            raise ValueError("No tif files in the folder: {}".format(input_))
        (height, width) = np.shape(losa.load_image(list_file[0]))
        depth_dsp = depth // cell_size[0]
        height_dsp = height // cell_size[1]
        width_dsp = width // cell_size[2]
        num = 0
        if (depth_dsp != 0) and (height_dsp != 0) and (width_dsp != 0):
            if output is not None:
                file_base, file_ext = os.path.splitext(output)
                if file_ext != "":
                    if not (file_ext == '.hdf' or file_ext == '.h5'
                            or file_ext == ".nxs"):
                        raise ValueError(
                            "File extension must be hdf, h5, or nxs")
                    else:
                        output = file_base + file_ext
                        data_out = losa.open_hdf_stream(
                            output, (depth_dsp, height_dsp, width_dsp),
                            key_path="downsample/data", overwrite=False)
            data_dsp = []
            for i in range(0, depth, cell_size[0]):
                if (i + cell_size[0]) > depth:
                    break
                else:
                    mat = []
                    for j in range(i, i + cell_size[0]):
                        mat.append(losa.load_image(list_file[j]))
                    mat = np.asarray(mat)
                    mat = mat[:, :height_dsp * cell_size[1],
                              :width_dsp * cell_size[2]]
                    mat = mat.reshape(1, cell_size[0], height_dsp,
                                      cell_size[1], width_dsp, cell_size[2])
                    mat_dsp = dsp_method(
                        dsp_method(dsp_method(mat, axis=-1), axis=1), axis=2)
                    if output is None:
                        data_dsp.append(mat_dsp[0])
                    else:
                        if file_ext == "":
                            out_name = "0000" + str(num)
                            losa.save_image(
                                output + "/img_" + out_name[-5:] + ".tif",
                                mat_dsp[0])
                        else:
                            data_out[num] = mat_dsp[0]
                        num += 1
        else:
            raise ValueError("Incorrect cell size {}".format(cell_size))
    else:
        if isinstance(input_, str):
            file_ext = os.path.splitext(input_)[-1]
            if not (file_ext == '.hdf' or file_ext == '.h5'
                    or file_ext == ".nxs"):
                raise ValueError(
                    "Can't open this type of file format {}".format(file_ext))
            if key_path is None:
                raise ValueError(
                    "Please provide the key path to the dataset!!!")
            input_ = losa.load_hdf(input_, key_path)
        (depth, height, width) = input_.shape
        depth_dsp = depth // cell_size[0]
        height_dsp = height // cell_size[1]
        width_dsp = width // cell_size[2]
        if (depth_dsp != 0) and (height_dsp != 0) and (width_dsp != 0):
            if output is None:
                input_ = input_[:depth_dsp * cell_size[0],
                                :height_dsp * cell_size[1],
                                :width_dsp * cell_size[2]]
                input_ = input_.reshape(
                    depth_dsp, cell_size[0], height_dsp, cell_size[1],
                    width_dsp, cell_size[2])
                data_dsp = dsp_method(
                    dsp_method(dsp_method(input_, axis=-1), axis=1), axis=2)
            else:
                file_base, file_ext = os.path.splitext(output)
                if file_ext != "":
                    if not (file_ext == '.hdf' or file_ext == '.h5'
                            or file_ext == ".nxs"):
                        raise ValueError(
                            "File extension must be hdf, h5, or nxs")
                    else:
                        output = file_base + file_ext
                        data_out = losa.open_hdf_stream(
                            output, (depth_dsp, height_dsp, width_dsp),
                            key_path="downsample/data", overwrite=False)
                num = 0
                for i in range(0, depth, cell_size[0]):
                    if (i + cell_size[0]) > depth:
                        break
                    else:
                        mat = input_[i:i + cell_size[0],
                                     :height_dsp * cell_size[1],
                                     :width_dsp * cell_size[2]]
                        mat = mat.reshape(1, cell_size[0], height_dsp,
                                          cell_size[1], width_dsp, cell_size[2])
                        mat_dsp = dsp_method(dsp_method(
                            dsp_method(mat, axis=-1), axis=1), axis=2)
                        if file_ext != "":
                            data_out[num] = mat_dsp[0]
                        else:
                            out_name = "0000" + str(num)
                            losa.save_image(
                                output + "/img_" + out_name[-5:] + ".tif",
                                mat_dsp[0])
                        num += 1
        else:
            raise ValueError("Incorrect cell size {}".format(cell_size))
    if output is None:
        return np.asarray(data_dsp)


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


def rescale_dataset(input_, output, nbit=16, minmax=None, skip=None,
                    key_path=None):
    """
    Rescale a dataset to 8-bit or 16-bit data-type. The dataset can be a
    folder of tif files, a hdf file, or a 3D array.

    Parameters
    ----------
    input_ : str, array_like
        It can be a folder path to tif files, a hdf file, or 3D array.
    output : str, None
        It can be a folder path, a hdf file path, or None (memory consuming).
    nbit : {8,16}
        Rescaled data-type: 8-bit or 16-bit.
    minmax : tuple of float, or None
        Minimum and maximum values used for rescaling. They are calculated if
        None is given.
    skip : int or None
        Skipping step of reading input used for getting statistical information.
    key_path : str, optional
        Key path to the dataset if the input is the hdf file.

    Returns
    -------
    array_like or None
        If output is None, returning an 3D array.
    """
    if output is not None:
        file_base, file_ext = os.path.splitext(output)
        if file_ext != "":
            file_base = os.path.dirname(output)
        if os.path.exists(file_base):
            raise ValueError("Folder exists!!! Please choose another path!!!")
    if isinstance(input_, str) and (os.path.splitext(input_)[-1] == ""):
        list_file = losa.find_file(input_ + "/*.tif*")
        depth = len(list_file)
        if depth == 0:
            raise ValueError("No tif files in the folder: {}".format(input_))
        if minmax is None:
            if skip is None:
                skip = int(np.ceil(0.15 * depth))
            (gmin, gmax) = get_statical_information_dataset(input_, skip=skip)[
                           0:2]
        else:
            (gmin, gmax) = minmax
        if output is not None:
            file_base, file_ext = os.path.splitext(output)
            if file_ext != "":
                if not (file_ext == '.hdf' or file_ext == '.h5'
                        or file_ext == ".nxs"):
                    raise ValueError("File extension must be hdf, h5, or nxs")
                output = file_base + file_ext
                (height, width) = np.shape(losa.load_image(list_file[0]))
                if nbit == 8:
                    data_type = "uint8"
                else:
                    data_type = "uint16"
                data_out = losa.open_hdf_stream(output, (depth, height, width),
                                                key_path="rescale/data",
                                                data_type=data_type,
                                                overwrite=False)
        data_res = []
        for i in range(0, depth):
            mat = rescale(
                losa.load_image(list_file[i]), nbit=nbit, minmax=(gmin, gmax))
            if output is None:
                data_res.append(mat)
            else:
                file_base, file_ext = os.path.splitext(output)
                if file_ext == "":
                    out_name = "0000" + str(i)
                    losa.save_image(output + "/img_" + out_name[-5:] + ".tif",
                                    mat)
                else:
                    data_out[i] = mat
    else:
        if isinstance(input_, str):
            file_ext = os.path.splitext(input_)[-1]
            if not (file_ext == '.hdf' or file_ext == '.h5'
                    or file_ext == ".nxs"):
                raise ValueError(
                    "Can't open this type of file format {}".format(file_ext))
            if key_path is None:
                raise ValueError(
                    "Please provide the key path to the dataset!!!")
            input_ = losa.load_hdf(input_, key_path)
        (depth, height, width) = input_.shape
        if minmax is None:
            if skip is None:
                skip = int(np.ceil(0.15 * depth))
            f_alias = get_statical_information_dataset
            (gmin, gmax) = f_alias(input_,skip=skip,key_path=key_path)[0:2]
        else:
            (gmin, gmax) = minmax
        data_res = []
        if output is not None:
            file_base, file_ext = os.path.splitext(output)
            if file_ext != "":
                if not (file_ext == '.hdf' or file_ext == '.h5'
                        or file_ext == ".nxs"):
                    raise ValueError("File extension must be hdf, h5, or nxs")
                output = file_base + file_ext
                if nbit == 8:
                    data_type = "uint8"
                else:
                    data_type = "uint16"
                data_out = losa.open_hdf_stream(
                    output, (depth, height, width), key_path="rescale/data",
                    data_type=data_type, overwrite=False)
        for i in range(0, depth):
            mat = rescale(input_[i], nbit=nbit, minmax=(gmin, gmax))
            if output is None:
                data_res.append(mat)
            else:
                file_base, file_ext = os.path.splitext(output)
                if file_ext != "":
                    data_out[i] = mat
                else:
                    out_name = "0000" + str(i)
                    losa.save_image(output + "/img_" + out_name[-5:] + ".tif",
                                    mat)
    if output is None:
        return np.asarray(data_res)


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
        Apply sorting (Ref. [2]_) if True.

    Returns
    -------
    array_like
        Ring-removed image.

    References
    ----------
    .. [1] https://doi.org/10.1063/1.1149043
    .. [2] https://doi.org/10.1364/OE.26.028396
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
    transform and the wavelet-fft-based method (Ref. [1]_).

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
        Apply sorting (Ref. [2]_) if True.

    Returns
    -------
    array_like
        Ring-removed image.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.17.008567
    .. [2] https://doi.org/10.1364/OE.26.028396
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
