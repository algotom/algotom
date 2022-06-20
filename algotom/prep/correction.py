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
# Description: Python implementations of preprocessing techniques.
# Contributors:
# ============================================================================

"""
Module of correction methods in the preprocessing stage:
- Flat-field correction.
- Distortion correction.
- MTF deconvolution.
- Tilted sinogram generation.
- Tilted 1D intensity-profile generation.
- Beam hardening correction.
"""

import numpy as np
import numpy.fft as fft
from scipy.ndimage import map_coordinates
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.util.utility as util


def flat_field_correction(proj, flat, dark, ratio=1.0, use_dark=True,
                          **options):
    """
    Do flat-field correction with options to remove zinger arifacts and/or
    stripe artifacts.

    Parameters
    ----------
    proj : array_like
        3D or 2D array. Projection images or a sinogram image.
    flat : array_like
        2D or 1D array. Flat-field image or a single row of it.
    dark : array_like
        2D or 1D array. Dark-field image or a single row of it.
    ratio : float
        Ratio between exposure time used for recording projections
        and exposure time used for recording flat field.
    use_dark : bool
        Subtracting dark field if True. May no need in some cases.
    options : dict, optional
        Apply a zinger removal method and/or ring removal methods.
        E.g option1={"method": "dezinger", "para1": 0.001, "para2": 1},
        option2={"method": "remove_stripe_based_sorting",
                "para1": 15, "para2": 1}

    Returns
    -------
    array_like
        3D or 2D array. Corrected projections or corrected sinograms.
    """
    flat = ratio * flat
    if use_dark:
        flatdark = flat - dark
        try:
            proj_corr = (np.float32(proj) - dark) / flatdark
        except ZeroDivisionError:
            nmean = np.mean(flatdark)
            if nmean != 0.0:
                flatdark[flatdark == 0.0] = nmean
            else:
                flatdark[flatdark == 0.0] = 1
            proj_corr = (np.float32(proj) - dark) / flatdark
    else:
        try:
            proj_corr = np.float32(proj) / flat
        except ZeroDivisionError:
            nmean = np.mean(flat)
            if nmean != 0.0:
                flat[flat == 0.0] = nmean
            else:
                flat[flat == 0.0] = 1
            proj_corr = np.float32(proj) / flat
    if len(options) != 0:
        for opt_name in options:
            opt = options[opt_name]
            if opt is not None:
                if 'method' in opt.keys():
                    method = opt['method']
                    list_para = tuple(opt.values())[1:]
                    if proj_corr.ndim == 2:
                        if method in dir(remo):
                            proj_corr = getattr(remo, method)(proj_corr, *list_para)
                        elif method in dir(filt):
                            proj_corr = getattr(filt, method)(proj_corr, *list_para)
                        else:
                            raise ValueError("Can't find the method: '{}' in the"
                                             " namespace".format(method))
                    else:
                        for i in np.arange(proj_corr.shape[1]):
                            if method in dir(remo):
                                proj_corr[:, i, :] = getattr(remo, method)(
                                    proj_corr[:, i, :], *list_para)
                            elif method in dir(filt):
                                proj_corr[:, i, :] = getattr(filt, method)(
                                    proj_corr[:, i, :], *list_para)
                            else:
                                raise ValueError("Can't find the method: '{}' in "
                                                 "the namespace".format(method))
                else:
                    raise ValueError("Incorrect option: {}".format(opt))
    return proj_corr


def unwarp_projection(proj, xcenter, ycenter, list_fact):
    """
    Apply distortion correction to a projection image using the polynomial
    backward model (Ref. [1]_).

    Parameters
    ----------
    proj : array_like
        2D array. Projection image.
    xcenter : float
        Center of distortion in x-direction.
    ycenter : float
        Center of distortion in y-direction.
    list_fact : list of float
        Polynomial coefficients of the backward model.

    Returns
    -------
    array_like
        2D array. Distortion corrected.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.23.032859

    """
    (height, width) = proj.shape
    xu_list = np.arange(width) - xcenter
    yu_list = np.arange(height) - ycenter
    xu_mat, yu_mat = np.meshgrid(xu_list, yu_list)
    ru_mat = np.sqrt(xu_mat ** 2 + yu_mat ** 2)
    fact_mat = np.sum(np.asarray(
        [factor * ru_mat ** i for i, factor in enumerate(list_fact)]), axis=0)
    xd_mat = np.float32(np.clip(xcenter + fact_mat * xu_mat, 0, width - 1))
    yd_mat = np.float32(np.clip(ycenter + fact_mat * yu_mat, 0, height - 1))
    indices = np.reshape(yd_mat, (-1, 1)), np.reshape(xd_mat, (-1, 1))
    proj = map_coordinates(proj, indices, order=1, mode='reflect')
    return proj.reshape((height, width))


def unwarp_sinogram(data, index, xcenter, ycenter, list_fact, **option):
    """
    Unwarp sinogram [:,index.:] of a 3D tomographic dataset or
    a hdf/nxs object.

    Parameters
    ----------
    data : array_like or hdf object
        3D array.
    index : int
        Index of the sinogram.
    xcenter : float
        Center of distortion in x-direction.
    ycenter : float
        Center of distortion in y-direction.
    list_fact : list of float
        Polynomial coefficients of the backward model.
    option : list or tuple of int
        To extract subset data along axis 0 from a hdf object. E.g option =
        (start, stop, step)

    Returns
    -------
    array_like
        2D array. Distortion-corrected sinogram.
    """
    if data.ndim != 3:
        raise ValueError("Input must be a 3D data !!!")
    (depth, height, width) = data.shape
    if len(option) != 0:
        opt = option[list(option.keys())[0]]
        start, stop, step = opt
    else:
        start, stop, step = 0, depth, 1
    list_idx = range(start, stop, step)
    depth = len(list_idx)
    xu_list = np.arange(0, width) - xcenter
    yu = index - ycenter
    ru_list = np.sqrt(xu_list ** 2 + yu ** 2)
    flist = np.sum(np.asarray(
        [factor * ru_list ** i for i, factor in enumerate(list_fact)]), axis=0)
    xd_list = np.clip(xcenter + flist * xu_list, 0, width - 1)
    yd_list = np.clip(ycenter + flist * yu, 0, height - 1)
    yd_min = np.int16(np.floor(np.amin(yd_list)))
    yd_max = np.int16(np.ceil(np.amax(yd_list))) + 1
    yd_list = yd_list - yd_min
    indices = yd_list, xd_list
    sinogram = np.zeros((depth, width), dtype=np.float32)
    for i, idx in enumerate(list_idx):
        sinogram[i] = map_coordinates(
            data[idx, yd_min:yd_max, :], indices, order=1, mode='reflect')
    return sinogram


def unwarp_sinogram_chunk(data, start_index, stop_index, xcenter, ycenter,
                          list_fact, **option):
    """
    Unwarp chunk of sinograms [:, start_index: stop_index, :]
    of a 3D tomographic dataset or a hdf/nxs object.

    Parameters
    ----------
    data : array_like or hdf object
        3D array.
    start_index : int
        Starting index of sinograms.
    stop_index : int
        Stopping index of sinograms.
    xcenter : float
        Center of distortion in x-direction.
    ycenter : float
        Center of distortion in y-direction.
    list_fact : list of float
        Polynomial coefficients of the backward model.
    option : list or tuple of int
        To extract subset data along axis 0 from a hdf object. E.g option =
        [start, stop, step]

    Returns
    -------
    array_like
        3D array. Distortion corrected.
    """
    if data.ndim != 3:
        raise ValueError("Input must be a 3D data !!!")
    (depth, height, width) = data.shape
    if len(option) != 0:
        opt = option[list(option.keys())[0]]
        start, stop, step = opt
    else:
        start, stop, step = 0, depth, 1
    list_idx = range(start, stop, step)
    if stop_index == -1:
        stop_index = height
    xu_list = np.arange(0, width) - xcenter
    yu1 = start_index - ycenter
    ru_list = np.sqrt(xu_list ** 2 + yu1 ** 2)
    flist = np.sum(np.asarray(
        [factor * ru_list ** i for i, factor in enumerate(list_fact)]), axis=0)
    yd_list1 = np.clip(ycenter + flist * yu1, 0, height - 1)
    yu2 = stop_index - ycenter
    ru_list = np.sqrt(xu_list ** 2 + yu2 ** 2)
    flist = np.sum(np.asarray(
        [factor * ru_list ** i for i, factor in enumerate(list_fact)]), axis=0)
    yd_list2 = np.clip(ycenter + flist * yu2, 0, height - 1)
    yd_min = np.int16(np.floor(np.amin(yd_list1)))
    yd_max = np.int16(np.ceil(np.amax(yd_list2)))
    yu_list = np.arange(start_index, stop_index) - ycenter
    xu_mat, yu_mat = np.meshgrid(xu_list, yu_list)
    ru_mat = np.sqrt(xu_mat ** 2 + yu_mat ** 2)
    fact_mat = np.sum(np.asarray(
        [factor * ru_mat ** i for i, factor in enumerate(list_fact)]), axis=0)
    xd_mat = np.float32(np.clip(xcenter + fact_mat * xu_mat, 0, width - 1))
    yd_mat = np.float32(
        np.clip(ycenter + fact_mat * yu_mat, 0, height - 1)) - yd_min
    sino_chunk = np.asarray(
        [util.mapping(data[i, yd_min:yd_max, :], xd_mat, yd_mat) for i in
         list_idx])
    return sino_chunk


def mtf_deconvolution(mat, window, pad):
    """
    Deconvolve an projection image using division in the Fourier domain.
    Window can be determined using the approach in Ref. [1]_.

    Parameters
    ----------
    mat : array_like
        2D array. Projection image.
    window : array_like
        2D array. MTF function.
    pad : int
        Padding width to reduce the side effects of the Fourier transform.

    Returns
    -------
    array_like
        2D array. Deconvolved image.

    References
    ----------
    .. [1] https://doi.org/10.1117/12.2530324
    """
    (height1, width1) = mat.shape
    (height2, width2) = window.shape
    if (height1 > height2) or (width1 > width2):
        raise ValueError(
            "The sizes of the image are larger than the sizes of the window")
    else:
        pad_row = height2 - height1
        pad_col = width2 - width1
        mat = np.pad(mat, ((0, pad_row), (0, pad_col)), mode="edge")
    mat_pad = np.pad(mat, pad, mode="reflect")
    win_pad = np.pad(window, pad, mode="constant", constant_values=1.0)
    mat_dec = np.real(fft.ifft2(fft.fft2(mat_pad) / fft.ifftshift(win_pad)))
    mat_dec = mat_dec[pad: height2 + pad, pad: width2 + pad]
    return mat_dec[:height1, :width1]


def generate_tilted_sinogram(data, index, angle, **option):
    """
    Generate a tilted sinogram of a 3D tomographic dataset or a hdf/nxs object.

    Parameters
    ----------
    data : array_like or hdf object
        3D array.
    index : int
        Index of the sinogram.
    angle : float
        Tilted angle in degree.
    option : list or tuple of int
        To extract subset data along axis 0 from a hdf object. E.g option =
        (start, stop, step)

    Returns
    -------
    array_like
        2D array. Tilted sinogram.
    """
    if data.ndim != 3:
        raise ValueError("Input must be a 3D data !!!")
    (depth, height, width) = data.shape
    if len(option) != 0:
        opt = option[list(option.keys())[0]]
        start, stop, step = opt
    else:
        start, stop, step = 0, depth, 1
    list_idx = range(start, stop, step)
    depth = len(list_idx)
    x_cen = (width - 1.0) / 2
    y_cen = (height - 1.0) / 2
    x_list = np.arange(0, width) - x_cen
    y = index - y_cen
    angle = angle * np.pi / 180.0
    xt_list = x_list * np.cos(angle) - y * np.sin(angle)
    yt_list = x_list * np.sin(angle) + y * np.cos(angle)
    xt_list = np.clip(x_cen + xt_list, 0, width - 1)
    yt_list = np.clip(y_cen + yt_list, 0, height - 1)
    yt_min = np.int16(np.floor(np.amin(yt_list)))
    yt_max = np.int16(np.ceil(np.amax(yt_list))) + 1
    yt_list = yt_list - yt_min
    indices = yt_list, xt_list
    sinogram = np.zeros((depth, width), dtype=np.float32)
    for i, idx in enumerate(list_idx):
        sinogram[i] = map_coordinates(
            data[idx, yt_min:yt_max, :], indices, order=1, mode='reflect')
    return sinogram


def generate_tilted_sinogram_chunk(data, start_index, stop_index, angle,
                                   **option):
    """
    Generate a chunk of tilted sinograms of a 3D tomographic dataset or a
    hdf/nxs object.

    Parameters
    ----------
    data : array_like or hdf object
        3D array.
    start_index : int
        Starting index of sinograms.
    stop_index : int
        Stopping index of sinograms.
    angle : float
        Tilted angle in degree.
    option : list or tuple of int
        To extract subset data along axis 0 from a hdf object. E.g option =
        (start, stop, step)

    Returns
    -------
    array_like
        3D array. Chunk of tilted sinograms.
    """
    if data.ndim != 3:
        raise ValueError("Input must be a 3D data !!!")
    (depth, height, width) = data.shape
    if len(option) != 0:
        opt = option[list(option.keys())[0]]
        start, stop, step = opt
    else:
        start, stop, step = 0, depth, 1
    list_idx = range(start, stop, step)
    x_cen = (width - 1.0) / 2
    y_cen = (height - 1.0) / 2
    angle = angle * np.pi / 180.0
    x_list = np.arange(0, width) - x_cen
    y_list = np.arange(start_index, stop_index + 1) - y_cen
    x_mat, y_mat = np.meshgrid(x_list, y_list)
    x_mat1 = x_mat * np.cos(angle) - y_mat * np.sin(angle)
    y_mat1 = x_mat * np.sin(angle) + y_mat * np.cos(angle)
    x_mat1 = np.clip(x_cen + x_mat1, 0, width - 1)
    y_mat1 = np.clip(y_cen + y_mat1, 0, height - 1)
    y_min = np.int16(np.floor(np.amin(y_mat1)))
    y_max = np.int16(np.ceil(np.amax(y_mat1))) + 1
    y_mat1 = y_mat1 - y_min
    sino_chunk = np.asarray(
        [util.mapping(data[i, y_min:y_max, :], x_mat1, y_mat1) for i in
         list_idx])
    return sino_chunk


def generate_tilted_profile_line(mat, index, angle):
    """
    Generate a tilted horizontal intensity-profile of an image.

    Parameters
    ----------
    mat : array_like
        2D array.
    index : int
        Index of the line.
    angle : float
        Tilted angle in degree.

    Returns
    -------
    array_like
        1D array.
    """
    if mat.ndim != 2:
        raise ValueError("Input must be a 2D array !!!")
    (height, width) = mat.shape
    x_cen = (width - 1.0) / 2
    y_cen = (height - 1.0) / 2
    x_list = np.arange(0, width) - x_cen
    y = index - y_cen
    angle = angle * np.pi / 180.0
    xt_list = x_list * np.cos(angle) - y * np.sin(angle)
    yt_list = x_list * np.sin(angle) + y * np.cos(angle)
    xt_list = np.clip(x_cen + xt_list, 0, width - 1)
    yt_list = np.clip(y_cen + yt_list, 0, height - 1)
    yt_min = np.int16(np.floor(np.amin(yt_list)))
    yt_max = np.int16(np.ceil(np.amax(yt_list))) + 1
    yt_list = yt_list - yt_min
    indices = yt_list, xt_list
    profile_line = map_coordinates(mat[yt_min:yt_max, :], indices, order=1,
                                   mode='reflect')
    return profile_line


def generate_tilted_profile_chunk(mat, start_index, stop_index, angle):
    """
    Generate a chunk of tilted horizontal intensity-profiles of an image.

    Parameters
    ----------
    mat : array_like
        2D array.
    start_index : int
        Starting index of lines.
    stop_index : int
        Stopping index of lines.
    angle : float
        Tilted angle in degree.

    Returns
    -------
    array_like
        2D array.
    """
    if mat.ndim != 2:
        raise ValueError("Input must be a 2D data !!!")
    (height, width) = mat.shape
    x_cen = (width - 1.0) / 2
    y_cen = (height - 1.0) / 2
    angle = angle * np.pi / 180.0
    x_list = np.arange(0, width) - x_cen
    y_list = np.arange(start_index, stop_index + 1) - y_cen
    x_mat, y_mat = np.meshgrid(x_list, y_list)
    x_mat1 = x_mat * np.cos(angle) - y_mat * np.sin(angle)
    y_mat1 = x_mat * np.sin(angle) + y_mat * np.cos(angle)
    x_mat1 = np.clip(x_cen + x_mat1, 0, width - 1)
    y_mat1 = np.clip(y_cen + y_mat1, 0, height - 1)
    y_min = np.int16(np.floor(np.amin(y_mat1)))
    y_max = np.int16(np.ceil(np.amax(y_mat1))) + 1
    y_mat1 = y_mat1 - y_min
    profile_chunk = util.mapping(mat[y_min:y_max, :], x_mat1, y_mat1)
    return profile_chunk


def non_linear_function(intensity, q, n, opt=True):
    """
    Function used to define the response curve.

    Parameters
    ----------
    intensity : float
        Values stay in the range of [0; 1]
    q : float
        Positive number.
    n : float
        Positive number. Must larger than 1.
    opt : bool
        True: Curve more to values closer to 1.0.
        False: Curve more to values closer to 0.0
    Returns
    -------
    float
    """
    if opt:
        x = 1.0 - intensity
        num = np.log(1.0 - q * (1 - n))
        result = 1.0 - (np.log(1.0 - q * (x**n - n * x)) / num)
    else:
        x = intensity
        num = np.log(1.0 - q * (1 - n))
        result = (np.log(1.0 - q * (x**n - n * x)) / num)
    return result


def beam_hardening_correction(mat, q, n, opt=True):
    """
    Correct the grayscale values of a normalized image using a non-linear
    function.

    Parameters
    ----------
    mat : array_like
        Normalized projection image or sinogram image.
    q : float
        Positive number. Recommended range [0.005, 50].
    n : float
        Positive number. Must larger than 1.
    opt : bool
        True: Curve towards 0.0.
        False: Curve towards 1.0.

    Returns
    -------
    array_like
        Corrected image.
    """
    if np.max(mat) >= 2.0:
        raise ValueError("!!! Input image must be normalized, i.e. gray-scales "
                         "are in the range of [0.0, 1.0]) !!!")
    if n < 2.0:
        raise ValueError("!!! n must be larger than or equal to 2 !!!")
    return np.asarray([non_linear_function(x, q, n, opt) for x in mat])
