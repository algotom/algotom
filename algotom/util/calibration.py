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
# Description: Calibration methods
# Contributors:
# ============================================================================

"""
Module of calibration methods:

    -   Correcting the non-uniform background of an image.
    -   Binarizing an image.
    -   Calculating the distance between two point-like objects segmented from
        two images. Useful for determining pixel-size in helical scans.
"""

import numpy as np
import scipy.ndimage as ndi
import algotom.util.utility as util
import scipy.signal as sig


def normalize_background(mat, size=51):
    """
    Correct a non-uniform background of an image using the median filter.

    Parameters
    ----------
    mat : array_like
        2D array.
    size : int
        Size of the median filter.

    Returns
    -------
    array_like
        2D array. Corrected image.
    """
    mat_bck = ndi.median_filter(mat, size, mode="reflect")
    mean_val = np.mean(mat_bck)
    try:
        mat_cor = mean_val * mat / mat_bck
    except ZeroDivisionError:
        mat_bck[mat_bck == 0.0] = mean_val
        mat_cor = mean_val * mat / mat_bck
    return mat_cor


def normalize_background_based_fft(mat, sigma=5, pad=None, mode="reflect"):
    """
    Correct a non-uniform background of an image using a Fourier Gaussian
    filter.

    Parameters
    ----------
    mat : array_like
        2D array.
    sigma : int
        Sigma of the Gaussian.
    pad : int
        Padding for the Fourier transform.
    mode : str, list of str, or tuple of str
        Padding method. One of options : 'reflect', 'edge', 'constant'. Full
        list is at:
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Returns
    -------
    array_like
        2D array. Corrected image.
    """
    (height, width) = mat.shape
    if height <= width:
        ratio = 1.0 * height / width
        sigma_x = int(np.ceil(sigma / ratio))
        sigma_y = sigma
    else:
        ratio = 1.0 * width / height
        sigma_y = int(np.ceil(sigma / ratio))
        sigma_x = sigma
    mat_bck = util.apply_gaussian_filter(mat, sigma_x, sigma_y,
                                         pad=pad, mode=mode)
    mean_val = np.mean(mat_bck)
    try:
        mat_cor = mean_val * mat / mat_bck
    except ZeroDivisionError:
        mat_bck[mat_bck == 0.0] = mean_val
        mat_cor = mean_val * mat / mat_bck
    return mat_cor


def invert_dot_contrast(mat):
    """
    Invert the contrast of a 2D binary array to make sure that a dot is white.

    Parameters
    ----------
    mat : array_like
        2D binary array.

    Returns
    -------
    array_like
        2D array.
    """
    (height, width) = mat.shape
    ratio = np.sum(mat) / (height * width)
    max_val = np.max(mat)
    if ratio > 0.5:
        mat = max_val - mat
    return mat


def calculate_threshold(mat, bgr="bright"):
    """
    Calculate threshold value based on Algorithm 4 in Ref. [1].

    Parameters
    ----------
    mat : array_like
        2D array.
    bgr : {"bright", "dark"}
        To indicate the brightness of the background against image features.

    Returns
    -------
    float
        Threshold value.

    References
    ----------
    [1] : https://doi.org/10.1364/OE.26.028396
    """
    size = max(mat.shape)
    list1 = np.sort(np.ndarray.flatten(mat))
    list1 = ndi.zoom(list1, (1.0 * size) / len(list1), mode='nearest')
    list2 = sig.savgol_filter(list1, 2 * (len(list1) // 2) - 1, 3)
    if bgr == "bright":
        threshold = list2[0]
    else:
        threshold = list2[-1]
    return threshold


def binarize_image(mat, threshold=None, bgr="bright", norm=False, denoise=True,
                   invert=True):
    """
    Binarize an image.

    Parameters
    ----------
    mat : array_like
        2D array.
    threshold : float, optional
        Threshold value for binarization. Automatically calculated using
        Algorithm 4 in Ref. [1] if None.
    bgr : {"bright", "dark"}
        To indicate the brightness of the background against image features.
    norm : bool, optional
        Apply normalization if True.
    denoise : bool, optional
        Apply denoising if True.
    invert : bool, optional
        Invert the contrast if needed.

    Returns
    -------
    array_like
        2D binary array.

    References
    ----------
    [1] : https://doi.org/10.1364/OE.26.028396
    """
    if denoise is True:
        mat = ndi.median_filter(np.abs(mat), (3, 3))
    if norm is True:
        mat = normalize_background_based_fft(mat)
    if threshold is None:
        threshold = calculate_threshold(mat, bgr)
    else:
        num_min = np.min(mat)
        num_max = np.max(mat)
        if threshold < num_min or threshold > num_max:
            raise ValueError("Selected threshold value is out of the range of"
                             " [{0}, {1}]".format(num_min, num_max))
    mat = np.asarray(mat > threshold, dtype=np.float32)
    if invert is True:
        mat = invert_dot_contrast(mat)
    mat = np.int16(ndi.binary_fill_holes(mat))
    return mat


def get_dot_size(mat, size_opt="max"):
    """
    Get size of binary dots given the option.

    Parameters
    ----------
    mat : array_like
        2D binary array.
    size_opt : {"max", "min", "median", "mean"}
        Select options.

    Returns
    -------
    dot_size : float
        Size of the dot.
    """
    mat = np.int16(mat)
    mat_label, num_dots = ndi.label(mat)
    list_index = np.arange(1, num_dots + 1)
    list_sum = ndi.measurements.sum(mat, labels=mat_label, index=list_index)
    if size_opt == "median":
        dot_size = np.median(list_sum)
    elif size_opt == "mean":
        dot_size = np.mean(list_sum)
    elif size_opt == "min":
        dot_size = np.min(list_sum)
    else:
        dot_size = np.max(list_sum)
    return dot_size


def check_dot_size(mat, min_size, max_size):
    """
    Check if the size of a dot is in a range.

    Parameters
    ----------
    mat : array_like
        2D array.
    min_size : float
        Minimum size.
    max_size : float
        Maximum size.

    Returns
    -------
    bool
    """
    check = False
    dot_size = mat.sum()
    if (dot_size >= min_size) and (dot_size <= max_size):
        check = True
    return check


def select_dot_based_size(mat, dot_size, ratio=0.01):
    """
    Select dots having a certain size.

    Parameters
    ----------
    mat : array_like
        2D array.
    dot_size : float
        Size of the standard dot.
    ratio : float
        Used to calculate the acceptable range.
        [dot_size - ratio*dot_size; dot_size + ratio*dot_size]

    Returns
    -------
    array_like
        2D array. Selected dots.
    """
    mat = np.int16(mat)
    min_size = np.clip(np.int32(dot_size - ratio * dot_size), 1, None)
    max_size = np.clip(np.int32(dot_size + ratio * dot_size), 1, None)
    mat_label, _ = ndi.label(mat)
    list_dots = ndi.find_objects(mat_label)
    dots_selected = [dot for dot in list_dots
                     if check_dot_size(mat[dot], min_size, max_size)]
    mat1 = np.zeros_like(mat)
    for _, j in enumerate(dots_selected):
        mat1[j] = mat[j]
    return mat1


def calculate_distance(mat1, mat2, size_opt="max", threshold=None,
                       bgr='bright', norm=False, denoise=True, invert=True):
    """
    Calculate the distance between two point-like objects segmented from
    two images. Useful for measuring pixel-size in helical scans (Ref. [1]).

    Parameters
    ----------
    mat1 : array_like
        2D array.
    mat2 : array_like
        2D array.
    size_opt : {"max", "min", "median", "mean"}
        Options to select binary objects based on their size.
    threshold : float, optional
        Threshold value for binarization. Automatically calculated using
        Algorithm 4 in Ref. [2] if None.
    bgr : {"bright", "dark"}
        To indicate the brightness of the background against image features.
    norm : bool, optional
        Apply normalization if True.
    denoise : bool, optional
        Apply denoising if True.
    invert : bool, optional
        Invert the contrast if needed.

    References
    ----------
    [1] : https://doi.org/10.1364/OE.418448
    [2] : https://doi.org/10.1364/OE.26.028396
    """
    mat_bin1 = binarize_image(mat1, threshold=threshold, bgr=bgr, norm=norm,
                              denoise=denoise, invert=invert)
    dot_size1 = get_dot_size(mat_bin1, size_opt=size_opt)
    mat_bin1 = select_dot_based_size(mat_bin1, dot_size1)
    mat_bin2 = binarize_image(mat2, bgr=bgr, norm=norm, threshold=threshold,
                              denoise=denoise, invert=invert)
    dot_size2 = get_dot_size(mat_bin2, size_opt=size_opt)
    mat_bin2 = select_dot_based_size(mat_bin2, dot_size2)
    com1 = ndi.center_of_mass(mat_bin1)
    com2 = ndi.center_of_mass(mat_bin2)
    distance = np.sqrt((com1[0] - com2[0]) ** 2 + (com1[1] - com2[1]) ** 2)
    return distance
