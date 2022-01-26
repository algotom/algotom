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
# Description: Python implementations of phase-imaging-related techniques.
# Contributors:
# ============================================================================

"""
Module for phase contrast imaging:
- Unwrap phase images.
- Generate a weighting map.
"""

import numpy as np
import numpy.fft as fft
from scipy.fft import dctn, idctn
import scipy.ndimage as ndi


def _wrap_to_pi(mat):
    """
    Wrap image values in the range of [-Pi; Pi]
    """
    return (mat + np.pi) % (2 * np.pi) - np.pi


def _make_window(height, width, direction="forward"):
    """
    Make a window for a FFT-based filter.

    Parameters
    ----------
    height : int
        Height of the window.
    width : int
        Width of the window.
    direction : {"forward", "backward"}
        Specify if the window is used for multiplication (forward) or
        division (backward).
    """
    xcenter = width // 2
    ycenter = height // 2
    ulist = (1.0 * np.arange(0, width) - xcenter) / xcenter
    vlist = (1.0 * np.arange(0, height) - ycenter) / ycenter
    u, v = np.meshgrid(ulist, vlist)
    window = u ** 2 + v ** 2
    if direction != "forward":
        window[ycenter, xcenter] = 1.0
    return window


def _forward_operator(mat, window):
    mat_res = fft.ifft2(fft.ifftshift(fft.fftshift(
        fft.fft2(mat)) * window))
    return mat_res


def _backward_operator(mat, window):
    mat_res = fft.ifft2(fft.ifftshift(fft.fftshift(
        fft.fft2(mat)) / window))
    return mat_res


def _double_image(mat):
    mat1 = np.hstack((mat, np.fliplr(mat)))
    mat2 = np.vstack((np.flipud(mat1), mat1))
    return mat2


def _make_cosine_window(height, width):
    """
    Make a window for cosine transform.
    """
    y_mat, x_mat = np.ogrid[0:height, 0:width]
    window = 2.0 * (np.cos(np.pi * y_mat / height) + np.cos(
        np.pi * x_mat / width) - 2.0)
    window[0, 0] = 1.0
    return window


def get_quality_map(mat, size):
    """
    Generate a quality map using the phase derivative variance (PDV) as
    described in Ref. [1].

    Parameters
    ----------
    mat : array_like
        2D array.
    size : int
        Window size. e.g. size=5.

    Returns
    -------
    array_like
        2D array.

    References
    ----------
    .. [1] Dennis Ghiglia and Mark Pritt, "Two-dimensional Phase Unwrapping:
           Theory, Algorithms, and Software", Wiley, New York,1998.
    """
    (height, width) = mat.shape
    win_size = 2 * (size // 2) + 1
    mat_pad = np.pad(mat, 1, mode="reflect")
    rho_x = _wrap_to_pi(np.diff(mat_pad, axis=1))[:height, :width]
    rho_y = _wrap_to_pi(np.diff(mat_pad, axis=0))[:height, :width]
    kernel = 1.0 * np.ones((win_size, win_size)) / (win_size ** 2)
    mean_x = ndi.convolve(rho_x, kernel, mode="reflect")
    mean_y = ndi.convolve(rho_y, kernel, mode="reflect")
    rad = win_size // 2
    sum_x = np.zeros_like(mat, dtype=np.float32)
    sum_y = np.zeros_like(mat, dtype=np.float32)
    for i in range(-rad, rad + 1):
        for j in range(-rad, rad + 1):
            sum_x += np.square(
                np.roll(np.roll(rho_x, i, axis=0), j, axis=1) - mean_x)
            sum_y += np.square(
                np.roll(np.roll(rho_y, i, axis=0), j, axis=1) - mean_y)
    return (np.sqrt(sum_x) + np.sqrt(sum_y)) / win_size ** 2


def get_weight_mask(mat, snr=1.5):
    """
    Generate a binary weight-mask based on a provided quality map. Threshold
    value is calculated based on Algorithm 4 in Ref. [1].

    Parameters
    ----------
    mat : array_like
        2D array. e.g. a quality map.
    snr : float
        Ratio used to calculate the threshold value. Greater is less sensitive.

    Returns
    -------
    array_like
        2D binary array.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    size = max(mat.shape)
    list_sort = np.sort(np.ndarray.flatten(mat))
    list_dsp = ndi.zoom(list_sort, 1.0 * size / len(list_sort), mode='nearest')
    npoint = len(list_dsp)
    xlist = np.arange(0, npoint, 1.0)
    ndrop = int(0.25 * npoint)
    (slope, intercept) = np.polyfit(xlist[ndrop:-ndrop - 1],
                                    list_dsp[ndrop:-ndrop - 1], 1)[0:2]
    y_end = intercept + slope * xlist[-1]
    noise_level = np.abs(y_end - intercept)
    threshold = y_end + noise_level * snr * 0.5
    mask = np.asarray(mat > threshold, dtype=np.float32)
    return mask


def phase_unwrap_based_cosine_transform(mat, window=None):
    """
    Unwrap a phase image using the cosine transform as described in Ref. [1].

    Parameters
    ----------
    mat : array_like
        2D array. Wrapped phase-image in the range of [-Pi; Pi].
    window : array_like
        2D array. Window is used for the cosine transform. Generated if None.

    Returns
    -------
    array_like
        2D array. Unwrapped phase-image.

    References
    ----------
    .. [1] https://doi.org/10.1364/JOSAA.11.000107
    """
    (height, width) = mat.shape
    if window is None:
        window = _make_cosine_window(height, width)
    else:
        if window.shape != mat.shape:
            raise ValueError("Window must be the same size as the image!!!")
    rho_x = _wrap_to_pi(np.diff(mat, axis=1))
    rho_y = _wrap_to_pi(np.diff(mat, axis=0))
    rho_x2 = np.diff(rho_x, axis=1, prepend=0, append=0)
    rho_y2 = np.diff(rho_y, axis=0, prepend=0, append=0)
    rho = rho_x2 + rho_y2
    mat_unwrap = idctn(dctn(rho) / window, overwrite_x=True)
    return mat_unwrap


def phase_unwrap_based_fft(mat, win_for=None, win_back=None):
    """
    Unwrap a phase image using the Fourier transform as described in Ref. [1].

    Parameters
    ----------
    mat : array_like
        2D array. Wrapped phase-image in the range of [-Pi; Pi].
    win_for : array_like
        2D array. FFT-window for the forward transform. Generated if None.
    win_back : array_like
        2D array. FFT-window for the backward transform. Making sure there are
        no zero-values. Generated if None.

    Returns
    -------
    array_like
        2D array. Unwrapped phase-image.

    References
    ----------
    .. [1] https://doi.org/10.1109/36.297989
    """
    height, width = mat.shape
    mat2 = _double_image(mat)
    height2, width2 = mat2.shape
    if win_for is None:
        win_for = _make_window(height2, width2, direction="forward")
    else:
        if win_for.shape != mat2.shape:
            raise ValueError("Window-size must be double the image-size!!!")
    if win_back is None:
        win_back = _make_window(height2, width2, direction="backward")
    else:
        if win_back.shape != mat2.shape:
            raise ValueError("Window-size must be double the image-size!!!")
    mat_unwrap = np.real(
        _backward_operator(np.imag(_forward_operator(
            np.exp(mat2 * 1j), win_for) * np.exp(-1j * mat2)), win_back))
    mat_unwrap = mat_unwrap[height:, 0:width]
    return mat_unwrap


def phase_unwrap_iterative_fft(mat, iteration=4, win_for=None, win_back=None,
                               weight_map=None):
    """
    Unwrap a phase image using an iterative FFT-based method as described in
    Ref. [1].

    Parameters
    ----------
    mat : array_like
        2D array. Wrapped phase-image in the range of [-Pi; Pi].
    iteration : int
        Number of iteration.
    win_for : array_like
        2D array. FFT-window for the forward transform. Generated if None.
    win_back : array_like
        2D array. FFT-window for the backward transform. Making sure there are
        no zero-values. Generated if None.
    weight_map : array_like
        2D array. Using a weight map if provided.

    Returns
    -------
    array_like
        2D array. Unwrapped phase-image.

    References
    ----------
    .. [1] https://doi.org/10.1364/AO.56.007079
    """
    height, width = mat.shape
    if win_for is None:
        win_for = _make_window(2 * height, 2 * width, direction="forward")
    if win_back is None:
        win_back = _make_window(2 * height, 2 * width, direction="backward")
    if weight_map is None:
        weight_map = np.ones_like(mat)
    mat_unwrap = phase_unwrap_based_fft(mat * weight_map, win_for, win_back)
    for i in range(iteration):
        mat_wrap = _wrap_to_pi(mat_unwrap)
        mat_diff = mat - mat_wrap
        nmean = np.mean(mat_diff)
        mat_diff = _wrap_to_pi(mat_diff - nmean)
        phase_diff = phase_unwrap_based_fft(mat_diff * weight_map, win_for,
                                            win_back)
        mat_unwrap = mat_unwrap + phase_diff
    return mat_unwrap
