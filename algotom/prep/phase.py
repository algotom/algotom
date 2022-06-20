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
    - Generate a quality map, weight mask.
    - Methods for speckle-based phase-contrast imaging.
        + Find shifts between two stacks of images.
        + Find shifts between sample-images.
        + Align between two stacks of images.
        + Retrieve phase image.
        + Generate transmission-signal and dark-signal images.
"""

import multiprocessing as mp
import numpy as np
import numpy.fft as fft
from scipy.fft import dctn, idctn
import scipy.ndimage as ndi
from numba import jit
from joblib import Parallel, delayed
import algotom.util.correlation as corl


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
    described in Ref. [1]_.

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
    value is calculated based on Algorithm 4 in Ref. [1]_.

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


def unwrap_phase_based_cosine_transform(mat, window=None):
    """
    Unwrap a phase image using the cosine transform as described in Ref. [1]_.

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


def unwrap_phase_based_fft(mat, win_for=None, win_back=None):
    """
    Unwrap a phase image using the Fourier transform as described in Ref. [1]_.

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


def unwrap_phase_iterative_fft(mat, iteration=4, win_for=None, win_back=None,
                               weight_map=None):
    """
    Unwrap a phase image using an iterative FFT-based method as described in
    Ref. [1]_.

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
    mat_unwrap = unwrap_phase_based_fft(mat * weight_map, win_for, win_back)
    for i in range(iteration):
        mat_wrap = _wrap_to_pi(mat_unwrap)
        mat_diff = mat - mat_wrap
        nmean = np.mean(mat_diff)
        mat_diff = _wrap_to_pi(mat_diff - nmean)
        phase_diff = unwrap_phase_based_fft(mat_diff * weight_map, win_for,
                                            win_back)
        mat_unwrap = mat_unwrap + phase_diff
    return mat_unwrap


def _make_window_FC_method(height, width):
    """
    Make a window for a normal integration method:
    the FC (Frankot and Chellappa) method.
    """
    xcenter = width // 2
    ycenter = height // 2
    ulist = (1.0 * np.arange(0, width) - xcenter) / width
    vlist = (1.0 * np.arange(0, height) - ycenter) / width
    u, v = np.meshgrid(ulist, vlist)
    window = u ** 2 + v ** 2
    window[ycenter, xcenter] = 1.0
    window = 1 / window
    window[ycenter, xcenter] = 0.0
    return u, v, window


def reconstruct_surface_from_gradient_FC_method(grad_x, grad_y,
                                                correct_negative=True,
                                                window=None):
    """
    Reconstruct a surface from the gradients in x and y-direction using the
    Frankot-Chellappa method (Ref. [1]_). Note that the DC-component
    (average value of an image) of the reconstructed image is unidentified
    because the DC-component of the FFT-window is zero.

    Parameters
    ----------
    grad_x : array_like
        2D array. Gradient in x-direction.
    grad_y : array_like
        2D array. Gradient in y-direction.
    correct_negative : bool, optional
        Correct negative offset if True.
    window : list of array_like
        list of three 2D-arrays. Spatial frequencies in x, y, and the window
        for the Fourier transform. Generated if None.

    Returns
    -------
    array_like
        2D array. Reconstructed surface.

    References
    ----------
    .. [1] https://doi.org/10.1109/34.3909
    """
    height, width = grad_x.shape
    if grad_x.shape != grad_y.shape:
        raise ValueError("Input gradients must be the same size!!!")
    grad2_x = _double_image(grad_x)
    grad2_y = _double_image(grad_y)
    height2, width2 = grad2_x.shape
    if window is None:
        u, v, win = _make_window_FC_method(height2, width2)
    else:
        err_msg = "Input must be a list of 3 arrays (u, v, window)!!!"
        if not (isinstance(window, tuple) or isinstance(window, list)):
            raise ValueError(err_msg)
        else:
            if len(window) != 3:
                raise ValueError(err_msg)
            else:
                (u, v, win) = window
            if win.shape != grad2_x.shape:
                raise ValueError("Window-size {0} must be double the "
                                 "image-size {1}!!!".format(win.shape,
                                                            grad_x.shape))
    fmat_x = -1j * u * fft.fftshift(fft.fft2(grad2_x))
    fmat_y = -1j * v * fft.fftshift(fft.fft2(grad2_y))
    rec_surf = (0.5 / np.pi) * np.real(
        fft.ifft2(fft.ifftshift((fmat_x + fmat_y) * win)))[height:, 0:width]
    if correct_negative:
        nmin = np.min(rec_surf)
        if nmin < 0.0:
            rec_surf = rec_surf - 2 * nmin
    return np.float32(rec_surf)


def _make_window_SCS_method(height, width):
    """
    Make a window for a normal integration method:
    the SCS (Simchony, Chellappa, and Shao) method.
    """
    ulist = 1.0 * np.arange(0, width) / width
    vlist = 1.0 * np.arange(0, height) / height
    u, v = np.meshgrid(ulist, vlist)
    sin_u = np.sin(2 * np.pi * u)
    sin_v = np.sin(2 * np.pi * v)
    sin_u2 = np.power(np.sin(np.pi * u), 2)
    sin_v2 = np.power(np.sin(np.pi * v), 2)
    window = (sin_u2 + sin_v2)
    window[0, 0] = 1.0
    window = 1 / (4 * 1j * window)
    window[0, 0] = 0.0
    return sin_u, sin_v, window


def reconstruct_surface_from_gradient_SCS_method(grad_x, grad_y,
                                                 correct_negative=True,
                                                 window=None, pad=0,
                                                 pad_mode="linear_ramp"):
    """
    Reconstruct a surface from the gradients in x and y-direction using the
    Simchony-Chellappa-Shao method (Ref. [1]_). Note that the DC-component
    (average value of an image) of the reconstructed image is unidentified
    because the DC-component of the FFT-window is zero.

    Parameters
    ----------
    grad_x : array_like
        2D array. Gradient in x-direction.
    grad_y : array_like
        2D array. Gradient in y-direction.
    correct_negative : bool, optional
        Correct negative offset if True.
    window : list of array_like
        List of three 2D-arrays. Spatial frequencies in x, y, and the window
        for the Fourier transform. Generated if None.
    pad : int
        Padding width.
    pad_mode : str
        Padding method. Full list can be found at numpy.pad documentation.

    Returns
    -------
    array_like
        2D array. Reconstructed surface.

    References
    ----------
    .. [1] https://doi.org/10.1109/34.55103
    """
    if grad_x.shape != grad_y.shape:
        raise ValueError("Input gradients must be the same size!!!")
    if pad != 0:
        grad_x = np.pad(grad_x, pad, mode=pad_mode)
        grad_y = np.pad(grad_y, pad, mode=pad_mode)
    (height, width) = grad_x.shape
    if window is None:
        sin_u, sin_v, win = _make_window_SCS_method(height, width)
    else:
        err_msg = "Input must be a list of 3 arrays (sin_u, sin_v, window)!!!"
        if not (isinstance(window, tuple) or isinstance(window, list)):
            raise ValueError(err_msg)
        else:
            if len(window) != 3:
                raise ValueError(err_msg)
            else:
                (sin_u, sin_v, win) = window
            if win.shape != grad_x.shape:
                raise ValueError("Window-size {0} is not the same as the "
                                 "image-size {1}. Note to take into account "
                                 "the pad value of {2}!!!"
                                 "".format(win.shape, grad_x.shape, pad))
    fmat_x = sin_u * fft.fft2(grad_x)
    fmat_y = sin_v * fft.fft2(grad_y)
    fmat = fmat_x + fmat_y
    rec_surf = np.real(fft.ifft2(fmat * win))
    if pad != 0:
        rec_surf = rec_surf[pad:-pad, pad:-pad]
    if correct_negative:
        nmin = np.min(rec_surf)
        if nmin < 0.0:
            rec_surf = rec_surf - 2 * nmin
    return np.float32(rec_surf)


def find_shift_between_image_stacks(ref_stack, sam_stack, win_size, margin,
                                    list_ij, global_value="median", gpu=False,
                                    block=32, sub_pixel=True, method="diff",
                                    size=3, ncore=None, norm=True):
    """
    Find shifts between each pair of two image-stacks. Can be used to
    align reference-images and sample-images in speckle-based imaging
    technique.
    The method finds the shift between two images by finding local shifts
    between small areas of the images given by a list of middle points.

    Parameters
    ----------
    ref_stack : array_like
        3D array. Reference images.
    sam_stack : array_like
        3D array. Sample images.
    win_size : int
        To define the size of the area around a selected pixel of the sample
        image.
    margin : int
        To define the size of the area of the reference image for searching,
        i.e. size = 2 * margin + win_size.
    list_ij : list of list of int
        List of indices of points used for local search. Accept the value of
        [i_index, j_index] for a single point or
        [[i_index0, i_index1,...], [j_index0, j_index1,...]]
        for multiple points.
    global_value : {"median", "mean", "mixed"}
        Method for calculating the global value from local values.
    gpu : bool, optional
        Use GPU for computing if True.
    block : int
        Size of a GPU block. E.g. 16, 32, 64, ...
    sub_pixel : bool, optional
        Enable sub-pixel location.
    method : {"diff", "poly_fit"}
        Method for finding 1d sub-pixel position. Two options: a differential
        method or a polynomial method.
    size : int
        Window size around the integer location of the maximum value used for
        sub-pixel searching.
    ncore: int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    norm : bool, optional
        Normalize the input images if True.

    Returns
    -------
    array_like
        List of [[x_shift0, y_shift0], [x_shift1, y_shift1],...]
    """
    if ref_stack.shape != sam_stack.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_stack.shape) == 2:
        ref_stack = np.expand_dims(ref_stack, axis=0)
        sam_stack = np.expand_dims(sam_stack, axis=0)
    num_point = len(ref_stack)
    xy_shifts = []
    f_alias = corl.find_global_shift_based_local_shifts
    for i in range(num_point):
        (x_shift, y_shift) = f_alias(ref_stack[i], sam_stack[i], win_size,
                                     margin, list_ij=list_ij,
                                     global_value=global_value,
                                     gpu=gpu, block=block,
                                     sub_pixel=sub_pixel, method=method,
                                     size=size, ncore=ncore, norm=norm,
                                     return_list=False)
        xy_shifts.append([x_shift, y_shift])
    return np.asarray(xy_shifts)


def find_shift_between_sample_images(ref_stack, sam_stack, sr_shifts, win_size,
                                     margin, list_ij, global_value="median",
                                     gpu=False, block=32, sub_pixel=True,
                                     method="diff", size=3, ncore=None,
                                     norm=True):
    """
    Find shifts between sample-images in a stack and the first sample-image.
    It is used to align sample-images of the same rotation angle from multiple
    tomographic datasets. Reference-images are used for normalization before
    finding the shifts.

    Parameters
    ----------
    ref_stack : array_like
        3D array. Reference images.
    sam_stack : array_like
        3D array. Sample images.
    sr_shifts : array_like
        List of shifts between each pair of reference-images and sample-images.
    win_size : int
        To define the size of the area around a selected pixel of the sample
        image.
    margin : int
        To define the size of the area of the reference image for searching,
        i.e. size = 2 * margin + win_size.
    list_ij : list of list of int
        List of indices of points used for local search. Accept the value of
        [i_index, j_index] for a single point or
        [[i_index0, i_index1,...], [j_index0, j_index1,...]]
        for multiple points.
    global_value : {"median", "mean", "mixed"}
        Method for calculating the global value from local values.
    gpu : bool, optional
        Use GPU for computing if True.
    block : int
        Size of a GPU block. E.g. 16, 32, 64, ...
    sub_pixel : bool, optional
        Enable sub-pixel location.
    method : {"diff", "poly_fit"}
        Method for finding 1d sub-pixel position. Two options: a differential
        method or a polynomial method.
    size : int
        Window size around the integer location of the maximum value used for
        sub-pixel searching.
    ncore: int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    norm : bool, optional
        Normalize the input images if True.

    Returns
    -------
    array_like
        List of [[0.0, 0.0], [x_shift1, y_shift1],...]. For convenient usage,
        the shift of the first image in the stack with itself, [0.0, 0.0], is
        added to the result.
    """
    if ref_stack.shape != sam_stack.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_stack.shape) == 2:
        ref_stack = np.expand_dims(ref_stack, axis=0)
        sam_stack = np.expand_dims(sam_stack, axis=0)
    eps = 1.0e-09
    xy_shifts = [[0.0, 0.0]]
    num_image = len(ref_stack)
    crop = 1 + int(np.max(np.abs(sr_shifts)))
    sam_mat0 = ndi.shift(sam_stack[0], np.flipud(sr_shifts[0])) / (
            ref_stack[0] + eps)
    sam_mat0 = sam_mat0[crop:-crop, crop:-crop]
    f_alias = corl.find_global_shift_based_local_shifts
    for i in range(1, num_image):
        sam_mat1 = ndi.shift(sam_stack[i], np.flipud(sr_shifts[i])) / (
                ref_stack[i] + eps)
        sam_mat1 = sam_mat1[crop:-crop, crop:-crop]
        (x_shift, y_shift) = f_alias(sam_mat0, sam_mat1, win_size, margin,
                                     list_ij=list_ij,
                                     global_value=global_value, gpu=gpu,
                                     block=block, sub_pixel=sub_pixel,
                                     method=method, size=size, ncore=ncore,
                                     norm=norm, return_list=False)
        xy_shifts.append([x_shift, y_shift])
    return np.asarray(xy_shifts)


def align_image_stacks(ref_stack, sam_stack, sr_shifts, sam_shifts=None,
                       mode="reflect"):
    """
    Align each pair of two image-stacks using provided reference-sample shifts
    with an option to correct the shifts between sample-images.

    Parameters
    ----------
    ref_stack : array_like
        3D array. Reference images.
    sam_stack : array_like
        3D array. Sample images.
    sr_shifts : array_like
        List of shifts between each pair of reference-images and sample-images.
    sam_shifts : array_like, optional
        List of shifts between each sample-image and the first sample-image.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        Method to fill up empty areas caused by shifting the images.

    Returns
    -------
    ref_stack : array_like
        3D array. Aligned reference-images.
    sam_stack : array_like
        3D array. Aligned sample-images.
    """
    msg = "Number of shifts and number of images must be the same !!!"
    if ref_stack.shape != sam_stack.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_stack) != len(sr_shifts):
        raise ValueError(msg)
    if sam_shifts is not None:
        if len(ref_stack) != len(sam_shifts):
            raise ValueError(msg)
    if len(ref_stack.shape) == 2:
        ref_stack = np.expand_dims(ref_stack, axis=0)
        sam_stack = np.expand_dims(sam_stack, axis=0)
    num_image = len(ref_stack)
    if num_image == 1:
        sam_stack[0] = ndi.shift(sam_stack[0],
                                 np.flipud(np.squeeze(sr_shifts)), mode=mode)
    else:
        for i in range(num_image):
            mat1 = ndi.shift(sam_stack[i], np.flipud(sr_shifts[i]), mode=mode)
            if sam_shifts is not None:
                mat1 = ndi.shift(mat1, np.flipud(sam_shifts[i]), mode=mode)
                ref1 = ndi.shift(ref_stack[i], np.flipud(sam_shifts[i]),
                                       mode=mode)
            else:
                ref1 = ref_stack[i]
            sam_stack[i] = mat1
            ref_stack[i] = ref1
    return ref_stack, sam_stack


def retrieve_phase_based_speckle_tracking(ref_stack, sam_stack, dim=1,
                                          win_size=7, margin=10, method="diff",
                                          size=3, gpu=False, block=(16, 16),
                                          ncore=None, norm=True,
                                          norm_global=True, chunk_size=None,
                                          surf_method="SCS",
                                          correct_negative=True, window=None,
                                          pad=0, pad_mode="linear_ramp",
                                          return_shift=False):
    """
    Retrieve the phase image from two stacks of speckle-images and
    sample-images where the shift of each pixel is determined using a
    correlation-based technique (Ref. [1-2]).

    Parameters
    ----------
    ref_stack : array_like
        3D array. Reference images (speckle images).
    sam_stack : array_like
        3D array. Sample images.
    dim : {1, 2}
        To find the shifts (in x and y) separately (1D) or together (2D).
    win_size : int
        Size of local areas in the sample image for finding shifts.
    margin : int
        To define the searching range of the sample images in finding the
        shifts compared to the reference images.
    method : {"diff", "poly_fit"}
        Method for finding sub-pixel shift. Two options: a differential
        method (Ref. [3]) or a polynomial method (Ref. [4]). The "poly_fit"
        option is not available if using GPU.
    size : int
        Window size around the integer location of the maximum value used for
        sub-pixel location. Adjustable if using the polynomial method.
    gpu : {False, True, "hybrid"}
        Use GPU for computing if True or in "hybrid" mode.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (8, 8), (16, 16), (32, 32), ...
    ncore: int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    norm : bool, optional
        Normalizing the inputs if True.
    norm_global : bool, optional
        Normalize by using the full size of the inputs if True.
    chunk_size : int or None
        Size of each chunk extracted along the height of the image.
    surf_method : {"SCS", "FC"}
        Select method for surface reconstruction: "SCS" (Ref. [5]) or "FC"
        (Ref. [6])
    correct_negative : bool, optional
        Correct negative offset if True.
    window : list of array_like
        List of three 2D-arrays. Spatial frequencies in x, y, and the window
        in the Fourier space for the surface reconstruction method. Generated
        if None.
    pad : int
        Padding-width used for the "SCS" method.
    pad_mode : str
        Padding-method used for the "SCS" method. Full list can be found at
        numpy.pad documentation.
    return_shift : bool, optional
        Return a list of 3 arrays: x-shifts, y-shifts, and phase image if True.
        The shifts can be used to determine transmission-signal and dark-signal
        image.

    Returns
    -------
    x_shifts : array_like
        Return if return_shift is True. Shifts in x-direction.
    y_shifts : array_like
        return if return_shift is True. Shifts in y-direction.
    phase : array_like
        Phase image.

    References
    ----------
    .. [1] https://doi.org/10.1038/srep08762
    .. [2] https://doi.org/10.1103/PhysRevApplied.5.044014
    .. [3] https://doi.org/10.48550/arXiv.0712.4289
    .. [4] https://doi.org/10.1088/0957-0233/17/6/045
    .. [5] https://doi.org/10.1109/34.55103
    .. [6] https://doi.org/10.1109/34.3909
    """
    win_size = np.clip(win_size, 1, None)
    margin = np.clip(margin, 1, None)
    size = np.clip(size, 3, None)
    (x_shifts, y_shifts) = corl.find_local_shifts(ref_stack, sam_stack,
                                                  dim=dim, win_size=win_size,
                                                  margin=margin, method=method,
                                                  size=size, gpu=gpu,
                                                  block=block, ncore=ncore,
                                                  norm=norm,
                                                  norm_global=norm_global,
                                                  chunk_size=chunk_size)
    edge = margin + 2
    x_shifts = np.pad(x_shifts[edge:-edge, edge:-edge], edge,
                      mode="reflect")
    y_shifts = np.pad(y_shifts[edge:-edge, edge:-edge], edge,
                      mode="reflect")
    if surf_method == "SCS":
        f_alias = reconstruct_surface_from_gradient_SCS_method
        phase = f_alias(x_shifts, y_shifts, correct_negative=correct_negative,
                        window=window, pad=pad, pad_mode=pad_mode)
    else:
        f_alias = reconstruct_surface_from_gradient_FC_method
        phase = f_alias(x_shifts, y_shifts, correct_negative=correct_negative,
                        window=window)
    if return_shift:
        return x_shifts, y_shifts, phase
    else:
        return phase


@jit(nopython=True, parallel=False, cache=True)
def _calculate_transmission_dark_field_values(ref_stack, sam_stack):
    """
    Supplementary method for determining transmission-signal image and
    dark-signal image.
    """
    num1 = np.mean(ref_stack)
    num2 = np.mean(sam_stack)
    trans = 1.0
    dark = 1.0
    if (num1 != 0.0) and (num2 != 0):
        trans = num2 / num1
        num = np.std(ref_stack)
        if num != 0.0:
            dark = (1 / trans) * (np.std(sam_stack) / num)
        else:
            dark = 1.0
    return trans, dark


@jit(nopython=True, parallel=False, cache=True)
def _get_transmission_dark_field_signal(ref_stack, sam_stack, x_shifts,
                                        y_shifts, win_size, margin):
    """
    Supplementary method for determining transmission-signal image and
    dark-signal image.
    """
    win_size = 2 * (win_size // 2) + 1
    radi = win_size // 2
    start = radi + margin
    radi1 = radi + 1
    (height, width) = ref_stack.shape[-2:]
    stop_col, stop_row = width - start, height - start
    f_alias = _calculate_transmission_dark_field_values
    trans = np.ones((height - 2 * start, width - 2 * start), dtype=np.float32)
    dark = np.ones_like(trans)
    if len(ref_stack.shape) == 2:
        for i in range(start, stop_row):
            for j in range(start, stop_col):
                i1 = i + int(y_shifts[i, j])
                j1 = j + int(x_shifts[i, j])
                mat1 = ref_stack[i - radi:i + radi1, j - radi:j + radi1]
                mat2 = sam_stack[i1 - radi:i1 + radi1, j1 - radi:j1 + radi1]
                (val1, val2) = f_alias(mat1, mat2)
                i2, j2 = i - start, j - start
                trans[i2, j2], dark[i2, j2] = val1, val2
    else:
        for i in range(start, stop_row):
            for j in range(start, stop_col):
                i1 = i + int(y_shifts[i, j])
                j1 = j + int(x_shifts[i, j])
                mat1 = ref_stack[:, i - radi:i + radi1, j - radi:j + radi1]
                mat2 = sam_stack[:, i1 - radi:i1 + radi1, j1 - radi:j1 + radi1]
                (val1, val2) = f_alias(mat1, mat2)
                i2, j2 = i - start, j - start
                trans[i2, j2], dark[i2, j2] = val1, val2
    return trans, dark


def get_transmission_dark_field_signal(ref_stack, sam_stack, x_shifts,
                                       y_shifts, win_size, ncore=None):
    """
    Get the transmission-signal image and dark-signal image from two stacks of
    speckle-images and sample-images.

    Parameters
    ----------
    ref_stack : array_like
        3D array. Reference images (speckle images).
    sam_stack : array_like
        3D array. Sample images.
    x_shifts : array_like
        x-shift image.
    y_shifts : array_like
        y-shift image.
    win_size : int
        Window size used for calculating signals.
    ncore: int or None
        Number of cpu-cores used for computing. Automatically selected if None.

    Returns
    -------
    trans : array_like
        Transmission-signal image
    dark : array_like
        Dark-signal image
    """
    if len(ref_stack.shape) != 2 and len(ref_stack.shape) != 3:
        raise ValueError("Input data must be 2D or 3D array !!!")
    (height, width) = ref_stack.shape[-2:]
    win_size = 2 * (win_size // 2) + 1
    radi = win_size // 2
    margin = int(max(np.max(np.abs(x_shifts)), np.max(np.abs(y_shifts))))
    pad = radi + margin
    if ncore is None:
        ncore = np.clip(mp.cpu_count() - 1, 1, None)
    chunk_size = (height - 2 * pad) // ncore
    f_alias = _get_transmission_dark_field_signal
    if ncore == 1 or chunk_size < 20:
        trans, dark = f_alias(ref_stack, sam_stack, x_shifts, y_shifts,
                              win_size,
                              margin)
    else:
        trans = np.ones((height - 2 * pad, width - 2 * pad), dtype=np.float32)
        dark = np.ones_like(trans)
        list_index = np.array_split(np.arange(pad, height - pad), ncore)
        b_e = np.asarray([[pos[0], pos[-1] + 1] for pos in list_index])
        ntime = len(b_e)
        if len(ref_stack.shape) == 2:
            ref_stack = np.expand_dims(ref_stack, 0)
            sam_stack = np.expand_dims(sam_stack, 0)
        results = Parallel(n_jobs=ncore)(
            delayed(f_alias)(ref_stack[:, b_e[i, 0] - pad:b_e[i, 1] + pad, :],
                             sam_stack[:, b_e[i, 0] - pad:b_e[i, 1] + pad, :],
                             x_shifts[b_e[i, 0] - pad:b_e[i, 1] + pad, :],
                             y_shifts[b_e[i, 0] - pad:b_e[i, 1] + pad, :],
                             win_size, margin) for i in range(ntime))
        for i in range(ntime):
            trans[b_e[i, 0] - pad:b_e[i, 1] - pad] = results[i][0]
            dark[b_e[i, 0] - pad:b_e[i, 1] - pad] = results[i][1]
    trans = np.pad(trans, pad, mode="edge")
    dark = np.pad(dark, pad, mode="edge")
    return trans, dark
