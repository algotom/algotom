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
# Description: Utility methods
# Contributors:
# ============================================================================

"""
Module of utility methods:
    - Methods for parallel computing, geometric transformation, masking.
    - Methods for customizing stripe/ring removal methods
        + sort_forward
        + sort_backward
        + separate_frequency_component
        + generate_fitted_image
        + detect_stripe
        + calculate_regularization_coefficient
        + make_2d_butterworth_window
        + make_2d_damping_window
        + apply_wavelet_decomposition
        + apply_wavelet_reconstruction
        + apply_filter_to_wavelet_component
        + interpolate_inside_stripe
        + transform_slice_forward
        + transform_slice_backward
    - Customized smoothing filters:
        + apply_gaussian_filter (in the Fourier space)
        + apply_regularization_filter
    - Methods for grid scans:
        + detect_sample
        + fix_non_sample_areas
        + locate_slice
        + locate_slice_chunk
 """

import sys
import multiprocessing as mp
import pywt
import numpy as np
import scipy.ndimage as ndi
from scipy import interpolate
import scipy.signal.windows as win
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
import numpy.fft as fft
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.rec.reconstruction as reco


def apply_method_to_multiple_sinograms(data, method, para, ncore=None):
    """
    Apply a processing method (in "filtering", "removal", and "reconstruction"
    module) to multiple sinograms or multiple slices in parallel.

    Parameters
    ----------
    data : array_like or hdf object
        3D array data where sinograms/slices are extracted along axis 1,
        e.g [:, i, :].
    method : str
        Name of a method. e.g. "remove_stripe_based_sorting".
    para : list
        Parameters of the method. e.g. [21, 1]
    ncore: int or None
        Number of cores used for computing. Automatically selected if None.

    Returns
    -------
    array_like
        Same axis-definition as the input.
    """
    if ncore is None:
        ncore = np.clip(mp.cpu_count() - 1, 1, None)
    else:
        ncore = np.clip(ncore, 1, None)
    if not isinstance(para, list):
        para = tuple(list([para]))
    else:
        para = tuple(para)
    (depth, height, width) = data.shape
    if method in dir(remo):
        method_used = getattr(remo, method)
    elif method in dir(filt):
        method_used = getattr(filt, method)
    elif method in dir(reco):
        method_used = getattr(reco, method)
    else:
        raise ValueError("Can't find the method: '{}' in the namespace"
                         "".format(method))
    data_out = Parallel(n_jobs=ncore, backend="threading")(
        delayed(method_used)(data[:, i, :], *para) for i in range(height))
    data_out = np.moveaxis(np.asarray(data_out), 0, 1)
    return data_out


def mapping(mat, x_mat, y_mat, order=1, mode="reflect"):
    """
    Apply a geometric transformation to a 2D array

    Parameters
    ----------
    mat : array_like
        2D array.
    x_mat : array_like
        2D array of the x-coordinates.
    y_mat : array_like
        2D array of the y-coordinates.
    order : int, optional
        The order of the spline interpolation, default is 1.
        The order has to be in the range 0-5.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the input array is extended beyond
        its boundaries. Default is 'reflect'.

    Returns
    -------
    array_like
        2D array.
    """
    coords = np.vstack((np.ndarray.flatten(y_mat), np.ndarray.flatten(x_mat)))
    mat = ndi.map_coordinates(mat, coords, order=order, mode=mode)
    return mat.reshape(x_mat.shape)


def make_circle_mask(width, ratio):
    """
    Create a circle mask.

    Parameters
    -----------
    width : int
        Width of a square array.
    ratio : float
        Ratio between the diameter of the mask and the width of the array.

    Returns
    ------
    array_like
         Square array.
    """
    mask = np.zeros((width, width), dtype=np.float32)
    center = width // 2
    radius = ratio * center
    y, x = np.ogrid[-center:width - center, -center:width - center]
    mask_check = x * x + y * y <= radius * radius
    mask[mask_check] = 1.0
    return mask


def sort_forward(mat, axis=0):
    """
    Sort grayscales of an image along an axis.
    e.g. axis=0 is to sort along each column.

    Parameters
    ----------
    mat : array_like
        2D array.
    axis : int
        Axis along which to sort.

    Returns
    --------
    mat_sort : array_like
        2D array. Sorted image.
    mat_index : array_like
        2D array. Index array used for sorting backward.
    """
    if axis == 0:
        mat = np.transpose(mat)
    (nrow, ncol) = mat.shape
    list_index = np.arange(0.0, ncol, 1.0)
    mat_index = np.tile(list_index, (nrow, 1))
    mat_comb = np.asarray(np.dstack((mat_index, mat)))
    mat_comb_sort = np.asarray(
        [row[row[:, 1].argsort()] for row in mat_comb])
    mat_sort = mat_comb_sort[:, :, 1]
    mat_index = mat_comb_sort[:, :, 0]
    if axis == 0:
        mat_sort = np.transpose(mat_sort)
        mat_index = np.transpose(mat_index)
    return mat_sort, mat_index


def sort_backward(mat, mat_index, axis=0):
    """
    Sort grayscales of an image using an index array provided.
    e.g axis=0 is to sort each column.

    Parameters
    ----------
    mat : array_like
        2D array.
    mat_index : array_like
        2D array. Index array used for sorting.
    axis : int
        Axis along which to sort.

    Returns
    --------
    mat_sort : array_like
        2D array. Sorted image.
    """
    if axis == 0:
        mat = np.transpose(mat)
        mat_index = np.transpose(mat_index)
    mat_comb = np.asarray(np.dstack((mat_index, mat)))
    mat_comb_sort = np.asarray(
        [row[row[:, 0].argsort()] for row in mat_comb])
    mat_sort = mat_comb_sort[:, :, 1]
    if axis == 0:
        mat_sort = np.transpose(mat_sort)
    return mat_sort


def separate_frequency_component(mat, axis=0,
                                 window={"name": "gaussian", "sigma": 5}):
    """
    Separate low and high frequency components of an image along an axis.
    e.g axis=0 is to apply the separation to each column.

    Parameters
    ----------
    mat : array_like
        2D array.
    axis : int
        Axis along which to apply the filter.
    window : array_like or dict
        1D array or a dictionary which given the name of a window in
        the scipy.signal.window list and its parameters (without window-length).

    Returns
    -------
    mat_low : array_like
        2D array. Low-frequency image.
    mat_high : array_like
        2D array. High-frequency image.
    """
    if axis == 0:
        mat = np.transpose(mat)
    (nrow, ncol) = mat.shape
    pad = min(150, int(0.1 * ncol))
    if not isinstance(window, dict):
        if len(window) != ncol:
            raise ValueError("Window-length is not the same as the"
                             " axis-length!")
        else:
            window = np.pad(window, (pad, pad), mode='constant',
                            constant_values=1.0)
    else:
        win_name = tuple(window.values())[0]
        para = tuple(window.values())[1:]
        window = getattr(win, win_name)(ncol + 2 * pad, *para)

    list_sign = np.power(-1.0, np.arange(ncol + 2 * pad))
    mat_pad = np.pad(mat, ((0, 0), (pad, pad)), mode='reflect')
    mat_smooth = np.copy(mat)
    for i, line in enumerate(mat_pad):
        mat_smooth[i] = np.real(
            fft.ifft(fft.fft(line * list_sign) * window) *
            list_sign)[pad:ncol + pad]
    mat_sharp = mat - mat_smooth
    if axis == 0:
        mat_smooth = np.transpose(mat_smooth)
        mat_sharp = np.transpose(mat_sharp)
    return mat_smooth, mat_sharp


def generate_fitted_image(mat, order, axis=0, num_chunk=1):
    """
    Apply a polynomial fitting along an axis of an image.
    e.g. axis=0 is to apply the fitting to each column.

    Parameters
    ----------
    mat : array_like
        2D array.
    order : int
        Order of the polynomial used to fit.
    axis : int
        Axis along which to apply the filter.
    num_chunk : int
        Number of chunks of rows or columns to apply the fitting.

    Returns
    -------
    mat_fit : array_like
    """
    (nrow, ncol) = mat.shape
    if axis == 0:
        size = nrow // num_chunk
        length = size if (size % 2 == 1) else size - 1
        mat_fit = savgol_filter(mat, length, order, axis=axis, mode='mirror')
    else:
        size = ncol // num_chunk
        length = size if (size % 2 == 1) else size - 1
        mat_fit = savgol_filter(mat, length, order, axis=axis, mode='mirror')
    return mat_fit


def detect_stripe(list_data, snr):
    """
    Locate stripe positions using Algorithm 4 in Ref. [1]

    Parameters
    ----------
    list_data : array_like
        1D array. Normalized data.
    snr : float
        Ratio (>1.0) used to detect stripe locations. Greater is less sensitive.

    Returns
    -------
    array_like
        1D binary mask.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    npoint = len(list_data)
    list_sort = np.sort(list_data)
<<<<<<< HEAD
    xlist = np.arange(0, npoint, 1.0)
    ndrop = int(0.25 * npoint)
    (slope, intercept) = np.polyfit(xlist[ndrop:-ndrop - 1],
=======
    x_list = np.arange(0, npoint, 1.0)
    ndrop = np.int16(0.25 * npoint)
    (slope, intercept) = np.polyfit(x_list[ndrop:-ndrop - 1],
>>>>>>> 726dc2124838194423b8532fab80ee1c902e144d
                                    list_sort[ndrop:-ndrop - 1], 1)
    y_end = intercept + slope * x_list[-1]
    noise_level = np.abs(y_end - intercept)
    if noise_level == 0.0:
        raise ValueError("The method doesn't work on noise-free data. If you "
                         "apply the method on simulated data, please add"
                         " noise!")
    val1 = np.abs(list_sort[-1] - y_end) / noise_level
    val2 = np.abs(intercept - list_sort[0]) / noise_level
    list_mask = np.zeros(npoint, dtype=np.float32)
    if val1 >= snr:
        upper_thresh = y_end + noise_level * snr * 0.5
        list_mask[list_data > upper_thresh] = 1.0
    if val2 >= snr:
        lower_thresh = intercept - noise_level * snr * 0.5
        list_mask[list_data <= lower_thresh] = 1.0
    return list_mask


def calculate_regularization_coefficient(width, alpha):
    """
    Calculate coefficients used for the regularization-based method.
    Eq. (7) in Ref. [1].

    Parameters
    ----------
    width : int
        Width of a square array.
    alpha : float
        Regularization parameter.

    Returns
    -------
    float
         Square array.

    References
    ----------
    .. [1] https://doi.org/10.1016/j.aml.2010.08.022
    """
    tau = 2.0 * np.arcsinh(np.sqrt(alpha) * 0.5)
    ilist = np.arange(0, width)
    jlist = np.arange(0, width)
    matjj, matii = np.meshgrid(jlist, ilist)
    mat1 = np.abs(matii - matjj)
    mat2 = matii + matjj
    mat1a = np.cosh((width - 1 - mat1) * tau)
    mat2a = np.cosh((width - mat2) * tau)
    matcoe = - (np.tanh(
        0.5 * tau) / (alpha * np.sinh(width * tau))) * (mat1a + mat2a)
    return matcoe


def make_2d_butterworth_window(width, height, u, v, n):
    """
    Create a 2d window from the 1D Butterworth window.

    Parameters
    ----------
    height : int
        Height of the window.
    width : int
        Width of the window.
    u : int
        Cutoff frequency.
    n : int
        Filter order.
    v : int
        Number of rows (= 2*v) around the height middle are the
        1D Butterworth windows.

    Returns
    -------
    array_like
        2D array.
    """
    xcenter = np.ceil(width / 2.0) - 1.0
    ycenter = np.int16(np.ceil(height / 2.0) - 1)
    x_list = np.arange(width) - xcenter
    window = 1.0 / (1.0 + np.power(x_list / u, 2 * n))
    row1 = ycenter - np.int16(v)
    row2 = ycenter + np.int16(v) + 1

    window_2d = np.ones((height, width), dtype=np.float32)
    window_2d[row1:row2] = window
    return window_2d


def make_2d_damping_window(width, height, size, window_name="gaussian"):
    """
    Make 2D damping window from a list of 1D window for a Fourier-space filter,
    i.e. a high-pass filter.

    Parameters
    ----------
    height : int
        Height of the window.
    width : int
        Width of the window.
    size : int
        Sigma of a Gaussian window or cutoff frequency of a Butterworth window.
    window_name : str, optional
        Two options: "gaussian" or "butter".

    Returns
    -------
    array_like
        2D array of the window.
    """
    xcenter = np.ceil(width / 2.0) - 1.0
    x_list = np.arange(width) - xcenter
    if window_name == "butter":
        window = 1.0 - 1.0 / (1.0 + np.power(x_list / size, 2))
    else:
        window = 1.0 - np.exp(-x_list ** 2 / (2 * (size ** 2)))
    return np.tile(window, (height, 1))


def apply_wavelet_decomposition(mat, wavelet_name, level=None):
    """
    Apply 2D wavelet decomposition.

    Parameters
    ----------
    mat : array_like
        2D array.
    wavelet_name : str
        Name of a wavelet. E.g. "db5"
    level : int, optional
        Decomposition level. It is constrained to return an array with
        a minimum size of larger than 16 pixels.

    Returns
    -------
    list
        The first element is an 2D-array, next elements are tuples of three
        2D-arrays. i.e [mat_n, (cH_level_n, cV_level_n, cD_level_n), ...,
        (cH_level_1, cV_level_1, cD_level_1)]
    """
    (nrow, ncol) = mat.shape
    max_level = int(
        min(np.floor(np.log2(nrow / 16.0)), np.floor(np.log2(ncol / 16.0))))
    if (level is None) or (level > max_level) or (level < 1):
        level = max_level
    return pywt.wavedec2(mat, wavelet_name, level=level)


def apply_wavelet_reconstruction(data, wavelet_name, ignore_level=None):
    """
    Apply 2D wavelet reconstruction.

    Parameters
    ----------
    data : list or tuple
        The first element is an 2D-array, next elements are tuples of three
        2D-arrays. i.e [mat_n, (cH_level_n, cV_level_n, cD_level_n), ...,
        (cH_level_1, cV_level_1, cD_level_1)].
    wavelet_name : str
        Name of a wavelet. E.g. "db5"
    ignore_level : int, optional
        Decomposition level to be ignored for reconstruction.

    Returns
    -------
    array_like
        2D array. Note that the sizes of the array are always even numbers.
    """
    if ignore_level is not None:
        level = len(data[1:])
        if level >= ignore_level > 0:
            data[-ignore_level] = tuple(
                [np.zeros_like(v) for v in data[-ignore_level]])
    return pywt.waverec2(data, wavelet_name)


def check_level(level, n_level):
    """
    Supplementary method for the method of "apply_filter_to_wavelet_component".
    To check if the provided level is in the right format.
    """
    if level is None:
        level = list(range(1, n_level + 1))
    else:
        if isinstance(level, int):
            if 0 < level <= n_level:
                level = [level]
            else:
                raise ValueError(
                    "Level is out of range: [1:{}]!".format(n_level))
        elif isinstance(level, list):
            level = [(i if (0 < i <= n_level) else 0) for i in level]
            if 0 in level:
                raise ValueError(
                    "Level is out of range: [1:{}]!".format(n_level))
        elif isinstance(level, tuple):
            level = [(i if (0 < i <= n_level) else 0) for i in level]
            if 0 in level:
                raise ValueError(
                    "Level is out of range: [1:{}]!".format(n_level))
        else:
            raise ValueError("Level-input format is incorrect!")
    return level


def apply_filter_to_wavelet_component(data, level=None, order=1,
                                      method="gaussian_filter",
                                      para=[(1, 11)]):
    """
    Apply a filter to a component of the wavelet decomposition of an image.

    Parameters
    ----------
    data : list or tuple
        The first element is an 2D-array, next elements are tuples of three
        2D-arrays. i.e [mat_n, (cH_level_n, cV_level_n, cD_level_n), ...,
        (cH_level_1, cV_level_1, cD_level_1)].
    level : int, list of int, or None
        Decomposition level to be applied the filter.
    order : {0, 1, 2}
        Specify which component in a tuple, (cH_level_n, cV_level_n, cD_level_n)
        to be filtered.
    method : str
        Name of the filter in the namespace.
    para : list or tuple
        Parameters of the filter.

    Returns
    -------
    list or tuple
        The first element is an 2D-array, next elements are tuples of three
        2D-arrays. i.e [mat_n, (cH_level_n, cV_level_n, cD_level_n), ...,
        (cH_level_1, cV_level_1, cD_level_1)].
    """
    n_level = len(data[1:])
    level = check_level(level, n_level)
    order = np.clip(order, 0, 2)
    data = [list(i_data) for i_data in data]
    if not isinstance(para, list):
        para = tuple(list([para]))
    else:
        para = tuple(para)
    for i in level:
        if method in dir(ndi):
            data[i][order] = getattr(ndi, method)(data[i][order], *para)
        else:
            if method in dir():
                obj = sys.modules[__name__]
                data[i][order] = getattr(obj, method)(data[i][order], *para)
            else:
                raise ValueError("Can't find the method: '{}' in the"
                                 " namespace!".format(method))
    data = [tuple(i_data) for i_data in data]
    data[0] = np.asarray(data[0])
    return data


def interpolate_inside_stripe(mat, list_mask, kind="linear"):
    """
    Interpolate gray-scales inside vertical stripes of an image. Stripe
    locations given by a binary 1D-mask.

    Parameters
    ----------
    mat : array_like
        2D array.
    list_mask : array_like
        1D array. Must equal the width of an image.
    kind : {'linear', 'cubic', 'quintic'}, optional
        The kind of spline interpolation to use. Default is 'linear'.

    Returns
    -------
    array_like
    """
    (nrow, ncol) = mat.shape
    if len(list_mask) != ncol:
        raise ValueError("Length of a binary 1D-mask is not the same as the "
                         "width of an image!")
    mat = np.copy(mat)
    list_mask = np.copy(list_mask)
    list_mask[0:2] = 0.0
    list_mask[-2:] = 0.0
    x_list = np.where(list_mask < 1.0)[0]
    ylist = np.arange(nrow)
    zmat = mat[:, x_list]
    finter = interpolate.interp2d(x_list, ylist, zmat, kind=kind)
    xlist_miss = np.where(list_mask > 0.0)[0]
    if len(xlist_miss) > 0:
        mat[:, xlist_miss] = finter(xlist_miss, ylist)
    return mat


def rectangular_from_polar(width_reg, height_reg, width_pol, height_pol):
    """
    Generate coordinates of a rectangular grid from polar coordinates.

    Parameters
    -----------
    width_reg : int
        Width of an image in the Cartesian coordinate system.
    height_reg : int
        Height of an image in the Cartesian coordinate system.
    width_pol : int
        Width of an image in the polar coordinate system.
    height_pol : int
        Height of an image in the polar coordinate system.

    Returns
    ------
    x_mat : array_like
         2D array. Broadcast of the x-coordinates.
    y_mat : array_like
         2D array. Broadcast of the y-coordinates.
    """
    xcenter = (width_reg - 1.0) * 0.5
    ycenter = (height_reg - 1.0) * 0.5
    r_max = np.floor(max(xcenter, ycenter))
    r_list = np.linspace(0.0, r_max, width_pol)
    theta_list = np.arange(0.0, height_pol, 1.0) * 2 * np.pi / (height_pol - 1)
    r_mat, theta_mat = np.meshgrid(r_list, theta_list)
    x_mat = np.float32(
        np.clip(xcenter + r_mat * np.cos(theta_mat), 0, width_reg - 1))
    y_mat = np.float32(
        np.clip(ycenter + r_mat * np.sin(theta_mat), 0, height_reg - 1))
    return x_mat, y_mat


def polar_from_rectangular(width_pol, height_pol, width_reg, height_reg):
    """
    Generate polar coordinates from grid coordinates.

    Parameters
    -----------
    width_pol : int
        Width of an image in the polar coordinate system.
    height_pol : int
        Height of an image in the polar coordinate system.
    width_reg : int
        Width of an image in the Cartesian coordinate system.
    height_reg : int
        Height of an image in the Cartesian coordinate system.

    Returns
    ------
    r_mat : array_like
         2D array. Broadcast of the r-coordinates.
    theta_mat : array_like
         2D array. Broadcast of the theta-coordinates.
    """
    xcenter = (width_reg - 1.0) * 0.5
    ycenter = (height_reg - 1.0) * 0.5
    r_max = np.floor(max(xcenter, ycenter))
    x_list = (np.flipud(np.arange(width_reg)) - xcenter) * width_pol / r_max
    y_list = (np.flipud(np.arange(height_reg)) - ycenter) * width_pol / r_max
    x_mat, y_mat = np.meshgrid(x_list, y_list)
    r_mat = np.float32(
        np.clip(np.sqrt(x_mat ** 2 + y_mat ** 2), 0, width_pol - 1))
    theta_mat = np.float32(np.clip(
        (np.pi + np.arctan2(y_mat, x_mat)) * (height_pol - 1) / (2 * np.pi), 0,
        height_pol - 1))
    return r_mat, theta_mat


def transform_slice_forward(mat, coord_mat=None):
    """
    Transform a reconstructed image into polar coordinates.

    Parameters
    ----------
    mat : array_like
        Square array. Reconstructed image.
    coord_mat : tuple of array_like, optional
        (Square array of x-coordinates , square array of y-coordinates) or
        generated if None.

    Returns
    -------
    array_like
        Transformed image.
    """
    (nrow, ncol) = mat.shape
    if nrow != ncol:
        raise ValueError("Height and width of the image is not the same!")
    if coord_mat is None:
        (x_mat, y_mat) = rectangular_from_polar(ncol, ncol, ncol, ncol)
    else:
        (x_mat, y_mat) = coord_mat
        if (x_mat.shape != mat.shape) or (y_mat.shape != mat.shape):
            raise ValueError("Shape of the coordinate array is not the same as"
                             " the shape of the image!")
    return mapping(mat, x_mat, y_mat)


def transform_slice_backward(mat, coord_mat=None):
    """
    Transform a reconstructed image in polar coordinates back to rectangular
    coordinates.

    Parameters
    ----------
    mat : array_like
        Square array. Reconstructed image in polar coordinates.
    coord_mat : tuple of array_like, optional
        (Square array of r-coordinates , square array of theta-coordinates) or
        generated if None.

    Returns
    -------
    array_like
        Transformed image.
    """
    (nrow, ncol) = mat.shape
    if nrow != ncol:
        raise ValueError("Height and width of the image is not the same!")
    if coord_mat is None:
        (r_mat, theta_mat) = polar_from_rectangular(ncol, ncol, ncol, ncol)
    else:
        (r_mat, theta_mat) = coord_mat
        if (r_mat.shape != mat.shape) or (theta_mat.shape != mat.shape):
            raise ValueError("Shape of the coordinate array is not the same as"
                             " the shape of the image!")
    return mapping(mat, r_mat, theta_mat)


def make_2d_gaussian_window(height, width, sigmax, sigmay):
    """
    Create a 2D Gaussian window.

    Parameters
    ----------
    height : int
        Height of the image.
    width : int
        Width of the image.
    sigmax : int
        Sigma in the x-direction.
    sigmay : int
        Sigma in the y-direction.

    Returns
    -------
    array_like
        2D array.
    """
    xcenter = (width - 1.0) / 2.0
    ycenter = (height - 1.0) / 2.0
    y, x = np.ogrid[-ycenter:height - ycenter, -xcenter:width - xcenter]
    window = np.exp(-(x ** 2 / (2 * sigmax ** 2) + y ** 2 / (2 * sigmay ** 2)))
    return window


def apply_gaussian_filter(mat, sigmax, sigmay, pad=None, mode=None):
    """
    Filtering an image in the Fourier domain using a 2D Gaussian window.
    Smaller is stronger.

    Parameters
    ----------
    mat : array_like
        2D array.
    sigmax : int
        Sigma in the x-direction.
    sigmay : int
        Sigma in the y-direction.
    pad : int or None
        Padding for the Fourier transform.
    mode : str, list of str, or tuple of str
        Padding method. One of options : 'reflect', 'edge', 'constant'. Full
        list is at:
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Returns
    -------
    array_like
        2D array. Filtered image.
    """
    if mode is None:
        # Default for a sinogram image.
        mode1 = "edge"
        mode2 = "mean"
    else:
        if isinstance(mode, list) or isinstance(mode, tuple):
            mode1 = mode[0]
            mode2 = mode[1]
        else:
            mode1 = mode2 = mode
    if pad is None:
        pad = min(150, int(0.1 * min(mat.shape)))
    mat_pad = np.pad(mat, ((0, 0), (pad, pad)), mode=mode1)
    mat_pad = np.pad(mat_pad, ((pad, pad), (0, 0)), mode=mode2)
    (nrow, ncol) = mat_pad.shape
    window = make_2d_gaussian_window(nrow, ncol, sigmax, sigmay)
    listx = np.arange(0, ncol)
    listy = np.arange(0, nrow)
    x, y = np.meshgrid(listx, listy)
    mat_sign = np.power(-1.0, x + y)
    mat_filt = np.real(
        fft.ifft2(fft.fft2(mat_pad * mat_sign) * window) * mat_sign)
    return mat_filt[pad:nrow - pad, pad:ncol - pad]


def apply_1d_regularizer(list_data, sijmat):
    """
    Supplementary method for the method of "apply_regularization_filter".
    To apply a regularizer to an 1D-array.
    """
    ncol = len(list_data)
    list_grad = np.zeros(ncol, dtype=np.float32)
    list_grad[1:-1] = (-1) * np.diff(list_data, 2)
    list_grad[0] = list_data[0] - list_data[1]
    list_grad[-1] = list_data[-1] - list_data[-2]
    list_coe = np.sum(np.tile(list_grad, (ncol, 1)) * sijmat, axis=1)
    return list_data + list_coe


def apply_regularization_filter(mat, alpha, axis=1, ncore=None):
    """
    Apply a regularization filter using the method in Ref. [1].
    Note that it's computationally costly.

    Parameters
    ----------
    mat : array_like
        2D array
    alpha : float
        Regularization parameter, e.g. 0.001. Smaller is stronger.
    axis : int
        Axis along which to apply the filter.
    ncore: int or None
        Number of cores used for computing. Automatically selected if None.

    Returns
    -------
    array_like
        2D array. Smoothed image.

    References
    ----------
    .. [1] https://doi.org/10.1016/j.aml.2010.08.022
    """
    if ncore is None:
        ncore = np.clip(mp.cpu_count() - 1, 1, None)
    if axis == 0:
        mat = np.transpose(mat)
    (nrow, ncol) = mat.shape
    sijmat = calculate_regularization_coefficient(ncol, alpha)
    mat = np.asarray(Parallel(n_jobs=ncore, backend="threading")(
        delayed(apply_1d_regularizer)(mat[i], sijmat) for i in range(nrow)))
    if axis == 0:
        mat = np.transpose(mat)
    return mat


def transform_1d_window_to_2d(win_1d):
    """
    Transform a 1d-window to 2d-window.
    Useful for designing a Fourier filter.

    Parameters
    ----------
    win_1d : array_like
        1D array.

    Returns
    --------
    win_2d : array_like
        Square array, a 2D version of the 1d-window.
    """
    width0 = len(win_1d)
    if width0 % 2 == 0:
        width = width0 + 1
    else:
        width = width0
    center = width // 2
    x_list = (1.0 * np.flipud(np.arange(width)) - center)
    y_list = (1.0 * np.arange(width) - center)
    x_mat, y_mat = np.meshgrid(x_list, y_list)
    r_mat = np.float32(np.clip(np.sqrt(x_mat ** 2 + y_mat ** 2), 0, center))
    theta_mat = np.arctan2(y_mat, x_mat)
    r_mat[theta_mat < 0] *= -1
    r_mat = np.float32(np.clip(r_mat + center, 0, width - 1))
    theta_mat = np.clip(np.float32(theta_mat * (width - 1.0) / np.pi), 0,
                        width - 1)
    mat = np.tile(win_1d, (width, 1))
    win_2d = mapping(mat, r_mat, theta_mat)
    return win_2d[0:width0, 0:width0]


def detect_sample(sinogram, sino_type="180"):
    """
    To check if there is a sample in a sinogram using the "double-wedge"
    property of the Fourier transform of the sinogram (Ref. [1]).

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image
    sino_type : {"180", "360"}
        Sinogram type : 180-degree or 360-degree.

    Returns
    -------
    bool
        True if there is a sample.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.418448

    """
    check = True
    if not (sino_type == "180" or sino_type == "360"):
        raise ValueError("!!! Use only one of two options: '180' or '360'!!!")
    if sino_type == "180":
        sinogram = 1.0 * np.vstack((sinogram, np.fliplr(sinogram)))
    sino_fft = np.abs(fft.fftshift(fft.fft2(sinogram)))
    (nrow, ncol) = sino_fft.shape
    ycenter = nrow // 2
    xcenter = ncol // 2
    radi = min(20, int(np.ceil(0.05 * min(ycenter, xcenter))))
    sino_fft = sino_fft[:ycenter - radi, :xcenter - radi]
    size = min(30, int(np.ceil(0.02 * min(nrow, ncol))))
    sino_smooth = ndi.gaussian_filter(sino_fft, size)
    (nrow1, _) = sino_smooth.shape
    row = int(0.8 * nrow1)
    list_check = np.zeros(nrow1 - row, dtype=np.float32)
    pos = np.argmax(sino_smooth[row])
    for i in np.arange(row, nrow1):
        pos1 = np.argmax(sino_smooth[i])
        if pos1 > pos:
            list_check[i - row] = 1.0
    ratio = (np.sum(list_check) / len(list_check))
    if ratio < 0.4:
        check = False
    return check


def fix_non_sample_areas(overlap_metadata):
    """
    Used to fix overlap values of grid-cells without sample by copying from
    its neighbours

    Parameters
    ---------
    overlap_metadata : array_like
        A matrix of overlap values of each grid-cell where each element is a
        list of [overlap, side].

    Returns
    -------
    metadata : array_like
    """
    (g_nrow, g_ncol, _) = overlap_metadata.shape
    metadata = np.copy(overlap_metadata)
    for i in np.arange(g_nrow):
        i1 = i - 1
        i2 = i + 1
        for j in np.arange(g_ncol):
            (area, _) = overlap_metadata[i, j]
            j1 = j - 1
            j2 = j + 1
            if area == 0:
                if 0 <= i1 < g_nrow:
                    (area1, side1) = overlap_metadata[i1, j]
                    if area1 != 0:
                        metadata[i, j] = np.asarray([area1, side1])
                        continue
                if 0 <= i2 < g_nrow:
                    (area1, side1) = overlap_metadata[i2, j]
                    if area1 != 0:
                        metadata[i, j] = np.asarray([area1, side1])
                        continue
                if 0 <= j1 < g_ncol:
                    (area1, side1) = overlap_metadata[i, j1]
                    if area1 != 0:
                        metadata[i, j] = np.asarray([area1, side1])
                        continue
                if 0 <= j2 < g_ncol:
                    (area1, side1) = overlap_metadata[i, j2]
                    if area1 != 0:
                        metadata[i, j] = np.asarray([area1, side1])
                        continue
    # Run the same above routine but in reverse order.
    for i in np.arange(g_nrow - 1, -1, -1):
        i1 = i - 1
        i2 = i + 1
        for j in np.arange(g_ncol - 1, -1, -1):
            (area, _) = metadata[i, j]
            j1 = j - 1
            j2 = j + 1
            if area == 0:
                if 0 <= i1 < g_nrow:
                    (area1, side1) = metadata[i1, j]
                    if area1 != 0:
                        metadata[i, j] = np.asarray([area1, side1])
                        continue
                if 0 <= i2 < g_nrow:
                    (area1, side1) = metadata[i2, j]
                    if area1 != 0:
                        metadata[i, j] = np.asarray([area1, side1])
                        continue
                if 0 <= j1 < g_ncol:
                    (area1, side1) = metadata[i, j1]
                    if area1 != 0:
                        metadata[i, j] = np.asarray([area1, side1])
                        continue
                if 0 <= j2 < g_ncol:
                    (area1, side1) = metadata[i, j2]
                    if area1 != 0:
                        metadata[i, j] = np.asarray([area1, side1])
                        continue
        return metadata


def locate_slice(slice_idx, height, overlap_metadata):
    """
    Locate slice indices in grid-rows given a slice index of the reconstruction
    data as a whole.

    Parameters
    ----------
    slice_idx : int
        Slice index of full reconstruction data.
    height : int
        Height of a projection image of each grid-cell.
    overlap_metadata : array_like
        A matrix of overlap values of each grid-row where each element is a
        list of [overlap, side]. Used to stitch the grid-data along the
        row-direction.

    Returns
    -------
    list of int and float
        If the slice is not in the overlapping area between two grid-rows, the
        result is a list of [grid_row_index, slice_index, weight_factor]. If the
        slice is in the overlapping area between two grid-rows, the result is a
        list of [[grid_row_index_0, slice_index_0, weight_factor_0],
        [grid_row_index_1, slice_index_1, weight_factor_1]]
    """
    g_nrow = overlap_metadata.shape[0] + 1
    side = overlap_metadata[0, 0, 1]
    overlap_list = overlap_metadata[:, 0, 0]
    if side == 1:
        list_slices = [(np.arange(i * height, i * height + height) -
                        np.sum(overlap_list[0: i])) for i in range(g_nrow)]
    else:
        list_slices = [
            (np.arange(i * height + height - 1, i * height - 1, -1) -
             np.sum(overlap_list[0: i])) for i in range(g_nrow)]
    list_slices = np.asarray(list_slices)
    results = []
    for i, list1 in enumerate(list_slices):
        pos = np.squeeze(np.where(list1 == slice_idx)[0])
        if pos.size == 1:
            results.append([i, pos, 1.0])
    if len(results) == 2:
        if side == 1:
            results[0][2] = (1.0 * list_slices[results[0][0]][
                -1] - slice_idx) / (overlap_list[results[0][0]] - 1.0)
            results[1][2] = (slice_idx - 1.0 * list_slices[results[1][0]][
                0]) / (overlap_list[results[0][0]] - 1.0)
        else:
            results[0][2] = (- slice_idx + 1.0 * list_slices[results[0][0]][
                0]) / (overlap_list[results[0][0]] - 1.0)
            results[1][2] = (-1.0 * list_slices[results[1][0]][
                -1] + slice_idx) / (overlap_list[results[0][0]] - 1.0)
    return results


def locate_slice_chunk(slice_start, slice_stop, height, overlap_metadata):
    """
    Locate slice indices in grid-rows given slice indices of the reconstruction
    data as a whole.

    Parameters
    ----------
    slice_start : int
        Starting index of full reconstruction data.
    slice_stop : int
        Stopping index of full reconstruction data.
    height : int
        Height of a projection image of each grid-cell.
    overlap_metadata : array_like
        A matrix of overlap values of each grid-row where each element is a
        list of [overlap, side]. Used to stitch the grid-data along the
        row-direction.

    Returns
    -------
    list of list of int and float
        List of results for each slice index. If a slice is not in the
        overlapping area between two grid-rows, the result is a list of
        [grid_row_index, slice_index, weight_factor]. If a slice is in the
        overlapping area between two grid-rows, the result is a list of
        [[grid_row_index_0, slice_index_0, weight_factor_0],
        [grid_row_index_1, slice_index_1, weight_factor_1]].
    """
    if slice_stop < slice_start:
        raise ValueError(
            "Stopping index must be larger than the starting index!!!")
    g_nrow = overlap_metadata.shape[0] + 1
    side = overlap_metadata[0, 0, 1]
    overlap_list = overlap_metadata[:, 0, 0]
    if side == 1:
        list_slices = [(np.arange(i * height, i * height + height) -
                        np.sum(overlap_list[0: i])) for i in range(g_nrow)]
    else:
        list_slices = [
            (np.arange(i * height + height - 1, i * height - 1, -1) -
             np.sum(overlap_list[0: i])) for i in range(g_nrow)]
    list_slices = np.asarray(list_slices)
    results = []
    for i, list1 in enumerate(list_slices):
        result1 = []
        if side == 1:
            for slice_idx in range(slice_start, slice_stop):
                pos = np.squeeze(np.where(list1 == slice_idx)[0])
                if pos.size == 1:
                    fact = 1.0
                    if i == 0:
                        ver_overlap = overlap_list[i]
                        dis1 = len(list1) - pos - 1
                        if dis1 < ver_overlap:
                            fact = dis1 / (ver_overlap - 1)
                    elif i == (g_nrow - 1):
                        ver_overlap = overlap_list[i - 1]
                        if pos < ver_overlap:
                            fact = pos / (ver_overlap - 1)
                    else:
                        ver_overlap1 = overlap_list[i]
                        dis1 = len(list1) - pos - 1
                        if dis1 < ver_overlap1:
                            fact = dis1 / (ver_overlap1 - 1)
                        if pos < ver_overlap1:
                            fact = pos / (ver_overlap1 - 1)
                        ver_overlap2 = overlap_list[i - 1]
                        dis1 = len(list1) - pos - 1
                        if dis1 < ver_overlap2:
                            fact = dis1 / (ver_overlap2 - 1)
                        if pos < ver_overlap2:
                            fact = pos / (ver_overlap2 - 1)
                    result1.append([i, pos, fact])
        else:
            for slice_idx in range(slice_start, slice_stop):
                pos = np.squeeze(np.where(list1 == slice_idx)[0])
                if pos.size == 1:
                    fact = 1.0
                    if i == 0:
                        ver_overlap = overlap_list[i]
                        if pos < ver_overlap:
                            fact = 1.0 * pos / (ver_overlap - 1)
                    elif i == (g_nrow - 1):
                        ver_overlap = overlap_list[i - 1]
                        dis1 = len(list1) - pos - 1
                        if dis1 < ver_overlap:
                            fact = 1.0 * dis1 / (ver_overlap - 1)
                    else:
                        ver_overlap1 = overlap_list[i]
                        dis1 = len(list1) - pos - 1
                        if dis1 < ver_overlap1:
                            fact = 1.0 * dis1 / (ver_overlap1 - 1)
                        if pos < ver_overlap1:
                            fact = 1.0 * pos / (ver_overlap1 - 1)
                        ver_overlap2 = overlap_list[i - 1]
                        dis1 = len(list1) - pos - 1
                        if dis1 < ver_overlap2:
                            fact = 1.0 * dis1 / (ver_overlap2 - 1)
                        if pos < ver_overlap2:
                            fact = 1.0 * pos / (ver_overlap2 - 1)
                    result1.append([i, pos, fact])
        if len(result1) > 0:
            results.append(result1)
    return results
