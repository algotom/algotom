# ============================================================================
# ============================================================================
# Copyright (c) 2022 Nghia T. Vo. All rights reserved.
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
# Description: Python module of correlation-related methods.
# Publication date: 20th June 2022
# Contributors:
# ============================================================================

"""
Module of correlation-based methods for finding shifts between images or
stacks of images. The methods are designed to be flexible to:

    -   Run on multicore CPU or GPU.
    -   Use small/large RAM or small/large GPU memory.
    -   Work with small/large size of data.
    -   Find shifts locally or globally.
"""

import math
import warnings
import multiprocessing as mp
import numpy as np
from numba import jit, cuda
from joblib import Parallel, delayed
from scipy.signal import correlate
from algotom.rec.reconstruction import make_smoothing_window


def normalize_image(mat):
    """
    Normalize an image.

    Parameters
    ----------
    mat : array_like
        2D or 3D array.

    Returns
    -------
    array_like
        2D or 3D array. Normalized image.
    """
    if len(mat.shape) == 3:
        mat1 = (mat - np.mean(mat)) / np.std(mat)
        mat1 = mat1 - np.min(mat1)
    else:
        rlist = np.sum(mat, axis=0)
        clist = np.transpose([np.sum(mat, axis=1)])
        mat1 = (mat / rlist) / clist
        mat1 = (mat1 - np.mean(mat1)) / np.std(mat1)
    return mat1


@jit(nopython=True, parallel=False, cache=True)
def _generate_correlation_map_2d_input_cpu(ref_mat, mat):  # pragma: no cover
    """
    CPU-function to generate a correlation map (Pearson coefficients) by
    shifting the second image around the first image.

    Parameters
    ----------
    ref_mat : array_like
        2D array. Reference image (size of height0 x width0).
    mat : array_like
        2D array. Image to be translated (size of height1 x width1). Its
        size should be smaller than the reference image.

    Returns
    -------
    array_like
        2D array with the size of (height0-height1+1) x (width0-width1+1)
    """
    (height0, width0) = ref_mat.shape
    (height1, width1) = mat.shape
    height2, width2 = height0 - height1 + 1, width0 - width1 + 1
    mat1 = mat - np.mean(mat)
    num = np.sum(mat1 * mat1)
    coef_mat = np.zeros((height2, width2), dtype=np.float32)
    for i in range(height2):
        for j in range(width2):
            row0, row1 = i, i + height1
            col0, col1 = j, j + width1
            ref_mat1 = ref_mat[row0:row1, col0:col1]
            ref_mat1 = ref_mat1 - np.mean(ref_mat1)
            ref_num = np.sum(ref_mat1 * ref_mat1)
            num1 = math.sqrt(num * ref_num)
            if num1 != 0.0:
                coef_mat[i, j] = np.sum(mat1 * ref_mat1) / num1
    return coef_mat


@jit(nopython=True, parallel=False, cache=True)
def _generate_correlation_map_3d_input_cpu(ref_mat, mat):  # pragma: no cover
    """
    CPU-function to generate a correlation map (Pearson coefficients) by
    shifting the second 3d-image around the first 3d-image along the axes of
    1 and 2.

    Parameters
    ----------
    ref_mat : array_like
        3D array. Reference image (size of depth0 x height0 x width0).
    mat : array_like
        3D array. Image to be translated (size of depth0 x height1 x width1).
        Its size (height1, width1) must be smaller than the reference image.

    Returns
    -------
    array_like
        2D array with the size of (height0-height1+1) x (width0-width1+1)
    """
    (height0, width0) = ref_mat.shape[-2:]
    (height1, width1) = mat.shape[-2:]
    height2, width2 = height0 - height1 + 1, width0 - width1 + 1
    coef_mat = np.zeros((height2, width2), dtype=np.float32)
    mat1 = mat - np.mean(mat)
    num = np.sum(mat1 * mat1)
    for i in range(height2):
        for j in range(width2):
            row0, row1 = i, i + height1
            col0, col1 = j, j + width1
            ref_mat1 = ref_mat[:, row0:row1, col0:col1]
            ref_mat1 = ref_mat1 - np.mean(ref_mat1)
            ref_num = np.sum(ref_mat1 * ref_mat1)
            num1 = math.sqrt(num * ref_num)
            if num1 != 0.0:
                coef_mat[i, j] = np.sum(mat1 * ref_mat1) / num1
    return coef_mat


@cuda.jit(device=True)
def __mean_2d(mat):  # pragma: no cover
    """
    GPU-kernel function to calculate mean of a 2d-array.
    """
    (height, width) = mat.shape
    num = 0.0
    for i in range(height):
        for j in range(width):
            num += mat[i, j]
    return num / (height * width)


@cuda.jit(device=True)
def __mean_3d(mat):  # pragma: no cover
    """
    GPU-kernel function to calculate mean of a 3d-array.
    """
    (depth, height, width) = mat.shape
    num = 0.0
    for i in range(depth):
        for j in range(height):
            for k in range(width):
                num += mat[i, j, k]
    return num / (depth * height * width)


@cuda.jit(device=True)
def __sum_square_2d(mat, mean):  # pragma: no cover
    """
    GPU-kernel function to calculate the sum of squares of a normed 2d-array.

    Parameters
    ----------
    mat : array_like
        2D array.
    mean : float
        Mean of the array.

    Returns
    -------
    float
    """
    (height, width) = mat.shape
    sum_sqr = 0.0
    for i in range(height):
        for j in range(width):
            val = mat[i, j] - mean
            sum_sqr += val * val
    return sum_sqr


@cuda.jit(device=True)
def __sum_square_3d(mat, mean):  # pragma: no cover
    """
    GPU-kernel function to calculate the sum of squares of a normed 3d-array.

    Parameters
    ----------
    mat : array_like
        3D array.
    mean : float
        Mean of the array.

    Returns
    -------
    float
    """
    (depth, height, width) = mat.shape
    sum_sqr = 0.0
    for i in range(depth):
        for j in range(height):
            for k in range(width):
                val = mat[i, j, k] - mean
                sum_sqr += val * val
    return sum_sqr


@cuda.jit(device=True)
def __sum_multiply_2d(ref_mat, mat, ref_mean, mat_mean):  # pragma: no cover
    """
    GPU-kernel function to calculate the sum of multiplies of two normed
    2d-arrays.

    Parameters
    ----------
    ref_mat : array_like
        2D array. The first image.
    mat : array_like
        2D array. The second image.
    ref_mean : float
        Mean of the first image.
    mat_mean : float
        Mean of the second image.

    Returns
    -------
    float
    """
    (height, width) = ref_mat.shape
    sum_mul = 0.0
    for i in range(height):
        for j in range(width):
            val = (ref_mat[i, j] - ref_mean) * (mat[i, j] - mat_mean)
            sum_mul += val
    return sum_mul


@cuda.jit(device=True)
def __sum_multiply_3d(ref_mat, mat, ref_mean, mat_mean):  # pragma: no cover
    """
    GPU-kernel function to calculate the sum of multiplies of two normed
    2d-arrays.

    Parameters
    ----------
    ref_mat : array_like
        2D array. The first image.
    mat : array_like
        2D array. The second image.
    ref_mean : float
        Mean of the first image.
    mat_mean : float
        Mean of the second image.

    Returns
    -------
    float
    """
    (depth, height, width) = ref_mat.shape
    sum_mul = 0.0
    for i in range(depth):
        for j in range(height):
            for k in range(width):
                val = (ref_mat[i, j, k] - ref_mean) * (mat[i, j, k] - mat_mean)
                sum_mul += val
    return sum_mul


@cuda.jit(device=True)
def __pearson_coefficient_2d(ref_mat, mat):  # pragma: no cover
    """
    Supplementary method. GPU-kernel function to calculate the Pearson
    correlation coefficient between two images.

    Parameters
    ----------
    ref_mat : array_like
        2D array. The first image.
    mat : array_like
        2D array. The second image.

    Returns
    -------
    float
        Correlation coefficient.
    """
    mat_mean = __mean_2d(mat)
    mat_sqr = __sum_square_2d(mat, mat_mean)
    ref_mean = __mean_2d(ref_mat)
    ref_sqr = __sum_square_2d(ref_mat, ref_mean)
    sum_mul = __sum_multiply_2d(ref_mat, mat, ref_mean, mat_mean)
    num = math.sqrt(ref_sqr * mat_sqr)
    coef_mat = 0.0
    if num != 0.0:
        coef_mat = sum_mul / num
    return coef_mat


@cuda.jit(device=True)
def __pearson_coefficient_3d(ref_mat, mat):  # pragma: no cover
    """
    Supplementary method. GPU-kernel function to calculate the Pearson
    correlation coefficient between two 3d-images.

    Parameters
    ----------
    ref_mat : array_like
        3D array. The first 3d-image.
    mat : array_like
        3D array. The second 3d-image.

    Returns
    -------
    float
        Correlation coefficient.
    """
    mat_mean = __mean_3d(mat)
    mat_sqr = __sum_square_3d(mat, mat_mean)
    ref_mean = __mean_3d(ref_mat)
    ref_sqr = __sum_square_3d(ref_mat, ref_mean)
    sum_mul = __sum_multiply_3d(ref_mat, mat, ref_mean, mat_mean)
    num = math.sqrt(ref_sqr * mat_sqr)
    coef_mat = 0.0
    if num != 0.0:
        coef_mat = sum_mul / num
    return coef_mat


@cuda.jit
def _generate_correlation_map_2d_input_gpu(coef_mat, ref_mat, mat, sum_sqr,
                                           height1, width1, height2,
                                           width2):  # pragma: no cover
    """
    GPU-CPU function to generate correlation map between two 2d-images.

    Parameters
    ----------
    coef_mat : array_like
        2D array. Coefficient map which is initialized and passed from CPU.
        In GPU global-memory.
    ref_mat : array_like
        2D array. Reference image, passed from CPU.
    mat : array_like
        2D array. The second image, passed from CPU.
    sum_sqr : float
        Sum of squares of values of the second image.
    height1 : int
        Height of the second image.
    width1 : int
        Width of the second image.
    height2 : int
        Height of the coefficient map.
    width2 : int
        Width of the coefficient map.

    Returns
    -------

        Update of the coefficient map passed from CPU.
    """
    (x_index, y_index) = cuda.grid(2)
    if (y_index < height2) and (x_index < width2):
        ref_mean = __mean_2d(
            ref_mat[y_index:y_index + height1, x_index:x_index + width1])
        num_sqr = 0.0
        num_mul = 0.0
        for i in range(y_index, y_index + height1):
            for j in range(x_index, x_index + width1):
                i1 = i - y_index
                j1 = j - x_index
                val = ref_mat[i, j] - ref_mean
                num_sqr += val * val
                num_mul += mat[i1, j1] * val
        num = math.sqrt(num_sqr * sum_sqr)
        if num != 0.0:
            coef_mat[y_index, x_index] = num_mul / num


@cuda.jit
def _generate_correlation_map_3d_input_gpu(coef_mat, ref_mat, mat, sum_sqr,
                                           depth, height1, width1, height2,
                                           width2):  # pragma: no cover
    """
    GPU-CPU function to generate correlation map between two 3d-images.

    Parameters
    ----------
    coef_mat : array_like
        3D array. Coefficient map which is initialized and passed from CPU.
        In GPU global-memory.
    ref_mat : array_like
        3D array. Reference image, passed from CPU.
    mat : array_like
        3D array. The second image, passed from CPU.
    sum_sqr : float
        Sum of squares of values of the second image.
    depth : int
        Number of images, or the size of the 3d-arrays along axis 0.
    height1 : int
        Height of the second image.
    width1 : int
        Width of the second image.
    height2 : int
        Height of the coefficient map.
    width2 : int
        Width of the coefficient map.

    Returns
    -------

        Update of the coefficient map passed from CPU.
    """
    (x_index, y_index) = cuda.grid(2)
    if (y_index < height2) and (x_index < width2):
        ref_mean = __mean_3d(
            ref_mat[:, y_index:y_index + height1, x_index:x_index + width1])
        num_sqr = 0.0
        num_mul = 0.0
        for i in range(depth):
            for j in range(y_index, y_index + height1):
                for k in range(x_index, x_index + width1):
                    j1 = j - y_index
                    k1 = k - x_index
                    val = ref_mat[i, j, k] - ref_mean
                    num_sqr += val * val
                    num_mul += mat[i, j1, k1] * val
        num = math.sqrt(num_sqr * sum_sqr)
        if num != 0.0:
            coef_mat[y_index, x_index] = num_mul / num


def generate_correlation_map(ref_mat, mat, gpu=False, block=(16, 16)):
    """
    Generate the correlation map (Pearson coefficients) between two images by
    shifting the second image over the reference image.

    Parameters
    ----------
    ref_mat : array_like
        2D or 3D array. The reference image (e.g. with height0 x width0).
    mat : array_like
        2D or 3D array. The second image (e.g. with height1 x width1). If 3D,
        the size of the first dimension (i.e. depth) must be the same as the
        reference image.
    gpu : bool, optional
        Use GPU for computing if True.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (4,4), (8, 8), ...

    Returns
    -------
    array_like
        2D array with the size of (height0-height1+1) x (width0-width1+1).
    """
    if gpu is True:
        if cuda.is_available() is False:
            warnings.warn("!!!No Nvidia GPU found!!!Run with CPU instead!!!")
            gpu = False
    dim = len(ref_mat.shape)
    if dim != 2 and dim != 3:
        raise ValueError("Inputs must be 2d or 3d arrays !!!")
    (height0, width0) = ref_mat.shape[-2:]
    (height1, width1) = mat.shape[-2:]
    height2, width2 = height0 - height1 + 1, width0 - width1 + 1
    if (height1 > height0) or (width1 > width0):
        raise ValueError("Size of the image must be equal or smaller than the "
                         "reference image !!!")
    if dim == 3:
        if ref_mat.shape[0] != mat.shape[0]:
            raise ValueError("Size of axis-0 must be the same !!!")
    if gpu is True:
        coef_mat = np.zeros((height2, width2), dtype=np.float32)
        if not isinstance(block, tuple):
            raise ValueError("Block must be a tuple of two integers !!!")
        block = block[-2:]
        size = max(height2, width2)
        grid = (int(np.ceil(1.0 * size / block[0])),
                int(np.ceil(1.0 * size / block[1])))
        if dim == 2:
            mat1 = mat - np.mean(mat)
            sum_sqr = np.sum(mat1 * mat1)
            ref_mat1 = np.float32(np.ascontiguousarray(ref_mat))
            mat1 = np.float32(np.ascontiguousarray(mat1))
            f_alias = _generate_correlation_map_2d_input_gpu
            f_alias[grid, block](coef_mat, ref_mat1, mat1, np.float32(sum_sqr),
                                 np.int32(height1), np.int32(width1),
                                 np.int32(height2), np.int32(width2))
        else:
            mat1 = mat - np.mean(mat)
            sum_sqr = np.sum(mat1 * mat1)
            ref_mat1 = np.float32(np.ascontiguousarray(ref_mat))
            mat1 = np.float32(np.ascontiguousarray(mat1))
            f_alias = _generate_correlation_map_3d_input_gpu
            f_alias[grid, block](coef_mat, ref_mat1, mat1, np.float32(sum_sqr),
                                 np.int32(ref_mat1.shape[0]),
                                 np.int32(height1),
                                 np.int32(width1), np.int32(height2),
                                 np.int32(width2))
    else:
        if dim == 2:
            coef_mat = _generate_correlation_map_2d_input_cpu(ref_mat, mat)
        else:
            coef_mat = _generate_correlation_map_3d_input_cpu(ref_mat, mat)
    return coef_mat


def _locate_1d_peak_subpixel(list_data, method="diff"):
    """
    Locate the position of the maximum value of a 1d-array with sub-pixel
    accuracy.

    Parameters
    ----------
    list_data : array_like
        1D array.
    method : {"diff", "poly_fit"}
        Method for finding sub-pixel shift. Two options: a differential
        method or a polynomial method.

    Returns
    -------
    float
        Sub-pixel position of the maximum value.
    """
    size = len(list_data)
    pos = np.argmax(list_data)
    pos_b, pos_f, size1 = pos - 1, pos + 1, size - 1
    if method == "poly_fit":
        if size > 9:
            warnings.warn("Large array can cause numerical error!!!")
        afact, bfact = np.polyfit(np.arange(size), list_data, 2)[:2]
        if afact != 0.0:
            sub_pos = - bfact / (2 * afact)
            if size1 > sub_pos > 0:
                pos = sub_pos
    else:
        if size1 > pos > 0:
            d1fx = (list_data[pos_f] - list_data[pos_b]) / 2.0
            d2fx = (list_data[pos_f] + list_data[pos_b] - 2 * list_data[pos])
            if d2fx != 0.0:
                x_off = - d1fx / d2fx
                pos = pos + x_off
    return pos


def _locate_2d_peak_subpixel(mat, method="diff"):
    """
    Locate the position of the maximum value of a 2d-array with sub-pixel
    accuracy.

    Parameters
    ----------
    mat : array_like
        2D array.
    method : {"diff", "poly_fit"}
        Method for finding sub-pixel shift. Two options: a differential
        method (Ref. [1]) or a polynomial method (Ref. [2]).

    Returns
    -------
    list of two floats
        Sub-pixel position (x, y), i.e. (column, row), of the maximum value.

    References
    ----------
    [1] : https://doi.org/10.48550/arXiv.0712.4289

    [2] : https://doi.org/10.1088/0957-0233/17/6/045
    """
    (height, width) = mat.shape
    (y0, x0) = np.unravel_index(np.argmax(mat, axis=None), mat.shape)
    y_pos, x_pos = 1.0 * y0, 1.0 * x0
    height1, width1 = height - 1, width - 1
    y_check = False if (y0 == 0 or y0 == height1) else True
    x_check = False if (x0 == 0 or x0 == width1) else True
    if (method == "poly_fit") and (y_check or x_check):
        size = min(height, width)
        if size > 9:
            warnings.warn("Large array can cause numerical error!!!")
        xlist, ylist = np.arange(0.0, width), np.arange(0.0, height)
        x_mat, y_mat = np.meshgrid(xlist, ylist)
        x, y = np.ndarray.flatten(x_mat), np.ndarray.flatten(y_mat)
        Amatrix = []
        for i in range(len(x)):
            Amatrix.append(
                [1.0, x[i], y[i], x[i] * x[i], x[i] * y[i], y[i] * y[i]])
        Amatrix = np.asarray(Amatrix)
        Bmatrix = np.transpose(np.ndarray.flatten(mat))
        coefs = np.linalg.lstsq(Amatrix, Bmatrix, rcond=1e-64)[0]
        a0, a1, a2, a3, a4, a5 = coefs
        num = a4 * a4 - 4 * a3 * a5
        if num != 0:
            x_pos1 = (2 * a1 * a5 - a2 * a4) / num
            y_pos1 = (2 * a2 * a3 - a1 * a4) / num
            if height1 > y_pos1 > 0:
                y_pos = y_pos1
            if width1 > x_pos1 > 0:
                x_pos = x_pos1
    else:
        d1fx, d2fx, d1fy, d2fy, d1fxy = 0.0, 0.0, 0.0, 0.0, 0.0
        if y_check and not x_check:
            d1fy = (mat[y0 + 1, x0] - mat[y0 - 1, x0]) / 2.0
            d2fy = (mat[y0 + 1, x0] + mat[y0 - 1, x0] - 2 * mat[y0, x0])
        if not y_check and x_check:
            d1fx = (mat[y0, x0 + 1] - mat[y0, x0 - 1]) / 2.0
            d2fx = (mat[y0, x0 + 1] + mat[y0, x0 - 1] - 2 * mat[y0, x0])
        if y_check and x_check:
            d1fx = 0.5 * (mat[y0, x0 + 1] - mat[y0, x0 - 1])
            d1fy = 0.5 * (mat[y0 + 1, x0] - mat[y0 - 1, x0])
            d2fx = (mat[y0, x0 + 1] + mat[y0, x0 - 1] - 2 * mat[y0, x0])
            d2fy = (mat[y0 + 1, x0] + mat[y0 - 1, x0] - 2 * mat[y0, x0])
            d1fxy = 0.25 * (mat[y0 + 1, x0 + 1] + mat[y0 - 1, x0 - 1] - mat[
                y0 + 1, x0 - 1] - mat[y0 - 1, x0 + 1])
        num = d1fxy ** 2 - d2fx * d2fy
        if num != 0:
            x_off = (d2fy * d1fx - d1fxy * d1fy) / num
            y_off = (d2fx * d1fy - d1fxy * d1fx) / num
            if 1.0 > abs(x_off) > 0.0:
                x_pos = x0 + x_off
            if 1.0 > abs(y_off) > 0.0:
                y_pos = y0 + y_off
    return x_pos, y_pos


def locate_peak(mat, sub_pixel=True, method="diff", dim=2, size=3,
                max_peak=True):
    """
    Locate the position of the maximum value of a 2d-array with sub-pixel
    accuracy.

    Parameters
    ----------
    mat : array_like
        2D array.
    sub_pixel : bool, optional
        Enable sub-pixel location.
    method : {"diff", "poly_fit"}
        Method for finding sub-pixel shift. Two options: a differential
        method (Ref. [1]) or a polynomial method (Ref. [2]).
    dim : {1, 2}
        Searching dimension for sub-pixel location.
    size : int
        Window size around the integer location of the maximum value used for
        sub-pixel searching.
    max_peak : bool, optional
        Used to locate the minimum value if False.

    Returns
    -------
    list of two floats
        Sub-pixel position (x, y), i.e. (column, row), of the maximum value.

    References
    ----------
    [1] : https://doi.org/10.48550/arXiv.0712.4289

    [2] : https://doi.org/10.1088/0957-0233/17/6/045
    """
    (height, width) = mat.shape
    size = 3 if method == "diff" else 2 * (size // 2) + 1
    size = np.clip(size, 3, 9)
    if not max_peak:
        mat = np.max(mat) - mat
    (y_pos, x_pos) = np.unravel_index(np.argmax(mat, axis=None), mat.shape)
    if sub_pixel is True:
        height1, width1 = height - 1, width - 1
        radius = size // 2
        if ((height1 > y_pos > 0) and (width1 > x_pos > 0)
                and (width >= size) and (height >= size)):
            row = np.clip(y_pos - radius, 0, height - size)
            col = np.clip(x_pos - radius, 0, width - size)
            if dim == 1:
                x_vals = mat[y_pos, col:col + size]
                y_vals = mat[row:row + size, x_pos]
                x_pos1 = _locate_1d_peak_subpixel(x_vals, method=method)
                y_pos1 = _locate_1d_peak_subpixel(y_vals, method=method)
                x_pos = col + x_pos1
                y_pos = row + y_pos1
            else:
                sub_mat = mat[row:row + size, col:col + size]
                (x_pos1, y_pos1) = _locate_2d_peak_subpixel(sub_mat,
                                                            method=method)
                x_pos = col + x_pos1
                y_pos = row + y_pos1
        if ((y_pos == 0 or y_pos == height1)
                and (width1 > x_pos > 0) and (width >= size)):
            col = np.clip(x_pos - radius, 0, width - size)
            x_vals = mat[y_pos, col:col + size]
            x_pos1 = _locate_1d_peak_subpixel(x_vals, method=method)
            x_pos = col + x_pos1
        if ((x_pos == 0 or x_pos == width1)
                and (height1 > y_pos > 0) and (height >= size)):
            row = np.clip(y_pos - radius, 0, height - size)
            y_vals = mat[row:row + size, x_pos]
            y_pos1 = _locate_1d_peak_subpixel(y_vals, method=method)
            y_pos = row + y_pos1
    return x_pos, y_pos


def find_shift_based_correlation_map(ref_mat, mat, margin=10, axis=None,
                                     sub_pixel=True, method="diff", dim=2,
                                     size=3, gpu=False, block=(16, 16)):
    """
    Find the relative translations of the second image against the first image
    using the correlation map generated by sliding the 2nd image over the 1st
    one. If the inputs are 3d-arrays, the size of the first axis must be the
    same.

    Parameters
    ----------
    ref_mat : array_like
        2D or 3D array. Reference image.
    mat : array_like
        2D or 3D array. The second image. If 3D, the size of the first
        dimension (i.e. depth) must be the same as the reference image.
    margin : int, optional
        If the second image and the first image are the same size, the second
        image will be cropped with the margin amount from the edges before
        sliding. Basically, this value defines the sliding range.
    axis : {0, 1, None}
        To select the axis for sliding. If the inputs are 3d-arrays, 0 and 1
        corresponding to axis-1 and axis-2 of a 3d-array.
    sub_pixel : bool, optional
        Enable sub-pixel location.
    method : {"diff", "poly_fit"}
        Method for finding sub-pixel shift. Two options: a differential
        method (Ref. [1]) or a polynomial method (Ref. [2]).
    dim : {1, 2}
        Searching dimension for sub-pixel location.
    size : int
        Window size around the integer location of the maximum value used for
        sub-pixel searching.
    gpu : bool, optional
        Use GPU for computing if True.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (4,4), (8, 8), ...

    Returns
    -------
    list of 2 floats
        The shifts in x and y-direction of the second image referred to the
        middle of the reference image.

    References
    ----------
    [1] : https://doi.org/10.48550/arXiv.0712.4289

    [2] : https://doi.org/10.1088/0957-0233/17/6/045
    """
    (height0, width0) = ref_mat.shape[-2:]
    (height1, width1) = mat.shape[-2:]
    check_3d = True if len(mat.shape) == 3 else False
    check1, check2 = True, True
    if (height0 == height1) and (width0 == width1):
        if axis == 0:
            if (height1 - 2 * margin) > 3:
                if check_3d:
                    mat = mat[:, margin:-margin, :]
                    height1 = mat.shape[1]
                else:
                    mat = mat[margin:-margin, :]
                    height1 = mat.shape[0]
            else:
                check1 = False
        elif axis == 1:
            if (width1 - 2 * margin) > 3:
                if check_3d:
                    mat = mat[:, :, margin:-margin]
                    width1 = mat.shape[-1]
                else:
                    mat = mat[:, margin:-margin]
                    width1 = mat.shape[-1]
            else:
                check1 = False
        else:
            if (height1 - 2 * margin) > 3 and (width1 - 2 * margin) > 3:
                if check_3d:
                    mat = mat[:, margin:-margin, margin:-margin]
                    (height1, width1) = mat.shape[-2:]
                else:
                    mat = mat[margin:-margin, margin:-margin]
                    (height1, width1) = mat.shape
            else:
                check1 = False
    if check1 is True:
        if axis == 0:
            check2 = False if (height0 - 2 * margin < height1) else True
        elif axis == 1:
            check2 = False if (width0 - 2 * margin < width1) else True
        else:
            check2 = False if ((height0 - 2 * margin < height1) and (
                    width0 - 2 * margin < width1)) else True
    if check2:
        match_img = generate_correlation_map(ref_mat, mat, gpu, block)
        (height2, width2) = match_img.shape
        y_mid, x_mid = height2 // 2, width2 // 2
        if axis is None:
            (x_peak, y_peak) = locate_peak(match_img, sub_pixel=sub_pixel,
                                           method=method, dim=dim, size=size)
        else:
            (x_peak, y_peak) = locate_peak(match_img, sub_pixel=sub_pixel,
                                           method=method, dim=1, size=size)
            x_peak = x_mid if axis == 0 else x_peak
            y_peak = y_mid if axis == 1 else y_peak
        y_shift, x_shift = y_peak - 1.0 * y_mid, x_peak - 1.0 * x_mid
    else:
        raise ValueError("The image shapes and the selected margin value "
                         "don't match !!!")
    return x_shift, y_shift


def _get_1d_shift_single_row_2d_input(ref_mat, mat, win_size=7, margin=10,
                                      sub_pixel=True, method="diff", size=3,
                                      gpu=False, block=(16, 16), pad=True,
                                      norm=True):
    """
    To find local 1d-shifts of the second image against the reference image
    where their shapes are the same. Each shifting value is associated with a
    column-index of the second image. The value is determined by selecting a
    slab of image-columns of the second image, defined by the 'win_size'
    parameter, and sliding over a slightly larger area of the reference image,
    defined by the 'win_size' and 'margin' parameters.

    Parameters
    ----------
    ref_mat : array_like
        2D array. Reference image.
    mat : array_like
        2D array. The 2nd image. Must be the same size as the reference image.
    win_size : int
        To define the size of the slab of image-columns around the selected
        column of the 2nd image.
    margin : int
        To define the size of the selected area of the reference image for
        sliding, i.e. size = 2 * margin + win_size
    sub_pixel : bool, optional
        Enable sub-pixel location.
    method : {"diff", "poly_fit"}
        Method for finding 1d sub-pixel position. Two options: a
        differential method or a polynomial method.
    size : int
        Window size around the integer location of the maximum value used for
        sub-pixel searching.
    gpu : bool, optional
        Use GPU for computing if True.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (4,4), (8, 8), ...
    pad : bool, optional
        Padding the result to the same width-size as the original images.
    norm : bool, optional
        Normalize the input images if True.

    Returns
    -------
    array_like
        1D array. X-shifting values associated with columns of the 2nd image.
        Starting index is at (margin + win_size // 2). Using the pad option
        to make it easier to find which value corresponding to which column.
    """
    (height, width) = ref_mat.shape
    win_size = 2 * (win_size // 2) + 1
    radi = win_size // 2
    start = radi + margin
    if ref_mat.shape != mat.shape:
        raise ValueError("Shapes of the inputs are not the same !!!")
    if width < (win_size + 2 * margin):
        raise ValueError("Width {0} of the inputs is smaller than the "
                         "requested size (win_size + 2*margin) = "
                         "{1}".format(width, win_size + 2 * margin))
    stop, start1, radi1 = width - start, start + 1, radi + 1
    f_alias = find_shift_based_correlation_map
    if norm:
        ref_mat = normalize_image(ref_mat)
        mat = normalize_image(mat)
    shifts = [f_alias(ref_mat[:, j - start:j + start1],
                      mat[:, j - radi:j + radi1], margin, 1, sub_pixel, method,
                      1, size, gpu, block)
              for j in range(start, stop)]
    shifting_line = np.asarray(shifts)[:, 0]
    if pad:
        shifting_line = np.pad(shifting_line, radi + margin, mode="constant")
    return shifting_line


def _get_1d_shift_multi_rows_3d_input(ref_mat, mat, direction="x", win_size=7,
                                      margin=10, sub_pixel=True, method="diff",
                                      size=3, gpu=False, block=(16, 16),
                                      pad=True, ncore=None, norm=True):
    """
    To find local 1d-shifts of the second 3d-image against the reference
    3d-image where their shapes are the same. Each shifting value is associated
    with a row- or column-index of the image. The value is determined by
    selecting a slab of columns or rows across the axis-0 (stack) of the
    second image, defined by the 'win_size' parameter, and sliding over a
    slightly larger area of the reference image, defined by the 'win_size' and
    'margin' parameters.

    Parameters
    ----------
    ref_mat : array_like
        3D array. Reference image.
    mat : array_like
        3D array. The 2nd image. Must be the same size as the reference image.
    direction : {"x", "y"}
        Select the direction for finding 1d-shift.
    win_size : int
        To define the size of the area around the selected pixel of the 2nd
        image.
    margin : int
        To define the size of the area of the reference image for
        sliding, i.e. size = 2 * margin + win_size
    sub_pixel : bool, optional
        Enable sub-pixel location.
    method : {"diff", "poly_fit"}
        Method for finding 1d sub-pixel position. Two options: a differential
        method or a polynomial method.
    size : int
        Window size around the integer location of the maximum value used for
        sub-pixel searching.
    gpu : bool, optional
        Use GPU for computing if True.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (4,4), (8, 8), ...
    pad : bool, optional
        Padding the result to the same as the original images.
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    norm : bool, optional
        Normalize the input images if True.

    Returns
    -------
    array_like:
        2D array. Shifting values associated with the pixel-positions of the
        2nd image. Starting index is at (margin + win_size // 2). Using the pad
        option to make it easier to find which value corresponding to which
        pixel-position.
    """
    if len(ref_mat.shape) != 3 or len(mat.shape) != 3:
        raise ValueError("Inputs must be 3d arrays !!!")
    if direction != "x" and direction != "y":
        raise ValueError("Only two options: 'x' or 'y''")
    (depth, height, width) = ref_mat.shape
    if ncore is None:
        ncore = mp.cpu_count() - 1
    f_alias = _get_1d_shift_single_row_2d_input
    if direction == "x":
        if ncore == 1:
            shifts = np.asarray(
                [f_alias(ref_mat[:, i, :], mat[:, i, :], win_size, margin,
                         sub_pixel, method, size, gpu, block, pad, norm)
                 for i in range(height)])
        else:
            shifts = np.asarray(Parallel(n_jobs=ncore)(
                delayed(f_alias)(ref_mat[:, i, :], mat[:, i, :], win_size,
                                 margin, sub_pixel, method, size, False, block,
                                 pad, norm) for i in range(height)))
    else:
        if ncore == 1:
            shifts = np.asarray(
                [f_alias(ref_mat[:, :, j], mat[:, :, j], win_size, margin,
                         sub_pixel, method, size, gpu, block, pad, norm)
                 for j in range(width)])
        else:
            shifts = np.asarray(Parallel(n_jobs=ncore)(
                delayed(f_alias)(ref_mat[:, :, j], mat[:, :, j], win_size,
                                 margin, sub_pixel, method, size, False, block,
                                 pad, norm) for j in range(width)))
        shifts = np.transpose(shifts)
    return shifts


def _get_1d_shift_full_image_3d_input_cpu(ref_mat, mat, direction="x",
                                          chunk_size=None, win_size=7,
                                          margin=10, sub_pixel=True,
                                          method="diff", size=3, ncore=None,
                                          norm=True, norm_global=False):
    """
    CPU-function to find local 1d-shifts of the second 3d-image against the
    reference 3d-image where their shapes are the same and their size may be
    too big for memory. Each shifting value is associated with a row- or
    column-index of the image. The value is determined by selecting a slab of
    columns or rows across the axis-0 (stack) of the second image, defined by
    the 'win_size' parameter, and sliding over a slightly larger area of the
    reference image, defined by the 'win_size' and 'margin' parameters.

    Parameters
    ----------
    ref_mat : array_like
        3D array, can be a numpy array or hdf object. Reference image.
    mat : array_like
        3D array, can be a numpy array or hdf object. The 2nd image. Must be
        the same size as the reference image.
    direction : {"x", "y"}
        Select the direction for finding 1d-shift.
    chunk_size : int or None
        Size of each chunk extracted along the axis-1 (height) of the 3d-image.
    win_size : int
        To define the size of the area around the selected pixel of the 2nd
        image.
    margin : int
        To define the size of the area of the reference image for
        sliding, i.e. size = 2 * margin + win_size
    sub_pixel : bool, optional
        Enable sub-pixel location.
    method : {"diff", "poly_fit"}
        Method for finding 1d sub-pixel position. Two options: a
        differential method or a polynomial method.
    size : int
        Window size around the integer location of the maximum value used for
        sub-pixel searching.
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    norm : bool, optional
        Normalize the input images if True.
    norm_global : bool, optional
        Normalize by using the full size of the inputs if True.

    Returns
    -------
    array_like:
        2D array. Shifting values along the x-direction or y-direction.
    """
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) != 3:
        raise ValueError("Inputs must be 3d-arrays !!!")
    (depth, height, width) = ref_mat.shape
    if chunk_size is None:
        chunk_size = height + 1
    else:
        chunk_size = np.clip(chunk_size, 1, height)
    num_chunk = np.clip(height // chunk_size + 1, 1, height)
    f_alias = _get_1d_shift_multi_rows_3d_input
    shifts = np.zeros((height, width), dtype=np.float32)
    edge = margin + win_size // 2
    if norm_global:
        ref_mat = normalize_image(ref_mat)
        mat = normalize_image(mat)
    if direction == "x":
        list_index = np.array_split(np.arange(height), num_chunk)
        for pos in list_index:
            bindex, eindex = pos[0], pos[-1] + 1
            ref_mat1 = ref_mat[:, bindex:eindex, :]
            mat1 = mat[:, bindex:eindex, :]
            shift_chunk = f_alias(ref_mat1, mat1, direction, win_size, margin,
                                  sub_pixel, method, size, False, None,
                                  True, ncore, norm)
            shifts[bindex:eindex] = shift_chunk
    else:
        list_index = np.array_split(np.arange(edge, height - edge), num_chunk)
        for pos in list_index:
            bindex, eindex = pos[0], pos[-1] + 1
            ref_mat1 = ref_mat[:, bindex - edge:eindex + edge, :]
            mat1 = mat[:, bindex - edge:eindex + edge, :]
            shift_chunk = f_alias(ref_mat1, mat1, direction, win_size, margin,
                                  sub_pixel, method, size, False, None,
                                  False, ncore, norm)
            shifts[bindex:eindex] = shift_chunk
    return shifts


def _get_2d_shift_full_image_2d_input(ref_mat, mat, win_size=7, margin=10,
                                      sub_pixel=True, method="diff", size=3,
                                      gpu=False, block=(16, 16), ncore=None,
                                      norm=True):
    """
    To find local (y,x)-shifts of the second image against the reference
    image where their shapes are the same. Each (y,x)-shifting value is
    associated with a pixel-location of the second image. The value is
    determined by selecting a small area of the second image, defined by
    the 'win_size' parameter, and sliding (y,x-direction) over a slightly
    larger area of the reference image, defined by the 'win_size' and 'margin'
    parameters.

    Parameters
    ----------
    ref_mat : array_like
        2D array. Reference image.
    mat : array_like
        2D array. The 2nd image. Must be the same size as the reference image.
    win_size : int
        To define the size of the area around the selected pixel of the 2nd
        image.
    margin : int
        To define the size of the area of the reference image for sliding,
        i.e. size = 2 * margin + win_size
    sub_pixel : bool, optional
        Enable sub-pixel location.
    method : {"diff", "poly_fit"}
        Method for finding sub-pixel shift. Two options: a differential
        method (Ref. [1]) or a polynomial method (Ref. [2]).
    size : int
        Window size around the integer location of the maximum value used for
        sub-pixel searching.
    gpu : bool, optional
        Use GPU for computing if True.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (4,4), (8, 8), ...
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    norm : bool, optional
        Normalizing the inputs.

    Returns
    -------
    list of two 2d-arrays
        x-shift array and y-shift array. Zeros at the outer area of the size of
        (margin + win_size // 2)

    References
    ----------
    [1] : https://doi.org/10.48550/arXiv.0712.4289

    [2] : https://doi.org/10.1088/0957-0233/17/6/045
    """
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) != 2:
        raise ValueError("Inputs must be 2d-arrays !!!")
    (height, width) = ref_mat.shape
    win_size = 2 * (win_size // 2) + 1
    radi = win_size // 2
    start = radi + margin
    als_size = win_size + 2 * margin
    if width < als_size or height < als_size:
        raise ValueError("Shapes of the inputs {0} are smaller than the "
                         "requested size (win_size + 2*margin) = "
                         "{1}".format(ref_mat.shape, als_size))
    stop_col, stop_row = width - start, height - start
    start1, radi1 = start + 1, radi + 1
    if norm:
        ref_mat = normalize_image(ref_mat)
        mat = normalize_image(mat)
    f_alias = find_shift_based_correlation_map
    if (gpu is True) or (ncore == 1):
        shifts = [
            [f_alias(ref_mat[i - start:i + start1, j - start:j + start1],
                     mat[i - radi:i + radi1, j - radi:j + radi1],
                     margin, None, sub_pixel, method, 2, size, gpu, block)
             for j in range(start, stop_col)]
            for i in range(start, stop_row)]
        shifts = np.asarray(shifts)
    else:
        if ncore is None:
            ncore = np.clip(mp.cpu_count() - 1, 1, None)
        shifts = np.asarray(Parallel(n_jobs=ncore)(
            delayed(f_alias)(
                ref_mat[i - start:i + start1, j - start:j + start1],
                mat[i - radi:i + radi1, j - radi:j + radi1],
                margin, None, sub_pixel, method, 2, size, False, block)
            for i in range(start, stop_row) for j in range(start, stop_col)))
        shifts = np.reshape(np.asarray(shifts),
                            (stop_row - start, stop_col - start, 2))
    (x_shifts, y_shifts) = np.moveaxis(shifts, 2, 0)
    x_shifts = np.pad(x_shifts, radi + margin, mode="constant")
    y_shifts = np.pad(y_shifts, radi + margin, mode="constant")
    return x_shifts, y_shifts


def _get_2d_shift_multi_rows_3d_input(ref_mat, mat, win_size=7, margin=10,
                                      sub_pixel=True, method="diff", size=3,
                                      gpu=False, block=(16, 16), pad=True,
                                      ncore=None, norm=False):
    """
    To find local (y,x)-shifts of the second 3d-image against the reference
    3d-image where their shapes are the same. Each (y,x)-shifting value is
    associated with a pixel-location of the second image. The value is
    determined by selecting a small volume of the second image, defined by
    the 'win_size' parameter, and sliding (y,x only) over a slightly larger
    volume of the reference image, defined by the 'win_size' and 'margin'
    parameters. Note that this function is computational expensive.

    Parameters
    ----------
    ref_mat : array_like
        3D array. Reference image.
    mat : array_like
        3D array. The 2nd image. Must be the same size as the reference image.
    win_size : int
        To define the size of the image volume around the selected pixel of the
        2nd image.
    margin : int
        To define the size of the selected volume of the reference image for
        sliding i.e. size = 2 * margin + win_size
    sub_pixel : bool, optional
        Enable sub-pixel location.
    method : {"diff", "poly_fit"}
        Method for finding sub-pixel shift. Two options: a differential
        method (Ref. [1]) or a polynomial method (Ref. [2]).
    size : int
        Window size around the integer location of the maximum value used for
        sub-pixel searching.
    gpu : bool, optional
        Use GPU for computing if True.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (4,4), (8, 8), ...
    pad : bool, optional
        Padding the result to the same as the original images.
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    norm : bool, optional
        Normalizing the inputs.

    Returns
    -------
    list of two 2d-arrays
        x-shift array and y-shift array, corresponding to the pixel-locations
        in the 2nd image where the starting location is at
        (margin + win_size/2, margin + win_size/2). Using the pad option to
        make it easier to find which value corresponding to which
        pixel-location.

    References
    ----------
    [1] : https://doi.org/10.48550/arXiv.0712.4289

    [2] : https://doi.org/10.1088/0957-0233/17/6/045
    """
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) != 3:
        raise ValueError("Inputs must be 3d-arrays !!!")
    (height, width) = ref_mat.shape[-2:]
    win_size = 2 * (win_size // 2) + 1
    radi = win_size // 2
    start = radi + margin
    als_size = win_size + 2 * margin
    if width < als_size or height < als_size:
        raise ValueError("Shapes of the inputs {0} are smaller than the "
                         "requested size (win_size + 2*margin) = "
                         "{1} x {1}".format((height, width), als_size))
    if norm:
        ref_mat = normalize_image(ref_mat)
        mat = normalize_image(mat)
    stop_col, stop_row = width - start, height - start
    start1, radi1 = start + 1, radi + 1
    f_alias = find_shift_based_correlation_map
    if (gpu is True) or (ncore == 1):
        shifts = [
            [f_alias(ref_mat[:, i - start:i + start1, j - start:j + start1],
                     mat[:, i - radi:i + radi1, j - radi:j + radi1],
                     margin, None, sub_pixel, method, 2, size, gpu, block)
             for j in range(start, stop_col)]
            for i in range(start, stop_row)]
        shifts = np.asarray(shifts)
    else:
        if ncore is None:
            ncore = np.clip(mp.cpu_count() - 1, 1, None)
        shifts = np.asarray(
            Parallel(n_jobs=ncore)(delayed(f_alias)(
                ref_mat[:, i - start:i + start1, j - start:j + start1],
                mat[:, i - radi:i + radi1, j - radi:j + radi1],
                margin, None, sub_pixel, method, 2, size, False, None)
                                   for i in range(start, stop_row)
                                   for j in range(start, stop_col)))
        shifts = np.reshape(np.asarray(shifts),
                            (stop_row - start, stop_col - start, 2))
    (x_shifts, y_shifts) = np.moveaxis(shifts, 2, 0)
    if pad:
        x_shifts = np.pad(x_shifts, radi + margin, mode="constant")
        y_shifts = np.pad(y_shifts, radi + margin, mode="constant")
    return x_shifts, y_shifts


def _get_2d_shift_full_image_3d_input_cpu(ref_mat, mat, chunk_size=None,
                                          win_size=7, margin=10,
                                          sub_pixel=True, method="diff",
                                          size=3, ncore=None, norm=False,
                                          norm_global=False):
    """
    CPU function to find local (y,x)-shifts of the second 3d-image against the
    reference 3d-image where their shapes are the same and their size may be
    too big for memory. Each (y,x)-shifting value is associated with a
    pixel-location of the second image. The value is determined by selecting a
    small volume of the second image, defined by the 'win_size' parameter, and
    sliding (y,x only) over a slightly larger volume of the reference image,
    defined by the 'win_size' and 'margin' parameters. Note that this function
    is computational expensive.

    Parameters
    ----------
    ref_mat : array_like
        3D array, can be a numpy array or hdf object. Reference image.
    mat : array_like
        3D array, can be a numpy array or hdf object. The 2nd image. Must be
        the same size as the reference image.
    chunk_size : int or None
        Size of each chunk extracted along the axis-1 (height) of the 3d-image.
    win_size : int
        To define the size of the area around the selected pixel of the 2nd
        image.
    margin : int
        To define the size of the area of the reference image for
        sliding, i.e. size = 2 * margin + win_size
    sub_pixel : bool, optional
        Enable sub-pixel location.
    method : {"diff", "poly_fit"}
        Method for finding sub-pixel position. Two options: a differential
        method or a polynomial method.
    size : int
        Window size around the integer location of the maximum value used for
        sub-pixel searching.
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    norm : bool, optional
        Normalize the input images if True.
    norm_global : bool, optional
        Normalize by using the full size of the inputs if True.

    Returns
    -------
    list of two 2d-arrays
        x-shift array and y-shift array. Zeros at the outer area of the size of
        (margin + win_size // 2).
    """
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) != 3:
        raise ValueError("Inputs must be 3d-arrays !!!")
    (height, width) = ref_mat.shape[-2:]
    if chunk_size is None:
        chunk_size = height + 1
    else:
        chunk_size = np.clip(chunk_size, 1, height)
    num_chunk = np.clip(height // chunk_size + 1, 1, height)
    edge = margin + win_size // 2
    if height < (2 * edge + 1):
        raise ValueError("Height of the inputs {0} are smaller than the "
                         "requested size (win_size + 2*margin + 1) = "
                         "{1} x {1}".format(height, 2 * edge + 1))
    list_index = np.array_split(np.arange(edge, height - edge), num_chunk)
    x_shifts = np.zeros((height, width), dtype=np.float32)
    y_shifts = np.zeros((height, width), dtype=np.float32)
    f_alias = _get_2d_shift_multi_rows_3d_input
    if norm_global:
        ref_mat = normalize_image(ref_mat)
        mat = normalize_image(mat)
    for pos in list_index:
        bindex, eindex = pos[0], pos[-1] + 1
        ref_mat1 = ref_mat[:, bindex - edge:eindex + edge, :]
        mat1 = mat[:, bindex - edge:eindex + edge, :]
        shift_chunk = f_alias(ref_mat1, mat1, win_size, margin, sub_pixel,
                              method, size, False, None, False, ncore, norm)
        x_shifts[bindex:eindex, edge:-edge] = shift_chunk[0]
        y_shifts[bindex:eindex, edge:-edge] = shift_chunk[1]
    return x_shifts, y_shifts


@cuda.jit(device=True)
def __gen_1d_corr_map_kernel(ref_mat, mat, list_coef):  # pragma: no cover
    """
    GPU-kernel function to generate a list of local correlation-coefficients
    between two images having the same height.

    Parameters
    ----------
    ref_mat : array_like
        2D array. The first image.
    mat : array_like
        2D array. The second image.
    list_coef :  array_like
        list of zeros, passed from GPU global memory.

    Returns
    -------
    array_like
        Calculated coefficients.
    """
    (height0, width0) = ref_mat.shape
    (height1, width1) = mat.shape
    width2 = width0 - width1 + 1
    mat_mean = __mean_2d(mat)
    mat_sqr = __sum_square_2d(mat, mat_mean)
    for j in range(width2):
        ref_mat1 = ref_mat[:, j: j + width1]
        ref_mean = __mean_2d(ref_mat1)
        ref_sqr = __sum_square_2d(ref_mat1, ref_mean)
        sum_mul = __sum_multiply_2d(ref_mat1, mat, ref_mean, mat_mean)
        num = math.sqrt(ref_sqr * mat_sqr)
        if num != 0.0:
            num_tmp = sum_mul / num
        list_coef[j] = num_tmp
    return list_coef


@cuda.jit(device=True)
def __locate_1d_peak_kernel(list_data):  # pragma: no cover
    """
    GPU-kernel function to locate the position of the maximum value of a
    1d-array with sub-pixel accuracy.

    Parameters
    ----------
    list_data : array_like
        1D array.

    Returns
    -------
    float
        Sub-pixel position of the maximum value.
    """
    num_point = len(list_data)
    pos_max = 0
    val_max = list_data[pos_max]
    for i in range(1, num_point):
        val = list_data[i]
        if val > val_max:
            val_max = val
            pos_max = i
    if (num_point - 1) > pos_max > 0:
        val1 = list_data[pos_max - 1]
        val2 = list_data[pos_max + 1]
        d1fx = (val2 - val1) / 2.0
        d2fx = (val2 + val1 - 2.0 * val_max)
        if d2fx != 0:
            x_off = - d1fx / d2fx
            pos_max = pos_max + x_off
    return pos_max


@cuda.jit
def _get_1d_shift_multi_rows_3d_input_kernel(shift_mat, ref_mat, mat,
                                             list_coef, height, width, radi,
                                             margin):  # pragma: no cover
    """
    GPU-CPU function to find local 1d-shifts of the second 3d-image against
    the reference 3d-image where their shapes are the same.

    Parameters
    ----------
    shift_mat : array_like
        2D array of zeros, initialized at CPU.
    ref_mat : array_like
        3D array. Reference image.
    mat : array_like
        3D array. The 2nd image. Must be the same size as the reference image.
    list_coef : array_like
        list of the list of zeros, initialized at CPU.
    height : int
        Height of the image.
    width : int
        Width of the image.
    radi : int
        Radius of the window to select a local area of the image.
    margin : int
        To define the size of the area of the reference image for
        sliding.

    Returns
    -------

        Update of the shifting image passed from CPU.
    """
    (x_index, y_index) = cuda.grid(2)
    if (y_index < height) and (x_index < width):
        edge = radi + margin
        pos = x_index + edge
        ref_mat1 = ref_mat[y_index, :, pos - edge: pos + edge + 1]
        mat1 = mat[y_index, :, pos - radi: pos + radi + 1]
        list_coef1 = list_coef[y_index, x_index, :]
        list_coef1 = __gen_1d_corr_map_kernel(ref_mat1, mat1, list_coef1)
        pos_max = __locate_1d_peak_kernel(list_coef1)
        shift_mat[y_index, x_index] = pos_max - margin


def _get_1d_shift_multi_rows_3d_input_gpu(ref_mat, mat, direction="x",
                                          win_size=7, margin=10,
                                          block=(16, 16), pad=True, norm=True):
    """
    GPU function to find local 1d-shifts of the second 3d-image against the
    reference 3d-image where their shapes are the same. Each shifting value is
    associated with a row- or column-index of the image. The value is
    determined by selecting a slab of columns or rows across the axis-0 (stack)
    of the second image, defined by the 'win_size' parameter, and sliding over
    a slightly larger area of the reference image, defined by the 'win_size'
    and 'margin' parameters.

    Parameters
    ----------
    ref_mat : array_like
        3D array. Reference image.
    mat : array_like
        3D array. The 2nd image. Must be the same size as the reference image.
    direction : {"x", "y"}
        Select the direction for finding 1d-shift.
    win_size : int
        To define the size of the area around the selected pixel of the 2nd
        image.
    margin : int
        To define the size of the area of the reference image for
        sliding, i.e. size = 2 * margin + win_size
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (4,4), (8, 8), ...
    pad : bool, optional
        Padding the result to the same as the original images.
    norm : bool, optional
        Normalize the inputs if True.

    Returns
    -------
    array_like:
        2D array. Shifting values associated with the pixel-positions of the
        2nd image. Starting index is at (margin + win_size // 2). Using the pad
        option to make it easier to find which value corresponding to which
        pixel-position.
    """
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) == 2:
        ref_mat = np.expand_dims(ref_mat, axis=0)
        mat = np.expand_dims(mat, axis=0)
    if len(ref_mat.shape) != 3:
        raise ValueError("Inputs must be 3d arrays !!!")
    if direction != "x" and direction != "y":
        raise ValueError("Only two options: 'x' or 'y''")
    if direction != "x":
        ref_mat = np.transpose(ref_mat, axes=(0, 2, 1))
        mat = np.transpose(mat, axes=(0, 2, 1))
    ref_mat = np.moveaxis(ref_mat, 1, 0)
    mat = np.moveaxis(mat, 1, 0)
    (height, depth, width) = ref_mat.shape
    if norm:
        ref_mat = np.asarray(
            [normalize_image(ref_mat[i]) for i in range(height)])
        mat = np.asarray([normalize_image(mat[i]) for i in range(height)])
    win_size = 2 * (win_size // 2) + 1
    radi = win_size // 2
    edge = radi + margin
    width1 = width - 2 * edge
    if width1 < 1:
        raise ValueError("Width {0} of the inputs is smaller than the "
                         "requested size (win_size + 2*margin) = "
                         "{1}".format(width, win_size + 2 * margin))
    shifts = np.zeros((height, width1), dtype=np.float32)
    grid = (int(np.ceil(1.0 * width1 / block[0])),
            int(np.ceil(1.0 * height / block[1])))
    f_alias = _get_1d_shift_multi_rows_3d_input_kernel
    ref_mat1 = np.float32(np.ascontiguousarray(ref_mat))
    mat1 = np.float32(np.ascontiguousarray(mat))
    list_coef = np.zeros((height, width1, 2 * margin + 1), dtype=np.float32)
    f_alias[grid, block](shifts, ref_mat1, mat1, list_coef, np.int32(height),
                         np.int32(width1), np.int32(radi), np.int32(margin))
    if pad:
        shifts = np.pad(shifts, ((0, 0), (edge, edge)), mode="constant")
    if direction != "x":
        shifts = np.transpose(shifts)
    return shifts


def _get_1d_shift_full_image_3d_input_gpu(ref_mat, mat, direction="x",
                                          chunk_size=None, win_size=7,
                                          margin=10, block=(16, 16), norm=True,
                                          norm_global=True):
    """
    GPU function to find local 1d-shifts of the second 3d-image against the
    reference 3d-image where their shapes are the same and their size may be
    too big for memory. Each shifting value is associated with a row- or
    column-index of the image. The value is determined by selecting a slab of
    columns or rows across the axis-0 (stack) of the second image, defined by
    the 'win_size' parameter, and sliding over a slightly larger area of the
    reference image, defined by the 'win_size' and 'margin' parameters.

    Parameters
    ----------
    ref_mat : array_like
        3D array, can be a numpy array or hdf object. Reference image.
    mat : array_like
        3D array, can be a numpy array or hdf object. The 2nd image. Must be
        the same size as the reference image.
    direction : {"x", "y"}
        Select the direction for finding 1d-shift.
    chunk_size : int or None
        Size of each chunk extracted along the axis-1 (height) of the 3d-image.
    win_size : int
        To define the size of the area around the selected pixel of the 2nd
        image.
    margin : int
        To define the size of the area of the reference image for sliding,
        i.e. size = 2 * margin + win_size.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (4,4), (8, 8), ...
    norm : bool, optional
        Normalize the input images if True.
    norm_global : bool, optional
        Normalize by using the full size of the inputs if True.

    Returns
    -------
    array_like:
        2D array. Shifting values along the x-direction or y-direction.
    """
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) != 3:
        raise ValueError("Inputs must be 3d-arrays !!!")
    (depth, height, width) = ref_mat.shape
    if chunk_size is None:
        chunk_size = height + 1
    else:
        chunk_size = np.clip(chunk_size, 1, height)
    num_chunk = np.clip(height // chunk_size + 1, 1, height)
    f_alias = _get_1d_shift_multi_rows_3d_input_gpu
    shifts = np.zeros((height, width), dtype=np.float32)
    edge = margin + win_size // 2
    if norm_global:
        ref_mat = normalize_image(ref_mat)
        mat = normalize_image(mat)
    if direction == "x":
        list_index = np.array_split(np.arange(height), num_chunk)
        for pos in list_index:
            bindex, eindex = pos[0], pos[-1] + 1
            ref_mat1 = ref_mat[:, bindex:eindex, :]
            mat1 = mat[:, bindex:eindex, :]
            shift_chunk = f_alias(ref_mat1, mat1, direction, win_size, margin,
                                  block, True, norm)
            shifts[bindex:eindex] = shift_chunk
    else:
        list_index = np.array_split(np.arange(edge, height - edge), num_chunk)
        for pos in list_index:
            bindex, eindex = pos[0], pos[-1] + 1
            ref_mat1 = ref_mat[:, bindex - edge:eindex + edge, :]
            mat1 = mat[:, bindex - edge:eindex + edge, :]
            shift_chunk = f_alias(ref_mat1, mat1, direction, win_size, margin,
                                  block, False, norm)
            shifts[bindex:eindex] = shift_chunk
    return shifts


@cuda.jit(device=True)
def __gen_2d_corr_map_2d_input(ref_mat, mat, coef_mat):  # pragma: no cover
    """
    GPU-kernel function to generate a correlation map between two images where
    the second image is smaller than the first image.

    Parameters
    ----------
    ref_mat : array_like
        2D array. The first image.
    mat : array_like
        2D array. The second image.
    coef_mat : array_like
        2D array of zeros, loaded from GPU global memory

    Returns
    -------
    array_like
        2D array. Calculated coefficient map.
    """
    (height0, width0) = ref_mat.shape
    (height1, width1) = mat.shape
    height2, width2 = height0 - height1 + 1, width0 - width1 + 1
    mat_mean = __mean_2d(mat)
    mat_sqr = __sum_square_2d(mat, mat_mean)
    # coef_mat is at GPU global memory with the size of (height2, width2)
    for i in range(height2):
        for j in range(width2):
            row0, row1 = i, i + height1
            col0, col1 = j, j + width1
            ref_mat1 = ref_mat[row0:row1, col0:col1]
            ref_mean = __mean_2d(ref_mat1)
            ref_sqr = __sum_square_2d(ref_mat1, ref_mean)
            sum_mul = __sum_multiply_2d(ref_mat1, mat, ref_mean, mat_mean)
            num = math.sqrt(ref_sqr * mat_sqr)
            if num != 0.0:
                coef_mat[i, j] = sum_mul / num
    return coef_mat


@cuda.jit(device=True)
def __gen_2d_corr_map_3d_input(ref_mat, mat, coef_mat):  # pragma: no cover
    """
    GPU-kernel function to generate a correlation map between two stacks of
    images where the number of images are the same but the image size of the
    second stack is smaller than the first one.

    Parameters
    ----------
    ref_mat : array_like
        3D array. The first image.
    mat : array_like
        3D array. The second image.
    coef_mat : array_like
        2D array, loaded from GPU global memory.

    Returns
    -------
    array_like
        2D array. Calculated coefficient map.
    """
    (depth, height0, width0) = ref_mat.shape
    (_, height1, width1) = mat.shape
    height2, width2 = height0 - height1 + 1, width0 - width1 + 1
    mat_mean = __mean_3d(mat)
    mat_sqr = __sum_square_3d(mat, mat_mean)
    # coef_mat is at GPU global memory with the size of (height2, width2)
    for i in range(height2):
        for j in range(width2):
            row0, row1 = i, i + height1
            col0, col1 = j, j + width1
            ref_mat1 = ref_mat[:, row0:row1, col0:col1]
            ref_mean = __mean_3d(ref_mat1)
            ref_sqr = __sum_square_3d(ref_mat1, ref_mean)
            sum_mul = __sum_multiply_3d(ref_mat1, mat, ref_mean, mat_mean)
            num = math.sqrt(ref_sqr * mat_sqr)
            if num != 0.0:
                coef_mat[i, j] = sum_mul / num
    return coef_mat


@cuda.jit(device=True)
def __locate_max_value(mat):  # pragma: no cover
    """
    GPU-kernel function to find the indices of the maximum value of a 2D array.

    Parameters
    ----------
    mat : array_like
        2D array.

    Returns
    -------
    list of two floats.
        (column-index, row-index)
    """
    (height, width) = mat.shape
    x_max = 0
    y_max = 0
    val_max = mat[y_max, x_max]
    for i in range(height):
        for j in range(width):
            val = mat[i, j]
            if val > val_max:
                y_max = i
                x_max = j
                val_max = val
    return x_max, y_max


@cuda.jit(device=True)
def __get_max_value(mat):  # pragma: no cover
    """
    GPU-kernel function to find the maximum value of a 2D array.

    Parameters
    ----------
    mat : array_like
        2D array.

    Returns
    -------
    list of two floats.
        (column-index, row-index)
    """
    (height, width) = mat.shape
    val_max = mat[0, 0]
    for i in range(height):
        for j in range(width):
            val = mat[i, j]
            if val > val_max:
                val_max = val
    return val_max


@cuda.jit(device=True)
def __inverse_values(mat, val_max):  # pragma: no cover
    """
    GPU-kernel function to inverse values of a 2D array.

    Parameters
    ----------
    mat : array_like
        2D array.
    val_max : float
        Maximum value of the array.

    Returns
    -------
    array_like
    """
    (height, width) = mat.shape
    for i in range(height):
        for j in range(width):
            mat[i, j] = val_max - mat[i, j]
    return mat


@cuda.jit(device=True)
def __locate_2d_peak_kernel(mat, x0, y0):  # pragma: no cover
    """
    GPU-kernel function to locate the position of the maximum value of a
    2d-array with sub-pixel accuracy (Ref. [1]).

    Parameters
    ----------
    mat : array_like
        2D array.
    x0 : int
        Column index of the maximum value.
    y0 : int
        Row index of the maximum value.

    Returns
    -------
    list of two floats
        Sub-pixel position (x, y), i.e. (column, row), of the maximum value.

    References
    ----------
    [1] : https://doi.org/10.48550/arXiv.0712.4289
    """
    (height, width) = mat.shape
    height1 = height - 1
    width1 = width - 1
    d1fx, d2fx, d1fy, d2fy, d1fxy = 0.0, 0.0, 0.0, 0.0, 0.0
    x_pos = x0
    y_pos = y0
    if (x0 == 0 or x0 == width1) and (height1 > y0 > 0):
        d1fy = (mat[y0 + 1, x0] - mat[y0 - 1, x0]) / 2.0
        d2fy = (mat[y0 + 1, x0] + mat[y0 - 1, x0] - 2 * mat[y0, x0])
    if (y0 == 0 or y0 == height1) and (width1 > x0 > 0):
        d1fx = (mat[y0, x0 + 1] - mat[y0, x0 - 1]) / 2.0
        d2fx = (mat[y0, x0 + 1] + mat[y0, x0 - 1] - 2 * mat[y0, x0])
    if (height1 > y0 > 0) and (width1 > x0 > 0):
        d1fx = 0.5 * (mat[y0, x0 + 1] - mat[y0, x0 - 1])
        d1fy = 0.5 * (mat[y0 + 1, x0] - mat[y0 - 1, x0])
        d2fx = (mat[y0, x0 + 1] + mat[y0, x0 - 1] - 2 * mat[y0, x0])
        d2fy = (mat[y0 + 1, x0] + mat[y0 - 1, x0] - 2 * mat[y0, x0])
        d1fxy = 0.25 * (mat[y0 + 1, x0 + 1] + mat[y0 - 1, x0 - 1] - mat[
            y0 + 1, x0 - 1] - mat[y0 - 1, x0 + 1])
    num = d1fxy ** 2 - d2fx * d2fy
    if num != 0:
        x_off = (d2fy * d1fx - d1fxy * d1fy) / num
        y_off = (d2fx * d1fy - d1fxy * d1fx) / num
        if 1.0 > abs(x_off) > 0.0:
            x_pos = x0 + x_off
        if 1.0 > abs(y_off) > 0.0:
            y_pos = y0 + y_off
    return x_pos, y_pos


@cuda.jit
def _get_2d_shift_multi_rows_2d_input_kernel(shifts, ref_mat, mat, coef_4d,
                                             height, width, radi,
                                             margin):  # pragma: no cover
    """
    GPU-CPU function to find local (y,x)-shifts of the second image against
    the reference image.

    Parameters
    ----------
    shifts : array_like
        Two of 2D arrays of zeros, initialized at CPU.
    ref_mat : array_like
        2D array. Reference image.
    mat : array_like
        2D array. The 2nd image. Must be the same size as the reference image.
    coef_4d : array_like
        4D array of zeros, initialized at CPU.
    height : int
        Height of the image.
    width : int
        Width of the image.
    radi : int
        Radius of the window to select a local area of the image.
    margin : int
        To define the size of the area of the reference image for sliding.

    Returns
    -------

        Update of the (y,x)-shifting images passed from CPU.
    """
    (x_index, y_index) = cuda.grid(2)
    if (y_index < height) and (x_index < width):
        radi_ref = radi + margin
        j = x_index + radi_ref
        i = y_index + radi_ref
        ref_mat1 = ref_mat[i - radi_ref:i + radi_ref + 1,
                   j - radi_ref: j + radi_ref + 1]
        mat1 = mat[i - radi:i + radi + 1, j - radi: j + radi + 1]
        coef_mat1 = coef_4d[y_index, x_index, :, :]
        coef_mat1 = __gen_2d_corr_map_2d_input(ref_mat1, mat1, coef_mat1)
        x_max, y_max = __locate_max_value(coef_mat1)
        x_pos, y_pos = __locate_2d_peak_kernel(coef_mat1, x_max, y_max)
        shifts[0, y_index, x_index] = x_pos - margin
        shifts[1, y_index, x_index] = y_pos - margin


def _get_2d_shift_multi_rows_2d_input_gpu(ref_mat, mat, win_size=7, margin=10,
                                          block=(16, 16), pad=True,
                                          norm=False):
    """
    Using gpu to find local (y,x)-shifts of the second image against the
    reference image where their shapes are the same. Each (y,x)-shifting value
    is associated with a pixel-location of the second image. The value is
    determined by selecting a small area of the second image, defined by the
    'win_size' parameter, and sliding (y,x-direction) over a slightly larger
    area of the reference image, defined by the 'win_size' and 'margin'
    parameters.

    Parameters
    ----------
    ref_mat : array_like
        2D array. Reference image.
    mat : array_like
        2D array. The 2nd image. Must be the same size as the reference image.
    win_size : int
        To define the size of the area around the selected pixel of the 2nd
        image.
    margin : int
        To define the size of the area of the reference image for sliding,
        i.e. size = 2 * margin + win_size
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (4,4), (8, 8), ...
    pad : bool, optional
        Pad the result to the same as the original images.
    norm : bool, optional
        Normalize the input images if True.

    Returns
    -------
    list of two 2d-arrays
        x-shift array and y-shift array. Zeros at the outer area of the size of
        (margin + win_size // 2) if using the pad option.
    """
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) != 2:
        raise ValueError("Inputs must be 2d-arrays !!!")
    (height, width) = ref_mat.shape
    win_size = 2 * (win_size // 2) + 1
    radi = win_size // 2
    edge, size = radi + margin, 2 * margin + 1
    height1, width1 = height - 2 * edge, width - 2 * edge
    if width1 < 1 or height1 < 1:
        raise ValueError("Shapes of the inputs {0} are smaller than the "
                         "requested size (win_size + 2*margin) = "
                         "{1}".format(ref_mat.shape, edge))
    if norm:
        ref_mat = normalize_image(ref_mat)
        mat = normalize_image(mat)
    shifts = np.zeros((2, height1, width1), dtype=np.float32)
    grid = (int(np.ceil(1.0 * width1 / block[0])),
            int(np.ceil(1.0 * height1 / block[1])))
    f_alias = _get_2d_shift_multi_rows_2d_input_kernel
    ref_mat1 = np.float32(np.ascontiguousarray(ref_mat))
    mat1 = np.float32(np.ascontiguousarray(mat))
    coef_4d = np.zeros((height1, width1, size, size), dtype=np.float32)
    f_alias[grid, block](shifts, ref_mat1, mat1, coef_4d, np.int32(height1),
                         np.int32(width1), np.int32(radi), np.int32(margin))
    if pad:
        shifts = np.pad(shifts, ((0, 0), (edge, edge), (edge, edge)),
                        mode="constant")
    return shifts[0], shifts[1]


def _get_2d_shift_full_image_2d_input_gpu(ref_mat, mat, chunk_size=None,
                                          win_size=7, margin=10,
                                          block=(16, 16), norm=False,
                                          norm_global=False):
    """
    GPU function to find local (y,x)-shifts of the second image against the
    reference image where their shapes are the same and their size may be
    too big for GPU memory. Each (y,x)-shifting value is associated with a
    pixel-location of the second image. The value is determined by selecting
    a small area of the second image, defined by the 'win_size' parameter, and
    sliding (y,x-direction) over a slightly larger area of the reference image,
    defined by the 'win_size' and 'margin' parameters.

    Parameters
    ----------
    ref_mat : array_like
        2D array, can be a numpy array or hdf object. Reference image.
    mat : array_like
        2D array, can be a numpy array or hdf object. The 2nd image. Must be
        the same size as the reference image.
    chunk_size : int or None
        Size of each chunk extracted along the height of the image.
    win_size : int
        To define the size of the area around the selected pixel of the 2nd
        image.
    margin : int
        To define the size of the area of the reference image for
        sliding, i.e. size = 2 * margin + win_size
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (4,4), (8, 8), ...
    norm : bool, optional
        Normalize the input images if True.
    norm_global : bool, optional
        Normalize by using the full size of the inputs if True.

    Returns
    -------
    list of two 2d-arrays
        x-shift array and y-shift array. Zeros at the outer area of the size of
        (margin + win_size // 2).
    """
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) != 2:
        raise ValueError("Inputs must be 2d-arrays !!!")
    (height, width) = ref_mat.shape
    if chunk_size is None:
        chunk_size = height + 1
    else:
        chunk_size = np.clip(chunk_size, 1, height)
    num_chunk = np.clip(height // chunk_size + 1, 1, height)
    edge = margin + win_size // 2
    if height < (2 * edge + 1):
        raise ValueError("Height of the inputs {0} are smaller than the "
                         "requested size (win_size + 2*margin + 1) = "
                         "{1} x {1}".format(height, 2 * edge + 1))
    list_index = np.array_split(np.arange(edge, height - edge), num_chunk)
    x_shifts = np.zeros((height, width), dtype=np.float32)
    y_shifts = np.zeros((height, width), dtype=np.float32)
    f_alias = _get_2d_shift_multi_rows_2d_input_gpu
    if norm_global:
        ref_mat = normalize_image(ref_mat)
        mat = normalize_image(mat)
    for pos in list_index:
        bindex, eindex = pos[0], pos[-1] + 1
        ref_mat1 = ref_mat[bindex - edge:eindex + edge, :]
        mat1 = mat[bindex - edge:eindex + edge, :]
        shift_chunk = f_alias(ref_mat1, mat1, win_size, margin, block,
                              False, norm)
        x_shifts[bindex:eindex, edge:-edge] = shift_chunk[0]
        y_shifts[bindex:eindex, edge:-edge] = shift_chunk[1]
    return x_shifts, y_shifts


@cuda.jit
def _get_2d_shift_multi_rows_3d_input_kernel(shifts, ref_mat, mat, coef_4d,
                                             height, width, radi,
                                             margin):  # pragma: no cover
    """
    GPU-CPU function to find local (y,x)-shifts of the second 3d-image against
    the reference 3d-image.

    Parameters
    ----------
    shifts : array_like
        Two of 2D arrays of zeros, initialized at CPU.
    ref_mat : array_like
        3D array. Reference image.
    mat : array_like
        3D array. The 2nd image. Must be the same size as the reference image.
    coef_4d : array_like
        4D array of zeros, initialized at CPU.
    height : int
        Height of the image.
    width : int
        Width of the image.
    radi : int
        Radius of the window to select a local area of the image.
    margin : int
        To define the size of the area of the reference image for sliding.

    Returns
    -------

        Update of the (y,x)-shifting images passed from CPU.
    """
    (x_index, y_index) = cuda.grid(2)
    if (y_index < height) and (x_index < width):
        radi_ref = margin + radi
        j = x_index + radi_ref
        i = y_index + radi_ref
        ref_mat1 = ref_mat[:, i - radi_ref:i + radi_ref + 1,
                   j - radi_ref: j + radi_ref + 1]
        mat1 = mat[:, i - radi:i + radi + 1, j - radi: j + radi + 1]
        coef_mat1 = coef_4d[y_index, x_index, :, :]
        coef_mat1 = __gen_2d_corr_map_3d_input(ref_mat1, mat1, coef_mat1)
        x_max, y_max = __locate_max_value(coef_mat1)
        x_pos, y_pos = __locate_2d_peak_kernel(coef_mat1, x_max, y_max)
        shifts[0, y_index, x_index] = x_pos - margin
        shifts[1, y_index, x_index] = y_pos - margin


def _get_2d_shift_multi_rows_3d_input_gpu(ref_mat, mat, win_size=7, margin=10,
                                          block=(16, 16), pad=True,
                                          norm=False):
    """
    GPU function to find local (y,x)-shifts of the second 3d-image against the
    reference 3d-image where their shapes are the same. Each (y,x)-shifting
    value is associated with a pixel-location of the second image. The value is
    determined by selecting a small volume of the second image, defined by
    the 'win_size' parameter, and sliding (y,x only) over a slightly larger
    volume of the reference image, defined by the 'win_size' and 'margin'
    parameters. This function may be computational expensive.

    Parameters
    ----------
    ref_mat : array_like
        3D array. Reference image.
    mat : array_like
        3D array. The 2nd image. Must be the same size as the reference image.
    win_size : int
        To define the size of the image volume around the selected pixel of the
        2nd image.
    margin : int
        To define the size of the selected volume of the reference image for
        sliding i.e. size = 2 * margin + win_size
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (4,4), (8, 8), ...
    pad : bool, optional
        Padding the result to the same as the original images.
    norm : bool, optional
        Normalizing the inputs.

    Returns
    -------
    list of two 2d-arrays
        x-shift array and y-shift array, corresponding to the pixel-locations
        in the 2nd image where the starting location is at
        (margin + win_size/2, margin + win_size/2). Using the pad option to
        make it easier to find which value corresponding to
        which pixel-location.
    """
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) != 3:
        raise ValueError("Inputs must be 3d-arrays !!!")
    (height, width) = ref_mat.shape[-2:]
    win_size = 2 * (win_size // 2) + 1
    radi = win_size // 2
    edge, size = radi + margin, 2 * margin + 1
    height1, width1 = height - 2 * edge, width - 2 * edge
    if width1 < 1 or height1 < 1:
        if width1 < 1 or height1 < 1:
            raise ValueError("Shapes of the inputs {0} are smaller than the "
                             "requested size (win_size + 2*margin) = "
                             "{1}".format(ref_mat.shape, edge))
    if norm:
        ref_mat = normalize_image(ref_mat)
        mat = normalize_image(mat)
    shifts = np.zeros((2, height1, width1), dtype=np.float32)
    grid = (int(np.ceil(1.0 * width1 / block[0])),
            int(np.ceil(1.0 * height1 / block[1])))
    f_alias = _get_2d_shift_multi_rows_3d_input_kernel
    ref_mat1 = np.float32(np.ascontiguousarray(ref_mat))
    mat1 = np.float32(np.ascontiguousarray(mat))
    coef_4d = np.zeros((height1, width1, size, size), dtype=np.float32)
    f_alias[grid, block](shifts, ref_mat1, mat1, coef_4d, np.int32(height1),
                         np.int32(width1), np.int32(radi), np.int32(margin))
    if pad:
        shifts = np.pad(shifts, ((0, 0), (edge, edge), (edge, edge)),
                        mode="constant")
    return shifts[0], shifts[1]


def _get_2d_shift_full_image_3d_input_gpu(ref_mat, mat, chunk_size=None,
                                          win_size=7, margin=10,
                                          block=(16, 16), norm=False,
                                          norm_global=False):
    """
    GPU function to find local (y,x)-shifts of the second 3d-image against the
    reference 3d-image where their shapes are the same and their size may be
    too big for memory. Each (y,x)-shifting value is associated with a
    pixel-location of the second image. The value is determined by selecting a
    small volume of the second image, defined by the 'win_size' parameter, and
    sliding (y,x only) over a slightly larger volume of the reference image,
    defined by the 'win_size' and 'margin' parameters. This function may be
    computational expensive.

    Parameters
    ----------
    ref_mat : array_like
        3D array, can be a numpy array or hdf object. Reference image.
    mat : array_like
        3D array, can be a numpy array or hdf object. The 2nd image. Must be
        the same size as the reference image.
    chunk_size : int or None
        Size of each chunk extracted along the height of the image.
    win_size : int
        To define the size of the area around the selected pixel of the 2nd
        image.
    margin : int
        To define the size of the area of the reference image for
        sliding, i.e. size = 2 * margin + win_size
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (4,4), (8, 8), ...
    norm : bool, optional
        Normalize the input images if True.
    norm_global : bool, optional
        Normalize by using the full size of the inputs if True.

    Returns
    -------
    list of two 2d-arrays
        x-shift array and y-shift array. Zeros at the outer area of the size of
        (margin + win_size // 2).
    """
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) != 3:
        raise ValueError("Inputs must be 3d-arrays !!!")
    (height, width) = ref_mat.shape[-2:]
    if chunk_size is None:
        chunk_size = height + 1
    else:
        chunk_size = np.clip(chunk_size, 1, height)
    num_chunk = np.clip(height // chunk_size + 1, 1, height)
    edge = margin + win_size // 2
    if height < (2 * edge + 1):
        raise ValueError("Height of the inputs {0} are smaller than the "
                         "requested size (win_size + 2*margin + 1) = "
                         "{1} x {1}".format(height, 2 * edge + 1))
    list_index = np.array_split(np.arange(edge, height - edge), num_chunk)
    x_shifts = np.zeros((height, width), dtype=np.float32)
    y_shifts = np.zeros((height, width), dtype=np.float32)
    f_alias = _get_2d_shift_multi_rows_3d_input_gpu
    if norm_global:
        ref_mat = normalize_image(ref_mat)
        mat = normalize_image(mat)
        norm = False
    for pos in list_index:
        bindex, eindex = pos[0], pos[-1] + 1
        ref_mat1 = ref_mat[:, bindex - edge:eindex + edge, :]
        mat1 = mat[:, bindex - edge:eindex + edge, :]
        shift_chunk = f_alias(ref_mat1, mat1, win_size, margin, block,
                              False, norm)
        x_shifts[bindex:eindex, edge:-edge] = shift_chunk[0]
        y_shifts[bindex:eindex, edge:-edge] = shift_chunk[1]
    return x_shifts, y_shifts


@cuda.jit
def _generate_4d_correlation_map_3d_input_kernel(coef_4d, ref_mat, mat, height,
                                                 width, radi,
                                                 margin):  # pragma: no cover
    """
    GPU-CPU function to calculate 2D correlation maps for each pixel location
    in height and width -> 4D array.

    Parameters
    ----------
    coef_4d : array_like
        4D array of zeros, initialized at CPU.
    ref_mat : array_like
        3D array. Reference image.
    mat : array_like
        3D array. The 2nd image. Must be the same size as the reference image.
    height : int
        Height of the image.
    width : int
        Width of the image.
    radi : int
        Radius of the window to select a local area of the image.
    margin : int
        To define the size of the area of the reference image for sliding.

    Returns
    -------

        Updated of 4D correlation map.
    """
    (x_index, y_index) = cuda.grid(2)
    if (y_index < height) and (x_index < width):
        radi_ref = margin + radi
        size = 2 * margin + 1
        j = x_index + radi_ref
        i = y_index + radi_ref
        ref_mat1 = ref_mat[:, i - radi_ref:i + radi_ref + 1,
                   j - radi_ref: j + radi_ref + 1]
        mat1 = mat[:, i - radi:i + radi + 1, j - radi: j + radi + 1]
        coef_mat1 = coef_4d[y_index, x_index, :, :]
        coef_mat1 = __gen_2d_corr_map_3d_input(ref_mat1, mat1, coef_mat1)
        for i in range(size):
            for j in range(size):
                coef_4d[y_index, x_index, i, j] = coef_mat1[i, j]


def _get_2d_shift_multi_rows_3d_input_cpu_gpu(ref_mat, mat, win_size=7,
                                              margin=10, method="diff", size=3,
                                              block=(16, 16), pad=True,
                                              ncore=None, norm=False):
    """
    Combine GPU and CPU to find local (y,x)-shifts of the second 3d-image
    against the reference 3d-image where their shapes are the same.
    Each (y,x)-shifting value is associated with a pixel-location of the second
    image. The value is determined by selecting a small volume of the second
    image, defined by the 'win_size' parameter, and sliding (y,x only) over a
    slightly larger volume of the reference image, defined by the 'win_size'
    and 'margin' parameters. This function may be computational expensive.

    Parameters
    ----------
    ref_mat : array_like
        3D array. Reference image.
    mat : array_like
        3D array. The 2nd image. Must be the same size as the reference image.
    win_size : int
        To define the size of the image volume around the selected pixel of the
        2nd image.
    margin : int
        To define the size of the selected volume of the reference image for
        sliding i.e. size = 2 * margin + win_size
    method : {"diff", "poly_fit"}
        Method for finding sub-pixel shift. Two options: a differential
        method (Ref. [1]) or a polynomial method (Ref. [2]).
    size : int
        Window size around the integer location of the maximum value used for
        sub-pixel searching.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (4,4), (8, 8), ...
    pad : bool, optional
        Padding the result to the same as the original images.
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    norm : bool, optional
        Normalizing the inputs.

    Returns
    -------
    list of two 2d-arrays
        x-shift array and y-shift array, corresponding to the pixel-locations
        in the 2nd image where the starting location is at
        (margin + win_size/2, margin + win_size/2). Using the pad option to
        make it easier to find which value corresponding to
        which pixel-location.

    References
    ----------
    [1] : https://doi.org/10.48550/arXiv.0712.4289

    [2] : https://doi.org/10.1088/0957-0233/17/6/045
    """
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) != 3:
        raise ValueError("Inputs must be 3d-arrays !!!")
    (height, width) = ref_mat.shape[-2:]
    win_size = 2 * (win_size // 2) + 1
    radi = win_size // 2
    edge, size_mag = radi + margin, 2 * margin + 1
    height1, width1 = height - 2 * edge, width - 2 * edge
    if width1 < 1 or height1 < 1:
        if width1 < 1 or height1 < 1:
            raise ValueError("Shapes of the inputs {0} are smaller than the "
                             "requested size (win_size + 2*margin) = "
                             "{1}".format(ref_mat.shape, edge))
    if norm:
        ref_mat = normalize_image(ref_mat)
        mat = normalize_image(mat)
    grid = (int(np.ceil(1.0 * width1 / block[0])),
            int(np.ceil(1.0 * height1 / block[1])))
    f_alias = _generate_4d_correlation_map_3d_input_kernel
    ref_mat1 = np.float32(np.ascontiguousarray(ref_mat))
    mat1 = np.float32(np.ascontiguousarray(mat))
    coef_4d = np.zeros((height1, width1, size_mag, size_mag), dtype=np.float32)
    f_alias[grid, block](coef_4d, ref_mat1, mat1, np.int32(height1),
                         np.int32(width1), np.int32(radi), np.int32(margin))
    if ncore is None:
        ncore = mp.cpu_count() - 1
    shifts = np.asarray(Parallel(n_jobs=ncore)(
        delayed(locate_peak)(coef_4d[i, j, :, :], True, method, 2, size)
        for i in range(height1) for j in range(width1)))
    shifts = np.moveaxis(np.reshape(
        np.asarray(shifts), (height1, width1, 2)), 2, 0) - margin
    if pad:
        shifts = np.pad(shifts, ((0, 0), (edge, edge), (edge, edge)),
                        mode="constant")
    return shifts[0], shifts[1]


def _get_2d_shift_full_image_3d_input_cpu_gpu(ref_mat, mat, chunk_size=None,
                                              win_size=7, margin=10,
                                              method="diff", size=3,
                                              block=(16, 16), ncore=None,
                                              norm=False, norm_global=False):
    """
    Combine GPU and CPU to find local (y,x)-shifts of the second 3d-image
    against the reference 3d-image where their shapes are the same and their
    size may be too big for memory. Each (y,x)-shifting value is
    associated with a pixel-location of the second image. The value is
    determined by selecting a small volume of the second image, defined by
    the 'win_size' parameter, and sliding (y,x only) over a slightly larger
    volume of the reference image, defined by the 'win_size' and 'margin'
    parameters. This function may be computational expensive.

    Parameters
    ----------
    ref_mat : array_like
        3D array, can be a numpy array or hdf object. Reference image.
    mat : array_like
        3D array, can be a numpy array or hdf object. The 2nd image. Must be
        the same size as the reference image.
    chunk_size : int or None
        Size of each chunk extracted along the height of the image.
    win_size : int
        To define the size of the area around the selected pixel of the 2nd
        image.
    margin : int
        To define the size of the area of the reference image for
        sliding, i.e. size = 2 * margin + win_size
    method : {"diff", "poly_fit"}
        Method for finding sub-pixel shift. Two options: a differential
        method (Ref. [1]) or a polynomial method (Ref. [2]).
    size : int
        Window size around the integer location of the maximum value used for
        sub-pixel searching.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (4,4), (8, 8), ...
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    norm : bool, optional
        Normalizing the inputs if True.
    norm_global : bool, optional
        Normalize by using the full size of the inputs if True.

    Returns
    -------
    list of two 2d-arrays
        x-shift array and y-shift array. Zeros at the outer area of the size of
        (margin + win_size // 2).

    References
    ----------
    [1] : https://doi.org/10.48550/arXiv.0712.4289

    [2] : https://doi.org/10.1088/0957-0233/17/6/045
    """
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) != 3:
        raise ValueError("Inputs must be 3d-arrays !!!")
    (height, width) = ref_mat.shape[-2:]
    if chunk_size is None:
        chunk_size = height + 1
    else:
        chunk_size = np.clip(chunk_size, 1, height)
    num_chunk = np.clip(height // chunk_size + 1, 1, height)
    edge = margin + win_size // 2
    if height < (2 * edge + 1):
        raise ValueError("Height of the inputs {0} are smaller than the "
                         "requested size (win_size + 2*margin + 1) = "
                         "{1} x {1}".format(height, 2 * edge + 1))
    list_index = np.array_split(np.arange(edge, height - edge), num_chunk)
    x_shifts = np.zeros((height, width), dtype=np.float32)
    y_shifts = np.zeros((height, width), dtype=np.float32)
    f_alias = _get_2d_shift_multi_rows_3d_input_cpu_gpu
    if norm_global:
        ref_mat = normalize_image(ref_mat)
        mat = normalize_image(mat)
        norm = False
    for pos in list_index:
        bindex, eindex = pos[0], pos[-1] + 1
        ref_mat1 = ref_mat[:, bindex - edge:eindex + edge, :]
        mat1 = mat[:, bindex - edge:eindex + edge, :]
        shift_chunk = f_alias(ref_mat1, mat1, win_size, margin, method, size,
                              block, False, ncore, norm)
        x_shifts[bindex:eindex, edge:-edge] = shift_chunk[0]
        y_shifts[bindex:eindex, edge:-edge] = shift_chunk[1]
    return x_shifts, y_shifts


def find_local_shifts(ref_mat, mat, dim=1, win_size=7, margin=10,
                      method="diff", size=3, gpu=False, block=(16, 16),
                      ncore=None, norm=True, norm_global=False,
                      chunk_size=100):
    """
    To find local shifts (in x and y direction) between two images by selecting
    a small area/volume of the second image and sliding over a slightly larger
    area/volume of the reference image.

    Parameters
    ----------
    ref_mat : array_like
        2D/3D array, can be a numpy array or hdf object. Reference image.
    mat : array_like
        2D/3D array, can be a numpy array or hdf object. The second image, must
        be the same size as the reference image.
    dim : {1, 2}
        To find the shifts (in x and y) separately or together.
    win_size : int
        Size of local areas in the second image.
    margin : int
        To define the sliding range of the second image.
    method : {"diff", "poly_fit"}
        Method for finding sub-pixel shift. Two options: a differential
        method (Ref. [1]) or a polynomial method (Ref. [2]). The "poly_fit"
        option is not available if using GPU.
    size : int
        Window size around the integer location of the maximum value used for
        sub-pixel location. Adjustable if using the polynomial method.
    gpu : {False, True, "hybrid"}
        Use GPU for computing if True or in "hybrid" mode.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (8, 8), (16, 16), (32, 32), ...
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    norm : bool, optional
        Normalizing the inputs if True.
    norm_global : bool, optional
        Normalize by using the full size of the inputs if True.
    chunk_size : int or None
        Size of each chunk extracted along the height of the image.

    Returns
    -------
    list of two 2d-arrays
        x-shift array and y-shift array. Zeros at the outer area of the size of
        (margin + win_size // 2).

    References
    ----------
    [1] : https://doi.org/10.48550/arXiv.0712.4289

    [2] : https://doi.org/10.1088/0957-0233/17/6/045
    """
    if gpu is True:
        if cuda.is_available() is False:
            warnings.warn("!!!No Nvidia GPU found!!!Run with CPU instead!!!")
            gpu = False
    if len(ref_mat.shape) == 2:
        if gpu is False:
            f_alias = _get_2d_shift_full_image_2d_input
            (x_shifts, y_shifts) = f_alias(ref_mat, mat, win_size=win_size,
                                           margin=margin, sub_pixel=True,
                                           method=method, size=size, gpu=False,
                                           ncore=ncore, norm=norm)
        else:
            f_alias = _get_2d_shift_full_image_2d_input_gpu
            (x_shifts, y_shifts) = f_alias(ref_mat, mat, chunk_size=chunk_size,
                                           win_size=win_size, margin=margin,
                                           block=block, norm=norm)
    else:
        if gpu is False:
            if dim == 1:
                f_alias = _get_1d_shift_full_image_3d_input_cpu
                x_shifts = f_alias(ref_mat, mat, direction="x",
                                   chunk_size=chunk_size, win_size=win_size,
                                   margin=margin, sub_pixel=True,
                                   method=method, size=size, ncore=ncore,
                                   norm=norm, norm_global=norm_global)
                y_shifts = f_alias(ref_mat, mat, direction="y",
                                   chunk_size=chunk_size, win_size=win_size,
                                   margin=margin, sub_pixel=True,
                                   method=method, size=size, ncore=ncore,
                                   norm=norm, norm_global=norm_global)
            else:
                f_alias = _get_2d_shift_full_image_3d_input_cpu
                (x_shifts, y_shifts) = f_alias(ref_mat, mat,
                                               chunk_size=chunk_size,
                                               win_size=win_size,
                                               margin=margin,
                                               sub_pixel=True, method=method,
                                               size=size, ncore=ncore,
                                               norm=norm,
                                               norm_global=norm_global)
        elif gpu == "Hybrid" or gpu == "hybrid":
            f_alias = _get_2d_shift_full_image_3d_input_cpu_gpu
            (x_shifts, y_shifts) = f_alias(ref_mat, mat, chunk_size=chunk_size,
                                           win_size=win_size, margin=margin,
                                           method=method, size=size,
                                           block=block, ncore=ncore,
                                           norm=norm, norm_global=norm_global)
        else:
            if dim == 1:
                f_alias = _get_1d_shift_full_image_3d_input_gpu
                x_shifts = f_alias(ref_mat, mat, direction="x",
                                   chunk_size=chunk_size, win_size=win_size,
                                   margin=margin, block=block, norm=norm,
                                   norm_global=norm_global)
                y_shifts = f_alias(ref_mat, mat, direction="y",
                                   chunk_size=chunk_size, win_size=win_size,
                                   margin=margin, block=block, norm=norm,
                                   norm_global=norm_global)
            else:
                f_alias = _get_2d_shift_full_image_3d_input_gpu
                (x_shifts, y_shifts) = f_alias(ref_mat, mat,
                                               chunk_size=chunk_size,
                                               win_size=win_size,
                                               margin=margin,
                                               block=block, norm=norm,
                                               norm_global=norm_global)
    return np.float32(x_shifts), np.float32(y_shifts)


def __get_input_list(list_ij, height, width, gap):
    """
    Supplementary method for methods of finding global_shift based on
    local_shifts.
    """
    input_format = "Please use the format of [i, j] for a single " \
                   "point or [list_i, list_j] for multiple points"
    box = [[gap, height - gap], [gap, width - gap]]
    msg = "The given list of points are not inside the box: {}".format(box)
    if isinstance(list_ij, list) or isinstance(list_ij, tuple) or \
            isinstance(list_ij, np.ndarray):
        if len(list_ij) != 2:
            raise ValueError(input_format)
        else:
            list_i = np.int32(np.asarray(list_ij[0]))
            list_j = np.int32(np.asarray(list_ij[1]))
            if isinstance(list_i, np.integer) and \
                    isinstance(list_j, np.integer):
                if ((height - gap) > list_i > gap) and \
                        ((width - gap) > list_j > gap):
                    list_i = [list_i]
                    list_j = [list_j]
                else:
                    raise ValueError(msg)
            else:
                if isinstance(list_i, np.integer) or \
                        isinstance(list_j, np.integer):
                    raise ValueError(input_format)
                else:
                    if len(list_i) != len(list_j):
                        raise ValueError("Size of each list of indices is not "
                                         "the same")
                    else:
                        i_tmp, j_tmp = [], []
                        for k in range(len(list_i)):
                            if ((height - gap) > list_i[k] > gap) and \
                                    ((width - gap) > list_j[k] > gap):
                                i_tmp.append(list_i[k])
                                j_tmp.append(list_j[k])
                        if len(i_tmp) == 0:
                            raise ValueError(msg)
                        else:
                            list_i, list_j = i_tmp, j_tmp
    else:
        raise ValueError(input_format)
    return list_i, list_j


def _find_global_shift_based_local_shifts_cpu(ref_mat, mat, win_size, margin,
                                              list_ij=None, num_point=None,
                                              global_value="mixed",
                                              sub_pixel=True, method="diff",
                                              size=3, ncore=None, norm=False,
                                              return_list=False):
    """
    CPU function to find global shift between two images based on finding
    local shifts.

    Parameters
    ----------
    ref_mat : array_like
        2D array. Reference image.
    mat : array_like
        2D array. The 2nd image. Must be the same size as the reference image.
    win_size : int
        To define the size of the area around the selected pixel of the 2nd
        image. E.g. 41, 61, ..
    margin : int
        To define the searching range (in pixel) for finding shift.
        E.g. 20, 30,...
    list_ij : list of lists of int or None
        List of indices of points used for local search. Accept the value of
        [i_index, j_index] for a single point or
        [[i_index0, i_index1,...], [j_index0, j_index1,...]]
        for multiple points. Automatically generated if None.
    num_point : int or None
        Number of points used for local search if list_ij is None.
    global_value : {"median", "mean", "mixed"}
        Method for calculating the global value from local values.
    sub_pixel : bool, optional
        Enable sub-pixel location.
    method : {"diff", "poly_fit"}
        Method for finding 1d sub-pixel position. Two options: a differential
        method (Ref. [1]) or a polynomial method (Ref. [2]).
    size : int, optional
        Window size around the integer location of the maximum value used for
        sub-pixel searching.
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    norm : bool, optional
        Normalize the input images if True.
    return_list : bool
        Return all local values if True.

    Returns
    -------
    float or list of float
        Shift in x-direction. Return a list of float if return_list is True.
    float or list of float
        Shift in y-direction. Return a list of float if return_list is True.

    References
    ----------
    [1] : https://doi.org/10.48550/arXiv.0712.4289

    [2] : https://doi.org/10.1088/0957-0233/17/6/045
    """
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) != 2:
        raise ValueError("Inputs must be 2d-arrays !!!")
    (height, width) = ref_mat.shape
    win_size = 2 * (win_size // 2) + 1
    radi = win_size // 2
    start = radi + margin
    als_size = win_size + 2 * margin
    if width < als_size or height < als_size:
        raise ValueError("Shapes of the inputs {0} are smaller than the "
                         "requested size (win_size + 2*margin) = "
                         "{1}".format(ref_mat.shape, als_size))
    start1, radi1 = start + 1, radi + 1
    if norm:
        ref_mat = normalize_image(ref_mat)
        mat = normalize_image(mat)
    if num_point is None:
        num_point = 30
    if list_ij is None:
        list_i = np.random.randint(start, height - start, size=num_point)
        list_j = np.random.randint(start, width - start, size=num_point)
    else:
        list_i, list_j = __get_input_list(list_ij, height, width, start)
        num_point = len(list_i)
    f_alias = find_shift_based_correlation_map
    if ncore == 1:
        shifts = [f_alias(ref_mat[list_i[k] - start:list_i[k] + start1,
                          list_j[k] - start:list_j[k] + start1],
                          mat[list_i[k] - radi:list_i[k] + radi1,
                          list_j[k] - radi:list_j[k] + radi1],
                          margin, None, sub_pixel, method, 2, size,
                          False, None) for k in range(num_point)]
        shifts = np.asarray(shifts)
    else:
        if ncore is None:
            ncore = np.clip(mp.cpu_count() - 1, 1, None)
        shifts = np.asarray(Parallel(n_jobs=ncore)(
            delayed(f_alias)(
                ref_mat[list_i[k] - start:list_i[k] + start1,
                list_j[k] - start:list_j[k] + start1],
                mat[list_i[k] - radi:list_i[k] + radi1,
                list_j[k] - radi:list_j[k] + radi1],
                margin, None, sub_pixel, method, 2, size, False, None)
            for k in range(num_point)))
    x_shifts, y_shifts = shifts[:, 0], shifts[:, 1]
    if return_list:
        return x_shifts, y_shifts
    else:
        if global_value == "median":
            global_xshift = np.median(x_shifts)
            global_yshift = np.median(y_shifts)
        elif global_value == "mixed":
            mid = num_point // 2
            begin = np.clip(mid - 3, 0, None)
            end = np.clip(mid + 3, mid, num_point)
            if end > begin:
                global_xshift = np.mean(np.sort(x_shifts)[begin:end])
                global_yshift = np.mean(np.sort(y_shifts)[begin:end])
            else:
                global_xshift = np.median(x_shifts)
                global_yshift = np.median(y_shifts)
        else:
            if num_point > 7:
                global_xshift = np.mean(np.sort(x_shifts)[2:-2])
                global_yshift = np.mean(np.sort(y_shifts)[2:-2])
            else:
                global_xshift = np.mean(x_shifts)
                global_yshift = np.mean(y_shifts)
        return global_xshift, global_yshift


@cuda.jit
def _get_local_shifts_gpu_kernel(shifts, ref_mat, mat, list_i, list_j,
                                 list_2d_coef, radi, margin,
                                 num_point):  # pragma: no cover
    """
    GPU-CPU function to find local (y,x)-shifts of the second image against
    the reference image.

    Parameters
    ----------
    shifts : array_like
        2D array of zeros, initialized at CPU.
    ref_mat : array_like
        2D array. Reference image.
    list_i : array_like
        1D array. i-index of points using for search.
    list_j : array_like
        1D array. j-index of points using for search.
    list_2d_coef : array_like
        List of 2D array of zeros, initialized at CPU.
    radi : int
        Radius of the window to select a local area of the image.
    margin : int
        To define the size of the area of the reference image for sliding.
    num_point : int
        Number of points using for search.

    Returns
    -------

        Update of the (y,x)-shifting images passed from CPU.
    """
    idx = cuda.grid(1)
    if idx < num_point:
        radi_ref = radi + margin
        i = list_i[idx]
        j = list_j[idx]
        ref_mat1 = ref_mat[i - radi_ref:i + radi_ref + 1,
                   j - radi_ref: j + radi_ref + 1]
        mat1 = mat[i - radi:i + radi + 1, j - radi: j + radi + 1]
        coef_mat1 = list_2d_coef[idx, :, :]
        coef_mat1 = __gen_2d_corr_map_2d_input(ref_mat1, mat1, coef_mat1)
        x_max, y_max = __locate_max_value(coef_mat1)
        x_pos, y_pos = __locate_2d_peak_kernel(coef_mat1, x_max, y_max)
        shifts[0, idx] = x_pos - margin
        shifts[1, idx] = y_pos - margin


def _find_global_shift_based_local_shifts_gpu(ref_mat, mat, win_size, margin,
                                              list_ij=None, num_point=None,
                                              global_value="mixed", block=32,
                                              norm=False, return_list=False):
    """
    GPU function to find global shift between two images based on finding
    local shifts.

    Parameters
    ----------
    ref_mat : array_like
        2D array. Reference image.
    mat : array_like
        2D array. The 2nd image. Must be the same size as the reference image.
    win_size : int
        To define the size of the area around the selected pixel of the 2nd
        image. E.g. 41, 61, ..
    margin : int
        To define the searching range (in pixel) for finding shift.
        E.g. 20, 30,...
    list_ij : list of lists of int or None
        List of indices of points used for local search. Accept the value of
        [i_index, j_index] for a single point or
        [[i_index0, i_index1,...], [j_index0, j_index1,...]]
        for multiple points. Automatically generated if None.
    num_point : int or None
        Number of points used for local search if list_ij is None.
    global_value : {"median", "mean", "mixed"}
        Method for calculating the global value from local values.
    block : int
        Size of a GPU block. E.g. 16, 32, 64, ...
    norm : bool, optional
        Normalize the input images if True.
    return_list : bool
        Return all local values if True.

    Returns
    -------
    float or list of float
        Shift in x-direction. Return a list of float if return_list is True.
    float or list of float
        Shift in y-direction. Return a list of float if return_list is True.
    """
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) != 2:
        raise ValueError("Inputs must be 2d-arrays !!!")
    (height, width) = ref_mat.shape
    win_size = 2 * (win_size // 2) + 1
    radi = win_size // 2
    edge, size = radi + margin, 2 * margin + 1
    height1, width1 = height - 2 * edge, width - 2 * edge
    if width1 < 1 or height1 < 1:
        raise ValueError("Shapes of the inputs {0} are smaller than the "
                         "requested size (win_size + 2*margin) = "
                         "{1}".format(ref_mat.shape, edge))
    if norm:
        ref_mat = normalize_image(ref_mat)
        mat = normalize_image(mat)
    if num_point is None:
        num_point = 30
    if list_ij is None:
        list_i = np.random.randint(edge, height - edge, size=num_point)
        list_j = np.random.randint(edge, width - edge, size=num_point)
    else:
        list_i, list_j = __get_input_list(list_ij, height, width, edge)
        num_point = len(list_i)
    shifts = np.zeros((2, num_point), dtype=np.float32)
    grid = int(np.ceil(1.0 * num_point / block))
    f_alias = _get_local_shifts_gpu_kernel
    ref_mat1 = np.float32(np.ascontiguousarray(ref_mat))
    mat1 = np.float32(np.ascontiguousarray(mat))
    list_2d_coef = np.zeros((num_point, size, size), dtype=np.float32)
    list_i, list_j = np.int32(list_i), np.int32(list_j)
    f_alias[grid, block](shifts, ref_mat1, mat1, list_i, list_j, list_2d_coef,
                         np.int32(radi), np.int32(margin), np.int32(num_point))
    x_shifts, y_shifts = shifts[0], shifts[1]
    if return_list:
        return x_shifts, y_shifts
    else:
        if global_value == "median":
            global_xshift = np.median(x_shifts)
            global_yshift = np.median(y_shifts)
        elif global_value == "mixed":
            mid = num_point // 2
            begin = np.clip(mid - 3, 0, None)
            end = np.clip(mid + 3, mid, num_point)
            if end > begin:
                global_xshift = np.mean(np.sort(x_shifts)[begin:end])
                global_yshift = np.mean(np.sort(y_shifts)[begin:end])
            else:
                global_xshift = np.median(x_shifts)
                global_yshift = np.median(y_shifts)
        else:
            if num_point > 7:
                global_xshift = np.mean(np.sort(x_shifts)[2:-2])
                global_yshift = np.mean(np.sort(y_shifts)[2:-2])
            else:
                global_xshift = np.mean(x_shifts)
                global_yshift = np.mean(y_shifts)
        return global_xshift, global_yshift


def find_global_shift_based_local_shifts(ref_mat, mat, win_size, margin,
                                         list_ij=None, num_point=None,
                                         global_value="mixed", gpu=False,
                                         block=32, sub_pixel=True,
                                         method="diff", size=3, ncore=None,
                                         norm=False, return_list=False):
    """
    Find global shift between two images based on finding local shifts.

    Parameters
    ----------
    ref_mat : array_like
        2D array. Reference image.
    mat : array_like
        2D array. The 2nd image. Must be the same size as the reference image.
    win_size : int
        To define the size of the area around the selected pixel of the 2nd
        image. E.g. 41, 61, ..
    margin : int
        To define the searching range (in pixel) for finding shift.
        E.g. 20, 30,...
    list_ij : list of lists of int or None
        List of indices of points used for local search. Accept the value of
        [i_index, j_index] for a single point or
        [[i_index0, i_index1,...], [j_index0, j_index1,...]]
        for multiple points. Automatically generated if None.
    num_point : int or None
        Number of points used for local search if list_ij is None.
    global_value : {"median", "mean", "mixed"}
        Method for calculating the global value from local values.
    gpu : bool, optional
        Use GPU for computing if True. If win_size and image size is large
        (e.g. > 201 x 2k x 2k), using CPU may be better.
    block : int, optional
        Size of a GPU block if using GPU. E.g. 16, 32, 64, ...
    sub_pixel : bool, optional
        Enable sub-pixel location.
    method : {"diff", "poly_fit"}
        Method for finding sub-pixel shift. Two options: a differential
        method (Ref. [1]) or a polynomial method (Ref. [2]). The "poly_fit"
        option is not available if using GPU.
    size : int, optional
        Window size around the integer location of the maximum value used for
        sub-pixel searching.
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    norm : bool, optional
        Normalize the input images if True.
    return_list : bool
        Return all local values if True.

    Returns
    -------
    float or list of float
        Shift in x-direction. Return a list of float if return_list is True.
    float or list of float
        Shift in y-direction. Return a list of float if return_list is True.

    References
    ----------
    [1] : https://doi.org/10.48550/arXiv.0712.4289

    [2] : https://doi.org/10.1088/0957-0233/17/6/045
    """

    if gpu is True:
        if cuda.is_available() is False:
            warnings.warn("!!!No Nvidia GPU found!!!Run with CPU instead!!!")
            gpu = False
    if gpu is True:
        f_alias = _find_global_shift_based_local_shifts_gpu
        x_out, y_out = f_alias(ref_mat, mat, win_size, margin, list_ij=list_ij,
                               num_point=num_point, global_value=global_value,
                               block=block, norm=norm, return_list=return_list)
    else:
        f_alias = _find_global_shift_based_local_shifts_cpu
        x_out, y_out = f_alias(ref_mat, mat, win_size, margin, list_ij=list_ij,
                               num_point=num_point, global_value=global_value,
                               sub_pixel=sub_pixel, method=method, size=size,
                               ncore=ncore, norm=norm, return_list=return_list)
    return x_out, y_out


def __calc_shift_umpa(ref_mat, mat, window, margin, t1, t3, t2, t4, t6,
                      dark_signal=False, method="poly_fit", size=3):
    """
    Supplementary CPU-function for finding local shifts using the UMPA
    approach.
    """
    t5 = np.zeros((2 * margin + 1, 2 * margin + 1), dtype=np.float32)
    for i in range(len(ref_mat)):
        t5 += correlate(ref_mat[i], window * mat[i], mode='valid')
    if dark_signal:
        mat_tmp = (t2 * t3 - t6 ** 2)
        K = (t2 * t5 - t4 * t6) / mat_tmp
        beta = (t3 * t4 - t5 * t6) / mat_tmp
        A = beta + K
        V = K / A
    else:
        K = t5 / t3
        beta = 0.0
    D = t1 + (beta ** 2) * t2 + (K ** 2) * t3 \
        - 2 * beta * t4 - 2 * K * t5 + 2 * beta * K * t6
    x_pos, y_pos = locate_peak(D, sub_pixel=True, method=method, dim=2,
                               size=size, max_peak=False)
    j, i = int(np.round(x_pos)), int(np.round(y_pos))
    x_shift, y_shift = x_pos - margin, y_pos - margin
    if dark_signal:
        return x_shift, y_shift, np.abs(A[i, j]), np.abs(V[i, j])
    else:
        return x_shift, y_shift


def __get_2d_shift_multi_rows_3d_input_umpa_cpu(ref_mat, mat, win_size, margin,
                                                window, L1, L3, L2, L4, L6,
                                                method="poly_fit", size=3,
                                                ncore=None, dark_signal=False):
    """
    Supplementary CPU-function for finding local shifts using the UMPA
    approach.
    """
    (height, width) = ref_mat.shape[-2:]
    win_size = 2 * (win_size // 2) + 1
    radi = win_size // 2
    start = radi + margin
    stop_col, stop_row = width - start, height - start
    start1, radi1, margin1 = start + 1, radi + 1, margin + 1
    f_alias = __calc_shift_umpa
    if ncore is None:
        ncore = np.clip(mp.cpu_count() - 1, 1, None)
    if dark_signal:
        results = np.asarray(
            Parallel(n_jobs=ncore)(delayed(f_alias)(
                ref_mat[:, i - start:i + start1, j - start:j + start1],
                mat[:, i - radi:i + radi1, j - radi:j + radi1], window, margin,
                L1[i, j], L3[i - margin:i + margin1, j - margin:j + margin1],
                L2, L4[i, j],
                L6[i - margin:i + margin1, j - margin:j + margin1],
                dark_signal, method=method, size=size)
                                   for i in range(start, stop_row)
                                   for j in range(start, stop_col)))
        results = np.reshape(np.asarray(results),
                             (stop_row - start, stop_col - start, 4))
        x_shifts, y_shifts = results[:, :, 0], results[:, :, 1]
        trans, dark = results[:, :, 2], results[:, :, 3]
    else:
        results = np.asarray(
            Parallel(n_jobs=ncore)(delayed(f_alias)(
                ref_mat[:, i - start:i + start1, j - start:j + start1],
                mat[:, i - radi:i + radi1, j - radi:j + radi1], window, margin,
                L1[i, j], L3[i - margin:i + margin1, j - margin:j + margin1],
                0.0, 0.0, 0.0, dark_signal, method=method, size=size)
                                   for i in range(start, stop_row)
                                   for j in range(start, stop_col)))
        results = np.reshape(np.asarray(results),
                             (stop_row - start, stop_col - start, 2))
        x_shifts, y_shifts = results[:, :, 0], results[:, :, 1]
    if dark_signal:
        return x_shifts, y_shifts, trans, dark
    else:
        return x_shifts, y_shifts


@cuda.jit(device=True)
def __sum_multiply_no_norm_2d(ref_mat, mat, window):  # pragma: no cover
    """
    GPU-kernel function to calculate the sum of multiplies of 2d-arrays.

    Parameters
    ----------
    ref_mat : array_like
        2D array. The first image.
    mat : array_like
        2D array. The second image.
    window : array_like
        2D array. Smoothing window.

    Returns
    -------
    float
    """
    (height, width) = ref_mat.shape
    sum_mul = 0.0
    for i in range(height):
        for j in range(width):
            val = ref_mat[i, j] * mat[i, j] * window[i, j]
            sum_mul += val
    return sum_mul


@cuda.jit(device=True)
def __accu_correlate(ref_mat, mat, window, coef_mat):  # pragma: no cover
    """
    GPU-kernel function to calculate cross-correlation between two images.

    Parameters
    ----------
    ref_mat : array_like
        2D array. The first image.
    mat : array_like
        2D array. The second image.
    window : array_like
        2D array. Smoothing window.
    coef_mat : array_like
        2D array. Resulting map.

    Returns
    -------
    array_like
    """
    (height0, width0) = ref_mat.shape
    (height1, width1) = mat.shape
    height2, width2 = height0 - height1 + 1, width0 - width1 + 1
    # coef_mat is at GPU global memory with the size of (height2, width2)
    for i in range(height2):
        for j in range(width2):
            row0, row1 = i, i + height1
            col0, col1 = j, j + width1
            ref_mat1 = ref_mat[row0:row1, col0:col1]
            sum_mul = __sum_multiply_no_norm_2d(ref_mat1, mat, window)
            coef_mat[i, j] += sum_mul
    return coef_mat


@cuda.jit
def __calc_shift_umpa_gpu_kernel(shifts, trans, dark, ref_mat, mat, coef_4d,
                                 depth, height, width, radi, margin, window,
                                 L1, L3, L2, L4, L6, A0, V0, D0,
                                 get_dark):  # pragma: no cover
    """
    Supplementary GPU-CPU function for finding local shifts using the UMPA
    approach.
    """
    (x_index, y_index) = cuda.grid(2)
    if (y_index < height) and (x_index < width):
        radi_ref = margin + radi
        radi_ref1 = radi_ref + 1
        margin1 = margin + 1
        radi1 = radi + 1
        size = 2 * margin + 1
        j = x_index + radi_ref
        i = y_index + radi_ref
        ref_mat1 = ref_mat[:, i - radi_ref:i + radi_ref1,
                   j - radi_ref: j + radi_ref1]
        mat1 = mat[:, i - radi:i + radi1, j - radi: j + radi1]
        A = A0[y_index, x_index, :, :]
        V = V0[y_index, x_index, :, :]
        D = D0[y_index, x_index, :, :]
        t5 = coef_4d[y_index, x_index, :, :]
        for k in range(depth):
            __accu_correlate(ref_mat1[k], mat1[k], window, t5)
        t1 = L1[i, j]
        t3 = L3[i - margin:i + margin1, j - margin:j + margin1]
        t2, t4 = L2, L4[i, j]
        t6 = L6[i - margin:i + margin1, j - margin:j + margin1]
        for u in range(size):
            for v in range(size):
                if get_dark == 1:
                    num = t2 * t3[u, v] - t6[u, v] ** 2
                    K = (t2 * t5[u, v] - t4 * t6[u, v]) / num
                    beta = (t3[u, v] * t4 - t5[u, v] * t6[u, v]) / num
                    A[u, v] = beta + K
                    if A[u, v] != 0.0:
                        V[u, v] = K / A[u, v]
                    num1 = t1 + ((beta ** 2) * t2 + (K ** 2) * t3[u, v]
                                 - 2 * beta * t4 - 2 * K * t5[u, v]
                                 + 2 * beta * K * t6[u, v])
                else:
                    K = t5[u, v] / t3[u, v]
                    num1 = t1 + (K ** 2) * t3[u, v] - 2 * K * t5[u, v]
                D[u, v] = num1
        val_max = __get_max_value(D)
        D = __inverse_values(D, val_max)
        x_max, y_max = __locate_max_value(D)
        x_pos, y_pos = __locate_2d_peak_kernel(D, x_max, y_max)
        j1, i1 = int(round(x_pos)), int(round(y_pos))
        shifts[0, y_index, x_index] = x_pos - margin
        shifts[1, y_index, x_index] = y_pos - margin
        if get_dark == 1:
            trans[y_index, x_index] = abs(A[i1, j1])
            dark[y_index, x_index] = abs(V[i1, j1])


def __get_2d_shift_multi_rows_3d_input_umpa_gpu(ref_mat, mat, win_size, margin,
                                                window, L1, L3, L2, L4, L6,
                                                block=(16, 16),
                                                dark_signal=True):
    """
    Supplementary GPU-function for finding local shifts using the UMPA
    approach.
    """
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) != 3:
        raise ValueError("Inputs must be 3d-arrays !!!")
    (depth, height, width) = ref_mat.shape
    win_size = 2 * (win_size // 2) + 1
    radi = win_size // 2
    edge, size = radi + margin, 2 * margin + 1
    height1, width1 = height - 2 * edge, width - 2 * edge
    if width1 < 1 or height1 < 1:
        if width1 < 1 or height1 < 1:
            raise ValueError("Shapes of the inputs {0} are smaller than the "
                             "requested size (win_size + 2*margin) = "
                             "{1}".format(ref_mat[0].shape, edge))
    shifts = np.zeros((2, height1, width1), dtype=np.float32)
    trans = np.ones((height1, width1), dtype=np.float32)
    dark = np.ones_like(trans)
    ref_mat1 = np.float32(np.ascontiguousarray(ref_mat))
    mat1 = np.float32(np.ascontiguousarray(mat))
    coef_4d = np.zeros((height1, width1, size, size), dtype=np.float32)
    A0 = np.zeros((height1, width1, size, size), dtype=np.float32)
    V0 = np.zeros((height1, width1, size, size), dtype=np.float32)
    D0 = np.zeros((height1, width1, size, size), dtype=np.float32)
    L1, L3 = np.float32(L1), np.float32(L3)
    L2, L4, L6 = np.float32(L2), np.float32(L4), np.float32(L6)
    window = np.float32(window)
    grid = (int(np.ceil(1.0 * width1 / block[0])),
            int(np.ceil(1.0 * height1 / block[1])))
    f_alias = __calc_shift_umpa_gpu_kernel
    if dark_signal:
        get_dark = 1
    else:
        get_dark = 0
    f_alias[grid, block](shifts, trans, dark, ref_mat1, mat1, coef_4d,
                         np.int32(depth), np.int32(height1),
                         np.int32(width1), np.int32(radi), np.int32(margin),
                         window, L1, L3, L2, L4, L6, A0, V0, D0,
                         np.int32(get_dark))
    return shifts[0], shifts[1], trans, dark


def find_local_shifts_umpa(ref_mat, mat, win_size=7, margin=10,
                           method="diff", size=3, gpu=True, block=(16, 16),
                           ncore=None, chunk_size=100, filter_name="hamming",
                           dark_signal=False):
    """
    To find local shifts (in x and y direction) of each pixel between
    two 3d-images by selecting a small volume of the second image and sliding
    over a slightly larger volume of the reference image. The cost function
    uses the formula in Ref. [1], known as UMPA.

    Parameters
    ----------
    ref_mat : array_like
        3D array, can be a numpy array or hdf object. Reference image.
    mat : array_like
        3D array, can be a numpy array or hdf object. The second image, must
        be the same size as the reference image.
    win_size : int
        Size of local areas in the second image.
    margin : int
        To define the sliding range of the second image.
    method : {"diff", "poly_fit"}
        Method for finding sub-pixel shift. Two options: a differential
        method (Ref. [2]) or a polynomial method (Ref. [3]). The "poly_fit"
        option is not available if using GPU.
    size : int
        Window size around the integer location of the maximum value used for
        sub-pixel location. Adjustable if using the polynomial method.
    gpu : bool
        Use GPU for computing if True.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (8, 8), (16, 16), (32, 32), ...
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    chunk_size : int or None
        Size of each chunk extracted along the height of the image. Use to
        avoid the out of memory problem.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming",\
                  "nuttall", "parzen", "triang"}
        To select a smoothing filter.
    dark_signal : bool
        Return both dark-signal image and transmission-signal image if True

    Returns
    -------
    list of two 2d-arrays or four 2d-arrays
        x-shift image and y-shift image. Zeros at the outer area of the size of
        (margin + win_size // 2). And/or dark-signal image and
        transmission-signal image If the 'dark_signal' option is True.

    References
    ----------
    [1] : https://doi.org/10.1103/PhysRevLett.118.203903

    [2] : https://doi.org/10.48550/arXiv.0712.4289

    [3] : https://doi.org/10.1088/0957-0233/17/6/045
    """
    if gpu is True:
        if cuda.is_available() is False:
            warnings.warn("!!!No Nvidia GPU found!!!Run with CPU instead!!!")
            gpu = False
    if ref_mat.shape != mat.shape:
        raise ValueError("Data shape must be the same !!!")
    if len(ref_mat.shape) != 3:
        raise ValueError("Inputs must be 3d-arrays !!!")
    (depth, height, width) = ref_mat.shape
    if chunk_size is None:
        chunk_size = height + 1
    else:
        chunk_size = np.clip(chunk_size, 1, height)
    num_chunk = np.clip(height // chunk_size + 1, 1, height)
    win_size = 2 * (win_size // 2) + 1
    edge = margin + win_size // 2
    als_size = win_size + 2 * margin
    if width < als_size or height < als_size:
        raise ValueError("Shapes of the inputs {0} are smaller than the "
                         "requested size (win_size + 2*margin) = "
                         "{1} x {1}".format((height, width), als_size))
    win_1d = make_smoothing_window(filter_name, win_size)
    window = np.multiply.outer(win_1d, win_1d)
    window = window / np.sum(window)
    S2 = np.sum(mat ** 2, axis=0)
    R2 = np.sum(ref_mat ** 2, axis=0)
    L1 = correlate(S2, window, mode="same")
    L3 = correlate(R2, window, mode="same")
    S1 = np.sum(mat, axis=0)
    R1 = np.sum(ref_mat, axis=0)
    Im = np.mean(R1) / depth
    L2 = (Im ** 2) * depth
    L4 = Im * correlate(S1, window, mode="same")
    L6 = Im * correlate(R1, window, mode="same")
    x_shifts = np.zeros((height, width), dtype=np.float32)
    y_shifts = np.zeros_like(x_shifts)
    trans = np.ones_like(x_shifts)
    dark = np.ones_like(x_shifts)
    list_index = np.array_split(np.arange(edge, height - edge), num_chunk)
    f_alias1 = __get_2d_shift_multi_rows_3d_input_umpa_gpu
    f_alias2 = __get_2d_shift_multi_rows_3d_input_umpa_cpu
    for pos in list_index:
        b, e = pos[0], pos[-1] + 1
        b1, e1 = b - edge, e + edge
        ref_mat1 = ref_mat[:, b1:e1, :]
        mat1 = mat[:, b1:e1, :]
        if gpu:
            results = f_alias1(ref_mat1, mat1, win_size, margin, window,
                               L1[b1:e1], L3[b1:e1], L2, L4[b1:e1], L6[b1:e1],
                               block=block, dark_signal=dark_signal)
        else:
            if dark_signal:
                results = f_alias2(ref_mat1, mat1, win_size, margin, window,
                                   L1[b1:e1], L3[b1:e1], L2, L4[b1:e1],
                                   L6[b1:e1], method=method, size=size,
                                   ncore=ncore, dark_signal=dark_signal)
            else:
                results = f_alias2(ref_mat1, mat1, win_size, margin, window,
                                   L1[b1:e1], L3[b1:e1], 0.0, 0.0, 0.0,
                                   method=method, size=size, ncore=ncore,
                                   dark_signal=dark_signal)
        x_shifts[b:e, edge:-edge] = results[0]
        y_shifts[b:e, edge:-edge] = results[1]
        if dark_signal:
            trans[b:e, edge:-edge] = results[2]
            dark[b:e, edge:-edge] = results[3]
    if dark_signal:
        trans = np.pad(trans[edge:-edge, edge:-edge], edge, mode="edge")
        dark = np.pad(dark[edge:-edge, edge:-edge], edge, mode="edge")
        return x_shifts, y_shifts, trans, dark
    else:
        return x_shifts, y_shifts
