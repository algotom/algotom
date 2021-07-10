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
# E-mail: algotomography@gmail.com
# Description: Python implementations of preprocessing techniques.
# Contributors:
# ============================================================================

"""
Module of calculation methods in the preprocessing stage:
    - Calculating the center-of-rotation (COR) in a 180-degree scan using a
      sinogram.
    - Determining the overlap-side and overlap-area between images.
    - Calculating the COR in a half-acquisition scan (360-degree scan with
      offset COR).
    - Using the similar technique as above to calculate the COR in a 180-degree
      scan from two projections.
    - Determining the relative translations between images using
      phase-correlation technique.
    - Calculating the COR in a 180-degree scan using phase-correlation technique.
"""

import numpy as np
from scipy import stats
import scipy.ndimage as ndi
import multiprocessing as mp
from joblib import Parallel, delayed
import numpy.fft as fft


def make_inverse_double_wedge_mask(height, width, radius):
    """
    Generate a double-wedge binary mask using Eq. (3) in Ref. [1].
    Values outside the double-wedge region correspond to 1.0.

    Parameters
    ----------
    height : int
        Image height.
    width : int
        Image width.
    radius : int
        Radius of an object, in pixel unit.

    Returns
    -------
    array_like
        2D binary mask.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.22.019078
    """
    du = 1.0 / width
    dv = (height - 1.0) / (height * 2.0 * np.pi)
    ndrop = min(20, np.int16(0.05 * height))
    ycenter = np.int16(np.ceil((height - 1) / 2.0))
    xcenter = np.int16(np.ceil((width - 1) / 2.0))
    mask = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        num = np.int16(np.ceil(((i - ycenter) * dv / radius) / du))
        (p1, p2) = np.int16(np.clip(
            np.sort((-num + xcenter, num + xcenter)), 0, width - 1))
        mask[i, p1:p2 + 1] = 1.0
    mask[ycenter - ndrop:ycenter + ndrop + 1, :] = 0.0
    mask[:, xcenter - 1:xcenter + 2] = 0.0
    return mask


def calculate_center_metric(center, sino_180, sino_flip, sino_comp, mask):
    """
    Calculate a metric of an estimated center-of-rotation.

    Parameters
    ----------
    center : float
        Estimated center.
    sino_180 : array_like
        2D array. 180-degree sinogram.
    sino_flip : array_like
        2D array. Flip the 180-degree sinogram in the left/right direction.
    sino_comp : array_like
        2D array. Used to fill the gap left by image shifting.
    mask : array_like
        2D array. Used to select coefficients in the double-wedge region.

    Returns
    -------
    float
        Metric.
    """
    ncol = sino_180.shape[1]
    center_flip = (ncol - 1.0) / 2.0
    shift_col = 2.0 * (center - center_flip)
    if np.abs(shift_col - np.floor(shift_col)) == 0.0:
        shift_col = int(shift_col)
        sino_shift = np.roll(sino_flip, shift_col, axis=1)
        if shift_col >= 0:
            sino_shift[:, :shift_col] = sino_comp[:, :shift_col]
        else:
            sino_shift[:, shift_col:] = sino_comp[:, shift_col:]
        mat = np.vstack((sino_180, sino_shift))
    else:
        sino_shift = ndi.shift(sino_flip, (0, shift_col), order=3,
                               prefilter=True)
        if shift_col >= 0:
            shift_int = int(np.ceil(shift_col))
            sino_shift[:, :shift_int] = sino_comp[:, :shift_int]
        else:
            shift_int = int(np.floor(shift_col))
            sino_shift[:, shift_int:] = sino_comp[:, shift_int:]
        mat = np.vstack((sino_180, sino_shift))
    metric = np.mean(
        np.abs(np.fft.fftshift(fft.fft2(mat))) * mask)
    return metric


def coarse_search_cor(sino_180, start, stop, ratio=0.5, denoise=True,
                      ncore=None):
    """
    Find the center-of-rotation (COR) using integer shifting.

    Parameters
    ----------
    sino_180 : array_like
        2D array. 180-degree sinogram.
    start : int
        Starting point for searching COR.
    stop : int
        Ending point for searching COR.
    ratio : float
        Ratio between a sample and the width of the sinogram.
    denoise : bool, optional
        Apply a smoothing filter.
    ncore: int or None
        Number of cores used for computing. Automatically selected if None.

    Returns
    -------
    float
        Center of rotation.
    """
    if denoise is True:
        sino_180 = ndi.gaussian_filter(sino_180, (3, 1), mode='reflect')
    if ncore is None:
        ncore = np.clip(mp.cpu_count() - 1, 1, None)
    (nrow, ncol) = sino_180.shape
    start_cor = np.int16(np.clip(start, 0, ncol - 1))
    stop_cor = np.int16(np.clip(stop, 0, ncol - 1))
    sino_flip = np.fliplr(sino_180)
    sino_comp = np.flipud(sino_180)
    list_cor = np.arange(start_cor, stop_cor + 1)
    list_metric = np.zeros(len(list_cor), dtype=np.float32)
    num_metric = len(list_metric)
    mask = make_inverse_double_wedge_mask(2 * nrow, ncol, 0.5 * ratio * ncol)
    if ncore == 1:
        for i, cor in enumerate(list_cor):
            list_metric[i] = calculate_center_metric(
                list_cor[i], sino_180, sino_flip, sino_comp, mask)
    else:
        list_metric = Parallel(n_jobs=ncore, backend="threading")(
            delayed(calculate_center_metric)(list_cor[i], sino_180, sino_flip,
                                             sino_comp, mask) for i in
            range(num_metric))
    return list_cor[np.argmin(list_metric)]


def fine_search_cor(sino_180, start, radius, step, ratio=0.5, denoise=True,
                    ncore=None):
    """
    Find the center-of-rotation (COR) using sub-pixel shifting.

    Parameters
    ----------
    sino_180 : array_like
        2D array. 180-degree sinogram.
    start : float
        Starting point for searching COR.
    radius : float
        Searching range: [start - radius; start + radius].
    step : float
        Searching step.
    ratio : float
        Ratio between a sample and the width of the sinogram.
    denoise : bool, optional
        Apply a smoothing filter.
    ncore: int or None
        Number of cores used for computing. Automatically selected if None.

    Returns
    -------
    float
        Center of rotation.
    """
    if denoise is True:
        sino_180 = ndi.gaussian_filter(sino_180, (2, 2), mode='reflect')
    if ncore is None:
        ncore = np.clip(mp.cpu_count() - 1, 1, None)
    (nrow, ncol) = sino_180.shape
    sino_flip = np.fliplr(sino_180)
    sino_comp = np.flipud(sino_180)
    list_cor = np.clip(
        start + np.arange(-radius, radius + step, step), 0.0, ncol - 1.0)
    list_metric = np.zeros(len(list_cor), dtype=np.float32)
    num_metric = len(list_metric)
    mask = make_inverse_double_wedge_mask(2 * nrow, ncol, 0.5 * ratio * ncol)
    if ncore == 1:
        for i, cor in enumerate(list_cor):
            list_metric[i] = calculate_center_metric(
                list_cor[i], sino_180, sino_flip, sino_comp, mask)
    else:
        list_metric = Parallel(n_jobs=ncore, backend="threading")(
            delayed(calculate_center_metric)(list_cor[i], sino_180, sino_flip,
                                             sino_comp, mask) for i in
            range(num_metric))
    return list_cor[np.argmin(list_metric)]


def downsample_cor(image, dsp_fact0, dsp_fact1):
    """
    Downsample an image by averaging.

    Parameters
    ----------
    image : array_like
        2D array.
    dsp_fact0 : int
        Downsampling factor along axis 0.
    dsp_fact1 : int
        Downsampling factor along axis 1.

    Returns
    -------
    array_like
        2D array. Downsampled image.
    """
    (height, width) = image.shape
    dsp_fact0 = np.clip(np.int16(dsp_fact0), 1, height // 2)
    dsp_fact1 = np.clip(np.int16(dsp_fact1), 1, width // 2)
    height_dsp = height // dsp_fact0
    width_dsp = width // dsp_fact1
    image_dsp = image[0:dsp_fact0 * height_dsp, 0:dsp_fact1 * width_dsp]
    image_dsp = image_dsp.reshape(
        height_dsp, dsp_fact0, width_dsp, dsp_fact1).mean(-1).mean(1)
    return image_dsp


def find_center_vo(sino_180, start=None, stop=None, step=0.25, radius=4,
                   ratio=0.5, dsp=True, ncore=None):
    """
    Find the center-of-rotation using the method described in Ref. [1].

    Parameters
    ----------
    sino_180 : array_like
        2D array. 180-degree sinogram.
    start : float
        Starting point for searching CoR.
    stop : float
        Ending point for searching CoR.
    step : float
        Sub-pixel accuracy of estimated CoR.
    radius : float
        Searching range with the sub-pixel step.
    ratio : float
        Ratio between the sample and the width of the sinogram.
    dsp : bool
        Enable/disable downsampling.
    ncore: int or None
        Number of cores used for computing. Automatically selected if None.

    Returns
    -------
    float
        Center-of-rotation.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.22.019078
    """
    (nrow, ncol) = sino_180.shape
    if start is None:
        start = ncol // 2 - ncol // 16
    if stop is None:
        stop = ncol // 2 + ncol // 16
    dsp_row = 1
    dsp_col = 1
    if dsp is True:
        if ncol > 1000:
            dsp_col = int(np.floor(ncol / 640.0))
        if nrow > 1000:
            dsp_row = int(np.floor(nrow / 900.0))
        sino_dns = ndi.gaussian_filter(sino_180, (3, 1), mode='reflect')
        sino_dsp = downsample_cor(sino_dns, dsp_row, dsp_col)
        radius = max(radius, dsp_col)
        off_set = 0.5 * dsp_col
        start = int(np.floor(1.0 * start / dsp_col))
        stop = int(np.ceil(1.0 * stop / dsp_col))
        raw_cor = coarse_search_cor(sino_dsp, start, stop, ratio,
                                    denoise=False, ncore=ncore)
        fine_cor = fine_search_cor(sino_180, raw_cor * dsp_col + off_set,
                                   radius, step, ratio, denoise=True,
                                   ncore=ncore)
    else:
        raw_cor = coarse_search_cor(sino_180, start, stop, ratio, denoise=True,
                                    ncore=ncore)
        fine_cor = fine_search_cor(sino_180, raw_cor, radius, step, ratio,
                                   denoise=True, ncore=ncore)
    return fine_cor


def calculate_curvature(list_metric):
    """
    Calculate the curvature of a fitted curve going through the minimum
    value of a metric list.

    Parameters
    ----------
    list_metric : array_like
        1D array. List of metrics.

    Returns
    -------
    curvature : float
        Quadratic coefficient of the parabola fitting.
    min_pos : float
        Position of the minimum value with sub-pixel accuracy.
    """
    radi = 2
    num_metric = len(list_metric)
    min_pos = np.clip(
        np.argmin(list_metric), radi, num_metric - radi - 1)
    list1 = list_metric[min_pos - radi:min_pos + radi + 1]
    (afact1, _, _) = np.polyfit(np.arange(0, 2 * radi + 1), list1, 2)
    list2 = list_metric[min_pos - 1:min_pos + 2]
    (afact2, bfact2, _) = np.polyfit(
        np.arange(min_pos - 1, min_pos + 2), list2, 2)
    curvature = np.abs(afact1)
    if afact2 != 0.0:
        num = - bfact2 / (2 * afact2)
        if (num >= min_pos - 1) and (num <= min_pos + 1):
            min_pos = num
    return curvature, np.float32(min_pos)


def correlation_metric(mat1, mat2):
    """
    Calculate the correlation metric. Smaller metric corresponds to better
    correlation.

    Parameters
    ---------
    mat1 : array_like
    mat2 : array_like

    Returns
    -------
    float
        Correlation metric.
    """
    metric = np.abs(
        1.0 - stats.pearsonr(mat1.flatten('F'), mat2.flatten('F'))[0])
    return metric


def search_overlap(mat1, mat2, win_width, side, denoise=True, norm=False,
                   use_overlap=False):
    """
    Calculate the correlation metrics between a rectangular region, defined
    by the window width, on the utmost left/right side of image 2 and the
    same size region in image 1 where the region is slided across image 1.

    Parameters
    ----------
    mat1 : array_like
        2D array. Projection image or sinogram image.
    mat2 : array_like
        2D array. Projection image or sinogram image.
    win_width : int
        Width of the searching window.
    side : {0, 1}
        Only two options: 0 or 1. It is used to indicate the overlap side
        respects to image 1. "0" corresponds to the left side. "1" corresponds
        to the right side.
    denoise : bool, optional
        Apply the Gaussian filter if True.
    norm : bool, optional
        Apply the normalization if True.
    use_overlap : bool, optional
        Use the combination of images in the overlap area for calculating
        correlation coefficients if True.

    Returns
    -------
    list_metric : array_like
        1D array. List of the correlation metrics.
    offset : int
        Initial position of the searching window where the position
        corresponds to the center of the window.
    """
    if denoise is True:
        mat1 = ndi.gaussian_filter(mat1, (2, 2), mode='reflect')
        mat2 = ndi.gaussian_filter(mat2, (2, 2), mode='reflect')
    (nrow1, ncol1) = mat1.shape
    (nrow2, ncol2) = mat2.shape
    if nrow1 != nrow2:
        raise ValueError("Two images are not at the same height!!!")
    win_width = np.int16(np.clip(win_width, 6, min(ncol1, ncol2) // 2 - 1))
    offset = win_width // 2
    win_width = 2 * offset  # Make it even
    ramp_down = np.linspace(1.0, 0.0, win_width)
    ramp_up = 1.0 - ramp_down
    wei_down = np.tile(ramp_down, (nrow1, 1))
    wei_up = np.tile(ramp_up, (nrow1, 1))
    if side == 1:
        mat2_roi = mat2[:, 0:win_width]
        mat2_roi_wei = mat2_roi * wei_up
    else:
        mat2_roi = mat2[:, ncol2 - win_width:]
        mat2_roi_wei = mat2_roi * wei_down
    list_mean2 = np.mean(np.abs(mat2_roi), axis=1)
    list_pos = np.arange(offset, ncol1 - offset)
    num_metric = len(list_pos)
    list_metric = np.ones(num_metric, dtype=np.float32)
    for i, pos in enumerate(list_pos):
        mat1_roi = mat1[:, pos - offset:pos + offset]
        if use_overlap is True:
            if side == 1:
                mat1_roi_wei = mat1_roi * wei_down
            else:
                mat1_roi_wei = mat1_roi * wei_up
        if norm is True:
            list_mean1 = np.mean(np.abs(mat1_roi), axis=1)
            list_fact = list_mean2 / list_mean1
            mat_fact = np.transpose(np.tile(list_fact, (win_width, 1)))
            mat1_roi = mat1_roi * mat_fact
            if use_overlap is True:
                mat1_roi_wei = mat1_roi_wei * mat_fact
        if use_overlap is True:
            mat_comb = mat1_roi_wei + mat2_roi_wei
            list_metric[i] = (correlation_metric(mat1_roi, mat2_roi)
                              + correlation_metric(mat1_roi, mat_comb)
                              + correlation_metric(mat2_roi, mat_comb)) / 3.0
        else:
            list_metric[i] = correlation_metric(mat1_roi, mat2_roi)
    min_metric = np.min(list_metric)
    if min_metric != 0.0:
        list_metric = list_metric / min_metric
    return list_metric, offset


def find_overlap(mat1, mat2, win_width, side=None, denoise=True, norm=False,
                 use_overlap=False):
    """
    Find the overlap area and overlap side between two images (Ref. [1]) where
    the overlap side referring to the first image.

    Parameters
    ----------
    mat1 : array_like
        2D array. Projection image or sinogram image.
    mat2 :  array_like
        2D array. Projection image or sinogram image.
    win_width : int
        Width of the searching window.
    side : {None, 0, 1}, optional
        Only there options: None, 0, or 1. "None" corresponding to fully
        automated determination. "0" corresponding to the left side. "1"
        corresponding to the right side.
    denoise : bool, optional
        Apply the Gaussian filter if True.
    norm : bool, optional
        Apply the normalization if True.
    use_overlap : bool, optional
        Use the combination of images in the overlap area for calculating
        correlation coefficients if True.

    Returns
    -------
    overlap : float
        Width of the overlap area between two images.
    side : int
        Overlap side between two images.
    overlap_position : float
        Position of the window in the first image giving the best
        correlation metric.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.418448
    """
    (_, ncol1) = mat1.shape
    (_, ncol2) = mat2.shape
    win_width = np.int16(np.clip(win_width, 6, min(ncol1, ncol2) // 2))
    if side == 1:
        (list_metric, offset) = search_overlap(mat1, mat2, win_width, side,
                                               denoise, norm, use_overlap)
        (_, overlap_position) = calculate_curvature(list_metric)
        overlap_position = overlap_position + offset
        overlap = ncol1 - overlap_position + win_width // 2
    elif side == 0:
        (list_metric, offset) = search_overlap(mat1, mat2, win_width, side,
                                               denoise, norm, use_overlap)
        (_, overlap_position) = calculate_curvature(list_metric)
        overlap_position = overlap_position + offset
        overlap = overlap_position + win_width // 2
    else:
        (list_metric1, offset1) = search_overlap(mat1, mat2, win_width, 1, norm,
                                                 denoise, use_overlap)
        (list_metric2, offset2) = search_overlap(mat1, mat2, win_width, 0, norm,
                                                 denoise, use_overlap)
        (curvature1, overlap_position1) = calculate_curvature(list_metric1)
        overlap_position1 = overlap_position1 + offset1
        (curvature2, overlap_position2) = calculate_curvature(list_metric2)
        overlap_position2 = overlap_position2 + offset2
        if curvature1 > curvature2:
            side = 1
            overlap_position = overlap_position1
            overlap = ncol1 - overlap_position + win_width // 2
        else:
            side = 0
            overlap_position = overlap_position2
            overlap = overlap_position + win_width // 2
    return overlap, side, overlap_position


def find_overlap_multiple(list_mat, win_width, side=None, denoise=True,
                          norm=False, use_overlap=False):
    """
    Find the overlap-areas and overlap-sides of a list of images where the
    overlap side referring to the previous image.

    Parameters
    ----------
    list_mat : list of array_like
        List of 2D array. Projection image or sinogram image.
    win_width : int
        Width of the searching window.
    side : {None, 0, 1}, optional
        Only there options: None, 0, or 1. "None" corresponding to fully
        automated determination. "0" corresponding to the left side. "1"
        corresponding to the right side.
    denoise : bool, optional
        Apply the Gaussian filter if True.
    norm : bool, optional
        Apply the normalization if True.
    use_overlap : bool, optional
        Use the combination of images in the overlap area for calculating
        correlation coefficients if True.

    Returns
    -------
    list_overlap : list of tuple of floats
        List of [overlap, side, overlap_position].
        overlap : Width of the overlap area between two images.
        side : Overlap side between two images.
        overlap_position : Position of the window in the first
        image giving the best correlation metric.
    """
    list_overlap = []
    num_mat = len(list_mat)
    if num_mat > 1:
        for i in range(num_mat-1):
            results = find_overlap(list_mat[i], list_mat[i + 1], win_width,
                                   side, denoise, norm, use_overlap)
            list_overlap.append(results)
    else:
        raise ValueError("Need at least 2 images to work!!!")
    return list_overlap


def find_center_360(sino_360, win_width, side=None, denoise=True, norm=False,
                    use_overlap=False):
    """
    Find the center-of-rotation (COR) in a 360-degree scan with offset COR use
    the method presented in Ref. [1].

    Parameters
    ----------
    sino_360 : array_like
        2D array. 360-degree sinogram.
    win_width : int
        Window width used for finding the overlap area.
    side : {None, 0, 1}, optional
        Overlap size. Only there options: None, 0, or 1. "None" corresponding
        to fully automated determination. "0" corresponding to the left side.
        "1" corresponding to the right side.
    denoise : bool, optional
        Apply the Gaussian filter if True.
    norm : bool, optional
        Apply the normalization if True.
    use_overlap : bool, optional
        Use the combination of images in the overlap area for calculating
        correlation coefficients if True.

    Returns
    -------
    cor : float
        Center-of-rotation.
    overlap : float
        Width of the overlap area between two halves of the sinogram.
    side : int
        Overlap side between two halves of the sinogram.
    overlap_position : float
        Position of the window in the first image giving the best
        correlation metric.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.418448
    """
    (nrow, ncol) = sino_360.shape
    nrow_180 = nrow // 2 + 1
    sino_top = sino_360[0:nrow_180, :]
    sino_bot = np.fliplr(sino_360[-nrow_180:, :])
    (overlap, side, overlap_position) = find_overlap(
        sino_top, sino_bot, win_width, side, denoise, norm, use_overlap)
    if side == 0:
        cor = overlap / 2.0 - 1.0
    else:
        cor = ncol - overlap / 2.0 - 1.0
    return cor, overlap, side, overlap_position


def complex_gradient(mat):
    """
    Return complex gradient of a 2D array.
    """
    mat1a = np.roll(mat, -2, axis=1)
    mat2a = mat1a - mat
    mat2a[:, :2] = 0.0
    mat2a[:, -2:] = 0.0
    mat1b = np.roll(mat, -2, axis=0)
    mat2b = mat1b - mat
    mat2b[:2] = 0.0
    mat2b[-2:] = 0.0
    mat2 = mat2a + 1j * mat2b
    return mat2


def find_shift_based_phase_correlation(mat1, mat2, gradient=True):
    """
    Find relative translation in x and y direction between images with
    haft-pixel accuracy (Ref. [1]).

    Parameters
    ----------
    mat1 : array_like
        2D array. Projection image or sinogram image.
    mat2 : array_like
        2D array. Projection image or sinogram image.
    gradient : bool, optional
        Use the complex gradient of the input image for calculation.

    Returns
    -------
    ty : float
        Translation in y-direction.
    tx : float
        Translation in x-direction.

    References
    ----------
    .. [1] https://doi.org/10.1049/el:20030666
    """
    if gradient is True:
        mat1 = complex_gradient(mat1)
        mat2 = complex_gradient(mat2)
    (nrow, ncol) = mat1.shape
    mat_tmp1 = fft.fft2(mat1) * np.conjugate(fft.fft2(mat2))
    mat_tmp2 = np.abs(mat_tmp1)
    mat_tmp2[mat_tmp2 == 0.0] = 1.0
    mat3 = np.abs(fft.ifft2(mat_tmp1 / mat_tmp2))
    (ty, tx) = np.unravel_index(np.argmax(mat3, axis=None), mat3.shape)
    list_x = np.asarray([tx - 1, tx, tx + 1])
    list_vx = np.asarray(
        [mat3[ty, tx - 1], mat3[ty, tx], mat3[ty, (tx + 1) % ncol]])
    list_y = np.asarray([ty - 1, ty, ty + 1])
    list_vy = np.asarray(
        [mat3[ty - 1, tx], mat3[ty, tx], mat3[(ty + 1) % nrow, tx]])
    (afact_x, bfact_x, _) = np.polyfit(list_x, list_vx, 2)
    (afact_y, bfact_y, _) = np.polyfit(list_y, list_vy, 2)
    if afact_x != 0.0:
        num = - bfact_x / (2 * afact_x)
        if (num >= tx - 1) and (num <= tx + 1):
            tx = num
    if afact_y != 0.0:
        num = - bfact_y / (2 * afact_y)
        if (num >= ty - 1) and (num <= ty + 1):
            ty = num
    xcenter = np.ceil((ncol - 1) * 0.5)
    ycenter = np.ceil((nrow - 1) * 0.5)
    if ty > ycenter:
        ty = -(nrow - ty)
    if tx > xcenter:
        tx = -(ncol - tx)
    return ty, tx


def find_center_based_phase_correlation(mat1, mat2, flip=True, gradient=True):
    """
    Find the center-of-rotation (COR) using projection images at 0-degree
    and 180-degree.

    Parameters
    ----------
    mat1 : array_like
        2D array. Projection image at 0-degree.
    mat2 : array_like
        2D array. Projection image at 180-degree.
    flip : bool, optional
        Flip the 180-degree projection in the left-right direction if True.
    gradient : bool, optional
        Use the complex gradient of the input image for calculation.

    Returns
    -------
    cor : float
        Center-of-rotation.
    """
    ncol = mat1.shape[-1]
    if flip is True:
        mat2 = np.fliplr(mat2)
    tx = find_shift_based_phase_correlation(mat1, mat2, gradient=gradient)[-1]
    cor = (ncol - 1.0 + tx) * 0.5
    return cor


def find_center_projection(mat1, mat2, flip=True, chunk_height=None,
                           start_row=None, denoise=True, norm=False,
                           use_overlap=False):
    """
    Find the center-of-rotation (COR) using projection images at 0-degree
    and 180-degree based on a method in Ref. [1].

    Parameters
    ----------
    mat1 : array_like
        2D array. Projection image at 0-degree.
    mat2 : array_like
        2D array. Projection image at 180-degree.
    flip : bool, optional
        Flip the 180-degree projection in the left-right direction if True.
    chunk_height : int or float, optional
        Height of the sub-area of projection images. If a float is given, it
        must be in the range of [0.0, 1.0].
    start_row : int, optional
        Starting row used to extract the sub-area.
    denoise : bool, optional
        Apply the Gaussian filter if True.
    norm : bool, optional
        Apply the normalization if True.
    use_overlap : bool, optional
        Use the combination of images in the overlap area for calculating
        correlation coefficients if True.

    Returns
    -------
    cor : float
        Center-of-rotation.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.418448
    """
    (nrow, ncol) = mat1.shape
    if flip is True:
        mat2 = np.fliplr(mat2)
    win_width = ncol // 2
    if chunk_height is None:
        chunk_height = int(0.1 * nrow)
    if isinstance(chunk_height, float):
        if 0.0 < chunk_height <= 1.0:
            chunk_height = int(chunk_height * nrow)
        else:
            chunk_height = int(0.1 * nrow)
    chunk_height = np.clip(chunk_height, 1, nrow - 1)
    if start_row is None:
        start = nrow // 2 - chunk_height // 2
    elif start_row < 0:
        start = nrow + start_row - chunk_height // 2
    else:
        start = start_row - chunk_height // 2
    stop = start + chunk_height
    start = np.clip(start, 0, nrow - chunk_height - 1)
    stop = np.clip(stop, chunk_height, nrow - 1)
    mat1_roi = mat1[start: stop]
    mat2_roi = mat2[start: stop]
    (overlap, side, _) = find_overlap(mat1_roi, mat2_roi, win_width, side=None,
                                      denoise=denoise, norm=norm,
                                      use_overlap=use_overlap)
    if side == 0:
        cor = overlap / 2.0 - 1.0
    else:
        cor = ncol - overlap / 2.0 - 1.0
    return cor


def calculate_reconstructable_height(y_start, y_stop, pitch, scan_type):
    """
    Calculate reconstructable height in a helical scan.

    Parameters
    ----------
    y_start : float
        Y-position of the stage at the beginning of the scan.
    y_stop : float
        Y-position of the stage at the end of the scan.
    pitch : float
        The distance which the y-stage is translated in one full rotation.
    scan_type : {"180", "360"}
        One of two options: "180" for generating a 180-degree sinogram or
        "360" for generating a 360-degree sinogram.

    Returns
    -------
    y_s : float
        Starting point of the reconstructable height.
    y_e : float
        End point of the reconstructable height.
    """
    if not(scan_type == "180" or scan_type == "360"):
        raise ValueError("!!! Please one of two options: '180' or '360'!!!")
    if scan_type == "360":
        y_s = y_start + pitch
        y_e = y_stop - pitch
    else:
        y_s = y_start + pitch / 2.0
        y_e = y_stop - pitch / 2.0
    return y_s, y_e


def calculate_maximum_index(y_start, y_stop, pitch, pixel_size, scan_type):
    """
    Calculate the maximum index of a reconstructable slice in a helical scan.

    Parameters
    ----------
    y_start : float
        Y-position of the stage at the beginning of the scan.
    y_stop : float
        Y-position of the stage at the end of the scan.
    pitch : float
        The distance which the y-stage is translated in one full rotation.
    pixel_size : float
        Pixel size. The unit must be the same as y-position.
    scan_type : {"180", "360"}
        One of two options: "180" for generating a 180-degree sinogram or
        "360" for generating a 360-degree sinogram.

    Returns
    -------
    int
        Maximum index of reconstructable slices.
    """
    if not(scan_type == "180" or scan_type == "360"):
        raise ValueError("!!! Please one of two options: '180' or '360'!!!")
    y_s, y_e = calculate_reconstructable_height(y_start, y_stop, pitch,
                                                scan_type)
    max_index = int(((y_e - y_s) / pixel_size)) + 1
    return max_index
