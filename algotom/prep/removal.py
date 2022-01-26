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
Module of removal methods in the preprocessing stage:
- Many methods for removing stripe artifact in a sinogram (<-> ring artifact
in a reconstructed image).
- A zinger removal method.
- Blob removal methods.
"""

import numpy as np
import scipy.ndimage as ndi
from scipy import interpolate
import numpy.fft as fft
import algotom.util.utility as util


def remove_stripe_based_sorting(sinogram, size=21, dim=1, **options):
    """
    Remove stripe artifacts in a sinogram using the sorting technique,
    algorithm 3 in Ref. [1]. Angular direction is along the axis 0.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    size : int
        Window size of the median filter.
    dim : {1, 2}, optional
        Dimension of the window.
    options : dict, optional
        Use another smoothing filter rather than the median filter.
        E.g. options={"method": "gaussian_filter", "para1": (1,21))}

    Returns
    -------
    array_like
        2D array. Stripe-removed sinogram.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    msg = "\n Please use the dictionary format: options={'method':" \
          " 'filter_name', 'para1': parameter_1, 'para2': parameter_2}"
    sino_sort, sino_index = util.sort_forward(np.float32(sinogram), axis=0)
    if len(options) == 0:
        if dim == 2:
            sino_sort = ndi.median_filter(sino_sort, (size, size))
        else:
            sino_sort = ndi.median_filter(sino_sort, (1, size))
    else:
        if not isinstance(options, dict):
            raise ValueError(msg)
        for opt_name in options:
            opt = options[opt_name]
            method = tuple(opt.values())[0]
            para = tuple(opt.values())[1:]
            if method in dir(ndi):
                try:
                    sino_sort = getattr(ndi, method)(sino_sort, *para)
                except:
                    raise ValueError(msg)
            else:
                if method in dir(util):
                    try:
                        sino_sort = getattr(util, method)(sino_sort, *para)
                    except:
                        raise ValueError(msg)
                else:
                    raise ValueError("Can't find the method: '{}' in the"
                                     " namespace".format(method))
    return util.sort_backward(sino_sort, sino_index, axis=0)


def remove_stripe_based_filtering(sinogram, sigma=3, size=21, dim=1, sort=True,
                                  **options):
    """
    Remove stripe artifacts in a sinogram using the filtering technique,
    algorithm 2 in Ref. [1]. Angular direction is along the axis 0.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image
    sigma : int
        Sigma of the Gaussian window used to separate the low-pass and
        high-pass components of the intensity profile of each column.
    size : int
        Window size of the median filter.
    dim : {1, 2}, optional
        Dimension of the window.
    sort : bool, optional
        Apply sorting if True.
    options : dict, optional
        Use another smoothing filter rather than the median filter.
        E.g. options={"method": "gaussian_filter", "para1": (1,21))}.

    Returns
    -------
    array_like
         2D array. Stripe-removed sinogram.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    msg = "\n Please use the dictionary format: options={'method':" \
          " 'filter_name', 'para1': parameter_1, 'para2': parameter_2}"
    window = {"name": "gaussian", "sigma": sigma}
    sino_smooth, sino_sharp = util.separate_frequency_component(
        np.float32(sinogram), axis=0, window=window)
    if sort is True:
        sino_smooth, sino_index = util.sort_forward(sino_smooth, axis=0)
    if len(options) == 0:
        if dim == 2:
            sino_smooth = ndi.median_filter(sino_smooth, (size, size))
        else:
            sino_smooth = ndi.median_filter(sino_smooth, (1, size))
    else:
        if not isinstance(options, dict):
            raise ValueError(msg)
        for opt_name in options:
            opt = options[opt_name]
            method = tuple(opt.values())[0]
            if method in dir(ndi):
                para = tuple(opt.values())[1:]
                try:
                    sino_smooth = getattr(ndi, method)(sino_smooth, *para)
                except:
                    raise ValueError(msg)
            else:
                if method in dir(util):
                    try:
                        sino_smooth = getattr(util, method)(sino_smooth, *para)
                    except:
                        raise ValueError(msg)
                else:
                    raise ValueError("Can't find the method: '{}' in the"
                                     " namespace".format(method))
    if sort is True:
        sino_smooth = util.sort_backward(sino_smooth, sino_index, axis=0)
    return sino_smooth + sino_sharp


def remove_stripe_based_fitting(sinogram, order=2, sigma=10, sort=False,
                                num_chunk=1, **options):
    """
    Remove stripe artifacts in a sinogram using the fitting technique,
    algorithm 1 in Ref. [1]. Angular direction is along the axis 0.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image
    order : int
        Polynomial fit order.
    sigma : int
        Sigma of the Gaussian window in the x-direction. Smaller is stronger.
    sort : bool, optional
        Apply sorting if True.
    num_chunk : int
        Number of chunks of rows to apply the fitting.
    options : dict, optional
        Use another smoothing filter rather than the Fourier gaussian filter.
        E.g. options={"method": "gaussian_filter", "para1": (1,21))}.

    Returns
    -------
    array_like
        2D array. Stripe-removed sinogram.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    msg = "\n Please use the dictionary format: options={'method':" \
          " 'filter_name', 'para1': parameter_1, 'para2': parameter_2}"
    (nrow, ncol) = sinogram.shape
    pad = min(150, int(0.1 * nrow))
    sigmay = np.clip(min(60, int(0.1 * ncol)), 10, None)
    if sort is True:
        sinogram, sino_index = util.sort_forward(sinogram, axis=0)
    sino_fit = util.generate_fitted_image(sinogram, order, axis=0,
                                          num_chunk=num_chunk)
    if len(options) == 0:
        sino_filt = util.apply_gaussian_filter(sino_fit, sigma, sigmay, pad)
    else:
        if not isinstance(options, dict):
            raise ValueError(msg)
        sino_filt = np.copy(sino_fit)
        for opt_name in options:
            opt = options[opt_name]
            method = tuple(opt.values())[0]
            if method in dir(ndi):
                para = tuple(opt.values())[1:]
                try:
                    sino_filt = getattr(ndi, method)(sino_filt, *para)
                except:
                    raise ValueError(msg)
            else:
                if method in dir(util):
                    try:
                        sino_filt = getattr(util, method)(sino_filt, *para)
                    except:
                        raise ValueError(msg)
                else:
                    raise ValueError("Can't find the method: '{}' in the"
                                     " namespace".format(method))
    sino_filt = np.mean(np.abs(sino_fit)) * sino_filt / np.mean(
        np.abs(sino_filt))
    sino_corr = ((sinogram / sino_fit) * sino_filt)
    if sort is True:
        sino_corr = util.sort_backward(sino_corr, sino_index, axis=0)
    return sino_corr


def remove_large_stripe(sinogram, snr=3.0, size=51, drop_ratio=0.1, norm=True,
                        **options):
    """
    Remove large stripe artifacts in a sinogram, algorithm 5 in Ref. [1].
    Angular direction is along the axis 0.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image
    snr : float
        Ratio (>1.0) used to detect stripe locations. Greater is less sensitive.
    size : int
        Window size of the median filter.
    drop_ratio : float, optional
        Ratio of pixels to be dropped, which is used to to reduce
        the possibility of the false detection of stripes.
    norm : bool, optional
        Apply normalization if True.
    options : dict, optional
        Use another smoothing filter rather than the median filter.
        E.g. options={"method": "gaussian_filter", "para1": (1,21))}.

    Returns
    -------
    array_like
        2D array. Stripe-removed sinogram.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    msg = "\n Please use the dictionary format: options={'method':" \
          " 'filter_name', 'para1': parameter_1, 'para2': parameter_2}"
    sinogram = np.copy(np.float32(sinogram))
    drop_ratio = np.clip(drop_ratio, 0.0, 0.8)
    (nrow, ncol) = sinogram.shape
    ndrop = int(0.5 * drop_ratio * nrow)
    sino_sort, sino_index = util.sort_forward(sinogram, axis=0)
    if len(options) == 0:
        sino_smooth = ndi.median_filter(sino_sort, (1, size))
    else:
        if not isinstance(options, dict):
            raise ValueError(msg)
        sino_smooth = np.copy(sino_sort)
        for opt_name in options:
            opt = options[opt_name]
            method = tuple(opt.values())[0]
            if method in dir(ndi):
                para = tuple(opt.values())[1:]
                try:
                    sino_smooth = getattr(ndi, method)(sino_smooth, *para)
                except:
                    raise ValueError(msg)
            else:
                if method in dir(util):
                    try:
                        sino_smooth = getattr(util, method)(sino_smooth, *para)
                    except:
                        raise ValueError(msg)
                else:
                    raise ValueError("Can't find the method: '{}' in the"
                                     " namespace".format(method))
    list1 = np.mean(sino_sort[ndrop:nrow - ndrop], axis=0)
    list2 = np.mean(sino_smooth[ndrop:nrow - ndrop], axis=0)
    list_fact = np.divide(list1, list2,
                          out=np.ones_like(list1), where=list2 != 0)
    list_mask = util.detect_stripe(list_fact, snr)
    list_mask = np.float32(ndi.binary_dilation(list_mask, iterations=1))
    if norm is True:
        sinogram = sinogram / np.tile(list_fact, (nrow, 1))
    sino_corr = util.sort_backward(sino_smooth, sino_index, axis=0)
    xlist_miss = np.where(list_mask > 0.0)[0]
    sinogram[:, xlist_miss] = sino_corr[:, xlist_miss]
    return sinogram


def remove_dead_stripe(sinogram, snr=3.0, size=51, residual=True):
    """
    Remove unresponsive or fluctuating stripe artifacts in a sinogram,
    algorithm 6 in Ref. [1]. Angular direction is along the axis 0.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    snr : float
        Ratio (>1.0) used to detect stripe locations. Greater is less sensitive.
    size : int
        Window size of the median filter.
    residual : bool, optional
        Removing residual stripes if True.

    Returns
    -------
    ndarray
        2D array. Stripe-removed sinogram.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    sinogram = np.copy(sinogram)  # Make it mutable
    (nrow, _) = sinogram.shape
    sino_smooth = np.apply_along_axis(ndi.uniform_filter1d, 0, sinogram, 10)
    list_diff = np.sum(np.abs(sinogram - sino_smooth), axis=0)
    list_diff_bck = ndi.median_filter(list_diff, size)
    nmean = np.mean(np.abs(list_diff_bck))
    list_diff_bck[list_diff_bck == 0.0] = nmean
    list_fact = list_diff / list_diff_bck
    list_mask = util.detect_stripe(list_fact, snr)
    list_mask = np.float32(ndi.binary_dilation(list_mask, iterations=1))
    list_mask[0:2] = 0.0
    list_mask[-2:] = 0.0
    xlist = np.where(list_mask < 1.0)[0]
    ylist = np.arange(nrow)
    mat = sinogram[:, xlist]
    finter = interpolate.interp2d(xlist, ylist, mat, kind='linear')
    xlist_miss = np.where(list_mask > 0.0)[0]
    if len(xlist_miss) > 0:
        sinogram[:, xlist_miss] = finter(xlist_miss, ylist)
    if residual is True:
        sinogram = remove_large_stripe(sinogram, snr, size)
    return sinogram


def remove_all_stripe(sinogram, snr=3.0, la_size=51, sm_size=21, drop_ratio=0.1,
                      dim=1, **options):
    """
    Remove all types of stripe artifacts in a sinogram by combining algorithm
    6, 5, 4, and 3 in Ref. [1]. Angular direction is along the axis 0.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    snr : float
        Ratio (>1.0) used to detect stripe locations. Greater is less sensitive.
    la_size : int
        Window size of the median filter to remove large stripes.
    sm_size : int
        Window size of the median filter to remove small-to-medium stripes.
    drop_ratio : float, optional
        Ratio of pixels to be dropped, which is used to to reduce
        the possibility of the false detection of stripes.
    dim : {1, 2}, optional
        Dimension of the window.
    options : dict, optional
        Use another smoothing filter rather than the median filter.
        E.g. options={"method": "gaussian_filter", "para1": (1,21))}

    Returns
    -------
    array_like
        2D array. Stripe-removed sinogram.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    sinogram = remove_dead_stripe(sinogram, snr, la_size, residual=False)
    sinogram = remove_large_stripe(sinogram, snr, la_size, drop_ratio,
                                   **options)
    sinogram = remove_stripe_based_sorting(sinogram, sm_size, dim, **options)
    return sinogram


def remove_stripe_based_2d_filtering_sorting(sinogram, sigma=3, size=21, dim=1,
                                             **options):
    """
    Remove stripes using a 2D low-pass filter and the sorting-based technique,
    algorithm in section 3.3.4 in Ref. [1].
    Angular direction is along the axis 0.

    Parameters
    ---------
    sinogram : array_like
        2D array. Sinogram image.
    sigma : int
        Sigma of the Gaussian window.
    size : int
        Window size of the median filter.
    dim : {1, 2}, optional
        Dimension of the window.

    Returns
    -------
    array_like
         2D array. Stripe-removed sinogram.

    References
    ----------
    .. [1] https://doi.org/10.1117/12.2530324
    """
    (nrow, ncol) = sinogram.shape
    pad = min(150, int(0.1 * min(nrow, ncol)))
    sino_smooth = util.apply_gaussian_filter(sinogram, sigma, sigma, pad)
    sino_sharp = sinogram - sino_smooth
    sino_sharp = remove_stripe_based_sorting(sino_sharp, size, dim, **options)
    return sino_smooth + sino_sharp


def remove_stripe_based_normalization(sinogram, sigma=15, num_chunk=1,
                                      sort=True, **options):
    """
    Remove stripes using the method in Ref. [1].
    Angular direction is along the axis 0.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    sigma : int
        Sigma of the Gaussian window.
    num_chunk : int
        Number of chunks of rows.
    sort : bool, optional
        Apply sorting (Ref. [2]) if True.
    options : dict, optional
        Use another smoothing 1D-filter rather than the Gaussian filter.
        E.g. options={"method": "median_filter", "para1": 21)}.

    Returns
    -------
    array_like
        2D array. Stripe-removed sinogram.

    References
    ----------
    .. [1] https://www.mcs.anl.gov/research/projects/X-ray-cmt/rivers/
           tutorial.html
    .. [2] https://doi.org/10.1364/OE.26.028396
    """
    msg = "\n Please use the dictionary format: options={'method':" \
          " 'filter_name', 'para1': parameter_1, 'para2': parameter_2}" \
          "\n Note that the filter must be a 1D-filter."
    (nrow, _) = sinogram.shape
    sinogram = np.copy(sinogram)
    if sort is True:
        sinogram, sino_index = util.sort_forward(sinogram, axis=0)
    list_index = np.array_split(np.arange(nrow), num_chunk)
    for pos in list_index:
        bindex = pos[0]
        eindex = pos[-1] + 1
        list_mean = np.mean(sinogram[bindex:eindex], axis=0)
        if len(options) == 0:
            list_filt = ndi.gaussian_filter(list_mean, sigma)
        else:
            if not isinstance(options, dict):
                raise ValueError(msg)
            list_filt = np.copy(list_mean)
            for opt_name in options:
                opt = options[opt_name]
                method = tuple(opt.values())[0]
                if method in dir(ndi):
                    para = tuple(opt.values())[1:]
                    try:
                        list_filt = getattr(ndi, method)(list_filt, *para)
                    except:
                        raise ValueError(msg)
                else:
                    if method in dir(util):
                        try:
                            list_filt = getattr(util, method)(list_filt, *para)
                        except:
                            raise ValueError(msg)
                    else:
                        raise ValueError("Can't find the method: '{}' in the"
                                         " namespace".format(method))
        list_coe = list_filt - list_mean
        matcoe = np.tile(list_coe, (eindex - bindex, 1))
        sinogram[bindex:eindex, :] = sinogram[bindex:eindex, :] + matcoe
    if sort is True:
        sinogram = util.sort_backward(sinogram, sino_index, axis=0)
    return sinogram


def remove_stripe_based_regularization(sinogram, alpha=0.0005, num_chunk=1,
                                       apply_log=True, sort=True):
    """
    Remove stripes using the method in Ref. [1].
    Angular direction is along the axis 0.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    alpha : float
        Regularization parameter, e.g. 0.0005. Smaller is stronger.
    num_chunk : int
        Number of chunks of rows.
    apply_log : bool
        Apply the logarithm function to the sinogram if True.
    sort : bool, optional
        Apply sorting (Ref. [2]) if True.

    Returns
    -------
    array_like
        2D array. Stripe-removed sinogram.

    References
    ----------
    .. [1] https://doi.org/10.1016/j.aml.2010.08.022
    .. [2] https://doi.org/10.1364/OE.26.028396
    """
    (nrow, ncol) = sinogram.shape
    sinogram = np.copy(sinogram)
    if sort is True:
        sinogram, sino_index = util.sort_forward(sinogram, axis=0)
    if apply_log is True:
        if np.any(sinogram <= 0.0):
            nmean = np.mean(np.abs(sinogram))
            sinogram[sinogram <= 0.0] = nmean
            sinogram = -np.log(sinogram)
        else:
            sinogram = -np.log(sinogram)
    sijmat = util.calculate_regularization_coefficient(ncol, alpha)
    list_index = np.array_split(np.arange(nrow), num_chunk)
    list_grad = np.zeros(ncol, dtype=np.float32)
    mat_grad = np.zeros((ncol, ncol), dtype=np.float32)
    for pos in list_index:
        bindex = pos[0]
        eindex = pos[-1] + 1
        list_mean = np.mean(sinogram[bindex:eindex], axis=0)
        list_grad[1:-1] = (-1) * np.diff(list_mean, 2)
        list_grad[0] = list_mean[0] - list_mean[1]
        list_grad[-1] = list_mean[-1] - list_mean[-2]
        mat_grad[:] = list_grad
        list_coe = np.sum(mat_grad * sijmat, axis=1)
        mat_coe = np.tile(list_coe, (eindex - bindex, 1))
        sinogram[bindex:eindex, :] = sinogram[bindex:eindex, :] + mat_coe
    if sort is True:
        sinogram = util.sort_backward(sinogram, sino_index, axis=0)
    if apply_log is True:
        sinogram = np.exp(-sinogram)
    return sinogram


def remove_stripe_based_fft(sinogram, u=20, n=8, v=1, sort=False):
    """
    Remove stripes using the method in Ref. [1].
    Angular direction is along the axis 0.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
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
    ndarray
        2D array. Stripe-removed sinogram.

    References
    ----------
    .. [1] https://doi.org/10.1063/1.1149043
    .. [2] https://doi.org/10.1364/OE.26.028396
    """
    if sort is True:
        sinogram, sino_index = util.sort_forward(sinogram, axis=0)
    pad = min(150, int(0.1 * np.min(sinogram.shape)))
    sinogram = np.pad(sinogram, ((pad, pad), (0, 0)), mode='mean')
    sinogram = np.pad(sinogram, ((0, 0), (pad, pad)), mode='edge')
    (nrow, ncol) = sinogram.shape
    window_2d = util.make_2d_butterworth_window(ncol, nrow, u, v, n)
    sinogram = fft.ifft2(
        np.fft.ifftshift(np.fft.fftshift(fft.fft2(sinogram)) * window_2d))
    sinogram = np.abs(sinogram[pad:nrow - pad, pad:ncol - pad])
    if sort is True:
        sinogram = util.sort_backward(sinogram, sino_index, axis=0)
    return sinogram


def remove_stripe_based_wavelet_fft(sinogram, level=5, size=1,
                                    wavelet_name="db9", window_name="gaussian",
                                    sort=False, **options):
    """
    Remove stripes using the method in Ref. [1].
    Angular direction is along the axis 0.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    level : int
        Wavelet decomposition level.
    size : int
        Damping parameter. Larger is stronger.
    wavelet_name : str
        Name of a wavelet. Search pywavelets API for a full list.
    window_name : str
        High-pass window. Two options: "gaussian" or "butter".
    sort : bool, optional
        Apply sorting (Ref. [2]) if True.

    Returns
    -------
    array_like
        2D array. Stripe-removed sinogram.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.17.008567
    .. [2] https://doi.org/10.1364/OE.26.028396
    """
    msg = "\n Please use the dictionary format: options={'method':" \
          " 'filter_name', 'para1': parameter_1, 'para2': parameter_2}"
    if sort is True:
        sinogram, sino_index = util.sort_forward(sinogram, axis=0)
    (nrow, ncol) = sinogram.shape
    pad = min(150, int(0.1 * min(nrow, ncol)))
    sinogram = np.pad(sinogram, ((pad, pad), (0, 0)), mode='mean')
    sinogram = np.pad(sinogram, ((0, 0), (pad, pad)), mode='edge')
    output_data = util.apply_wavelet_decomposition(sinogram, wavelet_name,
                                                   level=level)
    output_data = [list(data) for data in output_data]
    n_level = len(output_data[1:])
    for i in range(1, n_level + 1):
        if len(options) == 0:
            (height, width) = output_data[i][1].shape
            window = np.transpose(
                util.make_2d_damping_window(height, width, size, window_name))
            output_data[i][1] = np.real(np.fft.ifft2(np.fft.ifftshift(
                np.fft.fftshift(np.fft.fft2(output_data[i][1])) * window)))
        else:
            if not isinstance(options, dict):
                raise ValueError(msg)
            mat_smooth = np.copy(output_data[i][1])
            for opt_name in options:
                opt = options[opt_name]
                method = tuple(opt.values())[0]
                para = tuple(opt.values())[1:]
                if method in dir(ndi):
                    try:
                        mat_smooth = getattr(ndi, method)(mat_smooth, *para)
                    except:
                        raise ValueError(msg)
                elif method in dir(util):
                    try:
                        mat_smooth = getattr(util, method)(mat_smooth, *para)
                    except:
                        raise ValueError(msg)
                else:
                    raise ValueError("Can't find the method: '{}' in the"
                                     " namespace".format(method))
            output_data[i][1] = mat_smooth
    sinogram = util.apply_wavelet_reconstruction(output_data, wavelet_name)
    sinogram = sinogram[pad:nrow + pad, pad:ncol + pad]
    if sort is True:
        sinogram = util.sort_backward(sinogram, sino_index, axis=0)
    return sinogram


def remove_stripe_based_interpolation(sinogram, snr=3.0, size=51,
                                      drop_ratio=0.1, norm=True, kind="linear",
                                      **options):
    """
    Combination of algorithm 4, 5, and 6 in Ref. [1].
    Angular direction is along the axis 0.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image
    snr : float
        Ratio (>1.0) used to detect stripe locations. Greater is less sensitive.
    size : int
        Window size of the median filter used to detect stripes.
    drop_ratio : float, optional
        Ratio of pixels to be dropped, which is used to to reduce
        the possibility of the false detection of stripes.
    norm : bool, optional
        Apply normalization if True.
    kind : {'linear', 'cubic', 'quintic'}, optional
        The kind of spline interpolation to use. Default is 'linear'.
    options : dict, optional
        Use another smoothing filter rather than the median filter.
        E.g. options={"method": "gaussian_filter", "para1": (1,21))}

    Returns
    -------
    array_like
        2D array. Stripe-removed sinogram.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    msg = "\n Please use the dictionary format: options={'method':" \
          " 'filter_name', 'para1': parameter_1, 'para2': parameter_2}"
    drop_ratio = np.clip(drop_ratio, 0.0, 0.8)
    sinogram = np.copy(sinogram)
    (nrow, ncol) = sinogram.shape
    ndrop = int(0.5 * drop_ratio * nrow)
    sino_sort = np.sort(sinogram, axis=0)
    if len(options) == 0:
        sino_smooth = ndi.median_filter(sino_sort, (1, size))
    else:
        if not isinstance(options, dict):
            raise ValueError(msg)
        sino_smooth = np.copy(sino_sort)
        for opt_name in options:
            opt = options[opt_name]
            method = tuple(opt.values())[0]
            para = tuple(opt.values())[1:]
            if method in dir(ndi):
                try:
                    sino_smooth = getattr(ndi, method)(sino_smooth, *para)
                except:
                    raise ValueError(msg)
            else:
                if method in dir(util):
                    try:
                        sino_smooth = getattr(util, method)(sino_smooth, *para)
                    except:
                        raise ValueError(msg)
                else:
                    raise ValueError("Can't find the method: '{}' in the"
                                     " namespace".format(method))
    list1 = np.mean(sino_sort[ndrop:nrow - ndrop], axis=0)
    list2 = np.mean(sino_smooth[ndrop:nrow - ndrop], axis=0)
    list_fact = np.divide(list1, list2,
                          out=np.ones_like(list1), where=list2 != 0)
    list_mask = util.detect_stripe(list_fact, snr)
    list_mask = np.float32(ndi.binary_dilation(list_mask, iterations=1))
    mat_fact = np.tile(list_fact, (nrow, 1))
    if norm is True:
        sinogram = sinogram / mat_fact
    list_mask[0:2] = 0.0
    list_mask[-2:] = 0.0
    xlist = np.where(list_mask < 1.0)[0]
    ylist = np.arange(nrow)
    zmat = sinogram[:, xlist]
    finter = interpolate.interp2d(xlist, ylist, zmat, kind=kind)
    xlist_miss = np.where(list_mask > 0.0)[0]
    if len(xlist_miss) > 0:
        sinogram[:, xlist_miss] = finter(xlist_miss, ylist)
    return sinogram


def check_zinger_size(mat, max_size):
    """
    Check if the size of a zinger is smaller than a given size.

    Parameters
    ----------
    mat : array_like
        2D array.
    max_size : int
        Maximum size.

    Returns
    -------
    bool
    """
    check = False
    zinger_size = mat.sum()
    if zinger_size <= max_size:
        check = True
    return check


def select_zinger(mat, max_size):
    """
    Select zingers smaller than a certain size.

    Parameters
    ----------
    mat : array_like
        2D array.
    max_size : int
        Maximum size in pixel.

    Returns
    -------
    array_like
        2D binary array.
    """
    list_zin = ndi.find_objects(ndi.label(mat)[0])
    zin_sel = [zin for zin in list_zin
               if check_zinger_size(mat[zin], max_size)]
    mat_out = np.zeros_like(mat)
    for _, j in enumerate(zin_sel):
        mat_out[j] = mat[j]
    return mat_out


def remove_zinger(mat, threshold, size=2):
    """
    Remove zinger using the method in Ref. [1], working on a projection image
    or sinogram image.

    Parameters
    ----------
    mat : array_like
        2D array. Projection image or sinogram image.
    threshold : float
        Threshold to segment zingers. Smaller is more sensitive.
        Recommended range [0.05, 0.1].
    size : int
        Size of a zinger.

    Returns
    -------
    array_like
        2D array. Zinger-removed image.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.418448

    """
    step = np.clip(size, 1, None)
    mat = np.copy(mat)
    mat_ave = []
    for i in range(-size, size + 1, step):
        for j in range(-size, size + 1, step):
            if (i != 0) or (j != 0):
                mat_ave.append(np.roll(np.roll(mat, i, axis=0), j, axis=1))
    mat_ave = np.mean(np.asarray(mat_ave), axis=0)
    mat_ave[mat_ave == 0.0] = 1.0
    mat_nor = mat / mat_ave - 1.0
    mask = np.asarray(mat_nor > threshold, dtype=np.float32)
    mask = select_zinger(mask, size)
    mat[mask > 0.0] = mat_ave[mask > 0.0]
    return mat


def generate_blob_mask(flat, size, snr):
    """
    Generate a binary mask of blobs from a flat-field image (Ref. [1]).

    Parameters
    ----------
    flat : array_like
        2D array. Flat-field image.
    size : float
        Estimated size of the largest blob.
    snr : float
        Ratio used to segment blobs.

    Returns
    -------
    array_like
        2D array. Binary mask.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.418448
    """
    mat = ndi.median_filter(flat, (2, 2))
    mask = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        line = mat[i]
        line_fil = ndi.median_filter(line, size)
        line_fil[line_fil == 0.0] = np.mean(line_fil)
        line_norm = line / line_fil
        mask_1d = util.detect_stripe(line_norm, snr)
        mask_1d[0:2] = 0.0
        mask_1d[-2:] = 0.0
        mask[i] = np.float32(ndi.binary_dilation(mask_1d, iterations=1))
    return mask


def remove_blob_1d(sino_1d, mask_1d):
    """
    Remove blobs in one row of a sinogram, e.g. for a helical scan as shown in
    Ref. [1].

    Parameters
    ----------
    sino_1d : array_like
        1D array. A row of a sinogram.
    mask_1d : array_like
        1D binary mask.

    Returns
    -------
    array_like
        1D array.

    Notes
    -----
    The method is used to remove streak artifacts caused by blobs in
    a sinogram generated from a helical-scan data [1].

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.418448
    """
    sino_1d = np.copy(sino_1d)
    listx = np.where(mask_1d < 1.0)[0]
    listy = sino_1d[listx]
    finter = interpolate.interp1d(listx, listy)
    mask_1d[:2] = 0.0
    mask_1d[-2:] = 0.0
    listx_miss = np.where(mask_1d > 0.0)[0]
    if len(listx_miss) > 0:
        sino_1d[listx_miss] = finter(listx_miss)
    return sino_1d


def remove_blob(mat, mask):
    """
    Remove blobs in an image.

    Parameters
    ----------
    mat : array_like
        2D array. Projection image or sinogram image.
    mask : array_like
        2D binary mask.

    Returns
    -------
    array_like
        2D array.
    """
    mat = np.copy(mat)
    if mat.shape != mask.shape:
        raise ValueError("The image and the mask are not the same shape !!!")
    for i in range(mat.shape[0]):
        array_1d = mat[i]
        mask_1d = mask[i]
        mask_1d[:2] = 0.0
        mask_1d[-2:] = 0.0
        listx = np.where(mask_1d < 1.0)[0]
        listy = array_1d[listx]
        finter = interpolate.interp1d(listx, listy)
        listx_miss = np.where(mask_1d > 0.0)[0]
        if len(listx_miss) > 0:
            array_1d[listx_miss] = finter(listx_miss)
        mat[i] = array_1d
    return mat
