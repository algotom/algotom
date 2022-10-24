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
# Description: Python module of filtering techniques.
# Contributors:
# ============================================================================

"""
Module of filtering methods in the preprocessing stage:

    -   Fresnel filter (denoising or low-pass filter), a simplified version of
        the well-known Paganin's filter.
    -   Double-wedge filter.
"""

import numpy as np
import numpy.fft as fft
import algotom.prep.conversion as conv


def make_fresnel_window(height, width, ratio, dim):
    """
    Create a low pass window based on the Fresnel propagator.
    It is used to denoise a projection image (dim=2) or a
    sinogram image (dim=1).

    Parameters
    ----------
    height : int
        Image height
    width : int
        Image width
    ratio : float
        To define the shape of the window.
    dim : {1, 2}
        Use "1" if working on a sinogram image and "2" if working on
        a projection image.

    Returns
    -------
    array_like
        2D array.
    """
    ycenter = int(np.ceil((height - 1) * 0.5))
    xcenter = int(np.ceil((width - 1) * 0.5))
    if dim == 2:
        ulist = (1.0 * np.arange(0, width) - xcenter) / width
        vlist = (1.0 * np.arange(0, height) - ycenter) / height
        u, v = np.meshgrid(ulist, vlist)
        window = 1.0 + ratio * (u ** 2 + v ** 2)
    else:
        ulist = (1.0 * np.arange(0, width) - xcenter) / width
        win1d = 1.0 + ratio * ulist ** 2
        window = np.tile(win1d, (height, 1))
    return window


def fresnel_filter(mat, ratio, dim=1, window=None, pad=150, apply_log=True):
    """
    Apply a low-pass filter based on the Fresnel propagator to an image
    (Ref. [1]). It can be used for improving the contrast of an image.
    It's simpler than the well-known Paganin's filter (Ref. [2]).

    Parameters
    ----------
    mat : array_like
        2D array. Projection image or sinogram image.
    ratio : float
        Define the shape of the window. Larger is more smoothing.
    dim : {1, 2}
        Use "1" if working on a sinogram image and "2" if working on
        a projection image.
    window : array_like, optional
        Window for deconvolution.
    pad : int
        Padding width.
    apply_log : bool, optional
        Apply the logarithm function to the sinogram before filtering.

    Returns
    -------
    array_like
        2D array. Filtered image.

    References
    ----------
    [1] : https://doi.org/10.1364/OE.418448
    [2] : https://tinyurl.com/2f8nv875
    """
    if apply_log:
        mat = -np.log(mat)
    (nrow, ncol) = mat.shape
    if dim == 2:
        if window is None:
            window = make_fresnel_window(nrow, ncol, ratio, dim)
        mat_pad = np.pad(mat, pad, mode="edge")
        win_pad = np.pad(window, pad, mode="edge")
        mat_dec = fft.ifft2(fft.fft2(mat_pad) / fft.ifftshift(win_pad))
        mat_dec = np.real(mat_dec[pad:pad + nrow, pad:pad + ncol])
    else:
        if window is None:
            window = make_fresnel_window(nrow, ncol, ratio, dim)
        mat_pad = np.pad(mat, ((0, 0), (pad, pad)), mode='edge')
        win_pad = np.pad(window, ((0, 0), (pad, pad)), mode="edge")
        mat_fft = np.fft.fftshift(fft.fft(mat_pad), axes=1) / win_pad
        mat_dec = fft.ifft(np.fft.ifftshift(mat_fft, axes=1))
        mat_dec = np.real(mat_dec[:, pad:pad + ncol])
    if apply_log:
        mat_dec = np.exp(-mat_dec)
    return np.float32(mat_dec)


def make_double_wedge_mask(height, width, radius):
    """
    Generate a double-wedge binary mask using Eq. (3) in Ref. [1].
    Values outside the double-wedge region correspond to 0.0.

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
    [1] : https://doi.org/10.1364/OE.22.019078
    """
    du = 1.0 / width
    dv = (height - 1.0) / (height * 2.0 * np.pi)
    ndrop = 1
    ycenter = np.int16(np.ceil((height - 1) / 2.0))
    xcenter = np.int16(np.ceil((width - 1) / 2.0))
    mask = np.ones((height, width), dtype=np.float32)
    for i in range(height):
        num = np.int16(np.ceil(((i - ycenter) * dv / radius) / du))
        (p1, p2) = np.int16(np.clip(
            np.sort((-num + xcenter, num + xcenter)), 0, width - 1))
        mask[i, p1:p2 + 1] = 0.0
    mask[ycenter - ndrop:ycenter + ndrop + 1, :] = 1.0
    return mask


def double_wedge_filter(sinogram, center=0, sino_type="180", iteration=5,
                        mask=None, ratio=1.0, pad=250):
    """
    Apply double-wedge filter to a sinogram image (Ref. [1]).

    Parameters
    ----------
    sinogram : array_like
        2D array. 180-degree sinogram or 360-degree sinogram.
    center : float, optional
        Center-of-rotation. No need for a 360-sinogram.
    sino_type : {"180", "360"}
        Sinogram type : 180-degree or 360-degree.
    iteration : int
        Number of iteration.
    mask : array_like, optional
        Double-wedge binary mask.
    ratio : float, optional
        Define the cut-off angle of the double-wedge filter.
    pad : int
        Padding width.

    Returns
    -------
    array_like
        2D array. Filtered sinogram.

    References
    ----------
    [1] : https://doi.org/10.1364/OE.418448
    """
    if not (sino_type == "180" or sino_type == "360"):
        raise ValueError("!!! Use only one of two options: '180' or '360'!!!")
    if sino_type == "180":
        nrow0 = sinogram.shape[0]
        if center == 0:
            raise ValueError(
                "Please provide the location of the rotation axis")
        sinogram = conv.convert_sinogram_180_to_360(sinogram, center)
    (nrow, ncol) = sinogram.shape
    ncol_pad = ncol + 2 * pad
    if mask is None:
        mask = make_double_wedge_mask(nrow, ncol_pad, ratio * ncol / 2.0)
    else:
        if mask.shape != (nrow, ncol_pad):
            raise ValueError(
                "Shape of the left-right padded sinogram {0} and the mask "
                "{1} is not the same!!!".format((nrow, ncol_pad), mask.shape))
    sino_filt = np.copy(sinogram)
    for i in range(iteration):
        sino_filt = np.pad(sino_filt, ((0, 0), (pad, pad)), mode="edge")
        sino_filt = np.real(fft.ifft2(
            fft.ifftshift(fft.fftshift(fft.fft2(sino_filt)) * mask)))
        sino_filt = sino_filt[:, pad:ncol + pad]
    if sino_type == "180":
        sino_filt = sino_filt[:nrow0]
    return sino_filt
