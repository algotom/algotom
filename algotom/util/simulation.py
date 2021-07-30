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
# Description: Simulation methods
# Contributors:
# ============================================================================


"""
Module of simulation methods:
1- Methods for designing a customized 2D phantom.
2- Method for calculating a sinogram of a phantom based on the Fourier
   slice theorem.
3- Methods for adding artifacts to a simulated sinogram.
"""

import numpy as np
import scipy.signal.windows as win
import algotom.util.utility as util
import numpy.fft as fft


def make_elliptic_mask(size, center, length, angle):
    """
    Create an elliptic mask.

    Parameters
    -----------
    size : int
        Size of a square array.
    center : float or tuple of float
        Ellipse center.
    length : float or tuple of float
        Lengths of ellipse axes.
    angle : float
        Rotation angle (Degree) of the ellipse.

    Returns
    ------
    array_like
         Square array.
    """
    mask = np.zeros((size, size), dtype=np.float32)
    icenter = size // 2
    if isinstance(length, tuple):
        (x_len, y_len) = length
    else:
        x_len = y_len = length
    if isinstance(center, tuple):
        (x_cen, y_cen) = center
    else:
        x_cen = y_cen = center
    angle = - angle * np.pi / 180.0
    x_list = np.arange(size) - icenter - x_cen
    y_list = - np.arange(size) + icenter - y_cen
    x_mat, y_mat = np.meshgrid(x_list, y_list)
    x_mat1 = (x_mat * np.cos(angle) - y_mat * np.sin(angle)) / (0.5 * x_len)
    y_mat1 = (x_mat * np.sin(angle) + y_mat * np.cos(angle)) / (0.5 * y_len)
    r_mat = np.sqrt(x_mat1 ** 2 + y_mat1 ** 2)
    mask_check = r_mat <= 1.01
    mask[mask_check] = 1.0
    return mask


def make_rectangular_mask(size, center, length, angle):
    """
    Create a rectangular mask.

    Parameters
    -----------
    size : int
        Size of a square array.
    center : float or tuple of float
        Center of the mask.
    length : float or tuple of float
        Lengths of the rectangular mask.
    angle : float
        Rotation angle (Degree) of the mask.

    Returns
    ------
    array_like
         Square array.
    """
    mask = np.zeros((size, size), dtype=np.float32)
    icenter = size // 2
    if isinstance(length, tuple):
        (x_len, y_len) = length
    else:
        x_len = y_len = length
    if isinstance(center, tuple):
        (x_cen, y_cen) = center
    else:
        x_cen = y_cen = center
    angle = - angle * np.pi / 180.0
    x_list = np.arange(size) - icenter - x_cen
    y_list = - np.arange(size) + icenter - y_cen
    x_mat, y_mat = np.meshgrid(x_list, y_list)
    x_mat1 = np.abs(
        (x_mat * np.cos(angle) - y_mat * np.sin(angle)) / (0.5 * x_len))
    y_mat1 = np.abs(
        (x_mat * np.sin(angle) + y_mat * np.cos(angle)) / (0.5 * y_len))
    mask_check = (x_mat1 <= 1.01) & (y_mat1 <= 1.01)
    mask[mask_check] = 1.0
    return mask


def make_triangular_mask(size, center, length, angle):
    """
    Create an isosceles triangle mask.

    Parameters
    -----------
    size : int
        Size of a square array.
    center : float or tuple of float
        Center of the mask.
    length : float or tuple of float
        Lengths of the mask.
    angle : float
        Rotation angle (Degree) of the mask.

    Returns
    ------
    array_like
         Square array.
    """
    mask = make_rectangular_mask(size, center, length, angle)
    if isinstance(length, tuple):
        (x_len, y_len) = length
    else:
        x_len = y_len = length
    if isinstance(center, tuple):
        (x_cen, y_cen) = center
    else:
        x_cen = y_cen = center
    angle = np.deg2rad(angle)
    x_len1 = np.sqrt(x_len ** 2 + (0.5 * y_len) ** 2)
    angle1 = np.arctan2(0.5 * y_len, x_len)
    y_len1 = 2 * np.sin(angle1) * x_len
    x_off = - 0.5 * y_len1 * np.sin(angle1)
    y_off = 0.5 * y_len1 * np.cos(angle1) + 0.5 * x_len1 * np.sin(angle1)
    x_off1 = x_off * np.cos(angle) - y_off * np.sin(angle)
    y_off1 = x_off * np.sin(angle) + y_off * np.cos(angle)
    x_cen1 = x_cen + x_off1 + np.sign(x_off1) * 0.5
    y_cen1 = y_cen + y_off1 + np.sign(y_off1) * 0.5
    mask1 = make_rectangular_mask(size, (x_cen1, y_cen1), (x_len1, y_len1),
                                  np.rad2deg(angle + angle1))
    y_off = -y_off
    x_off1 = x_off * np.cos(angle) - y_off * np.sin(angle)
    y_off1 = x_off * np.sin(angle) + y_off * np.cos(angle)
    x_cen1 = x_cen + x_off1 + np.sign(x_off1) * 0.5
    y_cen1 = y_cen + y_off1 + np.sign(y_off1) * 0.5
    mask2 = make_rectangular_mask(size, (x_cen1, y_cen1), (x_len1, y_len1),
                                  np.rad2deg(angle - angle1))
    mask = np.clip(mask - mask1, 0.0, None)
    mask = np.clip(mask - mask2, 0.0, None)
    return mask


def make_line_target(size):
    """
    Create line patterns for testing the resolution of a reconstructed image.

    Parameters
    -----------
    size : int
        Size of a square array.

    Returns
    ------
    array_like
         Square array.
    """
    mask = np.zeros((size, size), dtype=np.float32)
    y_cen = 0
    line_hei = size // 16
    gap = 6
    check = True
    line_wid = 0
    start = line_hei // 2 + 1
    while check:
        line_wid = line_wid + 1
        stop = start + 3 * 2 * line_wid
        if stop > size // 2 - gap:
            check = False
        else:
            if line_wid % 2 == 1:
                for x_cen in np.arange(start, stop, 2 * line_wid):
                    mask += make_rectangular_mask(size, (x_cen, y_cen),
                                                  (line_wid, line_hei), 0.0)
            else:
                for x_cen in np.arange(start, stop, 2 * line_wid):
                    mask += make_rectangular_mask(size, (x_cen, y_cen),
                                                  (line_wid - 1, line_hei), 0.0)
                for x_cen in np.arange(start + line_wid // 2, stop,
                                       2 * line_wid):
                    mask += make_rectangular_mask(size, (x_cen, y_cen),
                                                  (1, line_hei), 0.0)
        start = stop + gap
    start = - line_hei // 2 - line_wid // 2
    line_wid = line_wid - 1
    while line_wid > 0:
        stop = start - 3 * 2 * line_wid
        if line_wid % 2 == 1:
            for x_cen in np.arange(start, stop, -2 * line_wid):
                mask += make_rectangular_mask(size, (x_cen, y_cen),
                                              (line_wid, line_hei), 0.0)
        else:
            for x_cen in np.arange(start, stop, -2 * line_wid):
                mask += make_rectangular_mask(size, (x_cen, y_cen),
                                              (line_wid - 1, line_hei), 0.0)
            for x_cen in np.arange(start - line_wid // 2, stop,
                                   -2 * line_wid):
                mask += make_rectangular_mask(size, (x_cen, y_cen),
                                              (1, line_hei), 0.0)
        start = stop - gap
        line_wid = line_wid - 1
    mask = mask + np.transpose(mask)
    list1 = mask[size // 2]
    list_pos = np.where(list1 == 1.0)[0]
    circle_mask = make_elliptic_mask(size, 0.0,
                                     list_pos[-1] - list_pos[0] + line_hei, 0.0)
    return circle_mask - mask


def make_face_phantom(size):
    """
    Create a face phantom for testing reconstruction methods.

    Parameters
    -----------
    size : int
        Size of a square array.

    Returns
    ------
    array_like
         Square array.
    """
    half = size // 2
    mask = np.zeros((size, size), dtype=np.float32)
    face1 = make_elliptic_mask(size, 0.0, (size / 1.3, 0.95 * size), 0.0)
    face2 = -0.6 * make_elliptic_mask(size, (0.0, -0.01 * size),
                                      (0.94 * size / 1.3, 0.94 * size), 0.0)
    face = face1 + face2
    x_rat_eye = 0.3
    y_rat_eye = 0.4
    eye1 = 0.6 * make_elliptic_mask(size, (-x_rat_eye * half, y_rat_eye * half),
                                    (0.15 * size, 0.05 * size), 0.0)
    pupil1a = -0.8 * make_elliptic_mask(size,
                                        (-x_rat_eye * half, y_rat_eye * half),
                                        (0.048 * size, 0.048 * size), 0.0)
    pupil1b = -0.2 * make_elliptic_mask(size,
                                        (-x_rat_eye * half, y_rat_eye * half),
                                        (0.015 * size, 0.015 * size), 0.0)
    pupil1 = pupil1a + pupil1b
    eyebrow1a = -0.3 * make_rectangular_mask(size,
                                             (-x_rat_eye * half, 0.54 * half),
                                             (0.18 * size, 0.02 * size), -5.0)
    eyebrow1b = -0.3 * make_rectangular_mask(size, (-x_rat_eye * half,
                                                    0.54 * half - 0.01 * half),
                                             (0.18 * size, 0.015 * size), -10.0)
    eyebrow1 = np.clip(eyebrow1a + eyebrow1b, -0.3, 0.0)
    eye2 = 0.6 * make_elliptic_mask(size, (x_rat_eye * half, y_rat_eye * half),
                                    (0.15 * size, 0.05 * size), 0.0)
    pupil2a = -0.8 * make_elliptic_mask(size,
                                        (x_rat_eye * half, y_rat_eye * half),
                                        (0.048 * size, 0.048 * size), 0.0)
    pupil2b = -0.2 * make_elliptic_mask(size,
                                        (x_rat_eye * half, y_rat_eye * half),
                                        (0.015 * size, 0.015 * size), 0.0)
    pupil2 = pupil2a + pupil2b
    eyebrow2a = -0.3 * make_rectangular_mask(size,
                                             (x_rat_eye * half, 0.54 * half),
                                             (0.18 * size, 0.02 * size), 5.0)
    eyebrow2b = -0.3 * make_rectangular_mask(size, (x_rat_eye * half,
                                                    0.54 * half - 0.01 * half),
                                             (0.18 * size, 0.015 * size), 10.0)
    eyebrow2 = np.clip(eyebrow2a + eyebrow2b, -0.3, 0.0)
    eye = eye1 + eye2 + pupil1 + pupil2 + eyebrow1 + eyebrow2
    nose1 = 0.2 * make_rectangular_mask(size, (0, 0),
                                        (0.05 * size, 0.25 * size), 0.0)
    nose2 = 0.2 * make_rectangular_mask(size, (0 + 0.015 * size, 0),
                                        (0.04 * size, 0.25 * size), 9.0)
    nose3 = 0.2 * make_rectangular_mask(size, (0 - 0.015 * size, 0),
                                        (0.04 * size, 0.25 * size), -9.0)
    nose = np.clip(nose1 + nose2 + nose3, 0.0, 0.2)
    mouth1 = 0.2 * make_rectangular_mask(size, (0.0, -0.22 * size),
                                         (0.24 * size, 0.055 * size), 0.0)
    mouth2 = 0.2 * make_elliptic_mask(size, (0.0, -0.22 * size + 0.025 * size),
                                      (0.24 * size, 0.07 * size), 0.0)
    mouth = mouth1 + mouth2
    mouth[mouth < 0.3] = 0.0
    beard1 = -0.4 * make_rectangular_mask(size, (0.0, -0.32 * size),
                                          (0.005 * size, 0.1 * size), 0.0)
    beard2 = -0.4 * make_rectangular_mask(size, (-0.02 * size, -0.32 * size),
                                          (0.005 * size, 0.1 * size), -10.0)
    beard3 = -0.4 * make_rectangular_mask(size, (0.02 * size, -0.32 * size),
                                          (0.005 * size, 0.1 * size), 10.0)
    beard = beard1 + beard2 + beard3
    mask += face + eye + nose + mouth + beard
    return mask


def make_sinogram(mat, angles, pad_rate=0.5, pad_mode="edge"):
    """
    Create a sinogram (series of 1D projections) from a 2D image based on the
    Fourier slice theorem (Ref. [1]).

    Parameters
    ----------
    mat : array_like
        Square array.
    angles : array_like
        1D array. List of angles (in radian) for projecting.
    pad_rate : float
        To apply padding before the FFT. The padding width equals to
        (pad_rate * image_width).
    pad_mode : str
        Padding method. Full list can be found at numpy.pad documentation.

    References
    ----------
    .. [1] https://doi.org/10.1071/PH560198
    """
    (nrow0, ncol0) = mat.shape
    if nrow0 != ncol0:
        raise ValueError(
            "Width and height of the image are not the same")
    if np.max(np.abs(angles)) > np.pi:
        print("!!! Warning !!!\nMaking sure that the angles are converted to "
              "Radian and in the range of [0; Pi]")
    pad = int(pad_rate * nrow0)
    mat_pad = np.pad(mat, pad, mode=pad_mode)
    if mat_pad.shape[0] % 2 == 0:
        mat_pad = np.pad(mat_pad, ((0, 1), (0, 1)), mode="edge")
    (nrow, ncol) = mat_pad.shape
    xcenter = (ncol - 1.0) * 0.5
    ycenter = (nrow - 1.0) * 0.5
    r_max = np.floor(max(xcenter, ycenter))
    r_list = np.linspace(-r_max, r_max, ncol)
    theta_list = - np.asarray(angles)
    r_mat, theta_mat = np.meshgrid(r_list, theta_list)
    x_mat = np.float32(
        np.clip(xcenter + r_mat * np.cos(theta_mat), 0, ncol - 1))
    y_mat = np.float32(
        np.clip(ycenter + r_mat * np.sin(theta_mat), 0, nrow - 1))
    mat_fft = fft.fftshift(fft.fft2(fft.ifftshift(mat_pad)))
    mat_real = np.real(mat_fft)
    mat_imag = np.imag(mat_fft)
    sino_real = util.mapping(mat_real, x_mat, y_mat, order=5, mode="reflect")
    sino_imag = util.mapping(mat_imag, x_mat, y_mat, order=5, mode="reflect")
    sinogram = np.real(fft.fftshift(
        fft.ifft(fft.ifftshift(sino_real + 1j * sino_imag, axes=1)), axes=1))
    return sinogram[:, pad:ncol0 + pad]


def add_noise(mat, noise_ratio=0.1):
    """
    Add Gaussian noise to an image.

    Parameters
    ----------
    mat : array_like
        2D array
    noise_ratio : float
        Ratio between the noise level and the mean of the array.

    Returns
    -------
    array_like
    """
    num_mean = np.mean(mat[mat != 0.0])
    noise_mean = num_mean * noise_ratio
    noise = np.random.normal(noise_mean, noise_mean * 0.5, size=mat.shape)
    return mat + noise


def add_stripe_artifact(sinogram, size, position, strength_ratio=0.2,
                        stripe_type="partial"):
    """
    Add stripe artifacts to a sinogram.

    Parameters
    ----------
    sinogram: array_like
        2D array. Sinogram image.
    size : int
        Size of stripe artifact.
    position : int
        Position of the stripe.
    strength_ratio : float
        To define the strength of the artifact. The value is in the range of
        [0.0, 1.0].
    stripe_type : {"partial", "full", "dead", "fluctuating"}
        Type of stripe as classified in Ref. [1].

    Returns
    -------
    array_like

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    sinogram = np.copy(sinogram)
    (nrow, ncol) = sinogram.shape
    position = np.clip(position, 0, ncol - size - 1)
    strength_ratio = np.clip(strength_ratio, 0.0, 1.0)
    stripe = sinogram[:, position: position + size]
    if stripe_type == "partial":
        stripe_sort, mat_idx = util.sort_forward(stripe, axis=0)
        pos = int((1.0 - strength_ratio) * nrow)
        list_ratio = np.ones(nrow, dtype=np.float32)
        list_ratio[pos:nrow] = np.linspace(1.0, 1.0 - strength_ratio,
                                           nrow - pos)
        mat_ratio = np.tile(list_ratio, (size, 1))
        stripe_sort = stripe_sort * np.transpose(mat_ratio)
        stripe_mod = util.sort_backward(stripe_sort, mat_idx, axis=0)
    elif stripe_type == "dead":
        stripe_mod = np.ones_like(stripe) * strength_ratio * np.max(sinogram)
    elif stripe_type == "fluctuating":
        std_dev = np.mean(sinogram[sinogram != 0.0]) * strength_ratio
        noise = np.random.normal(0.0, std_dev, size=stripe.shape)
        stripe_mod = stripe + noise
    else:
        list_ratio = (1 - strength_ratio) * np.ones(nrow, dtype=np.float32)
        mat_ratio = np.tile(list_ratio, (size, 1))
        stripe_mod = stripe * np.transpose(mat_ratio)
    sinogram[:, position: position + size] = stripe_mod
    return sinogram


def convert_to_Xray_image(sinogram, global_max=None):
    """
    Convert a simulated sinogram to an equivalent X-ray image.

    Parameters
    ----------
    sinogram : array_like
        2D array.
    global_max : float
        Maximum value used for normalizing array values to stay in the range
        of [0.0, 1.0].

    Returns
    -------
    array_like
    """
    if global_max is None:
        global_max = np.max(sinogram)
    sinogram = 1.0 * sinogram / global_max
    return np.exp(-sinogram)


def add_background_fluctuation(sinogram, strength_ratio=0.2):
    """
    Fluctuate the background of a sinogram image using a Gaussian profile beam.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    strength_ratio : float
        To define the strength of the variation. The value is in the range of
        [0.0, 1.0].

    Returns
    -------
    array_like
    """
    sinogram = np.copy(sinogram)
    (nrow, ncol) = sinogram.shape
    list_fact = 1.0 - np.random.rand(nrow) * strength_ratio
    list_shift = np.int16(
        (0.5 - np.random.rand(nrow)) * strength_ratio * ncol * 0.5)
    for i in range(nrow):
        sinogram[i] = sinogram[i] * np.roll(
            win.gaussian(ncol, 0.5 * list_fact[i] * ncol), list_shift[i])
    return sinogram
