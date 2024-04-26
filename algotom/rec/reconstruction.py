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
# Description: Python module of reconstruction methods
# Contributors:
# ============================================================================

"""
Module of reconstruction methods:

    -   Filtered back-projection (FBP) method for GPU and CPU.
    -   Back-projection filtering (BPF) method for GPU and CPU.
    -   Direct Fourier inversion (DFI) method.
    -   Wrapper for Astra-Toolbox reconstruction methods (optional)
    -   Wrapper for Tomopy-gridrec reconstruction method (optional)
    -   Automatic determination of the center of rotation.
    -   Tool to assist in manual determination of the center of rotation.
"""

import math
import warnings
import multiprocessing as mp
import numpy as np
import numpy.fft as fft
import scipy
from scipy import signal
from scipy import ndimage as ndi
from numba import jit, cuda, prange
from joblib import Parallel, delayed
import algotom.util.utility as util
import algotom.io.loadersaver as losa
from numba.core.errors import NumbaPerformanceWarning
# warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)


def make_smoothing_window(filter_name, width):
    """
    Make a 1d smoothing window.

    Parameters
    ----------
    filter_name : {"hann", "bartlett", "blackman", "hamming", "nuttall",\
                   "parzen", "triang"}
        Window function used for filtering.
    width : int
        Width of the window.

    Returns
    -------
    array_like
        1D array.
    """
    if filter_name == 'hann':
        window = signal.windows.hann(width)
    elif filter_name == 'bartlett':
        window = signal.windows.bartlett(width)
    elif filter_name == 'blackman':
        window = signal.windows.blackman(width)
    elif filter_name == 'hamming':
        window = signal.windows.hamming(width)
    elif filter_name == 'nuttall':
        window = signal.windows.nuttall(width)
    elif filter_name == 'parzen':
        window = signal.windows.parzen(width)
    elif filter_name == 'triang':
        window = signal.windows.triang(width)
    else:
        window = np.ones(width)
    return window


def make_2d_ramp_window(height, width, filter_name=None):
    """
    Make the 2d ramp window (in the Fourier space) by repeating the 1d ramp
    window with the option of adding a smoothing window.

    Parameters
    ----------
    height : int
        Height of the window.
    width : int
        Width of the window.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming",\
                  "nuttall", "parzen", "triang"}
         Name of a smoothing window used.

    Returns
    -------
    complex ndarray
        2D array.
    """
    ramp_win = np.arange(0.0, width) - np.ceil((width - 1.0) / 2)
    ramp_win[ramp_win == 0.0] = 0.25
    ramp_win[ramp_win % 2 == 0.0] = 0.0
    for i in range(width):
        if ramp_win[i] % 2 == 1.0:
            ramp_win[i] = - 1.0 / (ramp_win[i] * np.pi) ** 2
    window = make_smoothing_window(filter_name, width)
    ramp_fourier = fft.fftshift(fft.fft(ramp_win)) * window
    ramp_fourier_2d = np.tile(ramp_fourier, (height, 1))
    return ramp_fourier_2d


def apply_ramp_filter(sinogram, ramp_win=None, filter_name=None, pad=None,
                      pad_mode="edge"):
    """
    Apply the ramp filter to a sinogram with the option of adding a smoothing
    filter.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    ramp_win : complex ndarray or None
        Ramp window in the Fourier space.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming",\
                  "nuttall", "parzen", "triang"}
         Name of a smoothing window used.
    pad : int or None
        To apply padding before the FFT. The value is set to 10% of the image
        width if None is given.
    pad_mode : str
        Padding method. Full list can be found at numpy_pad documentation.

    Returns
    -------
    array_like
        Filtered sinogram.
    """
    (nrow, ncol) = sinogram.shape
    if pad is None:
        pad = min(int(0.15 * ncol), 150)
    sino_pad = np.pad(sinogram, ((0, 0), (pad, pad)), mode=pad_mode)
    if (ramp_win is None) or (ramp_win.shape != sino_pad.shape):
        ramp_win = make_2d_ramp_window(nrow, ncol + 2 * pad, filter_name)
    sino_fft = fft.fftshift(fft.fft(sino_pad), axes=1) * ramp_win
    sino_filtered = np.real(
        fft.ifftshift(fft.ifft(fft.ifftshift(sino_fft, axes=1)), axes=1))
    return np.ascontiguousarray(sino_filtered[:, pad:ncol + pad])


@cuda.jit
def __back_projection_gpu_kernel(recon, sinogram, angles, center, sino_height,
                                 sino_width, edge_pad):  # pragma: no cover
    """
    GPU-kernel function performing back-projection.
    """
    (x_index, y_index) = cuda.grid(2)
    if (x_index < sino_width) and (y_index < sino_width):
        sino_width1 = sino_width - 1
        icenter = 0.5 * sino_width1
        x_cor = (x_index - icenter)
        y_cor = (y_index - icenter)
        num = 0.0
        for i in range(sino_height):
            theta = -angles[i]
            x_pos = x_cor * math.cos(theta) + y_cor * math.sin(theta)
            f_pos = x_pos + center
            if 0 <= f_pos <= sino_width1:
                d_pos = int(math.floor(f_pos))
                u_pos = int(math.ceil(f_pos))
                if u_pos != d_pos:
                    yd = sinogram[i, d_pos]
                    yu = sinogram[i, u_pos]
                    val = yd + (yu - yd) * (f_pos - d_pos)
                else:
                    val = sinogram[i, d_pos]
                num += val
            else:
                if edge_pad:
                    if f_pos < 0:
                        val = sinogram[i, 0]
                    else:
                        val = sinogram[i, sino_width1]
                    num += val
        recon[y_index, x_index] = num


def back_projection_gpu(sinogram, angles, center, block=(16, 16),
                        edge_pad=False):  # pragma: no cover
    """
    Implement the back-projection algorithm using GPU.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    angles : array_like
        1D array. Angles (radian) corresponding to the sinogram.
    center : float
        Center of rotation.
    edge_pad : bool
        Enable/disable edge padding.
    block : tuple of int, optional
        Size of a GPU block. E.g. (8, 8), (16, 16), (32, 32), ...

    Returns
    -------
    recon : array_like
        Back-projected image.
    """
    (nrow, ncol) = sinogram.shape
    sinogram = np.ascontiguousarray(sinogram)
    grid = (int(np.ceil(1.0 * ncol / block[0])),
            int(np.ceil(1.0 * ncol / block[1])))
    recon = np.zeros((ncol, ncol), dtype=np.float32)
    __back_projection_gpu_kernel[grid, block](recon, np.float32(sinogram),
                                              np.float32(angles),
                                              np.float32(center),
                                              np.int32(nrow), np.int32(ncol),
                                              edge_pad)
    return recon


@cuda.jit
def __back_projection_gpu_chunk_kernel(recons, sinograms, angles, center,
                                       sino_height, sino_width,
                                       num_sino, edge_pad):  # pragma: no cover
    """
    GPU-kernel function performing back-projection for a chunk of sinograms.
    """
    (x_index, y_index) = cuda.grid(2)
    if (x_index < sino_width) and (y_index < sino_width):
        sino_width1 = sino_width - 1
        icenter = 0.5 * sino_width1
        x_cor = (x_index - icenter)
        y_cor = (y_index - icenter)
        for i in range(sino_height):
            theta = -angles[i]
            x_pos = x_cor * math.cos(theta) + y_cor * math.sin(theta)
            f_pos = x_pos + center
            if 0 <= f_pos <= sino_width1:
                d_pos = int(math.floor(f_pos))
                u_pos = int(math.ceil(f_pos))
                for j in range(num_sino):
                    if u_pos != d_pos:
                        yd = sinograms[i, j, d_pos]
                        yu = sinograms[i, j, u_pos]
                        val = yd + (yu - yd) * (f_pos - d_pos)
                    else:
                        val = sinograms[i, j, d_pos]
                    recons[y_index, j, x_index] += val
            else:
                if edge_pad:
                    for j in range(num_sino):
                        if f_pos < 0:
                            val = sinograms[i, j, 0]
                        else:
                            val = sinograms[i, j, sino_width1]
                        recons[y_index, j, x_index] += val


def back_projection_gpu_chunk(sinograms, angles, center, block=(16, 16),
                              edge_pad=False):  # pragma: no cover
    """
    Implement the back-projection algorithm for a chunk of sinograms using GPU.
    Axis of a sinogram/slice in the 3D array is 1.

    Parameters
    ----------
    sinograms : array_like
        3D array. Sinogram images.
    angles : array_like
        1D array. Angles (radian) corresponding to a sinogram.
    center : float
        Center of rotation.
    edge_pad : bool
        Enable/disable edge padding.
    block : tuple of int, optional
        Size of a GPU block. E.g. (8, 8), (16, 16), (32, 32), ...

    Returns
    -------
    recons : array_like
        Back-projected images.
    """
    (nrow, num_sino, ncol) = sinograms.shape
    sinograms = np.ascontiguousarray(sinograms)
    recons = np.zeros((ncol, num_sino, ncol), dtype=np.float32)
    grid = (int(np.ceil(1.0 * ncol / block[0])),
            int(np.ceil(1.0 * ncol / block[1])))
    __back_projection_gpu_chunk_kernel[grid, block](recons,
                                                    np.float32(sinograms),
                                                    np.float32(angles),
                                                    np.float32(center),
                                                    np.int32(nrow),
                                                    np.int32(ncol),
                                                    np.int32(num_sino),
                                                    edge_pad)
    return recons


@jit(nopython=True, parallel=True, cache=True)
def back_projection_cpu(sinogram, angles, center,
                        edge_pad=False):  # pragma: no cover
    """
    Implement the back-projection algorithm using CPU.

    Parameters
    ----------
    sinogram : array_like
        2D array. (Filtered) sinogram image.
    angles : array_like
        1D array. Angles (radian) corresponding to the sinogram.
    center : float
        Center of rotation.
    edge_pad : bool
        Enable/disable edge padding.

    Returns
    -------
    recon : array_like
        Square array, back-projected image.
    """
    (sino_height, sino_width) = sinogram.shape
    sino_width1 = sino_width - 1
    icenter = 0.5 * sino_width1
    recon = np.zeros((sino_width, sino_width), dtype=np.float32)
    for i in prange(sino_height):
        theta = - angles[i]
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        for y_index in range(sino_width):
            y_cor = y_index - icenter
            for x_index in range(sino_width):
                x_pos = (x_index - icenter) * cos_theta + y_cor * sin_theta
                f_pos = x_pos + center
                if 0 <= f_pos <= sino_width1:
                    d_pos = np.int32(np.floor(f_pos))
                    u_pos = np.int32(np.ceil(f_pos))
                    if u_pos != d_pos:
                        yd = sinogram[i, d_pos]
                        yu = sinogram[i, u_pos]
                        val = yd + (yu - yd) * (f_pos - d_pos)
                    else:
                        val = sinogram[i, d_pos]
                    recon[y_index, x_index] += val
                else:
                    if edge_pad:
                        if f_pos < 0:
                            val = sinogram[i, 0]
                        else:
                            val = sinogram[i, sino_width1]
                        recon[y_index, x_index] += val
    return recon


def fbp_reconstruction(sinogram, center, angles=None, ratio=1.0, ramp_win=None,
                       filter_name="hann", pad=None, pad_mode="edge",
                       apply_log=True, gpu=True, block=(16, 16), ncore=None):
    """
    Apply the FBP (filtered back-projection) reconstruction method to a
    sinogram-image or a chunk of sinogram-images. Angular axis is 0.
    If input is 3D array, the slicing axis of sinograms must be 1,
    e.g. data[:, index, :].

    Parameters
    ----------
    sinogram : array_like
        2D/3D array. Sinogram image.
    center : float
        Center of rotation.
    angles : array_like, optional
        1D array. List of angles (in radian) corresponding to the sinogram.
    ratio : float, optional
        To apply a circle mask to the reconstructed image.
    ramp_win : complex ndarray, optional
        Ramp window in the Fourier space. Generated if None.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming",\
                  "nuttall", "parzen", "triang"}
        Apply a smoothing filter.
    pad : int, optional
        To apply padding before the FFT. The value is set to 10% of the image
        width if None is given.
    pad_mode : str, optional
        Padding method. Full list can be found at numpy_pad documentation.
    apply_log : bool, optional
        Apply the logarithm function to the sinogram before reconstruction.
    gpu : bool, optional
        Use GPU for computing if True.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (8, 8), (16, 16), (32, 32), ...
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.

    Returns
    -------
    array_like
        Square array. Reconstructed image.
    """
    input_3d = False
    if len(sinogram.shape) == 3:
        (nrow, num_sino, ncol) = sinogram.shape
        input_3d = True
    else:
        num_sino = 1
        (nrow, ncol) = sinogram.shape
    if center < 0 or center >= ncol:
        raise ValueError("Center is out of the range [0, {}]".format(ncol - 1))
    if angles is None:
        angles = np.deg2rad(np.linspace(0.0, 180.0, nrow))
    else:
        if len(angles) != nrow:
            raise ValueError("!!!Number of angles is not the same as the row "
                             "number of the sinogram!!!")
    if gpu is True:
        if cuda.is_available() is False:
            warnings.warn("!!!No Nvidia GPU found! Run with CPU instead!!!")
            gpu = False
        else:
            grid = (int(np.ceil(1.0 * ncol / block[0])),
                    int(np.ceil(1.0 * ncol / block[1])))
    if ncore is None:
        ncore = np.clip(mp.cpu_count() - 1, 1, None)
    else:
        ncore = np.clip(ncore, 1, None)
    if apply_log is True:
        if np.any(sinogram <= 0.0):
            warnings.warn("!!!Applying logarithm is enabled but "
                          "there are values <= 0.0 in the data!!!")
            sinogram[sinogram <= 0.0] = np.float32(1.0)
        sinogram = -np.log(sinogram)
    if ratio is not None:
        if ratio == 0.0:
            ratio = min(center, ncol - center) / (0.5 * ncol)
        mask = util.make_circle_mask(ncol, ratio)
    if pad is None:
        pad = min(int(0.15 * ncol), 150)
    if ramp_win is None:
        ramp_win = make_2d_ramp_window(nrow, ncol + 2 * pad, filter_name)
    if num_sino == 1:
        sinogram = np.squeeze(sinogram)
        sino_filtered = apply_ramp_filter(sinogram, ramp_win, filter_name, pad,
                                          pad_mode)
        if gpu is True:
            sino_filtered = np.ascontiguousarray(sino_filtered)
            recon = np.zeros((ncol, ncol), dtype=np.float32)
            __back_projection_gpu_kernel[grid, block](recon, np.float32(
                sino_filtered), np.float32(angles), np.float32(center),
                np.int32(nrow), np.int32(ncol), False)
        else:
            recon = back_projection_cpu(np.float32(sino_filtered),
                                        np.float32(angles), np.float32(center))
        if ratio is not None:
            recon = recon * mask
    else:
        if ncore == 1:
            sino_filtered = np.zeros_like(sinogram)
            for i in range(num_sino):
                sino_filtered[:, i, :] = apply_ramp_filter(sinogram[:, i, :],
                                                           ramp_win,
                                                           filter_name, pad,
                                                           pad_mode)
        else:
            ncore = np.clip(ncore, 1, num_sino)
            sino_filtered = Parallel(n_jobs=ncore, prefer="threads")(
                delayed(apply_ramp_filter)(
                    sinogram[:, i, :], ramp_win, filter_name, pad,
                    pad_mode) for i in range(num_sino))
            sino_filtered = np.copy(
                np.moveaxis(np.asarray(sino_filtered), 0, 1))
        if gpu is True:
            sino_filtered = np.ascontiguousarray(sino_filtered)
            recon = np.zeros((ncol, num_sino, ncol), dtype=np.float32)
            __back_projection_gpu_chunk_kernel[grid, block](recon, np.float32(
                sino_filtered), np.float32(angles), np.float32(center),
                np.int32(nrow), np.int32(ncol), np.int32(num_sino), False)
        else:
            recon = np.zeros((ncol, num_sino, ncol), dtype=np.float32)
            for i in range(num_sino):
                recon[:, i, :] = back_projection_cpu(
                    np.float32(sino_filtered[:, i, :]), np.float32(angles),
                    np.float32(center))
        if ratio is not None:
            for i in range(num_sino):
                recon[:, i, :] = recon[:, i, :] * mask
    if input_3d is True and num_sino == 1:
        recon = np.expand_dims(recon, 1)
    return recon * np.pi / (nrow - 1)


def make_circular_ramp_window(width, filter_name=None):
    """
    Make a circular ramp window (2d) with the option of adding a smoothing
    window.

    Parameters
    ----------
    width : int
        Width of the window.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming",\
                  "nuttall", "parzen", "triang"}
         Name of a smoothing window used.

    Returns
    -------
    array_like
        Square array, size of (width, width)
    """
    ramp_win = np.arange(0.0, width) - np.ceil((width - 1.0) / 2)
    ramp_win[ramp_win == 0.0] = 0.25
    ramp_win[ramp_win % 2 == 0.0] = 0.0
    for i in range(width):
        if ramp_win[i] % 2 == 1.0:
            ramp_win[i] = - 1.0 / (ramp_win[i] * np.pi) ** 2
    window = make_smoothing_window(filter_name, width)
    ramp_1d = np.abs(fft.fftshift(fft.fft(ramp_win))) * window
    circ_ramp = util.transform_1d_window_to_2d(ramp_1d, order=1,
                                               mode="nearest")
    return circ_ramp


def apply_circular_ramp_filter(rec_img, ramp_win=None, filter_name=None,
                               pad=None, pad_mode="edge"):
    """
    Apply the circular ramp filter to a back-projected image.

    Parameters
    ----------
    rec_img : array_like
        Square array. back-projected image.
    ramp_win : array_like
        2d circular ramp window, generated if None given.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming",\
                  "nuttall", "parzen", "triang"}
         Name of a smoothing window used.
    pad : int or None
        To apply padding before the FFT. The value is set to 10% of the image
        width if None is given.
    pad_mode : str
        Padding method. Full list can be found at numpy_pad documentation.

    Returns
    -------
    array_like
        Square array.
    """
    ncol = rec_img.shape[0]
    if pad is None:
        pad = min(int(0.15 * ncol), 150)
    img_pad = np.pad(rec_img, pad, mode=pad_mode)
    if (ramp_win is None) or (ramp_win.shape != img_pad.shape):
        ramp_win = make_circular_ramp_window(ncol + 2 * pad, filter_name)
    filt_img = np.real(fft.ifft2(
        fft.ifftshift(fft.fftshift(np.fft.fft2(img_pad)) * ramp_win)))
    return filt_img[pad:ncol + pad, pad:ncol + pad]


def bpf_reconstruction(sinogram, center, angles=None, ratio=1.0, ramp_win=None,
                       filter_name="hann", pad=None, pad_mode="edge",
                       apply_log=True, gpu=True, block=(16, 16), ncore=None):
    """
    Apply the BPF (back-projection filtering) reconstruction method to a
    sinogram-image or a chunk of sinogram-images. Angular axis is 0.
    If input is 3D array, the slicing axis of sinograms must be 1,
    e.g. data[:, index, :].

    Parameters
    ----------
    sinogram : array_like
        2D/3D array. Sinogram image.
    center : float
        Center of rotation.
    angles : array_like, optional
        1D array. List of angles (in radian) corresponding to the sinogram.
    ratio : float, optional
        Apply a circle mask to the reconstructed image.
    ramp_win : complex ndarray, optional
        Circular ramp window, generated if None.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming",\
                  "nuttall", "parzen", "triang"}
        Apply a smoothing filter.
    pad : int, optional
        Apply padding before the FFT. The value is set to 10% of the image
        width if None is given.
    pad_mode : str, optional
        Padding method. Full list can be found at numpy_pad documentation.
    apply_log : bool, optional
        Apply logarithm to sinogram before reconstruction.
    gpu : bool, optional
        Use GPU for computing if True.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (8, 8), (16, 16), (32, 32), ...
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.

    Returns
    -------
    array_like
        Square array. Reconstructed image.
    """
    input_3d = False
    if len(sinogram.shape) == 3:
        (nrow, num_sino, ncol) = sinogram.shape
        input_3d = True
    else:
        num_sino = 1
        (nrow, ncol) = sinogram.shape
    if center < 0 or center >= ncol:
        raise ValueError("Center is out of the range [0, {}]".format(ncol - 1))
    if angles is None:
        angles = np.deg2rad(np.linspace(0.0, 180.0, nrow))
    else:
        if len(angles) != nrow:
            raise ValueError("!!!Number of angles is not the same as the row "
                             "number of the sinogram!!!")
    if gpu is True:
        if cuda.is_available() is False:
            warnings.warn("!!!No Nvidia GPU found! Run with CPU instead!!!")
            gpu = False
        else:
            grid = (int(np.ceil(1.0 * ncol / block[0])),
                    int(np.ceil(1.0 * ncol / block[1])))
    if ncore is None:
        ncore = np.clip(mp.cpu_count() - 1, 1, None)
    else:
        ncore = np.clip(ncore, 1, None)
    if apply_log is True:
        if np.any(sinogram <= 0.0):
            warnings.warn("!!!Applying logarithm is enabled but "
                          "there are values <= 0.0 in the data!!!")
            sinogram[sinogram <= 0.0] = np.float32(1.0)
        sinogram = -np.log(sinogram)
    if ratio is not None:
        if ratio == 0.0:
            ratio = min(center, ncol - center) / (0.5 * ncol)
        mask = util.make_circle_mask(ncol, ratio)
    if pad is None:
        pad = min(int(0.15 * ncol), 150)
    if ramp_win is None:
        ramp_win = make_circular_ramp_window(ncol + 2 * pad, filter_name)
    if num_sino == 1:
        sinogram = np.squeeze(sinogram)
        if gpu is True:
            sinogram = np.ascontiguousarray(sinogram)
            recon = np.zeros((ncol, ncol), dtype=np.float32)
            __back_projection_gpu_kernel[grid, block](recon, np.float32(
                sinogram), np.float32(angles), np.float32(center),
                np.int32(nrow), np.int32(ncol), True)
        else:
            recon = back_projection_cpu(np.float32(sinogram),
                                        np.float32(angles), np.float32(center),
                                        edge_pad=True)
        recon = apply_circular_ramp_filter(recon, ramp_win, pad=pad,
                                           pad_mode=pad_mode)
        if ratio is not None:
            recon = recon * mask
    else:
        if gpu is True:
            sinogram = np.ascontiguousarray(sinogram)
            recon = np.zeros((ncol, num_sino, ncol), dtype=np.float32)
            __back_projection_gpu_chunk_kernel[grid, block](recon, np.float32(
                sinogram), np.float32(angles), np.float32(center),
                np.int32(nrow), np.int32(ncol), np.int32(num_sino), True)
        else:
            recon = np.zeros((ncol, num_sino, ncol), dtype=np.float32)
            for i in range(num_sino):
                recon[:, i, :] = back_projection_cpu(
                    np.float32(sinogram[:, i, :]), np.float32(angles),
                    np.float32(center), edge_pad=True)
        recon = util.parallel_process_slices(recon, apply_circular_ramp_filter,
                                             [ramp_win, filter_name, pad,
                                              pad_mode], axis=1, ncore=ncore,
                                             prefer="threads")
        if ratio is not None:
            for i in range(num_sino):
                recon[:, i, :] = recon[:, i, :] * mask
    if input_3d is True and num_sino == 1:
        recon = np.expand_dims(recon, 1)
    return recon * np.pi / (nrow - 1)


def generate_mapping_coordinate(width_sino, height_sino, width_rec,
                                height_rec):
    """
    Calculate coordinates in the sinogram space from coordinates in the
    reconstruction space (in the Fourier domain). They are used for the
    DFI (direct Fourier inversion) reconstruction method.

    Parameters
    -----------
    width_sino : int
        Width of a sinogram image.
    height_sino : int
        Height of a sinogram image.
    width_rec : int
        Width of a reconstruction image.
    height_rec : int
        Height of a reconstruction image.

    Returns
    ------
    r_mat : array_like
         2D array. Broadcast of the r-coordinates.
    theta_mat : array_like
         2D array. Broadcast of the theta-coordinates.
    """
    xcenter = (width_rec - 1.0) * 0.5
    ycenter = (height_rec - 1.0) * 0.5
    r_max = np.floor(min(xcenter, ycenter))
    xlist = (np.flipud(np.arange(width_rec)) - xcenter)
    ylist = (np.arange(height_rec) - ycenter)
    x_mat, y_mat = np.meshgrid(xlist, ylist)
    r_mat = np.float32(np.clip(np.sqrt(x_mat ** 2 + y_mat ** 2), 0, r_max))
    theta_mat = np.pi + np.arctan2(y_mat, x_mat)
    r_mat[theta_mat > np.pi] *= -1
    r_mat = np.float32(np.clip(r_mat + r_max, 0, width_sino - 1))
    theta_mat[theta_mat > np.pi] -= np.pi
    theta_mat = np.float32(theta_mat * (height_sino - 1.0) / np.pi)
    return r_mat, theta_mat


def __dfi_handle_angles(sinogram, angles):
    """
    Supplementary method for the DFI reconstruction method.
    Allow to use real angles for reconstruction.
    """
    nrow = sinogram.shape[0]
    if len(angles) != nrow:
        raise ValueError("!!! Number of angles is not the same as the row "
                         "number of the sinogram !!!")
    t_ang = np.sum(np.abs(np.diff(angles * 180.0 / np.pi)))
    if abs(t_ang - 360) < 10:
        nrow = nrow // 2 + 1
        sinogram = (sinogram[:nrow] + np.fliplr(sinogram[-nrow:])) / 2
    step = np.mean(np.abs(np.diff(angles)))
    b_ang = angles[0] - (angles[0] // (2 * np.pi)) * (2 * np.pi)
    sino_360 = np.vstack((sinogram[: nrow - 1], np.fliplr(sinogram)))
    sinogram = ndi.shift(sino_360, (b_ang / step, 0), mode='wrap')[:nrow]
    if angles[-1] < angles[0]:
        sinogram = np.flipud(np.fliplr(sinogram))
    return sinogram


def __dfi_handle_sinogram(sinogram0, angles, center, pad_rate, pad_mode):
    """
    Supplementary method for the DFI reconstruction method.
    Shift and pad sinogram.
    """
    sinogram = np.squeeze(sinogram0)
    if len(sinogram.shape) == 3:
        (nrow, num_sino, ncol) = sinogram.shape
        if ncol % 2 == 0:
            sinogram = np.pad(sinogram, ((0, 0), (0, 0), (0, 1)), mode="edge")
    else:
        num_sino = 1
        (nrow, ncol) = sinogram.shape
        if ncol % 2 == 0:
            sinogram = np.pad(sinogram, ((0, 0), (0, 1)), mode="edge")
    ncol1 = sinogram.shape[-1]
    xshift = (ncol1 - 1) / 2.0 - center
    pad = int(pad_rate * ncol1)
    if num_sino > 1:
        sinogram = ndi.shift(sinogram, (0, 0, xshift), mode='nearest')
        if angles is not None:
            sino_tmp = []
            for i in range(num_sino):
                sino_tmp.append(__dfi_handle_angles(sinogram[:, i, :], angles))
            sinogram = np.copy(np.moveaxis(np.asarray(sino_tmp), 0, 1))
        sinogram = np.pad(sinogram, ((0, 0), (0, 0), (pad, pad)),
                          mode=pad_mode)
    else:
        sinogram = ndi.shift(sinogram, (0, xshift), mode='nearest')
        if angles is not None:
            sinogram = __dfi_handle_angles(sinogram, angles)
        sinogram = np.pad(sinogram, ((0, 0), (pad, pad)), mode=pad_mode)
    return sinogram, num_sino, pad


def __dfi_single_slice(sinogram, window, mask, r_mat, theta_mat):
    """
    Supplementary method for the DFI reconstruction method.
    Reconstruct a single slice.
    """
    sino_fft = fft.fftshift(fft.fft(fft.ifftshift(sinogram, axes=1)), axes=1)
    if window is not None:
        sino_fft = sino_fft * window
    scipy_ver = scipy.__version__
    scipy_ver = tuple(map(int, scipy_ver.split(".")[:2]))
    if scipy_ver < (1, 6):
        mat_real = np.real(sino_fft)
        mat_imag = np.imag(sino_fft)
        reg_real = util.mapping(mat_real, r_mat, theta_mat, order=5,
                                mode="reflect") * mask
        reg_imag = util.mapping(mat_imag, r_mat, theta_mat, order=5,
                                mode="reflect") * mask
        reg_comp = reg_real + 1j * reg_imag
    else:
        reg_comp = util.mapping(sino_fft, r_mat, theta_mat, order=5,
                                mode="reflect") * mask
    recon = np.real(fft.fftshift(fft.ifft2(fft.ifftshift(reg_comp))))
    return recon


def dfi_reconstruction(sinogram, center, angles=None, ratio=1.0,
                       filter_name="hann", pad_rate=0.25, pad_mode="edge",
                       apply_log=True, ncore=None):
    """
    Apply the DFI (direct Fourier inversion) reconstruction method (Ref. [1])
    to a sinogram-image or a chunk of sinogram-images. Angular axis is 0.
    If input is 3D array, the slicing axis of sinograms must be 1,
    e.g. data[:, index, :].

    Parameters
    ----------
    sinogram : array_like
        2D/3D array. Sinogram image.
    center : float
        Center of rotation.
    angles : array_like
        1D array. List of angles (in radian) corresponding to the sinogram.
    ratio : float
        To apply a circle mask to the reconstructed image.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming",\
                  "nuttall", "parzen", "triang"}
        Apply a smoothing filter.
    pad_rate : float
        To apply padding before the FFT. The padding width equals to
        (pad_rate * image_width).
    pad_mode : str
        Padding method. Full list can be found at numpy_pad documentation.
    apply_log : bool
        Apply the logarithm function to the sinogram before reconstruction.
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.

    Returns
    -------
    array_like
        Square array. Reconstructed image.

    References
    ----------
    [1] : https://doi.org/10.1071/PH560198
    """
    input_3d = False
    if len(sinogram.shape) == 3:
        input_3d = True
    nrow, ncol = sinogram.shape[0], sinogram.shape[-1]
    if center < 0 or center >= ncol:
        raise ValueError("Center is out of the range [0, {}]".format(ncol - 1))
    if ncol / nrow > 5.0:
        warnings.warn("!!!Sinogram is significantly undersampled!!! "
                      "Considering to use the 'upsample_sinogram' method "
                      "before reconstruction!")
    sinogram, num_sino, pad = __dfi_handle_sinogram(sinogram, angles, center,
                                                    pad_rate, pad_mode)
    if apply_log is True:
        if np.any(sinogram <= 0.0):
            warnings.warn("!!!Applying logarithm is enabled but "
                          "there are values <= 0.0 in the data!!!")
            sinogram[sinogram <= 0.0] = np.float32(1.0)
        sinogram = -np.log(sinogram)
    if ncore is None:
        ncore = np.clip(mp.cpu_count() - 1, 1, None)
    else:
        ncore = np.clip(ncore, 1, None)
    nrow, ncol2 = sinogram.shape[0], sinogram.shape[-1]
    mask = util.make_circle_mask(ncol2, 1.0)
    (r_mat, theta_mat) = generate_mapping_coordinate(ncol2, nrow, ncol2, ncol2)
    window = None
    if filter_name is not None:
        win1d = make_smoothing_window(filter_name, ncol2)
        window = np.tile(win1d, (nrow, 1))
    if num_sino > 1:
        if ncore > 1:
            ncore = np.clip(ncore, 1, num_sino)
            recon = Parallel(n_jobs=ncore, prefer="threads")(
                delayed(__dfi_single_slice)(
                    sinogram[:, i, :], window, mask, r_mat,
                    theta_mat) for i in range(num_sino))
        else:
            recon = []
            for i in range(num_sino):
                recon.append(__dfi_single_slice(sinogram[:, i, :], window,
                                                mask, r_mat, theta_mat))
        recon = np.copy(np.moveaxis(np.asarray(recon), 0, 1))
        recon = recon[pad:ncol + pad, :, pad:ncol + pad]
    else:
        recon = __dfi_single_slice(sinogram, window, mask, r_mat, theta_mat)
        recon = recon[pad:ncol + pad, pad:ncol + pad]
    if ratio is not None:
        if ratio == 0.0:
            ratio = min(center, ncol - center) / (0.5 * ncol)
        mask = util.make_circle_mask(ncol, ratio)
        if num_sino > 1:
            for i in range(num_sino):
                recon[:, i, :] = recon[:, i, :] * mask
        else:
            recon = recon * mask
    if input_3d is True and num_sino == 1:
        recon = np.expand_dims(recon, 1)
    return recon


def gridrec_reconstruction(sinogram, center, angles=None, ratio=1.0,
                           filter_name="shepp", apply_log=True, pad=100,
                           filter_par=0.9, ncore=1):  # pragma: no cover
    """
    Apply the gridrec method to a sinogram-image or a chunk of sinogram-images.
    Angular axis is 0. If input is 3D array, the slicing axis of sinograms
    must be 1, e.g. data[:, index, :]. This is the wrapper of the gridrec
    method implemented in the Tomopy package:
    https://tomopy.readthedocs.io/en/latest/api/tomopy.recon.algorithm.html.
    Users must install Tomopy before using this function.

    Parameters
    ----------
    sinogram : array_like
        2D/3D array. Sinogram image.
    center : float
        Center of rotation.
    angles : array_like
        1D array. List of angles (radian) corresponding to the sinogram.
    ratio : float
        To apply a circle mask to the reconstructed image.
    filter_name : str or None
        Apply a smoothing filter. Full list is at:
        https://github.com/tomopy/tomopy/blob/master/source/tomopy/recon/algorithm.py
    filter_par : float
        Adjust the strength of the filter. Smaller is stronger.
    apply_log : bool
        Apply the logarithm function to the sinogram before reconstruction.
    pad : bool or int
        Apply edge padding to the nearest power of 2.
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.

    Returns
    -------
    array_like
        Square array.
    """
    try:
        import tomopy
    except ImportError:
        raise ValueError("You must install Tomopy before using this function!")
    input_3d = False
    if len(sinogram.shape) == 3:
        input_3d = True
    if angles is None:
        angles = np.deg2rad(np.linspace(0.0, 180.0, sinogram.shape[0]))
    else:
        if len(angles) != sinogram.shape[0]:
            raise ValueError("!!! Number of angles is not the same as the row "
                             "number of the sinogram !!!")
    if apply_log is True:
        if np.any(sinogram <= 0.0):
            warnings.warn("!!!Applying logarithm is enabled but "
                          "there are values <= 0.0 in the data!!!")
            sinogram[sinogram <= 0.0] = np.float32(1.0)
        sinogram = -np.log(sinogram)
    if ncore is None:
        ncore = np.clip(mp.cpu_count() - 1, 1, None)
    else:
        ncore = np.clip(ncore, 1, None)
    if filter_name is None:
        filter_name = "shepp"
    pad_left = 0
    ncol = sinogram.shape[-1]
    if center < 0 or center >= ncol:
        raise ValueError("Center is out of the range [0, {}]".format(ncol - 1))
    if isinstance(pad, bool):
        if pad is True:
            ncol_pad = int(2 ** np.ceil(np.log2(1.0 * ncol)))
            pad_left = (ncol_pad - ncol) // 2
            pad_right = ncol_pad - ncol - pad_left
        else:
            pad_right = 0
    else:
        ncol_pad = int(2 ** np.ceil(np.log2(1.0 * ncol + 2 * pad)))
        pad_left = (ncol_pad - ncol) // 2
        pad_right = ncol_pad - ncol - pad_left
    if len(sinogram.shape) == 2:
        sinogram = np.expand_dims(sinogram, 1)
    sinogram = np.pad(sinogram, ((0, 0), (0, 0), (pad_left, pad_right)),
                      mode='edge')
    recon = tomopy.recon(sinogram, angles, center=center + pad_left,
                         algorithm='gridrec', filter_name=filter_name,
                         filter_par=filter_par, ncore=ncore)
    num_slice = len(recon)
    recon = recon[:, pad_left: pad_left + ncol, pad_left: pad_left + ncol]
    if ratio is not None:
        if ratio == 0.0:
            ratio = min(center, ncol - center) / (0.5 * ncol)
        mask = util.make_circle_mask(ncol, ratio)
        for i in range(num_slice):
            recon[i] = recon[i] * mask
    recon = np.copy(np.moveaxis(np.asarray(recon), 0, 1))
    if input_3d is False:
        recon = np.squeeze(recon)
    return recon


def __astra_recon_single(sinogram, center, angles, pad, method, filter_name,
                         num_iter):  # pragma: no cover
    try:
        import astra
    except ImportError:
        raise ValueError("You must install Astra-Toolbox before using this "
                         "function!")
    sinogram = np.pad(sinogram, ((0, 0), (pad, pad)), mode='edge')
    ncol = sinogram.shape[-1]
    proj_geom = astra.create_proj_geom('parallel', 1, ncol, angles)
    vol_geom = astra.create_vol_geom(ncol, ncol)
    cen_col = (ncol - 1.0) / 2.0
    sinogram = ndi.shift(sinogram, (0, cen_col - (center + pad)),
                         mode='nearest')
    sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom)
    proj_id = None
    if "CUDA" not in method:
        proj_id = astra.create_projector('line', proj_geom, vol_geom)
    cfg = astra.astra_dict(method)
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id
    if "CUDA" not in method:
        cfg['ProjectorId'] = proj_id
    if (method == "FBP_CUDA") or (method == "FBP"):
        cfg["FilterType"] = filter_name
    try:
        alg_id = astra.algorithm.create(cfg)
    except Exception:
        raise ValueError("Invalid selection of method!!!")
    astra.algorithm.run(alg_id, num_iter)
    recon = astra.data2d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(rec_id)
    return recon[pad:ncol - pad, pad:ncol - pad]


def astra_reconstruction(sinogram, center, angles=None, ratio=1.0,
                         method="FBP_CUDA", num_iter=1, filter_name="hann",
                         pad=None, apply_log=True,
                         ncore=1):  # pragma: no cover
    """
    Wrapper of reconstruction methods implemented in the astra toolbox package.
    https://www.astra-toolbox.com/docs/algs/index.html
    Users must install Astra Toolbox before using this function.
    Apply the method to a sinogram-image or a chunk of sinogram-images.
    Angular axis is 0. If input is 3D array, the slicing axis of sinograms
    must be 1, e.g. data[:, index, :]


    Parameters
    ----------
    sinogram : array_like
        2D/3D array. Sinogram image.
    center : float
        Center of rotation.
    angles : array_like
        1D array. List of angles (radian) corresponding to the sinogram.
    ratio : float
        To apply a circle mask to the reconstructed image.
    method : str
        Reconstruction algorithms. For CPU: 'FBP', 'SIRT', 'SART', 'ART', and
        'CGLS'. For GPU: 'FBP_CUDA', 'SIRT_CUDA', 'SART_CUDA', and 'CGLS_CUDA'.
    num_iter : int
        Number of iterations if using iteration methods.
    filter_name : str or None
        Apply filter if using FBP method. Options: 'ram-lak', 'hamming',
        'hann', 'lanczos', 'kaiser', 'parzen',...
    pad : int
        Padding to reduce the side effect of FFT.
    apply_log : bool
        Apply the logarithm function to the sinogram before reconstruction.
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.

    Returns
    -------
    array_like
        Square array.
    """
    try:
        import astra
    except ImportError:
        raise ValueError("You must install Astra-Toolbox before using this "
                         "function!")
    input_3d = False
    if len(sinogram.shape) == 3:
        (nrow, num_sino, ncol) = sinogram.shape
        input_3d = True
    else:
        num_sino = 1
        (nrow, ncol) = sinogram.shape
    if angles is None:
        angles = np.deg2rad(np.linspace(0.0, 180.0, nrow))
    else:
        if len(angles) != nrow:
            raise ValueError("!!! Number of angles is not the same as the row "
                             "number of the sinogram !!!")
    cpu_method = ["FBP", "SIRT", "SART", "ART", "CGLS"]
    gpu_method = ["FBP_CUDA", "SIRT_CUDA", "SART_CUDA", "CGLS_CUDA", "EM_CUDA"]
    gpu = True
    if "CUDA" in method:
        if cuda.is_available() is False:
            warnings.warn("!!!No Nvidia GPU found!!!Run with CPU instead!!!")
            method = method.replace("_CUDA", "")
            if "EM" in method:
                raise ValueError("No EM method for CPU-based algorithm!!!")
            gpu = False
    else:
        gpu = False
    if gpu is True:
        if method not in gpu_method:
            raise ValueError("No such option, {0}, in the available list "
                             "{1}".format(method, gpu_method))
    else:
        if method not in cpu_method:
            raise ValueError("No such option, {0}, in the available list "
                             "{1}".format(method, cpu_method))
    if ncore is None:
        ncore = np.clip(mp.cpu_count() - 1, 1, None)
    else:
        ncore = np.clip(ncore, 1, None)
    if center < 0 or center >= ncol:
        raise ValueError("Center is out of the range [0, {}]".format(ncol - 1))
    if apply_log is True:
        if np.any(sinogram <= 0.0):
            warnings.warn("!!!Applying logarithm is enabled but "
                          "there are values <= 0.0 in the data!!!")
            sinogram[sinogram <= 0.0] = np.float32(1.0)
        sinogram = -np.log(sinogram)
    if filter_name is None:
        filter_name = "ram-lak"
    if pad is None:
        pad = min(int(0.15 * sinogram.shape[-1]), 150)
    else:
        pad = np.clip(int(pad), 0, None)
    if input_3d is False:
        recon = __astra_recon_single(sinogram, center, angles, pad, method,
                                     filter_name, num_iter)
    else:
        if gpu is True:
            recon = []
            for i in range(num_sino):
                recon.append(__astra_recon_single(sinogram[:, i, :], center,
                                                  angles, pad, method,
                                                  filter_name, num_iter))
        else:
            ncore = np.clip(ncore, 1, num_sino)
            recon = Parallel(n_jobs=ncore, prefer="threads")(delayed(
                __astra_recon_single)(sinogram[:, i, :], center, angles, pad,
                                      method, filter_name,
                                      num_iter) for i in range(num_sino))
        recon = np.copy(np.moveaxis(np.asarray(recon), 0, 1))
    if ratio is not None:
        if ratio == 0.0:
            ratio = min(center, ncol - center) / (0.5 * ncol)
        mask = util.make_circle_mask(ncol, ratio)
        if input_3d is False:
            recon = recon * mask
        else:
            for i in range(num_sino):
                recon[:, i, :] = recon[:, i, :] * mask
    return recon


def _reconstruct_slice(sinogram, center, method, angles, ratio, filter_name,
                       apply_log, gpu, ncore):
    """
    Supplementary method for '_find_center_based_slice_metric'. Used to
    reconstruct a slice.
    """
    if method == "fbp":
        recon = fbp_reconstruction(sinogram, center, angles=angles,
                                   ratio=ratio, filter_name=filter_name,
                                   apply_log=apply_log, gpu=gpu, ncore=ncore)
    elif method == "astra":  # pragma: no cover
        if gpu is True:
            rec_method = "FBP_CUDA"
        else:
            rec_method = "FBP"
        recon = astra_reconstruction(sinogram, center, angles=angles,
                                     method=rec_method, ratio=ratio,
                                     filter_name=filter_name,
                                     apply_log=apply_log, ncore=ncore)
    elif method == "gridrec":  # pragma: no cover
        recon = gridrec_reconstruction(sinogram, center, angles=angles,
                                       ratio=ratio, filter_name=filter_name,
                                       apply_log=apply_log, ncore=ncore)
    else:
        recon = dfi_reconstruction(sinogram, center, angles=angles,
                                   ratio=ratio, filter_name=filter_name,
                                   apply_log=apply_log, ncore=ncore)
    return recon


def __calculate_histogram_entropy(recon_img, window=None):
    """
    Calculate a metric based on the entropy of histogram.
    """
    if window is None:
        window = signal.windows.boxcar(7)
    recon = np.uint8(recon_img * 255)
    hist = 1.0 + ndi.histogram(recon, 0, 255, 256)
    hist = signal.convolve(hist, window, mode='valid')
    metric = -np.sum(hist * np.log2(hist))
    return metric


def __calculate_edge_sharpness(image):
    """
    Calculate a sharpness metric using gradient.
    """
    gx, gy = np.gradient(np.float32(image))
    return np.mean(np.sqrt(gx**2 + gy**2))


def __get_slice_metric(sinogram, center, method, angles, ratio, filter_name,
                       apply_log, gpu, ncore, window, nmin=0.0, nmax=1.0,
                       metric="entropy", metric_function=None, **kwargs):
    """
    Supplementary method for '_find_center_based_slice_metric'. Used to
    reconstruct a slice and calculate its metric.
    """
    if metric_function is None:
        recon = _reconstruct_slice(sinogram, center, method, angles,
                                   ratio, filter_name, apply_log, gpu, ncore)
        if metric == "entropy":
            recon1 = (recon - nmin) / (nmax - nmin)
            recon1 = np.clip(recon1, 0, 1)
            value = __calculate_histogram_entropy(recon1, window)
        else:
            value = __calculate_edge_sharpness(recon)
    else:
        recon = _reconstruct_slice(sinogram, center, method, angles,
                                   ratio, filter_name, apply_log, gpu, ncore)
        value = metric_function(recon, **kwargs)
    return value


def _find_center_based_slice_metric(sinogram, start, stop, step,
                                    metric="entropy", method="fbp", gpu=True,
                                    angles=None, ratio=1.0, filter_name="hann",
                                    apply_log=True, ncore=None,
                                    prefer="threads", sigma=0,
                                    return_metric=True, invert_metric=False,
                                    metric_function=None, **kwargs):
    """
    Find the center-of-rotation (COR) using metrics of reconstructed slices
    at different CORs. The entropy of histogram (Ref. [1]) is used by default
    if the metric-function is set to None. If customized metrics are used, 
    not that the minimum value must be corresponding to the optimal center.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    start : float
        Starting point for searching CoR.
    stop : float
        Ending point for searching CoR.
    step : float
        Searching step.
    metric : {"entropy", "sharpness"}
        Which metric to use.
    method : {"dfi", "gridrec", "fbp", "astra"}
        To select a backend method for reconstruction.
    gpu : bool, optional
        Use GPU for computing if True.
    angles : array_like, optional
        1D array. List of angles (in radian) corresponding to the sinogram.
    ratio : float, optional
        To apply a circle mask to the reconstructed image.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming",\
                  "nuttall", "parzen", "triang"}
        Apply a smoothing filter before reconstruction.
    apply_log : bool, optional
        Apply the logarithm function to the sinogram before reconstruction.
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    prefer : {"threads", "processes"}
        Preferred parallel backend.
    sigma : int
        Denoising the sinogram before reconstruction.
    return_metric : bool
        Return a list of centers and their metrics if True.
    invert_metric : bool
        Invert the metric scale, used with a custom metric-function.
    metric_function : obj
        Custom function to calculate metric, accepts keyword
        arguments (**kwargs).

    Returns
    -------
    float or ndarray
        The optimal center or a list of centers and their metrics if
        return_metric=True.
    """
    list_center = np.arange(start, stop, step)
    num_center = len(list_center)
    if num_center == 0:
        raise ValueError("Invalid searching parameters: (start, stop, step)={}"
                         "!!!".format((start, stop, step)))
    if sigma > 0:
        sinogram = ndi.gaussian_filter1d(sinogram, sigma, axis=1)
    if apply_log:
        if np.any(sinogram <= 0.0):
            warnings.warn("!!!Applying logarithm is enabled but "
                          "there are values <= 0.0 in the data!!!")
            sinogram[sinogram <= 0.0] = np.float32(1.0)
        sinogram = -np.log(sinogram)
        apply_log = False
    if not cuda.is_available():
        gpu = False
    nmin, nmax = 0.0, 1.0
    window = signal.windows.boxcar(7)
    if (metric_function is None) and (metric == "entropy"):
        center = np.mean(list_center)
        recon = _reconstruct_slice(sinogram, center, method, angles, ratio,
                                   filter_name, apply_log, gpu, ncore)
        nmin, nmax = np.min(recon), np.max(recon)
        if nmin == nmax:
            raise ValueError("Empty image!!!")
    if ncore is None:
        ncore = np.clip(mp.cpu_count() - 1, 1, None)
    else:
        ncore = np.clip(ncore, 1, None)
    if (ncore > 1) and (gpu is False) and (method != "fbp"):
        list_metric = Parallel(n_jobs=ncore, prefer=prefer)(delayed(
            __get_slice_metric)(sinogram, center, method, angles, ratio,
                                filter_name, apply_log, gpu, 1, window, nmin,
                                nmax, metric, metric_function,
                                **kwargs) for center in list_center)
        list_metric = np.asarray(list_metric)
    else:
        list_metric = np.ones_like(list_center)
        for i, center in enumerate(list_center):
            list_metric[i] = __get_slice_metric(sinogram, center, method,
                                                angles, ratio, filter_name,
                                                apply_log, gpu, 1, window,
                                                nmin, nmax, metric,
                                                metric_function, **kwargs)
    if invert_metric:
        list_metric = np.max(list_metric) - list_metric
    opt_center = list_center[np.argmin(list_metric)]
    if return_metric:
        return list_center, list_metric
    else:
        return opt_center


def find_center_based_slice_metric(sinogram, start, stop, step=0.5,
                                   metric="entropy", radius=2, zoom=1.0,
                                   method="fbp", gpu=True, angles=None,
                                   ratio=1.0, filter_name="hann",
                                   apply_log=True, ncore=None, sigma=0,
                                   invert_metric=False, metric_function=None,
                                   **kwargs):
    """
    Find the center-of-rotation (COR) using metrics of reconstructed slices
    at different CORs. The entropy of histogram (Ref. [1]) is used by default.
    If customized metrics are used, the minimum value must be corresponding to
    the optimal center.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    start : float
        Starting point for searching CoR.
    stop : float
        Ending point for searching CoR.
    step : float
        Sub-pixel searching step.
    metric : {"entropy", "sharpness"}
        Which metric to use.
    radius : float
        Searching range with the sub-pixel step.
    zoom : float
        To resize the sinogram for fast coarse-searching. For example, 0.5 <=>
        reduce the size of the image by half.
    method : {"dfi", "gridrec", "fbp", "astra"}
        To select a backend method for reconstruction.
    gpu : bool, optional
        Use GPU for computing if True.
    angles : array_like, optional
        1D array. List of angles (in radian) corresponding to the sinogram.
    ratio : float, optional
        To apply a circle mask to the reconstructed image.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming",\
                  "nuttall", "parzen", "triang"}
        Apply a smoothing filter before reconstruction.
    apply_log : bool, optional
        Apply the logarithm function to the sinogram before reconstruction.
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    sigma : int
        Denoising the sinogram before reconstruction.
    invert_metric : bool
        Invert the metric scale, used with a custom metric-function.
    metric_function : obj
        Custom function to calculate metric, accepts keyword arguments
        (** kwargs).

    Returns
    -------
    float
        Center-of-rotation.

    References
    ----------
    [1] : https://doi.org/10.1364/JOSAA.23.001048
    """
    zoom = np.clip(zoom, 0.01, 1.0)
    angles_zoom = angles
    sino_zoom = np.copy(sinogram)
    if zoom < 1.0:
        sino_zoom = ndi.zoom(sinogram, zoom, order=1, mode="nearest")
        start, stop = start * zoom, stop * zoom
        if angles is not None:
            angles_zoom = ndi.zoom(np.tile(angles, (1, 1)), (1.0, zoom))[0]
    f_alias = _find_center_based_slice_metric
    if stop > start:
        coarse_center = f_alias(sino_zoom, start, stop + 1, 1.0, metric=metric,
                                method=method, gpu=gpu, angles=angles_zoom,
                                ratio=ratio, filter_name=filter_name,
                                apply_log=apply_log, ncore=ncore, sigma=sigma,
                                return_metric=False,
                                invert_metric=invert_metric,
                                metric_function=metric_function, **kwargs)
        coarse_center = coarse_center / zoom
    else:
        coarse_center = start / zoom
    if radius != 0.0:
        radius = max(radius, step + 1.0 / zoom)
        start, stop = coarse_center - radius, coarse_center + radius + step
        center = f_alias(sinogram, start, stop, step, metric=metric,
                         method=method, gpu=gpu, angles=angles, ratio=ratio,
                         filter_name=filter_name, apply_log=apply_log,
                         ncore=ncore, sigma=sigma, return_metric=False,
                         invert_metric=invert_metric,
                         metric_function=metric_function, **kwargs)
    else:
        center = coarse_center
    return center


def find_center_visual_slices(sinogram, output, start, stop, step=1, zoom=0.5,
                              method="fbp", gpu=False, angles=None, ratio=1.0,
                              filter_name="hann", apply_log=True, ncore=None,
                              display=False):
    """
    For visually finding the center-of-rotation (COR) using reconstructed
    slices at different CORs.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    output : str
        Base folder for saving reconstructed slices.
    start : float
        Starting point for searching CoR.
    stop : float
        Ending point for searching CoR.
    step : float
        Searching step.
    zoom : float
        To resize input and output images. For example, 0.5 <=> reduce the
        size of images by half.
    method : {"dfi", "gridrec", "fbp", "astra"}
        To select a backend method for reconstruction.
    gpu : bool, optional
        Use GPU for computing if True.
    angles : array_like, optional
        1D array. List of angles (in radian) corresponding to the sinogram.
    ratio : float, optional
        To apply a circle mask to the reconstructed image.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming",\
                  "nuttall", "parzen", "triang"}
        Apply a smoothing filter.
    apply_log : bool, optional
        Apply the logarithm function to the sinogram before reconstruction.
    ncore : int or None
        Number of cpu-cores used for computing. Automatically selected if None.
    display : bool
        Print the output if True.

    Returns
    -------
    str
        Folder path to tif images.
    """
    output_name = losa.make_folder_name(output, name_prefix="Find_center",
                                        zero_prefix=3)
    output_base = output + "/" + output_name + "/"
    (nrow, ncol) = sinogram.shape
    step = np.clip(step, 0.05, ncol - 1)
    start = np.clip(start, 0, ncol - 1)
    stop = np.clip(stop + step, start + step, ncol - 1)
    if zoom != 1.0:
        sinogram = ndi.zoom(sinogram, zoom, order=1, mode="nearest")
        start = start * zoom
        stop = stop * zoom
        step = step * zoom
        list_center = np.arange(start, stop, step)
        if angles is not None:
            angles = ndi.zoom(np.tile(angles, (1, 1)), (1.0, zoom))[0]
    else:
        list_center = np.arange(start, stop, step)
    if not cuda.is_available():
        gpu = False
    for center in list_center:
        rec_img = _reconstruct_slice(sinogram, center, method, angles, ratio,
                                     filter_name, apply_log, gpu, ncore)
        file_name = "center_{0:6.2f}".format(center / zoom) + ".tif"
        losa.save_image(output_base + file_name, rec_img)
        if display:
            print("Done: {}".format(output_base + file_name))
    return output_base
