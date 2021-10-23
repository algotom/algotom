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
# Description: Python implementations of FFT-based reconstruction methods
# Contributors:
# ============================================================================

"""
Module of FFT-based reconstruction methods in the reconstruction stage:
- Filtered back-projection (FBP) method for GPU (using numba and cuda) and CPU.
- Direct Fourier inversion (DFI) method.
- Wrapper for Astra Toolbox reconstruction (optional)
- Wrapper for Tomopy-gridrec reconstruction (optional)
"""

import math
import numpy as np
from numba import cuda
from numba import jit
from scipy import signal
from scipy.ndimage import shift
import numpy.fft as fft
import algotom.util.utility as util


def make_smoothing_window(filter_name, width):
    """
    Make a 1d smoothing window.

    Parameters
    ----------
    filter_name : {"hann", "bartlett", "blackman", "hamming", "nuttall",\\
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
    filter_name : {None, "hann", "bartlett", "blackman", "hamming", "nuttall",\\
                  "parzen", "triang"}
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
        2D rray. Sinogram image.
    ramp_win : complex ndarray or None
        Ramp window in the Fourier space.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming", "nuttall",\\
                  "parzen", "triang"}
         Name of a smoothing window used.
    pad : int or None
        To apply padding before the FFT. The value is set to 10% of the image
        width if None is given.
    pad_mode : str
        Padding method. Full list can be found at numpy.pad documentation.

    Returns
    -------
    array_like
        Filtered sinogram.
    """
    (nrow, ncol) = sinogram.shape
    if pad is None:
        pad = int(0.1 * ncol)
    sino_pad = np.pad(sinogram, ((0, 0), (pad, pad)), mode=pad_mode)
    if (ramp_win is None) or (ramp_win.shape != sinogram.shape):
        ramp_win = make_2d_ramp_window(nrow, ncol + 2 * pad, filter_name)
    sino_fft = fft.fftshift(fft.fft(sino_pad), axes=1) * ramp_win
    sino_filtered = np.real(
        fft.ifftshift(fft.ifft(fft.ifftshift(sino_fft, axes=1)), axes=1))
    return np.ascontiguousarray(sino_filtered[:, pad:ncol + pad])


@cuda.jit
def back_projection_gpu(recon, sinogram, angles, xlist, center, sino_height,
                        sino_width):
    """
    Implement the back-projection algorithm using GPU.

    Parameters:
    -----------
    recon : array_like
        Square array of zeros. Reconstruction image.
    sinogram : array_like
        2D array. (Filtered) sinogram image.
    angles : array_like
        1D array. Angles (radian) corresponding to the sinogram.
    xlist : array_like
        1D array. Distances of the integration lines to the image center.
    center : float
        Center of rotation.
    sino_height : int
        Height of the sinogram image.
    sino_width : int
        Width of the sinogram image.

    Returns
    -------
    recon : array_like
        Note that this is the GPU kernel function, i.e. no need of "return".
    """
    (x_index, y_index) = cuda.grid(2)
    icenter = math.ceil((sino_width - 1.0) / 2.0)
    x_cor = (x_index - icenter)
    y_cor = (y_index - icenter)
    x_min = max(-icenter, -center)
    x_max = min(sino_width - icenter - 1, sino_width - center - 1)
    if (x_index < sino_width) and (y_index < sino_width):
        num = 0.0
        for i in range(sino_height):
            theta = - angles[i]
            x_pos = x_cor * math.cos(theta) + y_cor * math.sin(theta)
            if (x_pos > x_min) and (x_pos < x_max):
                fpos = x_pos + center
                dpos = int(math.floor(fpos))
                upos = int(math.ceil(fpos))
                if upos != dpos:
                    xd = xlist[dpos]
                    xu = xlist[upos]
                    yd = sinogram[i, dpos]
                    yu = sinogram[i, upos]
                    val = yd + (yu - yd) * ((x_pos - xd) / (xu - xd))
                else:
                    val = sinogram[i, dpos]
                num += val
        recon[y_index, x_index] = num


@jit(nopython=True, parallel=True, cache=True)
def back_projection_cpu(sinogram, angles, xlist, center):
    """
    Implement the back-projection algorithm using CPU.

    Parameters:
    -----------
    sinogram : array_like
        2D array. (Filtered) sinogram image.
    angles : array_like
        1D array. Angles (radian) corresponding to the sinogram.
    xlist : array_like
        1D array. Distances of the integration lines to the image center.
    center : float
        Center of rotation.

    Returns
    -------
    recon : array_like
        Square array. Reconstructed image.
    """
    (sino_height, sino_width) = sinogram.shape
    icenter = np.ceil((sino_width - 1.0) / 2.0)
    x_min = max(-icenter, -center)
    x_max = min(sino_width - icenter - 1, sino_width - center - 1)
    recon = np.zeros((sino_width, sino_width), dtype=np.float32)
    for i in range(sino_height):
        theta = - angles[i]
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        for y_index in range(sino_width):
            y_cor = y_index - icenter
            for x_index in range(sino_width):
                x_pos = (x_index - icenter) * cos_theta + y_cor * sin_theta
                if (x_pos > x_min) and (x_pos < x_max):
                    fpos = x_pos + center
                    dpos = np.int32(np.floor(fpos))
                    upos = np.int32(np.ceil(fpos))
                    if upos != dpos:
                        xd = xlist[dpos]
                        xu = xlist[upos]
                        yd = sinogram[i, dpos]
                        yu = sinogram[i, upos]
                        val = yd + (yu - yd) * ((x_pos - xd) / (xu - xd))
                    else:
                        val = sinogram[i, dpos]
                    recon[y_index, x_index] += val
    return recon


def fbp_reconstruction(sinogram, center, angles=None, ratio=1.0, ramp_win=None,
                       filter_name="hann", pad=None, pad_mode="edge",
                       apply_log=True, gpu=True):
    """
    Apply the FBP (filtered back-projection) reconstruction method to a
    sinogram image.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    center : float
        Center of rotation.
    angles : array_like, optional
        1D array. List of angles (in radian) corresponding to the sinogram.
    ratio : float, optional
        To apply a circle mask to the reconstructed image.
    ramp_win : complex ndarray, optional
        Ramp window in the Fourier space. It will be generated if None is given.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming", "nuttall",\\
                  "parzen", "triang"}
        Apply a smoothing filter.
    pad : int, optional
        To apply padding before the FFT. The value is set to 10% of the image
        width if None is given.
    pad_mode : str, optional
        Padding method. Full list can be found at numpy.pad documentation.
    apply_log : bool, optional
        Apply the logarithm function to the sinogram before reconstruction.
    gpu : bool, optional
        Use GPU for computing if True.

    Returns
    -------
    array_like
        Square array. Reconstructed image.
    """
    if gpu is True:
        if cuda.is_available() is False:
            print("!!! No Nvidia GPU found !!! Run with CPU instead !!!")
            gpu = False
    if apply_log is True:
        sinogram = -np.log(sinogram)
    (nrow, ncol) = sinogram.shape
    if angles is None:
        angles = np.linspace(0.0, 180.0, nrow) * np.pi / 180.0
    else:
        num_pro = len(angles)
        if num_pro != nrow:
            msg = "!!!Number of angles is not the same as the row number of " \
                  "the sinogram!!!"
            raise ValueError(msg)
    xlist = np.float32(np.arange(0.0, ncol) - center)
    sino_filtered = apply_ramp_filter(sinogram, ramp_win, filter_name, pad,
                                      pad_mode)
    recon = np.zeros((ncol, ncol), dtype=np.float32)
    if gpu is True:
        fbp_block = (16, 16)
        fbp_grid = (int(np.ceil(1.0 * ncol / fbp_block[0])),
                    int(np.ceil(1.0 * ncol / fbp_block[1])))
        back_projection_gpu[fbp_grid, fbp_block](recon,
                                                 np.float32(sino_filtered),
                                                 np.float32(angles), xlist,
                                                 np.float32(center),
                                                 np.int32(nrow), np.int32(ncol))
    else:
        recon = back_projection_cpu(np.float32(sino_filtered),
                                    np.float32(angles), np.float32(xlist),
                                    np.float32(center))
    if ratio is not None:
        if ratio == 0.0:
            ratio = min(center, ncol - center) / (0.5 * ncol)
        mask = util.make_circle_mask(ncol, ratio)
        recon = recon * mask
    return recon * np.pi / (nrow - 1)


def generate_mapping_coordinate(width_sino, height_sino, width_rec, height_rec):
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
    x_list = (np.flipud(np.arange(width_rec)) - xcenter)
    y_list = (np.arange(height_rec) - ycenter)
    x_mat, y_mat = np.meshgrid(x_list, y_list)
    r_mat = np.float32(np.clip(np.sqrt(x_mat ** 2 + y_mat ** 2), 0, r_max))
    theta_mat = np.pi + np.arctan2(y_mat, x_mat)
    r_mat[theta_mat > np.pi] *= -1
    r_mat = np.float32(np.clip(r_mat + r_max, 0, width_sino - 1))
    theta_mat[theta_mat > np.pi] -= np.pi
    theta_mat = np.float32(theta_mat * (height_sino - 1.0) / np.pi)
    return r_mat, theta_mat


def dfi_reconstruction(sinogram, center, angles=None, ratio=1.0,
                       filter_name="hann", pad_rate=0.25, pad_mode="edge",
                       apply_log=True):
    """
    Apply the DFI (direct Fourier inversion) reconstruction method to a
    sinogram image (Ref. [1]). The method is a practical and direct
    implementation of the Fourier slice theorem (Ref. [2]).

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    center : float
        Center of rotation.
    angles : array_like
        1D array. List of angles (in radian) corresponding to the sinogram.
    ratio : float
        To apply a circle mask to the reconstructed image.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming", "nuttall",\\
                  "parzen", "triang"}
        Apply a smoothing filter.
    pad_rate : float
        To apply padding before the FFT. The padding width equals to
        (pad_rate * image_width).
    pad_mode : str
        Padding method. Full list can be found at numpy.pad documentation.
    apply_log : bool
        Apply the logarithm function to the sinogram before reconstruction.

    Returns
    -------
    array_like
        Square array. Reconstructed image.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.418448
    .. [2] https://doi.org/10.1071/PH560198
    """
    if apply_log is True:
        sinogram = -np.log(sinogram)
    (nrow, ncol) = sinogram.shape
    if ncol % 2 == 0:
        sinogram = np.pad(sinogram, ((0, 0), (0, 1)), mode="edge")
    ncol1 = sinogram.shape[1]
    xshift = (ncol1 - 1) / 2.0 - center
    sinogram = shift(sinogram, (0, xshift), mode='nearest')
    if angles is not None:
        t_ang = np.sum(np.abs(np.diff(angles * 180.0 / np.pi)))
        if abs(t_ang - 360) < 10:
            nrow = nrow // 2 + 1
            sinogram = (sinogram[:nrow] + np.fliplr(sinogram[-nrow:])) / 2
        step = np.mean(np.abs(np.diff(angles)))
        b_ang = angles[0] - (angles[0] // (2 * np.pi)) * (2 * np.pi)
        sino_360 = np.vstack((sinogram[: nrow - 1], np.fliplr(sinogram)))
        sinogram = shift(sino_360, (b_ang / step, 0), mode='wrap')[:nrow]
        if angles[-1] < angles[0]:
            sinogram = np.flipud(np.fliplr(sinogram))
    num_pad = int(pad_rate * ncol1)
    sinogram = np.pad(sinogram, ((0, 0), (num_pad, num_pad)), mode=pad_mode)
    ncol2 = sinogram.shape[1]
    mask = util.make_circle_mask(ncol2, 1.0)
    (r_mat, theta_mat) = generate_mapping_coordinate(ncol2, nrow, ncol2, ncol2)
    sino_fft = fft.fftshift(fft.fft(fft.ifftshift(sinogram, axes=1)), axes=1)
    if filter_name is not None:
        window = make_smoothing_window(filter_name, ncol2)
        sino_fft = sino_fft * np.tile(window, (nrow, 1))
    mat_real = np.real(sino_fft)
    mat_imag = np.imag(sino_fft)
    reg_real = util.mapping(mat_real, r_mat, theta_mat, order=5,
                            mode="reflect") * mask
    reg_imag = util.mapping(mat_imag, r_mat, theta_mat, order=5,
                            mode="reflect") * mask
    recon = np.real(
        fft.fftshift(fft.ifft2(fft.ifftshift(reg_real + 1j * reg_imag))))[
            num_pad:ncol + num_pad, num_pad:ncol + num_pad]
    if ratio is not None:
        if ratio == 0.0:
            ratio = min(center, ncol - center) / (0.5 * ncol)
        mask = util.make_circle_mask(ncol, ratio)
        recon = recon * mask
    return recon


def gridrec_reconstruction(sinogram, center, angles=None, ratio=1.0,
                           filter_name="shepp", apply_log=True, pad=True,
                           ncore=1):
    """
    Wrapper of the gridrec method implemented in the tomopy package:
    https://tomopy.readthedocs.io/en/latest/api/tomopy.recon.algorithm.html
    Users must install Tomopy before using this function.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    center : float
        Center of rotation.
    angles : array_like
        1D array. List of angles (radian) corresponding to the sinogram.
    ratio : float
        To apply a circle mask to the reconstructed image.
    filter_name : str
        Apply a smoothing filter. Full list is at:
        https://github.com/tomopy/tomopy/blob/master/source/tomopy/recon/algorithm.py
    apply_log : bool
        Apply the logarithm function to the sinogram before reconstruction.
    pad : bool
        Apply edge padding to the nearest power of 2.

    Returns
    -------
    array_like
        Square array.
    """
    try:
        import tomopy
    except ImportError:
        print("!!!!!! Error !!!!!!!")
        print("You must install Tomopy before using this function!")
        raise
    pad_left = 0
    ncol = sinogram.shape[-1]
    if isinstance(pad, bool):
        if pad is True:
            ncol_pad = int(2 ** np.ceil(np.log2(1.0 * ncol)))
            pad_left = (ncol_pad - ncol) // 2
            pad_right = ncol_pad - ncol - pad_left
            sinogram = np.pad(sinogram, ((0, 0), (pad_left, pad_right)),
                              mode='edge')
    else:
        pad_left = pad
        sinogram = np.pad(sinogram, ((0, 0), (pad, pad)),mode='edge')
    if apply_log is True:
        sinogram = -np.log(sinogram)
    if filter_name is None:
        filter_name = "shepp"
    if angles is None:
        angles = np.linspace(0.0, 180.0, sinogram.shape[0]) * np.pi / 180.0
    recon = tomopy.recon(np.expand_dims(sinogram, 1), angles,
                         center=center + pad_left, algorithm='gridrec',
                         filter_name=filter_name, ncore=ncore)[0]
    recon = recon[pad_left: pad_left + ncol, pad_left: pad_left + ncol]
    if ratio is not None:
        if ratio == 0.0:
            ratio = min(center, ncol - center) / (0.5 * ncol)
        mask = util.make_circle_mask(ncol, ratio)
        recon = recon * mask
    return recon


def astra_reconstruction(sinogram, center, angles=None, ratio=1.0,
                         method="FBP_CUDA", num_iter=1, filter_name="hann",
                         pad=None, apply_log=True):
    """
    Wrapper of reconstruction methods implemented in the astra toolbox package.
    https://www.astra-toolbox.com/docs/algs/index.html
    Users must install Astra Toolbox before using this function.

    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    center : float
        Center of rotation.
    angles : array_like
        1D array. List of angles (radian) corresponding to the sinogram.
    ratio : float
        To apply a circle mask to the reconstructed image.
    method : str
        Reconstruction algorithms. for CPU: 'FBP', 'SIRT', 'SART', 'ART',
        'CGLS'. for GPU: 'FBP_CUDA', 'SIRT_CUDA', 'SART_CUDA', 'CGLS_CUDA'.
    num_iter : int
        Number of iterations if using iteration methods.
    filter_name : str
        Apply filter if using FBP method. Options: 'hamming', 'hann',
        'lanczos', 'kaiser', 'parzen',...
    pad : int
        Padding to reduce the side effect of FFT.
    apply_log : bool
        Apply the logarithm function to the sinogram before reconstruction.

    Returns
    -------
    array_like
        Square array.
    """
    try:
        import astra
    except ImportError:
        print("!!!!!! Error !!!!!!!")
        print("You must install Astra Toolbox before using this function!")
        raise
    if apply_log is True:
        sinogram = -np.log(sinogram)
    if pad is None:
        pad = int(0.1 * sinogram.shape[1])
    sinogram = np.pad(sinogram, ((0, 0), (pad, pad)), mode='edge')
    (nrow, ncol) = sinogram.shape
    if angles is None:
        angles = np.linspace(0.0, 180.0, nrow) * np.pi / 180.0
    proj_geom = astra.create_proj_geom('parallel', 1, ncol, angles)
    vol_geom = astra.create_vol_geom(ncol, ncol)
    cen_col = (ncol - 1.0) / 2.0
    sinogram = shift(sinogram, (0, cen_col - (center + pad)), mode='nearest')
    sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom)
    if "CUDA" not in method:
        proj_id = astra.create_projector('line', proj_geom, vol_geom)
    cfg = astra.astra_dict(method)
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id
    if "CUDA" not in method:
        cfg['ProjectorId'] = proj_id
    if (method == "FBP_CUDA") or (method == "FBP"):
        cfg["FilterType"] = filter_name
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, num_iter)
    recon = astra.data2d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(rec_id)
    recon = recon[pad:ncol - pad, pad:ncol - pad]
    if ratio is not None:
        ncol0 = ncol - 2 * pad
        if ratio == 0.0:
            ratio = min(center, ncol0 - center) / (0.5 * ncol0)
        mask = util.make_circle_mask(ncol0, ratio)
        recon = recon * mask
    return recon
