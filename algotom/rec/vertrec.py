# ============================================================================
# ============================================================================
# Copyright (c) 2024 Nghia T. Vo. All rights reserved.
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
# Description: Python module for reconstructing vertical slices directly from
# tomography data.
# Publication date: 24th April 2024
# Contributors:
# ============================================================================

"""
Module of methods for directly reconstructing vertical slices without the need
for full reconstruction and reslicing of tomography data.

    -   Vertical back-projection methods for CPU and GPU that support single
        and multiple-slice reconstruction.
    -   Direct vertical reconstruction of a single slice or multiple slices
        (at the same or different angles) using Filtered Back-Projection (FBP)
        or Back-Projection Filtering (BPF) methods.
    -   Automatic determination of the center of rotation.
    -   Tool to assist in manual determination of the center of rotation.
"""

import sys
import math
import time
import warnings
import multiprocessing as mp
import numpy as np
import numpy.fft as fft
from scipy import signal
from numba import jit, cuda, prange
import algotom.util.utility as util
import algotom.io.loadersaver as losa
import algotom.rec.reconstruction as rec


@jit(nopython=True, parallel=True, cache=True)
def vertical_back_projection_cpu(projections, angles, xlist, ylist, center,
                                 edge_pad=False):
    """
    Perform vertical back-projection on CPU.

    Parameters
    ----------
    projections : array_like
        3D array of projection data with shape (depth, height, width)
    angles : array_like
        1D array. Angles (radian) corresponding to projections.
    xlist : array_like
        x-coordinates of points on the reconstructed line.
    ylist : array_like
        y-coordinates of points on the reconstructed line.
    center : float
        Center of rotation. (x-coordinate of the rotation axis)
    edge_pad : bool
        Enable/disable edge padding.

    Returns
    -------
    recon : array_like
        2D back-projected image, same size as projection (height, width).
    """
    (depth, height, width) = projections.shape
    recon = np.zeros((height, width), dtype=np.float32)
    width1 = width - 1
    for i in prange(height):
        for n in range(depth):
            theta = angles[n]
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            for j in range(width):
                x_cor = xlist[j]
                y_cor = ylist[j]
                r_pos = x_cor * cos_theta + y_cor * sin_theta
                f_pos = r_pos + center
                if 0 <= f_pos <= width1:
                    d_pos = int(math.floor(f_pos))
                    u_pos = int(math.ceil(f_pos))
                    if u_pos != d_pos:
                        yd = projections[n, i, d_pos]
                        yu = projections[n, i, u_pos]
                        val = yd + (yu - yd) * (f_pos - d_pos)
                    else:
                        val = projections[n, i, d_pos]
                    recon[i, j] += val
                else:
                    if edge_pad:
                        if f_pos < 0:
                            val = projections[n, i, 0]
                        else:
                            val = projections[n, i, width1]
                        recon[i, j] += val
    return recon


@jit(nopython=True, parallel=True, cache=True)
def vertical_back_projection_cpu_chunk(projections, angles, x_mat, y_mat,
                                       center, edge_pad=False):
    """
    Perform vertical back-projection on CPU for multiple slices.

    Parameters
    ----------
    projections : array_like
        3D array of projection data with shape (depth, height, width)
    angles : array_like
        1D array. Angles (radian) corresponding to projections.
    x_mat : array_like
        2D array (num_slice, width), each row contains x-coordinates for
        back-projection on each slice.
    y_mat : array_like
        2D array (num_slice, width), each row contains y-coordinates for
        back-projection on each slice.
    center : float
        Center of rotation. (x-coordinate of the rotation axis)
    edge_pad : bool
        Enable/disable edge padding.

    Returns
    -------
    recon : array_like
        3D back-projected image with size (num_slice, height, width)
    """
    (depth, height, width) = projections.shape
    num_slice = len(x_mat)
    width1 = width - 1
    recon = np.zeros((num_slice, height, width), dtype=np.float32)
    for s in prange(num_slice):
        for n in range(depth):
            theta = angles[n]
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            for j in range(width):
                x_cor = x_mat[s, j]
                y_cor = y_mat[s, j]
                r_pos = x_cor * cos_theta + y_cor * sin_theta
                f_pos = r_pos + center
                d_pos = int(math.floor(f_pos))
                u_pos = int(math.ceil(f_pos))
                for i in range(height):
                    if 0 <= f_pos <= width1:
                        if u_pos != d_pos:
                            yd = projections[n, i, d_pos]
                            yu = projections[n, i, u_pos]
                            val = yd + (yu - yd) * (f_pos - d_pos)
                        else:
                            val = projections[n, i, d_pos]
                        recon[s, i, j] += val
                    else:
                        if edge_pad:
                            if f_pos < 0:
                                val = projections[n, i, 0]
                            else:
                                val = projections[n, i, width1]
                            recon[s, i, j] += val
    return recon


@cuda.jit
def __vertical_back_projection_gpu_kernel(recon, projections, angles, xlist,
                                          ylist, center, depth, height, width,
                                          edge_pad):
    """
    GPU-kernel function performing back-projection for a single slice.
    """
    i_index, j_index = cuda.grid(2)
    if i_index >= height or j_index >= width:
        return
    x_cor = xlist[j_index]
    y_cor = ylist[j_index]
    width1 = width - 1
    val_acc = 0.0
    for k in range(depth):
        theta = angles[k]
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        r_pos = x_cor * cos_theta + y_cor * sin_theta
        f_pos = r_pos + center
        if 0 <= f_pos <= width1:
            d_pos = int(math.floor(f_pos))
            u_pos = int(math.ceil(f_pos))
            if u_pos != d_pos:
                yd = projections[k, i_index, d_pos]
                yu = projections[k, i_index, u_pos]
                val = yd + (yu - yd) * (f_pos - d_pos)
            else:
                val = projections[k, i_index, d_pos]
            val_acc += val
        else:
            if edge_pad:
                if f_pos < 0:
                    val = projections[k, i_index, 0]
                else:
                    val = projections[k, i_index, width1]
                val_acc += val
    recon[i_index, j_index] = val_acc


def vertical_back_projection_gpu(projections, angles, xlist, ylist, center,
                                 edge_pad=False, block=(16, 16)):
    """
    Perform vertical back-projection on GPU.

    Parameters
    ----------
    projections : array_like
        3D array of projection data with shape (depth, height, width)
    angles : array_like
        1D array. Angles (radian) corresponding to projections.
    xlist : array_like
        x-coordinates of points on the reconstructed line.
    ylist : array_like
        y-coordinates of points on the reconstructed line.
    center : float
        Center of rotation. (x-coordinate of the rotation axis)
    edge_pad : bool
        Enable/disable edge padding.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (8, 8), (16, 16), (32, 32), ...

    Returns
    -------
    recon : array_like
        2D back-projected image, same size as projection (height, width).
    """
    projections = np.ascontiguousarray(projections)
    (depth, height, width) = projections.shape
    recon = np.zeros((height, width), dtype=np.float32)
    grid = (int(np.ceil(1.0 * height / block[0])),
            int(np.ceil(1.0 * width / block[1])))
    __vertical_back_projection_gpu_kernel[
        grid, block](recon, np.float32(projections), np.float32(angles),
                     np.float32(xlist), np.float32(ylist),
                     np.float32(center), np.int32(depth), np.int32(height),
                     np.int32(width), edge_pad)
    return recon


@cuda.jit
def __vertical_back_projection_gpu_chunk_kernel(recons, projections, angles,
                                                x_mat, y_mat, center, depth,
                                                height, width, num_slice,
                                                edge_pad):
    """
    GPU-kernel function performing back-projection for multiple slices.
    """
    (i_index, j_index) = cuda.grid(2)
    if i_index >= height or j_index >= width:
        return
    width1 = width - 1
    for s in range(num_slice):
        x_cor = x_mat[s, j_index]
        y_cor = y_mat[s, j_index]
        val_acc = 0.0
        for n in range(depth):
            theta = angles[n]
            r_pos = x_cor * math.cos(theta) + y_cor * math.sin(theta)
            f_pos = r_pos + center
            if 0 <= f_pos <= width1:
                d_pos = int(math.floor(f_pos))
                u_pos = int(math.ceil(f_pos))
                if u_pos != d_pos:
                    yd = projections[n, i_index, d_pos]
                    yu = projections[n, i_index, u_pos]
                    val = yd + (yu - yd) * (f_pos - d_pos)
                else:
                    val = projections[n, i_index, d_pos]
                val_acc += val
            else:
                if edge_pad:
                    if f_pos < 0:
                        val = projections[n, i_index, 0]
                    else:
                        val = projections[n, i_index, width1]
                    val_acc += val
        recons[s, i_index, j_index] = val_acc


def vertical_back_projection_gpu_chunk(projections, angles, x_mat, y_mat,
                                       center, edge_pad=False, block=(16, 16)):
    """
    Perform vertical back-projection on GPU for multiple slices.

    Parameters
    ----------
    projections : array_like
        3D array of projection data with shape (depth, height, width)
    angles : array_like
        1D array. Angles (radian) corresponding to projections.
    x_mat : array_like
        2D array (num_slice, width), each row contains x-coordinates for
        back-projection on each slice.
    y_mat : array_like
        2D array (num_slice, width), each row contains y-coordinates for
        back-projection on each slice.
    center : float
        Center of rotation. (x-coordinate of the rotation axis)
    edge_pad : bool
        Enable/disable edge padding.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (8, 8), (16, 16), (32, 32), ...

    Returns
    -------
    recon : array_like
        3D back-projected image with size (num_slice, height, width)
    """
    projections = np.ascontiguousarray(projections)
    (depth, height, width) = projections.shape
    num_slice = len(x_mat)
    recons = np.zeros((num_slice, height, width), dtype=np.float32)
    grid = (int(np.ceil(1.0 * height / block[0])),
            int(np.ceil(1.0 * width / block[1])))
    __vertical_back_projection_gpu_chunk_kernel[
        grid, block](recons, np.float32(projections), np.float32(angles),
                     np.float32(x_mat), np.float32(y_mat),
                     np.float32(center), np.int32(depth), np.int32(height),
                     np.int32(width), np.int32(num_slice), edge_pad)
    return recons


def _get_points_single_line(slice_index, alpha, width):
    """
    Computes x and y coordinates of points along a specified slice at an angle.

    Parameters
    ----------
    slice_index : int
        Index of the line within image width.
    alpha : float
        Angle of the line in degree, between 0 and 180.
    width : int
        Width of the image area.

    Returns
    -------
    tuple of 1d-darray
        x and y coordinates along the specified line.
    """

    if alpha < 0.0 or alpha > 180.0:
        raise ValueError("Angle is out of range [0, 180] (degree)")
    if slice_index < 0 or slice_index > width - 1:
        raise ValueError(f"Slice index is out of range [0, {width - 1}]")
    center = np.ceil(0.5 * (width - 1))
    x_min, x_max = - center, width - center
    y_min, y_max = x_min, x_max
    angle = np.deg2rad(alpha)
    x0 = y0 = slice_index - center
    if alpha <= 45 or alpha >= 135:
        a_fact = np.tan(angle)
        if angle <= 45:
            b_fact = - y0 / np.cos(angle)
        else:
            b_fact = y0 / np.cos(angle)
        xcr = - a_fact * b_fact / (1 + a_fact ** 2)
        xlist = xcr + np.arange(x_min, x_max) * np.cos(angle)
        ylist = a_fact * xlist + b_fact
    else:
        a_fact = np.cos(angle) / np.sin(angle)
        if angle <= 90:
            b_fact = x0 / np.sin(angle)
        else:
            b_fact = - x0 / np.sin(angle)
        ycr = - a_fact * b_fact / (1 + a_fact ** 2)
        ylist = ycr + np.arange(y_min, y_max) * np.sin(angle)
        xlist = a_fact * ylist + b_fact
    return xlist, ylist


def _get_points_multiple_lines(start_index, stop_index, alpha, width,
                              step_index=1):
    """
    Computes x and y coordinates of points on multiple parallel slices at
    an angle.

    Parameters
    ----------
    start_index : int
        Start index of the lines within image width.
    stop_index : int
        Stop index of the lines within image width.
    alpha : float
        Angle of the lines in degree, between 0 and 180.
    width : int
        Width of the image area.
    step_index : int, optional
        Gap between lines.

    Returns
    -------
    tuple of 2d-array
        x and y coordinates along the specified lines, with each row
        corresponding to a line.
    """
    x_mat = []
    y_mat = []
    for idx in np.arange(start_index, stop_index + 1, step_index):
        xlist, ylist = _get_points_single_line(idx, alpha, width)
        x_mat.append(xlist)
        y_mat.append(ylist)
    return np.float32(x_mat), np.float32(y_mat)


def _get_points_multiple_lines_different_angles(list_index, list_alpha,
                                                width):
    """
    Computes x and y coordinates of points on multiple slices at different
    angles.

    Parameters
    ----------
    list_index : list of int
        Index list of the lines within image width.
    list_alpha : list of float
        List of angles in degree, between 0 and 180.
    width : int
        Width of the image area.

    Returns
    -------
    tuple of 2d-array
        x and y coordinates along the specified lines, with each row
        corresponding to a line.
    """
    x_mat = []
    y_mat = []
    for i, index in enumerate(list_index):
        xlist, ylist = _get_points_single_line(index, list_alpha[i], width)
        x_mat.append(xlist)
        y_mat.append(ylist)
    return np.float32(x_mat), np.float32(y_mat)


def vertical_reconstruction(projections, slice_index, center, alpha=0.0,
                            flat_field=None, dark_field=None, angles=None,
                            crop=(0, 0, 0, 0), proj_start=0, proj_stop=-1,
                            chunk_size=30, ramp_filter="after",
                            filter_name="hann", pad=None, pad_mode="edge",
                            apply_log=True, gpu=True, block=(16, 16),
                            ncore=None, prefer="threads", show_progress=True,
                            masking=False):
    """
    Reconstruct a vertical slice given a stack of projection-images
    (num_projection, height, width) with optional use of the ramp-filter.

    Parameters
    ----------
    projections : array_like
        3D array of projection data with shape (depth, height, width). Can be
        a numpy array or HDF-dataset object.
    slice_index : int
        Index of the slice for reconstruction. Referred to the cropped image.
    center : float
        Center of rotation, x-coordinate of the rotation axis. Referred to the
        cropped image.
    alpha : float, optional
        Angle of the slice in degree, between 0 and 180.
    flat_field : array_like, optional
        2D array for flat-field correction if not None.
    dark_field : array_like, optional
        2D array for dark-field correction if not None.
    angles : array_like, optional
        1D array. Angles corresponding to projections. The unit is radian
        to be consistent with other reconstruction methods.
    crop : tuple of int, optional
        Edges to crop from the images (top, bottom, left, right).
    proj_start : int, optional
        Start index for processing projections.
    proj_stop : int, optional
        End index for processing projections.
    chunk_size : int, optional
        Chunk size to manage memory usage.
    ramp_filter : {"after", "before", None}
        When to apply the ramp filter or not apply.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming",\
                  "nuttall", "parzen", "triang"}
        Type of smoothing filter used with the ramp filter.
    pad : int, optional
        Padding before FFT (defaults to 10% of width if None).
    pad_mode : str, optional
        Padding method (see numpy.pad documentation).
    apply_log : bool, optional
        Apply logarithm to projections before reconstruction.
    gpu : bool, optional
        Use GPU for computing if True.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (8, 8), (16, 16), (32, 32), ...
    ncore : int or None
        Number of CPU cores used (auto-selected if None).
    prefer : {"threads", "processes"}
        Preferred parallel backend.
    show_progress : bool
        Display reconstruction progress.
    masking : bool
        Mask non-reconstructable areas.

    Returns
    -------
    array_like
        Reconstructed image, same size as projection (height, width).
    """
    if not (ramp_filter == "before" or ramp_filter == "after"
            or ramp_filter is None):
        raise ValueError("Must use one of these options: "
                         "'before', 'after', None")
    if ncore is None:
        ncore = int(np.clip(mp.cpu_count() - 1, 1, None))
    else:
        ncore = int(np.clip(ncore, 1, None))
    (num_proj0, height0, width0) = projections.shape
    if proj_stop == -1:
        proj_stop = num_proj0
    num_proj = proj_stop - proj_start
    if num_proj < 1:
        raise ValueError("Wrong value of proj_start or proj_stop !!! Given "
                         "the number of projections {}".format(num_proj0))
    if angles is None:
        angles = np.deg2rad(np.linspace(0.0, 180.0, num_proj))
    else:
        if len(angles) != num_proj:
            raise ValueError("!!! Number of angles is not the same as the "
                             "number of projections !!!")
    (cr_top, cr_bottom, cr_left, cr_right) = crop
    top, bot = cr_top, height0 - cr_bottom
    left, right = cr_left, width0 - cr_right
    width = right - left
    height = bot - top
    if height < 1 or width < 1:
        raise ValueError("Can't crop images with the given parameters !!!")
    if center < 0 or center > (width - 1):
        raise ValueError("Center (relative to the cropped image) is out of "
                         "range {})".format((0, width - 1)))
    if slice_index < 0 or slice_index > (width - 1):
        raise ValueError("Index (relative to the cropped image) is out of "
                         "range {})".format((0, width - 1)))

    flat_correction = True
    if (flat_field is None) and (dark_field is None):
        flat_correction = False
    else:
        if flat_field is None:
            flat_field = np.ones((height0, width0), dtype=np.float32)
        else:
            if flat_field.shape != (height0, width0):
                raise ValueError("!!! Shape of flat-field and projection-image"
                                 " is not the same !!!")
        if dark_field is None:
            dark_field = np.zeros((height0, width0), dtype=np.float32)
        else:
            if dark_field.shape != (height0, width0):
                raise ValueError("!!! Shape of dark-field and projection-image"
                                 " is not the same !!!")
    if flat_correction:
        flat = flat_field[top:bot, left:right]
        dark = dark_field[top:bot, left:right]
        flat_dark = flat - dark
        flat_dark[flat_dark == 0.0] = np.float32(1.0)
        flat_dark = np.float32(flat_dark)

    if chunk_size is None:
        chunk_size = min(30, num_proj // 4 + 1)
    if chunk_size > num_proj or chunk_size < 1:
        chunk_size = num_proj
    num_iter = num_proj // chunk_size
    num_rest = num_proj - num_iter * chunk_size
    ver_slice = np.zeros((height, width), dtype=np.float32)
    xlist, ylist = _get_points_single_line(slice_index, alpha, width)

    if pad is None:
        pad = min(int(0.15 * width), 150)
    edge_pad = True
    if ramp_filter == "before":
        ramp_win = rec.make_2d_ramp_window(height, width + 2 * pad,
                                           filter_name=filter_name)
        edge_pad = False

    if show_progress:
        t0 = time.time()
    for i in range(num_iter):
        start = i * chunk_size + proj_start
        stop = start + chunk_size
        img_chunk = projections[start:stop, top:bot, left:right]
        if flat_correction:
            img_chunk = (img_chunk - dark) / flat_dark
        if apply_log:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                try:
                    img_chunk = -np.log(img_chunk)
                except RuntimeWarning:
                    warnings.warn("!!! Applying logarithm is enabled but "
                                  "there are values <= 0.0 in the data !!!")
                    img_chunk[img_chunk <= 0.0] = np.float32(1.0)
                    img_chunk = -np.log(img_chunk)
        if ramp_filter == "before":
            img_chunk = util.parallel_process_slices(img_chunk,
                                                     rec.apply_ramp_filter,
                                                     [ramp_win, filter_name,
                                                      pad, pad_mode], axis=0,
                                                     ncore=ncore,
                                                     prefer=prefer)
        sub_angles = angles[start - proj_start:stop - proj_start]
        if gpu:
            ver_slice += vertical_back_projection_gpu(img_chunk, sub_angles,
                                                      xlist, ylist, center,
                                                      block=block,
                                                      edge_pad=edge_pad)
        else:
            ver_slice += vertical_back_projection_cpu(img_chunk, sub_angles,
                                                      xlist, ylist, center,
                                                      edge_pad=edge_pad)
        if show_progress:
            t1 = time.time()
            elapsed_time = t1 - t0
            percent_complete = 100.0 * (stop - proj_start) / num_proj
            sys.stdout.write(f"\rProcessed {stop - proj_start}/{num_proj} "
                             f"images ({percent_complete:.0f}%) - "
                             f"Time elapsed: {elapsed_time:.2f} seconds")
            sys.stdout.flush()
    if num_rest != 0:
        start = num_iter * chunk_size + proj_start
        stop = start + num_rest
        img_chunk = projections[start:stop, top:bot, left:right]
        if flat_correction:
            img_chunk = (img_chunk - dark) / flat_dark
        if apply_log:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                try:
                    img_chunk = -np.log(img_chunk)
                except RuntimeWarning:
                    warnings.warn("Applying logarithm is enabled but "
                                  "there are values <= 0.0.")
                    img_chunk[img_chunk <= 0.0] = np.float32(1.0)
                    img_chunk = -np.log(img_chunk)
        if ramp_filter == "before":
            img_chunk = util.parallel_process_slices(img_chunk,
                                                     rec.apply_ramp_filter,
                                                     [ramp_win, filter_name,
                                                      pad, pad_mode], axis=0,
                                                     ncore=ncore,
                                                     prefer=prefer)
        sub_angles = angles[start - proj_start:stop - proj_start]
        if gpu:
            ver_slice += vertical_back_projection_gpu(img_chunk, sub_angles,
                                                      xlist, ylist, center,
                                                      block=block,
                                                      edge_pad=edge_pad)
        else:
            ver_slice += vertical_back_projection_cpu(img_chunk, sub_angles,
                                                      xlist, ylist, center,
                                                      edge_pad=edge_pad)
        if show_progress:
            t1 = time.time()
            elapsed_time = t1 - t0
            percent_complete = 100.0 * (stop - proj_start) / num_proj
            sys.stdout.write(f"\rProcessed {stop - proj_start}/{num_proj} "
                             f"images ({percent_complete:.0f}%) - "
                             f"Time elapsed: {elapsed_time:.2f} seconds")
            sys.stdout.flush()
    if ramp_filter == "after":
        ramp_win = np.abs(rec.make_2d_ramp_window(height, width + 2 * pad,
                                                  filter_name=filter_name))
        recon_pad = np.pad(ver_slice, ((0, 0), (pad, pad)), mode=pad_mode)
        recon_fft = fft.fftshift(fft.fft(recon_pad), axes=1) * ramp_win
        ver_slice = np.real(fft.ifft(
            fft.ifftshift(recon_fft, axes=1)))[:, pad:pad + width]
    if show_progress:
        t1 = time.time()
        print("\nDone! Total time elapsed: {0:.2f}".format(t1 - t0))
    if masking:
        rad = int(min(center, width - center))
        pad_left = (width - 2 * rad) // 2
        pad_right = width - 2 * rad - pad_left
        list_tmp = np.pad(np.ones(2 * rad, dtype=np.float32),
                          (pad_left, pad_right), mode="constant")
        mask = np.tile(list_tmp, (height, 1))
        ver_slice = ver_slice * mask
    return ver_slice * np.pi / (num_proj - 1)


def __apply_ver_ramp_filter(ver_slice, ramp_win, width, pad, pad_mode):
    """
    Supplementary method to apply the ramp filter to a vertical slice.
    """
    recon_pad = np.pad(ver_slice, ((0, 0), (pad, pad)), mode=pad_mode)
    recon_fft = fft.fftshift(fft.fft(recon_pad), axes=1) * ramp_win
    ver_slice = np.real(fft.ifft(
        fft.ifftshift(recon_fft, axes=1)))[:, pad:pad + width]
    return ver_slice


def vertical_reconstruction_multiple(projections, start_index, stop_index,
                                     center, alpha=0.0, step_index=1,
                                     flat_field=None, dark_field=None,
                                     angles=None, crop=(0, 0, 0, 0),
                                     proj_start=0, proj_stop=-1, chunk_size=30,
                                     ramp_filter="after", filter_name="hann",
                                     pad=None, pad_mode="edge", apply_log=True,
                                     gpu=True, block=(16, 16), ncore=None,
                                     prefer="threads", show_progress=True,
                                     masking=False):
    """
    Reconstruct multiple vertical-slices given a stack of projection-images
    (num_projection, height, width) with optional use of the ramp-filter.

    Parameters
    ----------
    projections : array_like
        3D array of projection data with shape (depth, height, width). Can be
        a numpy array or HDF-dataset object.
    start_index : int
        Start index of reconstructing slice. Referred to the cropped image.
    stop_index : int
        End index of reconstructing slice. Referred to the cropped image.
    center : float
        Center of rotation, x-coordinate of the rotation axis. Referred to the
        cropped image.
    alpha : float, optional
        Angle of the slices in degree, between 0 and 180.
    step_index : int
        Gap  between reconstructing slices. Referred to the cropped image.
    flat_field : array_like, optional
        2D array for flat-field correction if not None.
    dark_field : array_like, optional
        2D array for dark-field correction if not None.
    angles : array_like, optional
        1D array. Angles corresponding to projections. The unit is radian
        to be consistent with other reconstruction methods.
    crop : tuple of int, optional
        Edges to crop from the images (top, bottom, left, right).
    proj_start : int, optional
        Start index for processing projections.
    proj_stop : int, optional
        End index for processing projections.
    chunk_size : int, optional
        Chunk size to manage memory usage.
    ramp_filter : {"after", "before", None}
        When to apply the ramp filter or not apply.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming",\
                  "nuttall", "parzen", "triang"}
        Type of smoothing filter used with the ramp filter.
    pad : int, optional
        Padding before FFT (defaults to 10% of width if None).
    pad_mode : str, optional
        Padding method (see numpy.pad documentation).
    apply_log : bool, optional
        Apply logarithm to projections before reconstruction.
    gpu : bool, optional
        Use GPU for computing if True.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (8, 8), (16, 16), (32, 32), ...
    ncore : int or None
        Number of CPU cores used (auto-selected if None).
    prefer : {"threads", "processes"}
        Preferred parallel backend.
    show_progress : bool
        Display reconstruction progress.
    masking : bool
        Mask non-reconstructable areas.

    Returns
    -------
    array_like
        3D array. Multiple-reconstructed image.
    """
    if not (ramp_filter == "before" or ramp_filter == "after"
            or ramp_filter is None):
        raise ValueError("Must use one of these options: "
                         "'before', 'after', None")
    if ncore is None:
        ncore = int(np.clip(mp.cpu_count() - 1, 1, None))
    else:
        ncore = int(np.clip(ncore, 1, None))
    (num_proj0, height0, width0) = projections.shape
    if proj_stop == -1:
        proj_stop = num_proj0
    num_proj = proj_stop - proj_start
    if num_proj < 1:
        raise ValueError("Wrong value of proj_start or proj_stop !!! Given "
                         "the number of projections {}".format(num_proj0))
    if angles is None:
        angles = np.deg2rad(np.linspace(0.0, 180.0, num_proj))
    else:
        if len(angles) != num_proj:
            raise ValueError("!!! Number of angles is not the same as the "
                             "number of projections !!!")
    (cr_top, cr_bottom, cr_left, cr_right) = crop
    top, bot = cr_top, height0 - cr_bottom
    left, right = cr_left, width0 - cr_right
    width = right - left
    height = bot - top
    if height < 1 or width < 1:
        raise ValueError("Can't crop images with the given parameters !!!")
    if center < 0 or center > (width - 1):
        raise ValueError("Center (relative to the cropped image) is out of "
                         "range {})".format((0, width - 1)))
    if start_index < 0 or start_index > (width - 1):
        raise ValueError("Start index (relative to the cropped image) is out "
                         "of range {})".format((0, width - 1)))
    if stop_index == -1:
        stop_index = width - 1
    if stop_index < 0 or stop_index > (width - 1):
        raise ValueError("Start index (relative to the cropped image) is out "
                         "of range {})".format((0, width - 1)))

    flat_correction = True
    if (flat_field is None) and (dark_field is None):
        flat_correction = False
    else:
        if flat_field is None:
            flat_field = np.ones((height0, width0), dtype=np.float32)
        else:
            if flat_field.shape != (height0, width0):
                raise ValueError("!!! Shape of flat-field and projection-image"
                                 " is not the same !!!")
        if dark_field is None:
            dark_field = np.zeros((height0, width0), dtype=np.float32)
        else:
            if dark_field.shape != (height0, width0):
                raise ValueError("!!! Shape of dark-field and projection-image"
                                 " is not the same !!!")
    if flat_correction:
        flat = flat_field[top:bot, left:right]
        dark = dark_field[top:bot, left:right]
        flat_dark = flat - dark
        flat_dark[flat_dark == 0.0] = np.float32(1.0)
        flat_dark = np.float32(flat_dark)

    if chunk_size is None:
        chunk_size = min(30, num_proj // 4 + 1)
    if chunk_size > num_proj or chunk_size < 1:
        chunk_size = num_proj
    num_iter = num_proj // chunk_size
    num_rest = num_proj - num_iter * chunk_size
    x_mat, y_mat = _get_points_multiple_lines(start_index, stop_index, alpha,
                                              width, step_index)
    num_slice = len(x_mat)
    ver_slices = np.zeros((num_slice, height, width), dtype=np.float32)
    if pad is None:
        pad = min(int(0.15 * width), 150)
    edge_pad = True
    if ramp_filter == "before":
        ramp_win = rec.make_2d_ramp_window(height, width + 2 * pad,
                                           filter_name=filter_name)
        edge_pad = False

    if show_progress:
        t0 = time.time()
    for i in range(num_iter):
        start = i * chunk_size + proj_start
        stop = start + chunk_size
        img_chunk = projections[start:stop, top:bot, left:right]
        if flat_correction:
            img_chunk = (img_chunk - dark) / flat_dark
        if apply_log:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                try:
                    img_chunk = -np.log(img_chunk)
                except RuntimeWarning:
                    warnings.warn("!!! Applying logarithm is enabled but "
                                  "there are values <= 0.0 in the data !!!")
                    img_chunk[img_chunk <= 0.0] = np.float32(1.0)
                    img_chunk = -np.log(img_chunk)
        if ramp_filter == "before":
            img_chunk = util.parallel_process_slices(img_chunk,
                                                     rec.apply_ramp_filter,
                                                     [ramp_win, filter_name,
                                                      pad, pad_mode], axis=0,
                                                     ncore=ncore,
                                                     prefer=prefer)
        sub_angles = angles[start - proj_start:stop - proj_start]
        if gpu:
            ver_slices += vertical_back_projection_gpu_chunk(img_chunk,
                                                             sub_angles,
                                                             x_mat, y_mat,
                                                             center,
                                                             block=block,
                                                             edge_pad=edge_pad)
        else:
            ver_slices += vertical_back_projection_cpu_chunk(img_chunk,
                                                             sub_angles,
                                                             x_mat, y_mat,
                                                             center,
                                                             edge_pad=edge_pad)
        if show_progress:
            t1 = time.time()
            elapsed_time = t1 - t0
            percent_complete = 100.0 * (stop - proj_start) / num_proj
            sys.stdout.write(f"\rProcessed {stop - proj_start}/{num_proj} "
                             f"images ({percent_complete:.0f}%) - "
                             f"Time elapsed: {elapsed_time:.2f} seconds")
            sys.stdout.flush()
    if num_rest != 0:
        start = num_iter * chunk_size + proj_start
        stop = start + num_rest
        img_chunk = projections[start:stop, top:bot, left:right]
        if flat_correction:
            img_chunk = (img_chunk - dark) / flat_dark
        if apply_log:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                try:
                    img_chunk = -np.log(img_chunk)
                except RuntimeWarning:
                    warnings.warn("Applying logarithm is enabled but "
                                  "there are values <= 0.0.")
                    img_chunk[img_chunk <= 0.0] = np.float32(1.0)
                    img_chunk = -np.log(img_chunk)
        if ramp_filter == "before":
            img_chunk = util.parallel_process_slices(img_chunk,
                                                     rec.apply_ramp_filter,
                                                     [ramp_win, filter_name,
                                                      pad, pad_mode], axis=0,
                                                     ncore=ncore,
                                                     prefer=prefer)
        sub_angles = angles[start - proj_start:stop - proj_start]
        if gpu:
            ver_slices += vertical_back_projection_gpu_chunk(img_chunk,
                                                             sub_angles,
                                                             x_mat, y_mat,
                                                             center,
                                                             block=block,
                                                             edge_pad=edge_pad)
        else:
            ver_slices += vertical_back_projection_cpu_chunk(img_chunk,
                                                             sub_angles,
                                                             x_mat, y_mat,
                                                             center,
                                                             edge_pad=edge_pad)
        if show_progress:
            t1 = time.time()
            elapsed_time = t1 - t0
            percent_complete = 100.0 * (stop - proj_start) / num_proj
            sys.stdout.write(f"\rProcessed {stop - proj_start}/{num_proj} "
                             f"images ({percent_complete:.0f}%) - "
                             f"Time elapsed: {elapsed_time:.2f} seconds")
            sys.stdout.flush()
    if ramp_filter == "after":
        ramp_win = np.abs(rec.make_2d_ramp_window(height, width + 2 * pad,
                                                  filter_name=filter_name))
        slice_filtered = []
        for i in range(num_slice):
            slice_filtered.append(__apply_ver_ramp_filter(ver_slices[i],
                                                          ramp_win, width, pad,
                                                          pad_mode))
        ver_slices = np.float32(slice_filtered)
    if show_progress:
        t1 = time.time()
        print("\nDone! Total time elapsed: {0:.2f}".format(t1 - t0))
    if masking:
        rad = int(min(center, width - center))
        pad_left = (width - 2 * rad) // 2
        pad_right = width - 2 * rad - pad_left
        list_tmp = np.pad(np.ones(2 * rad, dtype=np.float32),
                          (pad_left, pad_right), mode="constant")
        mask = np.tile(list_tmp, (height, 1))
        for i in range(num_slice):
            ver_slices[i] = ver_slices[i] * mask
    return ver_slices * np.pi / (num_proj - 1)


def vertical_reconstruction_different_angles(projections, slice_indices,
                                             alphas, center, flat_field=None,
                                             dark_field=None, angles=None,
                                             crop=(0, 0, 0, 0), proj_start=0,
                                             proj_stop=-1, chunk_size=30,
                                             ramp_filter="after",
                                             filter_name="hann", pad=None,
                                             pad_mode="edge", apply_log=True,
                                             gpu=True, block=(16, 16),
                                             ncore=None, prefer="threads",
                                             show_progress=True,
                                             masking=False):
    """
    Reconstruct multiple vertical-slices at different slice-angles given a
    stack of projection-images (num_projection, height, width) with optional
    use of the ramp-filter.

    Parameters
    ----------
    projections : array_like
        3D array of projection data with shape (depth, height, width). Can be
        a numpy array or HDF-dataset object.
    slice_indices : list of int
        List of reconstructing slice indices. Referred to the cropped image.
    alphas : list of float
        List of angles (degrees, between 0 and 180) corresponding to the slice
        indices.
    center : float
        Center of rotation, x-coordinate of the rotation axis. Referred to the
        cropped image.
    flat_field : array_like, optional
        2D array for flat-field correction if not None.
    dark_field : array_like, optional
        2D array for dark-field correction if not None.
    angles : array_like, optional
        1D array. Angles corresponding to projections. The unit is radian
        to be consistent with other reconstruction methods.
    crop : tuple of int, optional
        Edges to crop from the images (top, bottom, left, right).
    proj_start : int, optional
        Start index for processing projections.
    proj_stop : int, optional
        End index for processing projections.
    chunk_size : int, optional
        Chunk size to manage memory usage.
    ramp_filter : {"after", "before", None}
        When to apply the ramp filter or not apply.
    filter_name : {None, "hann", "bartlett", "blackman", "hamming",\
                  "nuttall", "parzen", "triang"}
        Type of smoothing filter used with the ramp filter.
    pad : int, optional
        Padding before FFT (defaults to 10% of width if None).
    pad_mode : str, optional
        Padding method (see numpy.pad documentation).
    apply_log : bool, optional
        Apply logarithm to projections before reconstruction.
    gpu : bool, optional
        Use GPU for computing if True.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (8, 8), (16, 16), (32, 32), ...
    ncore : int or None
        Number of CPU cores used (auto-selected if None).
    prefer : {"threads", "processes"}
        Preferred parallel backend.
    show_progress : bool
        Display reconstruction progress.
    masking : bool
        Mask non-reconstructable areas.

    Returns
    -------
    array_like
        3D array. Multiple-reconstructed image.
    """
    if not isinstance(slice_indices, list):
        raise ValueError("Please provide a list of slice indices")
    if not isinstance(alphas, list):
        raise ValueError("Please provide a list of angles corresponding to"
                         "the slice indices")
    if len(slice_indices) != len(alphas):
        raise ValueError("!!! Number of slice indices is not the same as the "
                         "number of angles !!!")
    if not (ramp_filter == "before" or ramp_filter == "after"
            or ramp_filter is None):
        raise ValueError("Must use one of these options: "
                         "'before', 'after', None")
    if ncore is None:
        ncore = int(np.clip(mp.cpu_count() - 1, 1, None))
    else:
        ncore = int(np.clip(ncore, 1, None))
    (num_proj0, height0, width0) = projections.shape
    if proj_stop == -1:
        proj_stop = num_proj0
    num_proj = proj_stop - proj_start
    if num_proj < 1:
        raise ValueError("Wrong value of proj_start or proj_stop !!! Given "
                         "the number of projections {}".format(num_proj0))
    if angles is None:
        angles = np.deg2rad(np.linspace(0.0, 180.0, num_proj))
    else:
        if len(angles) != num_proj:
            raise ValueError("!!! Number of angles is not the same as the "
                             "number of projections !!!")
    (cr_top, cr_bottom, cr_left, cr_right) = crop
    top, bot = cr_top, height0 - cr_bottom
    left, right = cr_left, width0 - cr_right
    width = right - left
    height = bot - top
    width1 = width - 1
    if height < 1 or width < 1:
        raise ValueError("Can't crop images with the given parameters !!!")
    if center < 0 or center > width1:
        raise ValueError("Center (relative to the cropped image) is out of "
                         "range {})".format((0, width1)))
    if any(idx < 0 or idx > width1 for idx in slice_indices):
        raise ValueError("Slice index (relative to the cropped image) is out "
                         "of range {})".format((0, width1)))

    flat_correction = True
    if (flat_field is None) and (dark_field is None):
        flat_correction = False
    else:
        if flat_field is None:
            flat_field = np.ones((height0, width0), dtype=np.float32)
        else:
            if flat_field.shape != (height0, width0):
                raise ValueError("!!! Shape of flat-field and projection-image"
                                 " is not the same !!!")
        if dark_field is None:
            dark_field = np.zeros((height0, width0), dtype=np.float32)
        else:
            if dark_field.shape != (height0, width0):
                raise ValueError("!!! Shape of dark-field and projection-image"
                                 " is not the same !!!")
    if flat_correction:
        flat = flat_field[top:bot, left:right]
        dark = dark_field[top:bot, left:right]
        flat_dark = flat - dark
        flat_dark[flat_dark == 0.0] = np.float32(1.0)
        flat_dark = np.float32(flat_dark)

    if chunk_size is None:
        chunk_size = min(30, num_proj // 4 + 1)
    if chunk_size > num_proj or chunk_size < 1:
        chunk_size = num_proj
    num_iter = num_proj // chunk_size
    num_rest = num_proj - num_iter * chunk_size
    x_mat, y_mat = _get_points_multiple_lines_different_angles(slice_indices,
                                                               alphas, width)
    num_slice = len(x_mat)
    ver_slices = np.zeros((num_slice, height, width), dtype=np.float32)
    if pad is None:
        pad = min(int(0.15 * width), 150)
    edge_pad = True
    if ramp_filter == "before":
        ramp_win = rec.make_2d_ramp_window(height, width + 2 * pad,
                                           filter_name=filter_name)
        edge_pad = False

    if show_progress:
        t0 = time.time()
    for i in range(num_iter):
        start = i * chunk_size + proj_start
        stop = start + chunk_size
        img_chunk = projections[start:stop, top:bot, left:right]
        if flat_correction:
            img_chunk = (img_chunk - dark) / flat_dark
        if apply_log:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                try:
                    img_chunk = -np.log(img_chunk)
                except RuntimeWarning:
                    warnings.warn("!!! Applying logarithm is enabled but "
                                  "there are values <= 0.0 in the data !!!")
                    img_chunk[img_chunk <= 0.0] = np.float32(1.0)
                    img_chunk = -np.log(img_chunk)
        if ramp_filter == "before":
            img_chunk = util.parallel_process_slices(img_chunk,
                                                     rec.apply_ramp_filter,
                                                     [ramp_win, filter_name,
                                                      pad, pad_mode], axis=0,
                                                     ncore=ncore,
                                                     prefer=prefer)
        sub_angles = angles[start - proj_start:stop - proj_start]
        if gpu:
            ver_slices += vertical_back_projection_gpu_chunk(img_chunk,
                                                             sub_angles,
                                                             x_mat, y_mat,
                                                             center,
                                                             block=block,
                                                             edge_pad=edge_pad)
        else:
            ver_slices += vertical_back_projection_cpu_chunk(img_chunk,
                                                             sub_angles,
                                                             x_mat, y_mat,
                                                             center,
                                                             edge_pad=edge_pad)
        if show_progress:
            t1 = time.time()
            elapsed_time = t1 - t0
            percent_complete = 100.0 * (stop - proj_start) / num_proj
            sys.stdout.write(f"\rProcessed {stop - proj_start}/{num_proj} "
                             f"images ({percent_complete:.0f}%) - "
                             f"Time elapsed: {elapsed_time:.2f} seconds")
            sys.stdout.flush()
    if num_rest != 0:
        start = num_iter * chunk_size + proj_start
        stop = start + num_rest
        img_chunk = projections[start:stop, top:bot, left:right]
        if flat_correction:
            img_chunk = (img_chunk - dark) / flat_dark
        if apply_log:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                try:
                    img_chunk = -np.log(img_chunk)
                except RuntimeWarning:
                    warnings.warn("Applying logarithm is enabled but "
                                  "there are values <= 0.0.")
                    img_chunk[img_chunk <= 0.0] = np.float32(1.0)
                    img_chunk = -np.log(img_chunk)
        if ramp_filter == "before":
            img_chunk = util.parallel_process_slices(img_chunk,
                                                     rec.apply_ramp_filter,
                                                     [ramp_win, filter_name,
                                                      pad, pad_mode], axis=0,
                                                     ncore=ncore,
                                                     prefer=prefer)
        sub_angles = angles[start - proj_start:stop - proj_start]
        if gpu:
            ver_slices += vertical_back_projection_gpu_chunk(img_chunk,
                                                             sub_angles,
                                                             x_mat, y_mat,
                                                             center,
                                                             block=block,
                                                             edge_pad=edge_pad)
        else:
            ver_slices += vertical_back_projection_cpu_chunk(img_chunk,
                                                             sub_angles,
                                                             x_mat, y_mat,
                                                             center,
                                                             edge_pad=edge_pad)
        if show_progress:
            t1 = time.time()
            elapsed_time = t1 - t0
            percent_complete = 100.0 * (stop - proj_start) / num_proj
            sys.stdout.write(f"\rProcessed {stop - proj_start}/{num_proj} "
                             f"images ({percent_complete:.0f}%) - "
                             f"Time elapsed: {elapsed_time:.2f} seconds")
            sys.stdout.flush()
    if ramp_filter == "after":
        ramp_win = np.abs(rec.make_2d_ramp_window(height, width + 2 * pad,
                                                  filter_name=filter_name))
        slice_filtered = []
        for i in range(num_slice):
            slice_filtered.append(__apply_ver_ramp_filter(ver_slices[i],
                                                          ramp_win, width, pad,
                                                          pad_mode))
        ver_slices = np.float32(slice_filtered)
    if show_progress:
        t1 = time.time()
        print("\nDone! Total time elapsed: {0:.2f}".format(t1 - t0))
    if masking:
        rad = int(min(center, width - center))
        pad_left = (width - 2 * rad) // 2
        pad_right = width - 2 * rad - pad_left
        list_tmp = np.pad(np.ones(2 * rad, dtype=np.float32),
                          (pad_left, pad_right), mode="constant")
        mask = np.tile(list_tmp, (height, 1))
        for i in range(num_slice):
            ver_slices[i] = ver_slices[i] * mask
    return ver_slices * np.pi / (num_proj - 1)


def __calculate_autocorrelation_coefficient(image):
    """
    Calculate a metric based on auto-correlation coefficient.
    """
    fft_row = fft.fft(np.float32(image))
    autocorr = np.abs(fft.ifft(np.abs(fft_row) ** 2))
    return np.mean(np.max(autocorr, axis=1))


def find_center_vertical_slice(projections, slice_index, start, stop, step=1.0,
                               metric="entropy", alpha=0.0, angles=None,
                               chunk_size=30, ramp_filter="after",
                               apply_log=True, gpu=True, block=(16, 16),
                               ncore=None, prefer="threads",
                               show_progress=True, masking=True,
                               return_metric=False, invert_metric=False,
                               metric_function=None, **kwargs):
    """
    Find the center-of-rotation (COR) by evaluating reconstruction metrics
    across estimated centers. Minimum metric is corresponding to the optimal
    center.

    Parameters
    ----------
    projections : array_like
        3D array of projection data with shape (depth, height, width).
    slice_index : int
        Index of the slice for reconstruction.
    start : float
        Starting point for searching the center.
    stop : float
        Ending point for searching the center.
    step : float, optional
        Searching step.
    metric : {"entropy", "sharpness", "autocorrelation"}
        Which metric to use.
    alpha : float, optional
        Angle of the slice in degree, between 0 and 180.
    angles : array_like, optional
        1D array. Angles corresponding to projections. The unit is radian
        to be consistent with other reconstruction methods.
    chunk_size : int, optional
        Chunk size to manage memory usage.
    ramp_filter : {"after", "before"}
        When to apply the ramp filter.
    apply_log : bool, optional
        Apply logarithm to projections before reconstruction.
    gpu : bool, optional
        Use GPU for computing if True.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (8, 8), (16, 16), (32, 32), ...
    ncore : int or None
        Number of CPU cores used (auto-selected if None).
    prefer : {"threads", "processes"}
        Preferred parallel backend.
    show_progress : bool
        Display the computing progress.
    masking : bool
        Mask non-reconstructable areas.
    return_metric : bool
        Return list of centers and their metrics if True.
    invert_metric : bool
        Invert the metric scale.
    metric_function : obj
        Custom function to calculate metric, accepts keyword
        arguments (kwargs).

    Returns
    -------
    float or ndarray
        Optimal center or tuple of two lists; centers and their metrics if
        return_metric=True.
    """
    if (metric != "entropy" and metric != "sharpness"
            and metric != "autocorrelation" and metric_function is None):
        msg = "Please select one of three options: 'entropy', 'sharpness', " \
              "'autocorrelation'; or provide a custom method."
        raise ValueError(msg)
    if ramp_filter is None:
        ramp_filter = "after"
    list_center = np.arange(start, stop + step, step)
    list_metric = np.zeros_like(list_center)
    num_metric = len(list_metric)
    (num_proj, height, width) = projections.shape
    if angles is None:
        angles = np.deg2rad(np.linspace(0.0, 180.0, num_proj))

    if apply_log:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=RuntimeWarning)
            try:
                projections = -np.log(projections)
            except RuntimeWarning:
                warnings.warn("!!! Applying logarithm is enabled but "
                              "there are values <= 0.0 in the data !!!")
                projections[projections <= 0.0] = np.float32(1.0)
                projections = -np.log(projections)

    if masking:
        rad = int(min(np.min(list_center), width - np.min(list_center)))
        pad_left = (width - 2 * rad) // 2
        pad_right = width - 2 * rad - pad_left
        list_tmp = np.pad(np.ones(2 * rad, dtype=np.float32),
                          (pad_left, pad_right), mode="constant")
        mask = np.tile(list_tmp, (height, 1))

    pad = min(int(0.15 * width), 150)
    if metric_function is None:
        if metric == "entropy":
            center = np.mean(list_center)
            recon = vertical_reconstruction(projections, slice_index, center,
                                            alpha=alpha, flat_field=None,
                                            dark_field=None, angles=angles,
                                            crop=(0, 0, 0, 0), proj_start=0,
                                            proj_stop=-1,
                                            chunk_size=chunk_size,
                                            ramp_filter=ramp_filter, pad=pad,
                                            pad_mode="edge",
                                            filter_name="hann",
                                            apply_log=False, gpu=gpu,
                                            block=block, ncore=ncore,
                                            prefer=prefer, show_progress=False,
                                            masking=False)
            nmin, nmax = np.min(recon), np.max(recon)
            window = signal.windows.boxcar(7)

    if show_progress:
        t0 = time.time()
    for i, center in enumerate(list_center):
        recon = vertical_reconstruction(projections, slice_index, center,
                                        alpha=alpha, flat_field=None,
                                        dark_field=None, angles=angles,
                                        crop=(0, 0, 0, 0), proj_start=0,
                                        proj_stop=-1, chunk_size=chunk_size,
                                        ramp_filter=ramp_filter, pad=pad,
                                        pad_mode="edge", filter_name="hann",
                                        apply_log=False, gpu=gpu, block=block,
                                        ncore=ncore, prefer=prefer,
                                        show_progress=False, masking=False)
        if masking:
            recon = recon * mask
        if metric_function is None:
            if metric == "entropy":
                recon1 = (recon - nmin) / (nmax - nmin)
                recon1 = np.clip(recon1, 0, 1)
                value = rec.__calculate_histogram_entropy(recon1, window)
            elif metric == "autocorrelation":
                value = __calculate_autocorrelation_coefficient(recon)
            else:
                value = rec.__calculate_edge_sharpness(recon)
        else:
            value = metric_function(recon, **kwargs)
        list_metric[i] = value
        if show_progress:
            t1 = time.time()
            elapsed_time = t1 - t0
            percent_complete = 100.0 * (i + 1) / num_metric
            sys.stdout.write(
                f"\rReconstructed image using the center: {center}. "
                f"Progress: {percent_complete:.0f}% - "
                f"Time elapsed: {elapsed_time:.2f} seconds")
    if invert_metric:
        list_metric = np.max(list_metric) - list_metric
    min_pos = np.argmin(list_metric)
    center = list_center[min_pos]
    if show_progress:
        print("\nDone! Optimal center: {0}.".format(center))
    if return_metric is True:
        return list_center, list_metric
    else:
        return center


def find_center_visual_vertical_slices(projections, output_folder, slice_index,
                                       start, stop, step=1.0, alpha=0.0,
                                       angles=None, chunk_size=30,
                                       ramp_filter="after", apply_log=True,
                                       gpu=True, block=(16, 16), ncore=None,
                                       prefer="processes", display=True,
                                       masking=True):
    """
    For visually finding the center-of-rotation (COR) using reconstructed
    slices at different CORs.

    Parameters
    ----------
    projections : array_like
        3D array of projection data with shape (depth, height, width).
    output_folder : str
        Base folder for saving reconstructed slices.
    slice_index : int
        Index of the slice for reconstruction.
    start : float
        Starting point for searching the center.
    stop : float
        Ending point for searching the center.
    step : float, optional
        Searching step.
    alpha : float, optional
        Angle of the slice in degree, between 0 and 180.
    angles : array_like, optional
        1D array. Angles corresponding to projections. The unit is radian
        to be consistent with other reconstruction methods.
    chunk_size : int, optional
        Chunk size to manage memory usage.
    ramp_filter : {"after", "before"}
        When to apply the ramp filter.
    apply_log : bool, optional
        Apply logarithm to projections before reconstruction.
    gpu : bool, optional
        Use GPU for computing if True.
    block : tuple of two integer-values, optional
        Size of a GPU block. E.g. (8, 8), (16, 16), (32, 32), ...
    ncore : int or None
        Number of CPU cores used (auto-selected if None).
    prefer : {"threads", "processes"}
        Preferred parallel backend.
    display : bool
        Print the output if True.
    masking : bool
        Mask non-reconstructable areas.

    Returns
    -------
    output_base : str
        Folder path to tif images.
    """
    output_name = losa.make_folder_name(output_folder,
                                        name_prefix="Find_center",
                                        zero_prefix=3)
    output_base = output_folder + "/" + output_name + "/"
    list_center = np.arange(start, stop + step, step)
    (num_proj, height, width) = projections.shape
    if angles is None:
        angles = np.deg2rad(np.linspace(0.0, 180.0, num_proj))
    if apply_log:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=RuntimeWarning)
            try:
                projections = -np.log(projections)
            except RuntimeWarning:
                warnings.warn("!!! Applying logarithm is enabled but "
                              "there are values <= 0.0 in the data !!!")
                projections[projections <= 0.0] = np.float32(1.0)
                projections = -np.log(projections)
    if ramp_filter is None:
        ramp_filter = "after"
    if masking:
        rad = int(min(np.min(list_center), width - np.min(list_center)))
        pad_left = (width - 2 * rad) // 2
        pad_right = width - 2 * rad - pad_left
        list_tmp = np.pad(np.ones(2 * rad, dtype=np.float32),
                          (pad_left, pad_right), mode="constant")
        mask = np.tile(list_tmp, (height, 1))
    pad = min(int(0.15 * width), 150)
    if display:
        t0 = time.time()
    for i, center in enumerate(list_center):
        recon = vertical_reconstruction(projections, slice_index, center,
                                        alpha=alpha, flat_field=None,
                                        dark_field=None, angles=angles,
                                        crop=(0, 0, 0, 0), proj_start=0,
                                        proj_stop=-1, chunk_size=chunk_size,
                                        ramp_filter=ramp_filter, pad=pad,
                                        pad_mode="edge", filter_name="hann",
                                        apply_log=False, gpu=gpu,
                                        block=block, ncore=ncore,
                                        prefer=prefer, show_progress=False,
                                        masking=False)
        if masking:
            recon = recon * mask
        file_name = "center_{0:.2f}".format(center) + ".tif"
        losa.save_image(output_base + file_name, recon)
        if display:
            t1 = time.time()
            print("Done: {0}. Time elapsed {1:.2f}".format(
                  output_base + file_name, t1 - t0))
    return output_base
