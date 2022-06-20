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
Module of conversion methods in the preprocessing stage:
- Stitching images.
- Joining images if there is no overlapping.
- Converting a 360-degree sinogram with offset center-of-rotation (COR) to
  a 180-degree sinogram.
- Extending a 360-degree sinogram with offset COR for direct reconstruction
  instead of converting it to a 180-degree sinogram.
- Converting a 180-degree sinogram to a 360-sinogram.
- Generating a sinogram from a helical data.
"""

import numpy as np
from scipy import interpolate
from scipy.ndimage import shift
import algotom.prep.removal as remo
import algotom.prep.calculation as calc


def make_weight_matrix(mat1, mat2, overlap, side):
    """
    Generate a linear-ramp weighting matrix for image stitching.

    Parameters
    ----------
    mat1 : array_like
        2D array. Projection image or sinogram image.
    mat2 :  array_like
        2D array. Projection image or sinogram image.
    overlap : int
        Width of the overlap area between two images.
    side : {0, 1}
        Only two options: 0 or 1. It is used to indicate the overlap side
        respects to image 1. "0" corresponds to the left side. "1" corresponds
        to the right side.
    """
    overlap = int(np.floor(overlap))
    wei_mat1 = np.ones_like(mat1)
    wei_mat2 = np.ones_like(mat2)
    if side == 1:
        list_down = np.linspace(1.0, 0.0, overlap)
        list_up = 1.0 - list_down
        wei_mat1[:, -overlap:] = np.float32(list_down)
        wei_mat2[:, :overlap] = np.float32(list_up)
    else:
        list_down = np.linspace(1.0, 0.0, overlap)
        list_up = 1.0 - list_down
        wei_mat2[:, -overlap:] = np.float32(list_down)
        wei_mat1[:, :overlap] = np.float32(list_up)
    return wei_mat1, wei_mat2


def stitch_image(mat1, mat2, overlap, side, wei_mat1=None, wei_mat2=None,
                 norm=True, total_width=None):
    """
    Stitch projection images or sinogram images using a linear ramp.

    Parameters
    ----------
    mat1 : array_like
        2D array. Projection image or sinogram image.
    mat2 :  array_like
        2D array. Projection image or sinogram image.
    overlap : float
        Width of the overlap area between two images.
    side : {0, 1}
        Only two options: 0 or 1. It is used to indicate the overlap side
        respects to image 1. "0" corresponds to the left side. "1" corresponds
        to the right side.
    wei_mat1 : array_like, optional
        Weighting matrix used for image 1.
    wei_mat2 : array_like, optional
        Weighting matrix used for image 2.
    norm : bool, optional
        Enable/disable normalization before stitching.
    total_width : int, optional
        Final width of the stitched image.

    Returns
    -------
    array_like
        Stitched image.
    """
    (nrow1, ncol1) = mat1.shape
    (nrow2, ncol2) = mat2.shape
    overlap_int = int(np.floor(overlap))
    sub_pixel = overlap - overlap_int
    if sub_pixel > 0.0:
        if side == 1:
            mat1 = shift(mat1, (0, sub_pixel), mode='nearest')
            mat2 = shift(mat2, (0, -sub_pixel), mode='nearest')
        else:
            mat1 = shift(mat1, (0, -sub_pixel), mode='nearest')
            mat2 = shift(mat2, (0, sub_pixel), mode='nearest')
    if nrow1 != nrow2:
        raise ValueError("Two images are not at the same height!!!")
    if (wei_mat1 is None) or (wei_mat2 is None):
        (wei_mat1, wei_mat2) = make_weight_matrix(mat1, mat2, overlap_int, side)
    total_width0 = ncol1 + ncol2 - overlap_int
    if (total_width is None) or (total_width < total_width0):
        total_width = total_width0
    mat_comb = np.zeros((nrow1, total_width0), dtype=np.float32)
    if side == 1:
        if norm is True:
            factor1 = np.mean(mat1[:, -overlap_int:])
            factor2 = np.mean(mat2[:, :overlap_int])
            mat2 = mat2 * factor1 / factor2
        mat_comb[:, 0:ncol1] = mat1 * wei_mat1
        mat_comb[:, (ncol1 - overlap_int):total_width0] += mat2 * wei_mat2
    else:
        if norm is True:
            factor2 = np.mean(mat2[:, -overlap_int:])
            factor1 = np.mean(mat1[:, :overlap_int])
            mat2 = mat2 * factor1 / factor2
        mat_comb[:, 0:ncol2] = mat2 * wei_mat2
        mat_comb[:, (ncol2 - overlap_int):total_width0] += mat1 * wei_mat1
    if total_width > total_width0:
        mat_comb = np.pad(
            mat_comb, ((0, 0), (0, total_width - total_width0)), mode='edge')
    return mat_comb


def join_image(mat1, mat2, joint_width, side, norm=True, total_width=None):
    """
    Join projection images or sinogram images. This is useful for fixing the
    problem of non-overlap between images.

    Parameters
    ----------
    mat1 : array_like
        2D array. Projection image or sinogram image.
    mat2 :  array_like
        2D array. Projection image or sinogram image.
    joint_width : float
        Width of the joint area between two images.
    side : {0, 1}
        Only two options: 0 or 1. It is used to indicate the overlap side
        respects to image 1. "0" corresponds to the left side. "1" corresponds
        to the right side.
    norm : bool
        Enable/disable normalization before joining.
    total_width : int, optional
        Final width of the joined image.

    Returns
    -------
    array_like
        Stitched image.
    """
    (nrow1, ncol1) = mat1.shape
    (nrow2, ncol2) = mat2.shape
    joint_int = int(np.floor(joint_width))
    sub_pixel = joint_width - joint_int
    side = int(side)
    if sub_pixel > 0.0:
        if side == 1:
            mat1 = shift(mat1, (0, sub_pixel), mode='nearest')
            mat2 = shift(mat2, (0, -sub_pixel), mode='nearest')
        else:
            mat1 = shift(mat1, (0, -sub_pixel), mode='nearest')
            mat2 = shift(mat2, (0, sub_pixel), mode='nearest')
    if nrow1 != nrow2:
        raise ValueError("Two images are not at the same height!!!")
    total_width0 = ncol1 + ncol2 + joint_int
    if (total_width is None) or (total_width < total_width0):
        total_width = total_width0
    mat_comb = np.zeros((nrow1, total_width0), dtype=np.float32)
    if side == 1:
        if norm is True:
            factor1 = np.mean(mat1[:, -3:])
            factor2 = np.mean(mat2[:, :3])
            mat2 = mat2 * factor1 / factor2
        mat_comb[:, 0:ncol1] = mat1
        mat_comb[:, (ncol1 + joint_int):total_width0] += mat2
        list_mask = np.zeros(total_width0, dtype=np.float32)
        list_mask[ncol1 - 2:ncol1 + joint_int + 3] = 1.0
        listx = np.where(list_mask < 1.0)[0]
        listy = np.arange(nrow1)
        mat = mat_comb[:, listx]
        finter = interpolate.interp2d(listx, listy, mat, kind='linear')
        listx_miss = np.where(list_mask > 0.0)[0]
        if len(listx_miss) > 0:
            mat_comb[:, listx_miss] = finter(listx_miss, listy)
    else:
        if norm is True:
            factor2 = np.mean(mat2[:, -3:])
            factor1 = np.mean(mat1[:, :3])
            mat2 = mat2 * factor1 / factor2
        mat_comb[:, 0:ncol2] = mat2
        mat_comb[:, (ncol2 + joint_int):total_width0] += mat1
        list_mask = np.zeros(total_width0, dtype=np.float32)
        list_mask[ncol2 - 2:ncol2 + joint_int + 3] = 1.0
        listx = np.where(list_mask < 1.0)[0]
        listy = np.arange(nrow1)
        mat = mat_comb[:, listx]
        finter = interpolate.interp2d(listx, listy, mat, kind='linear')
        listx_miss = np.where(list_mask > 0.0)[0]
        if len(listx_miss) > 0:
            mat_comb[:, listx_miss] = finter(listx_miss, listy)
    if total_width > total_width0:
        mat_comb = np.pad(
            mat_comb, ((0, 0), (0, total_width - total_width0)), mode='edge')
    return mat_comb


def stitch_image_multiple(list_mat, list_overlap, norm=True, total_width=None):
    """
    Stitch list of projection images or sinogram images using a linear ramp.

    Parameters
    ----------
    list_mat : list of array_like
        List of 2D array. Projection image or sinogram image.
    list_overlap : list of tuple of floats
        List of [overlap, side].
        overlap : Width of the overlap area between two images.
        side : Overlap side between two images.
    norm : bool, optional
        Enable/disable normalization before stitching.
    total_width : int, optional
        Final width of the stitched image.

    Returns
    -------
    array_like
        Stitched image.
    """
    num_mat = len(list_mat)
    mat_comb = np.copy(list_mat[0])
    if num_mat > 1:
        for i in range(1, num_mat):
            (overlap, side) = list_overlap[i - 1][0:2]
            mat_comb = stitch_image(mat_comb, list_mat[i], overlap, side, norm)
        width = mat_comb.shape[1]
        if total_width is None:
            total_width = width
        if total_width > width:
            mat_comb = np.pad(
                mat_comb, ((0, 0), (0, total_width - width)), mode='edge')
    else:
        raise ValueError("Need at least 2 images to work!!!")
    return np.asarray(mat_comb)


def join_image_multiple(list_mat, list_joint, norm=True, total_width=None):
    """
    Join list of projection images or sinogram images. This is useful for
    fixing the problem of non-overlap between images.

    Parameters
    ----------
    list_mat : list of array_like
        List of 2D array. Projection image or sinogram image.
    list_joint : list of tuple of floats
        List of [joint_width, side].
        joint_width : Width of the joint area between two images.
        side : Overlap side between two images.
    norm : bool, optional
        Enable/disable normalization before stitching.
    total_width : int, optional
        Final width of the stitched image.

    Returns
    -------
    array_like
        Stitched image.
    """
    num_mat = len(list_mat)
    if num_mat > 1:
        mat_comb = np.copy(list_mat[0])
        for i in range(1, num_mat):
            (joint_width, side) = list_joint[i - 1][0:2]
            mat_comb = join_image(mat_comb, list_mat[i], joint_width, side,
                                  norm)
        width = mat_comb.shape[1]
        if total_width is None:
            total_width = width
        if total_width > width:
            mat_comb = np.pad(
                mat_comb, ((0, 0), (0, total_width - width)), mode='edge')
    else:
        raise ValueError("Need at least 2 images to work!!!")
    return np.asarray(mat_comb)


def convert_sinogram_360_to_180(sino_360, cor, wei_mat1=None, wei_mat2=None,
                                norm=True, total_width=None):
    """
    Convert a 360-degree sinogram to a 180-degree sinogram.

    Parameters
    ----------
    sino_360 : array_like
        2D array. 360-degree sinogram.
    cor : float or tuple of float
        Center-of-rotation or (Overlap_area, overlap_side).
    wei_mat1 : array_like, optional
        Weighting matrix used for the 1st haft of the sinogram.
    wei_mat2 : array_like, optional
        Weighting matrix used for the 2nd haft of the sinogram.
    norm : bool, optional
        Enable/disable normalization before stitching.
    total_width : int, optional
        Final width of the stitched image.

    Returns
    -------
    sino_stiched : array_like
        Converted sinogram.
    cor : float
        Updated center-of-rotation referred to the converted sinogram.
    """
    (nrow, ncol) = sino_360.shape
    xcenter = (ncol - 1.0) * 0.5
    nrow_180 = nrow // 2 + 1
    sino_top = sino_360[0:nrow_180, :]
    sino_bot = np.fliplr(sino_360[-nrow_180:, :])
    if isinstance(cor, tuple):
        (overlap, side) = cor
    else:
        if cor <= xcenter:
            overlap = 2 * (cor + 1)
            side = 0
        else:
            overlap = 2 * (ncol - cor - 1)
            side = 1
    sino_stitch = stitch_image(
        sino_top, sino_bot, overlap, side, wei_mat1=wei_mat1,
        wei_mat2=wei_mat2, norm=norm, total_width=total_width)
    cor = (2 * ncol - np.floor(overlap) - 1.0) / 2.0
    return sino_stitch, cor


def convert_sinogram_180_to_360(sino_180, center):
    """
    Convert a 180-degree sinogram to a 360-degree sinogram (Ref. [1]_).

    Parameters
    ----------
    sino_180 : array_like
        2D array. 180-degree sinogram.
    center : float
        Center-of-rotation.

    Returns
    -------
    array_like
        360-degree sinogram.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.22.019078
    """
    (nrow, ncol) = sino_180.shape
    xcenter = (ncol - 1.0) / 2.0
    shift_x = xcenter - center
    sino_flip = shift(np.fliplr(shift(sino_180, (0, shift_x), mode='nearest')),
                      (0, -shift_x), mode='nearest')
    return np.vstack((sino_180, sino_flip[1:]))


def extend_sinogram(sino_360, cor, apply_log=True):
    """
    Extend a 360-degree sinogram (with offset center-of-rotation) for
    later reconstruction (Ref. [1]_).

    Parameters
    ----------
    sino_360 : array_like
        2D array. 360-degree sinogram.
    cor : float or tuple of float
        Center-of-rotation or (Overlap_area, overlap_side).
    apply_log : bool, optional
        Apply the logarithm function if True.

    Returns
    -------
    sino_pad : array_like
        Extended sinogram.
    cor : float
        Updated center-of-rotation referred to the converted sinogram.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.418448
    """
    if apply_log is True:
        sino_360 = -np.log(sino_360)
    else:
        sino_360 = np.copy(sino_360)
    (nrow, ncol) = sino_360.shape
    xcenter = (ncol - 1.0) * 0.5
    if isinstance(cor, tuple):
        (overlap, side) = cor
    else:
        if cor <= xcenter:
            overlap = 2 * (cor + 1)
            side = 0
        else:
            overlap = 2 * (ncol - cor - 1)
            side = 1
    overlap_int = int(np.floor(overlap))
    sub_pixel = overlap - overlap_int
    if side == 1:
        if sub_pixel > 0.0:
            sino_360 = shift(sino_360, (0, sub_pixel), mode='nearest')
        wei_list = np.linspace(1.0, 0.0, overlap_int)
        wei_mat = np.tile(wei_list, (nrow, 1))
        sino_360[:, -overlap_int:] = sino_360[:, -overlap_int:] * wei_mat
        pad_wid = ncol - overlap_int
        sino_pad = np.pad(sino_360, ((0, 0), (0, pad_wid)), mode='edge')
    else:
        if sub_pixel > 0.0:
            sino_360 = shift(sino_360, (0, -sub_pixel), mode='nearest')
        wei_list = np.linspace(0.0, 1.0, overlap_int)
        wei_mat = np.tile(wei_list, (nrow, 1))
        sino_360[:, :overlap_int] = sino_360[:, :overlap_int] * wei_mat
        pad_wid = ncol - overlap_int
        sino_pad = np.pad(sino_360, ((0, 0), (pad_wid, 0)), mode='edge')
    cor = (sino_pad.shape[1] - 1.0) / 2.0
    return sino_pad, cor


def generate_sinogram_helical_scan(index, tomo_data, num_proj, pixel_size,
                                   y_start, y_stop, pitch, scan_type="180",
                                   angles=None, flat=None, dark=None,
                                   mask=None, crop=(0, 0, 0, 0)):
    """
    Generate a 180-degree sinogram or a 360-degree sinogram from a helical
    scan dataset which is a hdf/nxs object (Ref. [1]_).

    Parameters
    ----------
    index : int
        Index of the sinogram.
    tomo_data : hdf object.
        3D array.
    num_proj : int
        Number of projections per 180-degree.
    pixel_size : float
        Pixel size. The unit must be the same as y-position.
    y_start : float
        Y-position of the stage at the beginning of the scan.
    y_stop : float
        Y-position of the stage at the end of the scan.
    pitch : float
        The distance which the y-stage is translated in one full rotation.
    scan_type : {"180", "360"}
        One of two options: "180" for generating a 180-degree sinogram or
        "360" for generating a 360-degree sinogram.
    angles : array_like, optional
        1D array. List of angles (degree) corresponding to acquired projections.
    flat : array_like, optional
        Flat-field image used for flat-field correction.
    dark : array_like, optional
        Dark-field image used for flat-field correction.
    mask : array_like, optional
        Used for removing streak artifacts caused by blobs in the flat-field
        image.
    crop : tuple of int, optional
        Used for cropping images.

    Returns
    -------
    sinogram : array_like
        2D array. 180-degree sinogram or 360-degree sinogram.
    list_angle : array_like
        1D array. List of angles corresponding to the generated sinogram.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.418448
    """
    max_index = calc.calculate_maximum_index(y_start, y_stop, pitch,
                                             pixel_size, scan_type)
    (y_s, y_e) = calc.calculate_reconstructable_height(y_start, y_stop,
                                                       pitch, scan_type)
    if index < 0 or index > max_index:
        msg1 = "Requested index {0} is out of available index-range" \
               " [0, {1}]\n".format(index, max_index)
        msg2 = "corresponding to reconstructable heights" \
               " [{0}, {1}]".format(y_s, y_e)
        raise ValueError(msg1 + msg2)
    (depth0, height0, width0) = tomo_data.shape
    (crop_top, crop_bottom, crop_left, crop_right) = crop
    top = crop_top
    bottom = height0 - crop_bottom
    left = crop_left
    right = width0 - crop_right
    width = right - left
    height = bottom - top
    if flat is None:
        flat = np.ones((height0, width0), dtype=np.float32)
    if dark is None:
        dark = np.zeros((height0, width0), dtype=np.float32)
    if angles is None:
        step_angle = 180.0 / (num_proj - 1)
        angles = np.arange(0, depth0) * step_angle
    flat_dark = flat - dark
    FoV = pixel_size * height
    y_step = pitch / (2.0 * (num_proj - 1))
    if scan_type == "180":
        num_proj_used = num_proj
    else:
        num_proj_used = 2 * (num_proj - 1) + 1
    y_pos = (index - 1) * pixel_size + y_s
    i0 = int(np.ceil((y_e - y_pos) / y_step))
    if (i0 < 0) or (i0 >= depth0):
        raise ValueError(
            "Sinogram index {0} requests a projection index {1}"
            " which is out of the data range [0, {2}]".format(
                index, i0, depth0 - 1))
    sinogram = np.zeros((num_proj_used, width), dtype=np.float32)
    for i in range(i0, i0 + num_proj_used):
        j0 = (y_e + FoV - i * y_step - y_pos) / pixel_size - 1
        if (j0 < 0) or (j0 >= height):
            raise ValueError(
                "Requested row index {0} of projection {1} is out of the"
                " range [0, {2}]".format(j0, i0, height - 1))
        j0 = np.clip(j0, 0, height - 1)
        jd = int(np.floor(j0))
        ju = int(np.ceil(j0))
        list_down = (tomo_data[i, jd + crop_top, left: right]
                     - dark[jd + crop_top, left: right]) / flat_dark[
                                                           jd + crop_top,
                                                           left: right]
        if mask is not None:
            list_down = remo.remove_blob_1d(list_down,
                                            mask[jd + crop_top, left: right])
        if ju != jd:
            list_up = (tomo_data[i, ju + crop_top, left: right]
                       - dark[ju + crop_top, left: right]) \
                      / flat_dark[ju + crop_top, left: right]
            if mask is not None:
                list_up = remo.remove_blob_1d(list_up,
                                              mask[ju + crop_top, left: right])
            sinogram[i - i0] = list_down * (ju - j0) / (ju - jd) + list_up * (
                        j0 - jd) / (ju - jd)
        else:
            sinogram[i - i0] = list_down
    list_angle = angles[i0:i0 + num_proj_used]
    return sinogram, list_angle


def generate_full_sinogram_helical_scan(index, tomo_data, num_proj, pixel_size,
                                        y_start, y_stop, pitch, scan_type="180",
                                        angles=None, flat=None, dark=None,
                                        mask=None, crop=(0, 0, 0, 0)):
    """
    Generate a full sinogram from a helical scan dataset which is a hdf/nxs
    object (Ref. [1]_). Full sinogram is all 1D projection of the same slice of
    a sample staying inside the field of view.

    Parameters
    ----------
    index : int
        Index of the sinogram.
    tomo_data : hdf object.
        3D array.
    num_proj : int
        Number of projections per 180-degree.
    pixel_size : float
        Pixel size. The unit must be the same as y-position.
    y_start : float
        Y-position of the stage at the beginning of the scan.
    y_stop : float
        Y-position of the stage at the end of the scan.
    pitch : float
        The distance which the y-stage is translated in one full rotation.
    scan_type : {"180", "360"}
        Data acquired is the 180-degree type or 360-degree type [1].
    angles : array_like, optional
        1D array. List of angles (degree) corresponding to acquired projections.
    flat : array_like, optional
        Flat-field image used for flat-field correction.
    dark : array_like, optional
        Dark-field image used for flat-field correction.
    mask : array_like, optional
        Used for removing streak artifacts caused by blobs in the flat-field
        image.
    crop : tuple of int, optional
        Used for cropping images.

    Returns
    -------
    sinogram : array_like
        2D array. Full sinogram.
    list_angle : array_like
        1D array. List of angles corresponding to the generated sinogram.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.418448
    """
    (depth0, height0, width0) = tomo_data.shape
    (crop_top, crop_bottom, crop_left, crop_right) = crop
    top = crop_top
    bottom = height0 - crop_bottom
    left = crop_left
    right = width0 - crop_right
    width = right - left
    height = bottom - top
    if flat is None:
        flat = np.ones((height0, width0), dtype=np.float32)
    if dark is None:
        dark = np.zeros((height0, width0), dtype=np.float32)
    if angles is None:
        step_angle = 180.0 / (num_proj - 1)
        angles = np.arange(0, depth0) * step_angle
    flat_dark = flat - dark
    FoV = pixel_size * height
    y_step = pitch / (2.0 * (num_proj - 1))
    if scan_type == "180":
        y_e = y_stop - pitch / 2.0
        y_s = y_start + pitch / 2.0
    else:
        y_e = y_stop - pitch
        y_s = y_start + pitch
    num_proj_used = int(np.floor(FoV / y_step)) - 1
    y_pos = (index - 1) * pixel_size + y_s
    i0 = int(np.ceil((y_e - y_pos) / y_step))
    if (i0 < 0) or (i0 >= depth0):
        raise ValueError(
            "Sinogram index {0} requests a projection index {1} which "
            "is out of the projection range [0, {2}]".format(
                index, i0, depth0 - 1))
    if (i0 + num_proj_used) >= depth0:
        raise ValueError(
            "Sinogram index {0} requests projection-indices in the range of "
            "[{1}, {2}] which is out of the data range [0, {3}]".format(
                index, i0, i0 + num_proj_used, depth0 - 1))
    sinogram = np.zeros((num_proj_used, width), dtype=np.float32)
    for i in range(i0, i0 + num_proj_used):
        j0 = (y_e + FoV - i * y_step - y_pos) / pixel_size - 1
        if (j0 < 0) or (j0 >= height):
            raise ValueError(
                "Requested row index {0} of projection {1} is out of"
                " the range [0, {2}]".format(j0, i0, height))
        j0 = np.clip(j0, 0, height - 1)
        jd = int(np.floor(j0))
        ju = int(np.ceil(j0))
        list_down = (tomo_data[i, jd + crop_top, left: right]
                     - dark[jd + crop_top, left: right]) / flat_dark[
                                                           jd + crop_top,
                                                           left: right]
        if mask is not None:
            list_down = remo.remove_blob_1d(list_down,
                                            mask[jd + crop_top, left: right])
        if ju != jd:
            list_up = (tomo_data[i, ju + crop_top, left: right]
                       - dark[ju + crop_top, left: right]) / flat_dark[
                                                             ju + crop_top,
                                                             left: right]
            if mask is not None:
                list_up = remo.remove_blob_1d(list_up,
                                              mask[ju + crop_top, left: right])
            sinogram[i - i0] = list_down * (ju - j0) / (ju - jd) + list_up * (
                        j0 - jd) / (ju - jd)
        else:
            sinogram[i - i0] = list_down
    list_angle = angles[i0:i0 + num_proj_used]
    return sinogram, list_angle
