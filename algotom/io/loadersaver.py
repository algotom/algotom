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
# Description: Python module for loading and saving data.
# Contributors:
# ============================================================================

"""
Module for I/O tasks:

    -   Load data from an image file (tif, png, jpeg) or a hdf/nxs file.
    -   Get information from a hdf/nxs file.
    -   Search for datasets in a hdf/nxs file.
    -   Save a 2D array as a tif image or 2D, 3D array to a hdf/nxs file.
    -   Get file names, make file/folder name.
    -   Load distortion coefficients from a txt file.
    -   Get the tree view of a hdf/nxs file.
    -   Functions for loading stacks of images from multiple datasets, e.g. to
        be used by speckle-based phase contrast tomography.
"""

import os
import glob
import warnings
from collections import OrderedDict, deque
import h5py
import numpy as np
from PIL import Image


PIPE = "│"
ELBOW = "└──"
TEE = "├──"
PIPE_PREFIX = "│   "
SPACE_PREFIX = "    "


def load_image(file_path):
    """
    Load data from an image.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    float
        2D array.
    """
    if "\\" in file_path:
        raise ValueError("Please use the forward slash in the file path")
    try:
        mat = np.asarray(Image.open(file_path), dtype=np.float32)
    except IOError:
        raise ValueError("No such file or directory: {}".format(file_path))
    if len(mat.shape) > 2:
        axis_m = np.argmin(mat.shape)
        mat = np.mean(mat, axis=axis_m)
    return mat


def get_hdf_information(file_path, display=False):
    """
    Get information of datasets in a hdf/nxs file.

    Parameters
    ----------
    file_path : str
        Path to the file.
    display : bool
        Print the results onto the screen if True.

    Returns
    -------
    list_key : str
        Keys to the datasets.
    list_shape : tuple of int
        Shapes of the datasets.
    list_type : str
        Types of the datasets.
    """
    hdf_object = h5py.File(file_path, 'r')
    keys = []
    hdf_object.visit(keys.append)
    list_key, list_shape, list_type = [], [], []
    for key in keys:
        try:
            data = hdf_object[key]
            if isinstance(data, h5py.Group):
                list_tmp = list(data.items())
                if list_tmp:
                    for key2, _ in list_tmp:
                        list_key.append(key + "/" + key2)
                else:
                    list_key.append(key)
            else:
                list_key.append(data.name)
        except KeyError:
            list_key.append(key)
            pass
    for i, key in enumerate(list_key):
        shape, dtype = None, None
        try:
            data = hdf_object[list_key[i]]
            if isinstance(data, h5py.Dataset):
                shape, dtype = data.shape, data.dtype
            list_shape.append(shape)
            list_type.append(dtype)
        except KeyError:
            list_shape.append(shape)
            list_type.append(dtype)
            pass
    hdf_object.close()
    if display:
        if list_key:
            for i, key in enumerate(list_key):
                print(key + " : " + str(list_shape[i]) + " : " + str(
                    list_type[i]))
        else:
            print("Empty file !!!")
    return list_key, list_shape, list_type


def find_hdf_key(file_path, pattern, display=False):
    """
    Find datasets matching the name-pattern in a hdf/nxs file.

    Parameters
    ----------
    file_path : str
        Path to the file.
    pattern : str
        Pattern to find the full names of the datasets.
    display : bool
        Print the results onto the screen if True.

    Returns
    -------
    list_key : str
        Keys to the datasets.
    list_shape : tuple of int
        Shapes of the datasets.
    list_type : str
        Types of the datasets.
    """
    hdf_object = h5py.File(file_path, 'r')
    list_key, keys = [], []
    hdf_object.visit(keys.append)
    for key in keys:
        try:
            data = hdf_object[key]
            if isinstance(data, h5py.Group):
                list_tmp = list(data.items())
                if list_tmp:
                    for key2, _ in list_tmp:
                        list_key.append(key + "/" + key2)
                else:
                    list_key.append(key)
            else:
                list_key.append(data.name)
        except KeyError:
            pass
    list_dkey, list_dshape, list_dtype = [], [], []
    for _, key in enumerate(list_key):
        if pattern in key:
            list_dkey.append(key)
            shape, dtype = None, None
            try:
                data = hdf_object[key]
                if isinstance(data, h5py.Dataset):
                    shape, dtype = data.shape, data.dtype
                list_dtype.append(dtype)
                list_dshape.append(shape)
            except KeyError:
                list_dtype.append(dtype)
                list_dshape.append(shape)
                pass
    hdf_object.close()
    if display:
        if list_dkey:
            for i, key in enumerate(list_dkey):
                print(key + " : " + str(list_dshape[i]) + " : " + str(
                    list_dtype[i]))
        else:
            print("Can't find datasets with keys matching the "
                  "pattern: {}".format(pattern))
    return list_dkey, list_dshape, list_dtype


def load_hdf(file_path, key_path):
    """
    Load a hdf/nexus dataset as an object.

    Parameters
    ----------
    file_path : str
        Path to the file.
    key_path : str
        Key path to the dataset.

    Returns
    -------
    object
        hdf/nxs object.
    """
    try:
        hdf_object = h5py.File(file_path, 'r')
    except IOError:
        raise ValueError("Couldn't open file: {}".format(file_path))
    check = key_path in hdf_object
    if not check:
        raise ValueError(
            "Couldn't open object with the given key: {}".format(key_path))
    return hdf_object[key_path]


def make_folder(file_path):
    """
    Create a folder for saving file if the folder does not exist. This is a
    supplementary function for savers.

    Parameters
    ----------
    file_path : str
        Path to a file.
    """
    file_base = os.path.dirname(file_path)
    if not os.path.exists(file_base):
        try:
            os.makedirs(file_base)
        except OSError:
            raise ValueError("Can't create the folder: {}".format(file_base))


def make_file_name(file_path):
    """
    Create a new file name to avoid overwriting.

    Parameters
    ----------
    file_path : str

    Returns
    -------
    str
        Updated file path.
    """
    file_base, file_ext = os.path.splitext(file_path)
    if os.path.isfile(file_path):
        nfile = 0
        check = True
        while check:
            name_add = '0000' + str(nfile)
            file_path = file_base + "_" + name_add[-4:] + file_ext
            if os.path.isfile(file_path):
                nfile = nfile + 1
            else:
                check = False
    return file_path


def make_folder_name(folder_path, name_prefix="Output", zero_prefix=5):
    """
    Create a new folder name to avoid overwriting.
    E.g: Output_00001, Output_00002...

    Parameters
    ----------
    folder_path : str
        Path to the parent folder.
    name_prefix : str
        Name prefix
    zero_prefix : int
        Number of zeros to be added to file names.
    Returns
    -------
    str
        Name of the folder.
    """
    scan_name_prefix = name_prefix + "_"
    num_folder_exist = len(
        glob.glob(folder_path + "/" + scan_name_prefix + "*"))
    num_folder_new = num_folder_exist + 1
    name_tmp = "00000" + str(num_folder_new)
    scan_name = scan_name_prefix + name_tmp[-zero_prefix:]
    while os.path.isdir(folder_path + "/" + scan_name):
        num_folder_new = num_folder_new + 1
        name_tmp = "00000" + str(num_folder_new)
        scan_name = scan_name_prefix + name_tmp[-zero_prefix:]
    return scan_name


def find_file(path):
    """
    Search file

    Parameters
    ----------
    path : str
        Path and pattern to find files.

    Returns
    -------
    str or list of str
        List of files.
    """
    file_path = glob.glob(path)
    if len(file_path) == 0:
        raise ValueError("!!! No files found in: {}".format(path))
    for i in range(len(file_path)):
        file_path[i] = file_path[i].replace("\\", "/")
    return sorted(file_path)


def save_image(file_path, mat, overwrite=True):
    """
    Save a 2D array to an image.

    Parameters
    ----------
    file_path : str
        Path to the file.
    mat : int or float
        2D array.
    overwrite : bool
        Overwrite an existing file if True.

    Returns
    -------
    str
        Updated file path.
    """
    if "\\" in file_path:
        raise ValueError("Please use the forward slash in the file path")
    file_ext = os.path.splitext(file_path)[-1]
    if not ((file_ext == ".tif") or (file_ext == ".tiff")):
        mat = np.uint8(255 * (mat - np.min(mat)) / (np.max(mat) - np.min(mat)))
    else:
        data_type = str(mat.dtype)
        if not (data_type == "uint8" or data_type == "uint16"
                or data_type == "float32"):
            raise ValueError("Can't save to tiff with this "
                             "format: {}".format(data_type))
    make_folder(file_path)
    if not overwrite:
        file_path = make_file_name(file_path)
    image = Image.fromarray(mat)
    try:
        image.save(file_path)
    except IOError:
        raise ValueError("Couldn't write to file {}".format(file_path))
    return file_path


def open_hdf_stream(file_path, data_shape, key_path='entry/data',
                    data_type='float32', overwrite=True, **options):
    """
    Write an array to a hdf/nxs file with options to add metadata.

    Parameters
    ----------
    file_path : str
        Path to the file.
    data_shape : tuple of int
        Shape of the data.
    key_path : str
        Key path to the dataset.
    data_type: str
        Type of data.
    overwrite : bool
        Overwrite the existing file if True.
    options : dict, optional
        Add metadata. E.g options={"entry/angles": angles, "entry/energy": 53}.

    Returns
    -------
    object
        hdf object.
    """
    file_base, file_ext = os.path.splitext(file_path)
    if not (file_ext == '.hdf' or file_ext == '.h5' or file_ext == ".nxs"):
        file_ext = '.hdf'
    file_path = file_base + file_ext
    make_folder(file_path)
    if not overwrite:
        file_path = make_file_name(file_path)
    try:
        ofile = h5py.File(file_path, 'w')
    except IOError:
        raise ValueError("Couldn't write to file: {}".format(file_path))
    if len(options) != 0:
        for opt_name in options:
            opts = options[opt_name]
            for key in opts:
                if key_path in key:
                    msg = "!!!Selected key-path, '{0}', can not be a child " \
                          "key-path of '{1}'!!!\n!!!Change to make sure " \
                          "they are at the same level!!!".format(key, key_path)
                    raise ValueError(msg)
                ofile.create_dataset(key, data=opts[key])
    data_out = ofile.create_dataset(key_path, data_shape, dtype=data_type)
    return data_out


def load_distortion_coefficient(file_path):
    """
    Load distortion coefficients from a text file. The file must use the
    following format:
    x_center : float
    y_center : float
    factor0 : float
    factor1 : float
    ...

    Parameters
    ----------
    file_path : str
        Path to the file

    Returns
    -------
    tuple of float and list
        Tuple of (xcenter, ycenter, list_fact).
    """
    if "\\" in file_path:
        raise ValueError("Please use the forward slash in the file path")
    with open(file_path, 'r') as f:
        x = f.read().splitlines()
        list_data = []
        for i in x:
            list_data.append(float(i.split()[-1]))
    xcenter = list_data[0]
    ycenter = list_data[1]
    list_fact = list_data[2:]
    return xcenter, ycenter, list_fact


def save_distortion_coefficient(file_path, xcenter, ycenter, list_fact,
                                overwrite=True):
    """
    Write distortion coefficients to a text file.

    Parameters
    ----------
    file_path : str
        Path to the file.
    xcenter : float
        Center of distortion in x-direction.
    ycenter : float
        Center of distortion in y-direction.
    list_fact : float
        1D array. Coefficients of the polynomial fit.
    overwrite : bool
        Overwrite an existing file if True.

    Returns
    -------
    str
        Updated file path.
    """
    file_base, file_ext = os.path.splitext(file_path)
    if not ((file_ext == '.txt') or (file_ext == '.dat')):
        file_ext = '.txt'
    file_path = file_base + file_ext
    make_folder(file_path)
    if not overwrite:
        file_path = make_file_name(file_path)
    metadata = OrderedDict()
    metadata['xcenter'] = xcenter
    metadata['ycenter'] = ycenter
    for i, fact in enumerate(list_fact):
        kname = 'factor' + str(i)
        metadata[kname] = fact
    with open(file_path, "w") as f:
        for line in metadata:
            f.write(str(line) + " = " + str(metadata[line]))
            f.write('\n')
    return file_path


def _get_subgroups(hdf_object, key=None):
    """
    Supplementary method for building the tree view of a hdf5 file.
    Return the name of subgroups.
    """
    list_group = []
    if key is None:
        for group in hdf_object.keys():
            list_group.append(group)
        if len(list_group) == 1:
            key = list_group[0]
        else:
            key = ""
    else:
        if key in hdf_object:
            try:
                obj = hdf_object[key]
                if isinstance(obj, h5py.Group):
                    for group in hdf_object[key].keys():
                        list_group.append(group)
            except KeyError:
                pass
    if len(list_group) > 0:
        list_group = sorted(list_group)
    return list_group, key


def _add_branches(tree, hdf_object, key, key1, index, last_index, prefix,
                  connector, level, add_shape):
    """
    Supplementary method for building the tree view of a hdf5 file.
    Add branches to the tree.
    """
    shape = None
    key_comb = key + "/" + key1
    if add_shape is True:
        if key_comb in hdf_object:
            try:
                obj = hdf_object[key_comb]
                if isinstance(obj, h5py.Dataset):
                    shape = str(obj.shape)
            except KeyError:
                shape = str("-> ???External-link???")
    if shape is not None:
        tree.append(f"{prefix}{connector} {key1} {shape}")
    else:
        tree.append(f"{prefix}{connector} {key1}")
    if index != last_index:
        prefix += PIPE_PREFIX
    else:
        prefix += SPACE_PREFIX
    _make_tree_body(tree, hdf_object, prefix=prefix, key=key_comb,
                    level=level, add_shape=add_shape)


def _make_tree_body(tree, hdf_object, prefix="", key=None, level=0,
                    add_shape=True):
    """
    Supplementary method for building the tree view of a hdf5 file.
    Create the tree body.
    """
    entries, key = _get_subgroups(hdf_object, key)
    num_ent = len(entries)
    last_index = num_ent - 1
    level = level + 1
    if num_ent > 0:
        if last_index == 0:
            key = "" if level == 1 else key
            if num_ent > 1:
                connector = PIPE
            else:
                connector = ELBOW if level > 1 else ""
            _add_branches(tree, hdf_object, key, entries[0], 0, 0, prefix,
                          connector, level, add_shape)
        else:
            for index, key1 in enumerate(entries):
                connector = ELBOW if index == last_index else TEE
                if index == 0:
                    tree.append(prefix + PIPE)
                _add_branches(tree, hdf_object, key, key1, index, last_index,
                              prefix, connector, level, add_shape)


def get_hdf_tree(file_path, output=None, add_shape=True, display=True):
    """
    Get the tree view of a hdf/nxs file.

    Parameters
    ----------
    file_path : str
        Path to the file.
    output : str or None
        Path to the output file in a text-format file (.txt, .md,...).
    add_shape : bool
        Including the shape of a dataset to the tree if True.
    display : bool
        Print the tree onto the screen if True.

    Returns
    -------
    list of string
    """
    hdf_object = h5py.File(file_path, 'r')
    tree = deque()
    _make_tree_body(tree, hdf_object, add_shape=add_shape)
    if output is not None:
        make_folder(output)
        output_file = open(output, mode="w", encoding="UTF-8")
        with output_file as stream:
            for entry in tree:
                print(entry, file=stream)
    else:
        if display:
            for entry in tree:
                print(entry)
    return tree


def __get_ref_sam_stacks_dls(proj_idx, list_data_obj, list_sam_idx,
                             list_ref_idx, list_dark_idx, top, bot, left,
                             right, height, width, flat_field, dark_field,
                             fix_zero_div):
    """
    Supplementary method for the method of "get_reference_sample_stacks_dls"
    """
    ref_stack = []
    sam_stack = []
    num_img = len(list_data_obj)
    height1 = bot - top
    width1 = right - left
    if flat_field is not None:
        if flat_field.shape != (height, width):
            raise ValueError("Shape of flat-field image is not "
                             "the same as projection image "
                             "({0}, {1})".format(height, width))
        else:
            flat_ave = flat_field[top: bot, left:right]
    else:
        flat_ave = np.ones((height1, width1), dtype=np.float32)
    if dark_field is not None:
        if dark_field.shape != (height, width):
            raise ValueError("Shape of dark-field image is not "
                             "the same as projection image "
                             "({0}, {1})".format(height, width))
        else:
            dark_ave = dark_field[top: bot, left:right]
    for i in range(num_img):
        if dark_field is None:
            if len(list_dark_idx) != 0:
                idx1 = list_dark_idx[i][0]
                idx2 = list_dark_idx[i][-1] + 1
                dark_ave = np.mean(
                    list_data_obj[i][idx1:idx2, top:bot, left:right], axis=0)
            else:
                dark_ave = np.zeros((height1, width1), dtype=np.float32)
        if flat_field is not None:
            flat_dark = flat_ave - dark_ave
            nmean = np.mean(flat_dark)
            flat_dark[flat_dark == 0.0] = nmean
        if len(list_ref_idx) != 0:
            idx1 = list_ref_idx[i][0]
            idx2 = list_ref_idx[i][-1] + 1
            ref_ave = np.mean(list_data_obj[i][idx1:idx2, top:bot, left:right],
                              axis=0)
            if flat_field is not None:
                ref_ave = (ref_ave - dark_ave) / flat_dark
            else:
                ref_ave = ref_ave - dark_ave
            ref_stack.append(ref_ave)
        idx = list_sam_idx[i][proj_idx]
        proj = list_data_obj[i][idx, top:bot, left:right]
        if flat_field is not None:
            proj = (proj - dark_ave) / flat_dark
        else:
            proj = (proj - dark_ave)
        sam_stack.append(proj)
    sam_stack = np.asarray(sam_stack)
    if fix_zero_div:
        nmean = np.mean(sam_stack)
        sam_stack[sam_stack == 0.0] = nmean
    if ref_stack:
        ref_stack = np.asarray(ref_stack)
        if fix_zero_div:
            nmean = np.mean(ref_stack)
            ref_stack[ref_stack == 0.0] = nmean
    return ref_stack, sam_stack


def get_reference_sample_stacks_dls(proj_idx, list_path, data_key=None,
                                    image_key=None, crop=(0, 0, 0, 0),
                                    flat_field=None, dark_field=None,
                                    num_use=None, fix_zero_div=True):
    """
    A method for multi-position speckle-based phase-contrast tomography to get
    two stacks of reference images (speckle images) and sample images (at the
    same rotation angle from each tomographic dataset).

    The method is specific to tomographic datasets acquired at Diamond Light
    Source (DLS) where projection-images, flat-field images, and dark-field
    images are in the same 3d array. There is a dataset named "image_key"
    inside a hdf/nxs file used to distinguish image types.

    Parameters
    ----------
    proj_idx : int
        Index of a projection-image in a tomographic dataset.
    list_path : list of str
        List of file paths (hdf/nxs format) to tomographic datasets.
    data_key : str, optional
        Key to images. Automatically find the key if None.
    image_key : str, list, tuple, ndarray, optional
        Key to 1d-array dataset for specifying image types. Automatically
        find the key if None. Can be used to pass the 1d-array manually.
    crop : tuple of int, optional
        Crop the images from the edges, i.e.
        crop = (crop_top, crop_bottom, crop_left, crop_right).
    flat_field : ndarray, optional
        2D array or None. Used for flat-field correction if not None.
    dark_field : ndarray, optional
        2D array or None. Used for dark-field correction if not None.
    num_use : int, optional
        Number of datasets used for stacking.
    fix_zero_div : bool, optional
        Correct zeros to avoid zero-division problem down the processing line.

    Returns
    -------
    ref_stack : ndarray
        Return if reference-images found. 3D array.
    sam_stack : ndarray
        3D array. A stack of sample-images.
    """
    if not isinstance(list_path, list):
        raise ValueError("Input must be a list of strings!!!")
    num_file = len(list_path)
    if num_use is None:
        num_use = num_file
    else:
        num_use = np.clip(num_use, 1, num_file)
    if data_key is None:
        data_key = find_hdf_key(list_path[0], "data/data")[0]
        if len(data_key) != 0:
            data_key = data_key[0]
        else:
            raise ValueError("Please provide the key to dataset!!!")
    if image_key is None:
        image_key = find_hdf_key(list_path[0], "image_key")[0]
        if len(image_key) != 0:
            image_key = image_key[0]
        else:
            image_key = None
            warnings.warn("No image-key found!!!. Output will be a single "
                          "stack")
    (height, width) = load_hdf(list_path[0], data_key).shape[-2:]
    cr_top, cr_bot, cr_left, cr_right = crop
    top = cr_top
    bot = height - cr_bot
    left = cr_left
    right = width - cr_right
    height1 = bot - top
    width1 = right - left
    if height1 < 1 or width1 < 1:
        raise ValueError("Can't crop data with the given input!!!")
    list_data_obj = []
    list_sam_idx = []
    list_ref_idx = []
    list_dark_idx = []
    list_num_proj = []
    list_start_idx = []
    list_stop_idx = []
    for path in list_path[:num_use]:
        data_obj = load_hdf(path, data_key)
        num_img = len(data_obj)
        list_data_obj.append(data_obj)
        if image_key is not None:
            if isinstance(image_key, str):
                int_keys = load_hdf(path, image_key)[:]
            else:
                if not (isinstance(image_key, list) or
                        isinstance(image_key, tuple) or
                        isinstance(image_key, np.ndarray)):
                    raise ValueError("Input must be a string, list, tuple, or "
                                     "1D numpy array!!!")
                else:
                    int_keys = np.asarray(image_key, dtype=np.float32)
                    if len(int_keys) != num_img:
                        raise ValueError("Number of image-keys is not the same"
                                         " as the number of images {0}!!!"
                                         "".format(num_img))
            list_tmp = np.where(int_keys == 0.0)[0]
            if len(list_tmp) != 0:
                list_idx = np.sort(np.int32(np.squeeze(np.asarray(list_tmp))))
                list_sam_idx.append(list_idx)
                list_start_idx.append(list_idx[0])
                list_stop_idx.append(list_idx[-1])
            list_num_proj.append(len(list_tmp))
            list_tmp = np.where(int_keys == 1.0)[0]
            if len(list_tmp) != 0:
                list_ref_idx.append(
                    np.sort(np.int32(np.squeeze(np.asarray(list_tmp)))))
            list_tmp = np.where(int_keys == 2.0)[0]
            if len(list_tmp) != 0:
                list_dark_idx.append(
                    np.sort(np.int32(np.squeeze(np.asarray(list_tmp)))))
        else:
            num_proj = num_img
            list_sam_idx.append(np.arange(num_proj))
            list_start_idx.append(0)
            list_stop_idx.append(num_proj - 1)
            list_num_proj.append(num_proj)
    num_proj = np.min(np.asarray(list_num_proj))
    start_idx = np.max(np.asarray(list_start_idx))
    stop_idx = np.min(np.asarray(list_stop_idx))
    if (stop_idx - start_idx + 1) > num_proj:
        stop_idx = start_idx + num_proj - 1
    idx_off = proj_idx + start_idx
    if idx_off > stop_idx or idx_off < start_idx:
        raise ValueError("Requested projection-index is out of the range"
                         " [{0}, {1}] given the offset of "
                         "{2}".format(start_idx, stop_idx, start_idx))
    else:
        f_alias = __get_ref_sam_stacks_dls
        ref_stack, sam_stack = f_alias(proj_idx, list_data_obj, list_sam_idx,
                                       list_ref_idx, list_dark_idx, top, bot,
                                       left, right, height, width, flat_field,
                                       dark_field, fix_zero_div)
        if len(ref_stack) != 0:
            return ref_stack, sam_stack
        else:
            return sam_stack


def __check_dark_flat_field(flat_field, dark_field, height, width):
    """
    Supplementary method for checking dark-field image, flat-field image.
    """
    if flat_field is not None:
        if len(flat_field) == 3:
            flat_field = np.mean(flat_field, axis=0)
        (height2, width2) = flat_field.shape
        if height2 != height or width2 != width:
            raise ValueError("Shape of flat-field image is not "
                             "the same as projection image")
    else:
        flat_field = np.ones((height, width))
    if dark_field is not None:
        if len(dark_field) == 3:
            dark_field = np.mean(dark_field, axis=0)
        (height2, width2) = dark_field.shape
        if height2 != height or width2 != width:
            raise ValueError("Shape of dark-field image is not "
                             "the same as projection image")
    else:
        dark_field = np.zeros((height, width))
    return flat_field, dark_field


def get_reference_sample_stacks(proj_idx, ref_path, sam_path, ref_key, sam_key,
                                crop=(0, 0, 0, 0), flat_field=None,
                                dark_field=None, num_use=None,
                                fix_zero_div=True):
    """
    Get two stacks of reference images (speckle images) and sample images (at
    the same rotation angle from each tomographic dataset). A method for
    multi-position speckle-based phase-contrast tomography.

    Parameters
    ----------
    proj_idx : int
        Index of a projection-image in a tomographic dataset.
    ref_path : list of str
        List of file paths (hdf/nxs format) to reference-image datasets.
    sam_path : list of str
        List of file paths (hdf/nxs format) to tomographic datasets.
    ref_key : str
        Key to a reference-image dataset.
    sam_key : str
        Key to a projection-image dataset.
    crop : tuple of int, optional
        Crop the images from the edges, i.e.
        crop = (crop_top, crop_bottom, crop_left, crop_right).
    flat_field : ndarray, optional
        2D array or None. Used for flat-field correction if not None.
    dark_field : ndarray, optional
        2D array or None. Used for dark-field correction if not None.
    num_use : int, optional
        Number of datasets used for stacking.
    fix_zero_div : bool, optional
        Correct zeros to avoid zero-division problem down the processing line.

    Returns
    -------
    ref_stack : ndarray
        3D array. A stack of reference-images.
    sam_stack : ndarray
        3D array. A stack of sample-images.
    """
    if not isinstance(ref_path, list):
        raise ValueError("Input-path must be a list of strings!!!")
    if len(ref_path) != len(sam_path):
        raise ValueError("Number of inputs must be the same!!!")
    num_file = len(ref_path)
    if num_use is None:
        num_use = num_file
    else:
        num_use = np.clip(num_use, 1, num_file)
    (height, width) = load_hdf(ref_path[0], ref_key).shape[-2:]
    cr_top, cr_bot, cr_left, cr_right = crop
    top = cr_top
    bot = height - cr_bot
    left = cr_left
    right = width - cr_right
    height1 = bot - top
    width1 = right - left
    if height1 < 1 or width1 < 1:
        raise ValueError("Can't crop data with the given input!!!")
    fix_zeros = False if flat_field is None else True
    flat_field, dark_field = __check_dark_flat_field(flat_field, dark_field,
                                                     height, width)
    flat_field = flat_field[top:bot, left:right]
    dark_field = dark_field[top:bot, left:right]
    flat_dark = flat_field - dark_field
    if fix_zeros:
        nmean = np.mean(flat_dark)
        flat_dark[flat_dark == 0.0] = nmean
    ref_objs = []
    sam_objs = []
    for i in range(num_use):
        ref_objs.append(load_hdf(ref_path[i], ref_key))
        sam_objs.append(load_hdf(sam_path[i], sam_key))
    ref_stack = []
    sam_stack = []
    for i in range(num_use):
        if len(ref_objs[i].shape) == 3:
            ref_ave = np.mean(ref_objs[i][:, top:bot, left:right], axis=0)
        else:
            ref_ave = ref_objs[i][top:bot, left:right]
        proj = sam_objs[i][proj_idx, top:bot, left:right]
        if fix_zeros:
            ref_ave = (ref_ave - dark_field) / flat_dark
            proj = (proj - dark_field) / flat_dark
        else:
            ref_ave = ref_ave - dark_field
            proj = (proj - dark_field)
        ref_stack.append(ref_ave)
        sam_stack.append(proj)
    ref_stack = np.asarray(ref_stack)
    sam_stack = np.asarray(sam_stack)
    if fix_zero_div:
        nmean = np.mean(ref_stack)
        ref_stack[ref_stack == 0.0] = nmean
        nmean = np.mean(sam_stack)
        sam_stack[sam_stack == 0.0] = nmean
    return ref_stack, sam_stack


def get_tif_stack(file_base, idx=None, crop=(0, 0, 0, 0), flat_field=None,
                  dark_field=None, num_use=None, fix_zero_div=True):
    """
    Load tif images to a stack.

    Parameters
    ----------
    file_base : str
        Folder path to tif images.
    idx : int or None
        Load single or multiple images.
    crop : tuple of int, optional
        Crop the images from the edges, i.e.
        crop = (crop_top, crop_bottom, crop_left, crop_right).
    flat_field : ndarray, optional
        2D array or None. Used for flat-field correction if not None.
    dark_field : ndarray, optional
        2D array or None. Used for dark-field correction if not None.
    num_use : int, optional
        Number of images used for stacking.
    fix_zero_div : bool, optional
        Correct zeros to avoid zero-division problem down the processing line.

    Returns
    -------
    img_stack : ndarray
        3D array. A stack of images.
    """
    list_file = find_file(file_base + "/*tif*")
    num_file = len(list_file)
    if num_file != 0:
        (height, width) = np.shape(load_image(list_file[0]))
    else:
        raise ValueError("No tif-images in: {}".format(file_base))
    if idx is not None:
        if idx < 0:
            idx = num_file + idx
        if idx > (num_file - 1):
            raise ValueError("Requested index: {0} is out of "
                             "the range: {1}".format(idx, num_file - 1))
    if num_use is None:
        num_use = num_file
    else:
        num_use = np.clip(num_use, 1, num_file)
    cr_top, cr_bot, cr_left, cr_right = crop
    top = cr_top
    bot = height - cr_bot
    left = cr_left
    right = width - cr_right
    height1 = bot - top
    width1 = right - left
    if height1 < 1 or width1 < 1:
        raise ValueError("Can't crop data with the given input!!!")
    fix_zeros = False if flat_field is None else True
    flat_field, dark_field = __check_dark_flat_field(flat_field, dark_field,
                                                     height, width)
    flat_field = flat_field[top:bot, left:right]
    dark_field = dark_field[top:bot, left:right]
    flat_dark = flat_field - dark_field
    if fix_zeros:
        nmean = np.mean(flat_dark)
        flat_dark[flat_dark == 0.0] = nmean
    if idx is not None:
        img_stack = load_image(list_file[idx])[top:bot, left:right]
        if fix_zeros:
            img_stack = (img_stack - dark_field) / flat_dark
        else:
            img_stack = img_stack - dark_field
        img_stack = [img_stack]
    else:
        img_stack = []
        for file in list_file[:num_use]:
            img = load_image(file)[top:bot, left:right]
            if fix_zeros:
                img = (img - dark_field) / flat_dark
            else:
                img = img - dark_field
            img_stack.append(img)
    img_stack = np.asarray(img_stack)
    if fix_zero_div:
        nmean = np.mean(img_stack)
        img_stack[img_stack == 0.0] = nmean
    return img_stack


def get_image_stack(idx, list_path, data_key=None, average=False,
                    crop=(0, 0, 0, 0), flat_field=None, dark_field=None,
                    num_use=None, fix_zero_div=True):
    """
    Stack images having the same index from multiple datasets. For tif images,
    if only one dataset is provided (list_path is a string, not a list),
    there is an option, idx = None, to load the whole stack.

    Parameters
    ----------
    idx : int or None
        Index of an image in a dataset. Use None to load all images if only
        one dataset provided.
    list_path : list of str
        List of hdf/nxs file-paths or folders of tif-images to datasets.
    data_key : str
        Requested if input is a hdf/nxs files.
    average : bool, optional
        Average images in a dataset if True.
    crop : tuple of int, optional
        Crop the images from the edges, i.e.
        crop = (crop_top, crop_bottom, crop_left, crop_right).
    flat_field : ndarray, optional
        2D array or None. Used for flat-field correction if not None.
    dark_field : ndarray, optional
        2D array or None. Used for dark-field correction if not None.
    num_use : int, optional
        Number of datasets used for stacking.
    fix_zero_div : bool, optional
        Correct zeros to avoid zero-division problem down the processing line.

    Returns
    -------
    img_stack : ndarray
        3D array. A stack of images.
    """
    if isinstance(list_path, list):
        num_file = len(list_path)
        if num_use is None:
            num_use = num_file
        else:
            num_use = np.clip(num_use, 1, num_file)
        tif_format = False
        file_base, file_ext = os.path.splitext(list_path[0])
        if file_ext == "":
            tif_format = True
        else:
            if data_key is None:
                raise ValueError(
                    "Please provide the key to a dataset in the hdf/nxs file")
        if tif_format:
            list_file = find_file(file_base + "/*tif*")
            if len(list_file) != 0:
                (height, width) = np.shape(load_image(list_file[0]))
            else:
                raise ValueError("No tif-images in: {}".format(file_base))
        else:
            (height, width) = load_hdf(list_path[0], data_key).shape[-2:]
        cr_top, cr_bot, cr_left, cr_right = crop
        top = cr_top
        bot = height - cr_bot
        left = cr_left
        right = width - cr_right
        height1 = bot - top
        width1 = right - left
        if height1 < 1 or width1 < 1:
            raise ValueError("Can't crop data with the given input!!!")
        fix_zeros = False if flat_field is None else True
        flat_field, dark_field = __check_dark_flat_field(flat_field,
                                                         dark_field,
                                                         height, width)
        flat_field = flat_field[top:bot, left:right]
        dark_field = dark_field[top:bot, left:right]
        flat_dark = flat_field - dark_field
        if fix_zeros:
            nmean = np.mean(flat_dark)
            flat_dark[flat_dark == 0.0] = nmean
        img_stack = []
        if not tif_format:
            for i in range(num_use):
                data_obj = load_hdf(list_path[i], data_key)
                if len(data_obj.shape) == 3:
                    if average:
                        img = np.mean(data_obj[:, top:bot, left:right], axis=0)
                    else:
                        img = data_obj[idx, top:bot, left:right]
                else:
                    img = data_obj[top:bot, left:right]
                if fix_zeros:
                    img = (img - dark_field) / flat_dark
                else:
                    img = img - dark_field
                img_stack.append(img)
        else:
            for i in range(num_use):
                list_file = find_file(list_path[i] + "/*tif*")
                if average:
                    img = np.mean(
                        np.asarray([load_image(file)[top:bot, left:right]
                                    for file in list_file]), axis=0)
                else:
                    img = load_image(list_file[idx])[top:bot, left:right]
                if fix_zeros:
                    img = (img - dark_field) / flat_dark
                else:
                    img = img - dark_field
                img_stack.append(img)
        img_stack = np.asarray(img_stack)
        if fix_zero_div:
            nmean = np.mean(img_stack)
            img_stack[img_stack == 0.0] = nmean
    else:
        img_stack = get_tif_stack(list_path, idx=idx, crop=crop,
                                  flat_field=flat_field,
                                  dark_field=dark_field, num_use=num_use,
                                  fix_zero_div=fix_zero_div)
    return img_stack
