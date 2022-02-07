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
# Description: Python codes for loading and saving data.
# Contributors:
# ============================================================================

"""
Module for I/O tasks:
    - Load data from an image file (tif, png, jpeg) or a hdf/nxs file.
    - Get dataset information in a hdf/nxs file.
    - Search for datasets in a hdf/nxs file.
    - Save a 2D array as a tif image or 2D, 3D array to a hdf/nxs file.
    - Search file names, make a file/folder name.
    - Load distortion coefficients from a txt file.
    - Get the tree view of a hdf/nxs file.
"""

import os
import glob
import h5py
import numpy as np
from PIL import Image
from collections import OrderedDict, deque

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
        raise ValueError(
            "Please use the forward slash in the file path")
    try:
        mat = np.asarray(Image.open(file_path), dtype=np.float32)
    except IOError:
        print("No such file or directory: {}".format(file_path))
        raise
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
        try:
            data = hdf_object[list_key[i]]
            if isinstance(data, h5py.Dataset):
                shape, dtype = data.shape, data.dtype
            else:
                shape, dtype = None, None
            if isinstance(data, list):
                if len(data) == 1:
                    if not isinstance(data, np.ndarray):
                        dtype = str(list(data)[0])
                        dtype.replace("b'", "'")
            list_shape.append(shape)
            list_type.append(dtype)
        except KeyError:
            list_shape.append(None)
            list_type.append(None)
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
            try:
                data = hdf_object[key]
                if isinstance(data, h5py.Dataset):
                    shape, dtype = data.shape, data.dtype
                else:
                    shape, dtype = None, None
                if isinstance(data, list):
                    if len(data) == 1:
                        if not isinstance(data, np.ndarray):
                            dtype = str(list(data)[0])
                            dtype.replace("b'", "'")
                list_dtype.append(dtype)
                list_dshape.append(shape)
            except KeyError:
                list_dtype.append(None)
                list_dshape.append(None)
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
        print("Couldn't open file: {}".format(file_path))
        raise
    check = key_path in hdf_object
    if not check:
        print("Couldn't open object with the key path: {}".format(key_path))
        raise ValueError("!!! Wrong key !!!")
    return hdf_object[key_path]


def make_folder(file_path):
    """
    Create a folder if not exist.

    Parameters
    ----------
    file_path : str
    """
    file_base = os.path.dirname(file_path)
    if not os.path.exists(file_base):
        try:
            os.makedirs(file_base)
        except OSError:
            raise ValueError("Can't create the folder: {}".format(file_path))


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


def make_folder_name(folder_path, name_prefix="Output"):
    """
    Create a new folder name to avoid overwriting.
    E.g: Output_00001, Output_00002...

    Parameters
    ----------
    folder_path : str
        Path to the parent folder.
    name_prefix : str
        Name prefix

    Returns
    -------
    str
        Name of the folder.
    """
    zero_prefix = 5
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
        raise ValueError(
            "Please use the forward slash in the file path")
    _, file_ext = os.path.splitext(file_path)
    if not ((file_ext == ".tif") or (file_ext == ".tiff")):
        mat = np.uint8(255 * (mat - np.min(mat)) / (np.max(mat) - np.min(mat)))
    make_folder(file_path)
    if not overwrite:
        file_path = make_file_name(file_path)
    image = Image.fromarray(mat)
    try:
        image.save(file_path)
    except IOError:
        print("Couldn't write to file {}".format(file_path))
        raise
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
        Add metadata. E.g. options={"entry/angles": angles, "entry/energy": 53}.

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
        print("Couldn't write to file: {}".format(file_path))
        raise
    if len(options) != 0:
        for opt_name in options:
            opts = options[opt_name]
            for key in opts:
                if key_path in key:
                    msg = "!!! Selected key path, '{0}', can not be a child " \
                          "key-path of '{1}' !!!\n!!! Change to make sure " \
                          "they are at the same level !!!".format(key, key_path)
                    raise ValueError(msg)
                ofile.create_dataset(key, data=opts[key])
    data_out = ofile.create_dataset(key_path, data_shape, dtype=data_type)
    return data_out


def load_distortion_coefficient(file_path):
    """
    Load distortion coefficients from a text file.

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
        raise ValueError(
            "Please use the forward slash in the file path")
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
