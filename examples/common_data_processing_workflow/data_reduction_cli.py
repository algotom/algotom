#!/users/conda/envs/2023-3.3-py310/bin/python

import os
import shutil
import time
import timeit
import argparse
import numpy as np
import h5py

import algotom.io.loadersaver as losa
import algotom.post.postprocessing as post

usage = """
This CLI script is used for data reduction of a reconstructed volume: 
rescaling, downsampling, cropping, reslicing.
"""

parser = argparse.ArgumentParser(description=usage)
parser.add_argument("-i", dest="input_path",
                    help="Input path: folder of tifs or a hdf/h5/nxs file",
                    type=str, required=True)
parser.add_argument("-k", dest="hdf_key",
                    help="Key to the dataset if input is in the hdf/nxs/h5 format",
                    required=False, default=None)
parser.add_argument("-o", dest="output_path",
                    help="Output path: folder or a hdf/h5/nxs file",
                    type=str, required=True)

parser.add_argument("--crop", dest="crop",
                    help="To crop the volume from the edges '(top, bottom, left, right, front, back)'",
                    type=str, required=False, default="(0, 0, 0, 0, 0, 0)")
parser.add_argument("--rescale", dest="rescale",
                    help="Rescale to a 8/16-bit data-type",
                    type=int, required=False, default=8)
parser.add_argument("--downsample", dest="downsample",
                    help="Downsample. e.g, 2x2x2",
                    type=int, required=False, default=1)
parser.add_argument("--reslice", dest="reslice",
                    help="Reslice the volume along axis 1 or 2",
                    type=int, required=False, default=0)
parser.add_argument("--rotate", dest="rotate",
                    help="Rotate the volume (degree) if reslicing",
                    type=float, required=False, default=0.0)
args = parser.parse_args()

input_path = args.input_path
hdf_key = args.hdf_key
output_path = args.output_path
crop_str = args.crop
rescale = args.rescale
downsample = args.downsample
reslice = args.reslice
rotate_angle = args.rotate

# Check cropping parameters
crop_str = crop_str.strip().lstrip('(').rstrip(')')
elements = crop_str.split(',')
msg = "Provide cropping parameters using this format --crop '(top, bottom, left, right, front, back)'"
try:
    crop = tuple(int(element) for element in elements)
except ValueError:
    raise ValueError(msg)
if len(crop) != 6:
    raise ValueError(msg)
else:
    if not all(isinstance(cr, int) and cr >= 0 for cr in crop):
        raise ValueError("Cropping parameters must be integer and >=0")

# Check input path
path_exist = os.path.exists(input_path)
if not path_exist:
    raise ValueError(f"Input-path '{input_path}' does not exist.")
input_format = None
if os.path.isfile(input_path) and input_path.lower().endswith(
        ('.hdf', '.nxs', '.h5')):
    input_format = "hdf"
    if hdf_key is None:
        raise ValueError("Please provide key to the dataset in the hdf file!!!")
    else:
        try:
            with h5py.File(input_path, 'r') as file:
                if hdf_key in file and isinstance(file[hdf_key], h5py.Dataset):
                    if file[hdf_key].ndim != 3:
                        raise ValueError(f"The dataset '{hdf_key}' is not 3D !!!")
                else:
                    raise ValueError(f"The key '{hdf_key}' does not exist or is not a dataset.")
        except Exception as e:
            raise ValueError(e)
elif os.path.isdir(input_path):
    input_format = "folder"
    if len(losa.find_file(input_path + "/*.tif*")) == 0:
        raise ValueError(f"No tif files in the folder: '{input_path}'")
else:
    raise ValueError(f"Input-path '{input_path}' does not match the specified file formats or a folder.")

# Check output path
if output_path.lower().endswith(('.hdf', '.nxs', '.h5')):
    output_format = "hdf"
    if os.path.isfile(output_path):
        output_path = losa.make_file_name(output_path)
        # raise ValueError("File exists!!! Please choose another file path!!!")
else:
    _, file_extension = os.path.splitext(output_path)
    if file_extension != "":
        raise ValueError("Output-path must be a folder or a hdf/nxs/h5 file!!!")
    else:
        output_format = "folder"
        if os.path.isdir(output_path):
            num = 1
            output_path = os.path.normpath(output_path)
            parent_path = os.path.dirname(output_path)
            last_folder = os.path.basename(output_path)
            while True:
                name = ("0000" + str(num))[-5:]
                new_path = parent_path + "/" + last_folder + "_" + name
                if os.path.isdir(new_path):
                    num = num + 1
                else:
                    break
            output_path = new_path
            # raise ValueError("Folder exists!!! Please choose another path!!!")

# Check if rescaling and reslicing inputs are valid
if rescale != 8 and rescale != 16 and rescale != 32:
    raise ValueError(f"Invalid rescaling input: {rescale}!!! Only 3 options: 8, 16, 32")
if reslice != 0 and reslice != 1 and reslice != 2:
    raise ValueError(f"Invalid slicing input: {reslice}!!! Only 3 options: 0, 1, 2")

print("\n====================================================================")
print("          Run the script for data reduction")
print("          Time: {}".format(time.ctime(time.time())))
print("====================================================================\n")
print("Crop-(top, bottom, left, right, front, back) = {}".format(crop))
print("Rescale: {}-bit".format(rescale))
print("Downsample: {}".format(downsample))
print("Reslice: axis-{}".format(reslice))
print("Rotate: {}-degree".format(rotate_angle))
print("...")
print("...")
print("...")

t_start = timeit.default_timer()
downsample = int(np.clip(downsample, 1, 30))
if reslice == 0:
    if downsample > 1:
        if rescale == 32:
            post.downsample_dataset(input_path, output_path, downsample,
                                    key_path=hdf_key,
                                    method='mean', rescaling=False,
                                    skip=None, crop=crop)
        else:
            post.downsample_dataset(input_path, output_path, downsample,
                                    key_path=hdf_key,
                                    method='mean', rescaling=True,
                                    nbit=rescale,
                                    skip=None, crop=crop)
    else:
        if rescale == 32 and input_format == output_format and any(
                cr == 0 for cr in crop):
            raise ValueError("Input format and output format is the same!!!")
        else:
            post.rescale_dataset(input_path, output_path, nbit=rescale,
                                 crop=crop,
                                 key_path=hdf_key)
else:
    if downsample == 1:
        if rescale == 32:
            post.reslice_dataset(input_path, output_path, rescaling=False,
                                 key_path=hdf_key,
                                 rotate=rotate_angle, axis=reslice, crop=crop,
                                 chunk=40, show_progress=True, ncore=None)
        else:
            post.reslice_dataset(input_path, output_path, rescaling=True,
                                 key_path=hdf_key,
                                 rotate=rotate_angle, nbit=rescale,
                                 axis=reslice, crop=crop,
                                 chunk=40, show_progress=True, ncore=None)
    else:
        if output_format == "hdf":
            file_name = os.path.basename(output_path)
            folder_path = os.path.splitext(output_path)[0]
            dsp_folder = folder_path + "_dsp_tmp/"
            dsp_path = dsp_folder + file_name
        else:
            dsp_folder = dsp_path = os.path.normpath(output_path) + "_dsp_tmp/"
        try:
            if rescale == 32:
                post.downsample_dataset(input_path, dsp_path, downsample,
                                        key_path=hdf_key,
                                        method='mean', rescaling=False,
                                        skip=None, crop=crop)
            else:
                post.downsample_dataset(input_path, dsp_path, downsample,
                                        key_path=hdf_key,
                                        method='mean', rescaling=True,
                                        nbit=rescale,
                                        skip=None, crop=crop)
            if rescale == 32:
                post.reslice_dataset(dsp_path, output_path, rescaling=False,
                                     key_path="entry/data",
                                     rotate=rotate_angle, axis=reslice,
                                     chunk=40, show_progress=True, ncore=None)
            else:
                post.reslice_dataset(dsp_path, output_path, rescaling=True,
                                     key_path="entry/data",
                                     rotate=rotate_angle, nbit=rescale,
                                     axis=reslice,
                                     chunk=40, show_progress=True, ncore=None)
            shutil.rmtree(dsp_folder)
        except Exception as e:
            shutil.rmtree(dsp_folder)
            raise ValueError(e)

t_stop = timeit.default_timer()
print("\n====================================================================")
print("Output: {}".format(output_path))
print("!!! All Done. Total time cost {} !!!".format(t_stop - t_start))
print("\n====================================================================")
