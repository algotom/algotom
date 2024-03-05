import os
import shutil
import time
import timeit
import numpy as np
import h5py

import algotom.io.loadersaver as losa
import algotom.post.postprocessing as post


"""
This script is used for data reduction of a reconstructed volume: 
rescaling, downsampling, cropping, reslicing.
"""

input_path = "/tomography/scan_00008/full_reconstruction.hdf"
# input_path = "/tomography/scan_00008/full_reconstruction/" # For tifs

output_path = "/tomography/tmp/scan_00008/data_reduction"

crop = (0, 0, 0, 0, 0, 0)  #  (top, bottom, left, right, front, back)
rescale = 8  # 8-bit
downsample = 2  # Downsample
reslice = 1  # Reslice along axis-1
rotate_angle = 0.0  # Rotate the volume is reslice != 0
hdf_key = "entry/data/data"

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

# Check output pathw
if output_path.lower().endswith(('.hdf', '.nxs', '.h5')):
    output_format = "hdf"
    if os.path.isfile(output_path):
        output_path = losa.make_file_name(output_path)
else:
    _, file_extension = os.path.splitext(output_path)
    if file_extension != "":
        raise ValueError("Output-path must be a folder or a hdf/nxs/h5 file!!!")
    else:
        output_format = "folder"
        if os.path.isdir(output_path):
            num = 0
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

t_start = timeit.default_timer()
downsample = int(np.clip(downsample, 1, 30))
if reslice == 0:
    if downsample > 1:
        if rescale == 32:
            post.downsample_dataset(input_path, output_path, downsample, key_path=hdf_key,
                                    method='mean', rescaling=False,
                                    skip=None, crop=crop)
        else:
            post.downsample_dataset(input_path, output_path, downsample, key_path=hdf_key,
                                    method='mean', rescaling=True, nbit=rescale,
                                    skip=None,crop=crop)
    else:
        post.rescale_dataset(input_path, output_path, nbit=rescale, crop=crop,
                                key_path=hdf_key)
else:
    if downsample == 1:
        if rescale == 32:
            post.reslice_dataset(input_path, output_path, rescaling=False, key_path=hdf_key,
                                 rotate=rotate_angle, axis=reslice, crop=crop,
                                 chunk=40, show_progress=True, ncore=None)
        else:
            post.reslice_dataset(input_path, output_path, rescaling=True, key_path=hdf_key,
                                 rotate=rotate_angle, nbit=rescale, axis=reslice, crop=crop,
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
                post.downsample_dataset(input_path, dsp_path, downsample, key_path=hdf_key,
                                        method='mean', rescaling=False,
                                        skip=None, crop=crop)
            else:
                post.downsample_dataset(input_path, dsp_path, downsample, key_path=hdf_key,
                                        method='mean', rescaling=True, nbit=rescale,
                                        skip=None,crop=crop)
            if rescale == 32:
                post.reslice_dataset(dsp_path, output_path, rescaling=False, key_path="entry/data",
                                    rotate=rotate_angle, axis=reslice,
                                    chunk=40, show_progress=True, ncore=None)
            else:
                post.reslice_dataset(dsp_path, output_path, rescaling=True, key_path="entry/data",
                                    rotate=rotate_angle, nbit=rescale, axis=reslice,
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
