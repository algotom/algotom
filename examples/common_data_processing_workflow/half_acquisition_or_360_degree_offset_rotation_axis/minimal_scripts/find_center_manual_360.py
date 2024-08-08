#!/software/conda/hex_tomo/bin/python

"""
This script is for manually finding the center of rotation (rotation axis)
of a half-acquisition scan (360-degree scanning with offset rotation axis).
It can handle the case that the rotation axis is out of the field of view
(by mistake when acquired data).
"""

import time
import timeit
import argparse
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.util.utility as util
import algotom.prep.conversion as conv
import algotom.rec.reconstruction as rec


proj_file = "F:/Tomo_data/raw_data/scan_00008/proj_00000.hdf"
flat_file = "F:/tomography/raw_data/scan_00009/flat_00000.hdf"
dark_file = "F://tomography/raw_data/scan_00009/dark_00000.hdf"
output_base = "F:/tmp/processed/"

slice_index = 0

start_center = -40   # Could be negative or larger than the width of an image
                     # for the case the rotation axis is out of the FOV by mistake.
stop_center = 10
step_center = 1
crop_left = 0
crop_right = 0
ring_removal = "norm"
ratio = 0
view = "slice"

t_start = timeit.default_timer()
output_name = losa.make_folder_name(output_base, name_prefix='Find_center', zero_prefix=3)
output_base = output_base + "/" + output_name + "/"

print("====================================================================\n")
print("            Run the script for finding center manually                ")
print("            Time: {}".format(time.ctime(time.time())))
print("====================================================================\n")

# Keys to hdf/nxs/h5 datasets
proj_key = "entry/data/data"
flat_key = "entry/data/data"
dark_key = "entry/data/data"
angle_key = "entry/data/rotation_angle"

# Load data, average flat and dark images
proj_obj = losa.load_hdf(proj_file, proj_key)  # hdf object
(depth, height, width) = proj_obj.shape
left = crop_left
right = width - crop_right
width1 = right - left
angles = np.deg2rad(losa.load_hdf(proj_file, angle_key)[:])
flat_field = np.mean(np.asarray(losa.load_hdf(flat_file, flat_key)), axis=0)
dark_field = np.mean(np.asarray(losa.load_hdf(dark_file, dark_key)), axis=0)

if slice_index < 0 or slice_index > height - 1:
    raise ValueError("Slice index is out of the range [0, {}]".format(height - 1))

sinogram = corr.flat_field_correction(proj_obj[:, slice_index, left:right],
                                      flat_field[slice_index, left:right],
                                      dark_field[slice_index, left:right],
                                      use_dark=True)
sinogram = remo.remove_zinger(sinogram, 0.08)
# Apply ring removal
if ring_removal != "none":
    if ring_removal == "norm":
        sinogram = remo.remove_stripe_based_normalization(sinogram, 15)
    elif ring_removal == "sort":
        sinogram = remo.remove_stripe_based_sorting(sinogram, 21)
    else:
        sinogram = remo.remove_all_stripe(sinogram, 2.0, 51, 21)
if ratio > 0.0:
    # Apply contrast enhancement
    sinogram = filt.fresnel_filter(sinogram, ratio)

list_center = np.arange(start_center, stop_center + step_center, step_center)
min_center = np.min(list_center)
max_center = np.max(list_center)
pad = 0
if min_center < 0 or max_center < 0:
    pad = max(abs(min_center), abs(max_center))
    total_width = 2 * width1 + 2 * pad + 1
elif min_center > width1 or max_center > width1:
    pad = max(abs(width1 - min_center), abs(width1 - max_center))
    total_width = 2 * width1 + 2 * pad + 1
else:
    total_width = 2 * width1

nrow_180 = sinogram.shape[0] // 2 + 1
num = 0
for i, center in enumerate(list_center):
    if width1 > center >=0:
        sino_180, center1 = conv.convert_sinogram_360_to_180(sinogram, center,
                                                             total_width=total_width)
    else:
        sino_top = sinogram[0:nrow_180, :]
        sino_bot = np.fliplr(sinogram[-nrow_180:, :])
        if center < 0:
            join_width = 2 * abs(center) + 1
            sino_180 = conv.join_image(sino_top, sino_bot, join_width, 0,
                                       norm=True,
                                       total_width=total_width)
            center1 = width1 + abs(center) - 0.5
        else:
            join_width = 2 * (center - width1) - 1
            sino_180 = conv.join_image(sino_top, sino_bot, join_width, 0,
                                       norm=True,
                                       total_width=total_width)
            center1 = center
    rec_img = rec.gridrec_reconstruction(sino_180, center1,
                                         filter_name="hann",
                                         filter_par=0.95, apply_log=True)
    # name = "{0:.2f}".format(center) + ".tif"
    # out_file = output_base + "/center_" + name
    # losa.save_image(out_file, rec_img)
    # print("Done center {}".format(out_file))
    name = f"{num:05}" + ".tif"
    out_file = output_base + "/slice/rec_" + name
    losa.save_image(out_file, rec_img)
    out_file = output_base + "/sinogram/sino_" + name
    losa.save_image(out_file, sino_180)
    print(f"Done center: {center}. File: {out_file}")
    num = num + 1

t_stop = timeit.default_timer()
print("====================================================================\n")
print("All done! Time cost {}".format(t_stop - t_start))
print("Output {}".format(output_base))
print("====================================================================\n")