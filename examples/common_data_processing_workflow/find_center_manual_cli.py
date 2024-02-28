#!/users/conda/envs/2023-3.3-py310/bin/python

import time
import timeit
import argparse
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.util.utility as util

usage = """
This CLI script is used to find the center of rotation manually:
https://algotom.readthedocs.io/en/latest/toc/section4/section4_5.html#finding-the-center-of-rotation
"""

parser = argparse.ArgumentParser(description=usage)
parser.add_argument("-p", dest="proj_scan",
                    help="Scan number of tomographic data", type=str,
                    required=True)
parser.add_argument("-d", dest="df_scan", help="Scan number of dark-flat data",
                    type=str, required=True)
parser.add_argument("-s", dest="slice_index",
                    help="Index of the reconstructed slice for visualization",
                    type=int, required=True)
parser.add_argument("-r", dest="ratio",
                    help="Ratio between delta and beta for phase filter",
                    type=float, required=False, default=100.0)
parser.add_argument("-v", dest="view",
                    help="Select a sinogram-based method or a "
                         "reconstruction-based method: 'sino', 'rec'",
                    type=str, required=False, default="rec")

parser.add_argument("--start", dest="start_center", help="Start center",
                    type=int, required=True)
parser.add_argument("--stop", dest="stop_center", help="Stop center", type=int,
                    required=True)
parser.add_argument("--step", dest="step_center", help="Searching step",
                    type=float, required=False, default=1.0)

parser.add_argument("--left", dest="crop_left", help="Crop left", type=int,
                    required=False, default=0)
parser.add_argument("--right", dest="crop_right", help="Crop right", type=int,
                    required=False, default=0)
parser.add_argument("--ring", dest="ring_removal",
                    help="Select ring removal: 'sort', 'norm', 'all', 'none'",
                    type=str, required=False, default='sort')
parser.add_argument("--method", dest="recon_method",
                    help="Select reconstruction method: 'fbp' or 'gridrec'",
                    type=str, required=False, default='gridrec')
args = parser.parse_args()

input_base = "/facl/data/beamline/proposals/2024/pass-123456/tomography/raw_data/"
output_base0 = "/facl/data/beamline/proposals/2024/pass-123456/tomography/find_center/"

proj_scan_num = args.proj_scan
dark_flat_scan_num = args.df_scan
slice_index = args.slice_index
ratio = args.ratio
view = args.view

start_center = args.start_center
stop_center = args.stop_center
step_center = args.step_center
crop_left = args.crop_left
crop_right = args.crop_right
ring_removal = args.ring_removal
method = args.recon_method

t_start = timeit.default_timer()
output_base = output_base0 + "/" + proj_scan_num + "/"
print("\n====================================================================")
print("            Run the script for finding center manually")
print("            Time: {}".format(time.ctime(time.time())))
print("====================================================================\n")

proj_file = input_base + "/" + proj_scan_num + "/" + proj_scan_num + ".nxs"
flat_file = input_base + "/" + dark_flat_scan_num + "/flat_00000.hdf"
dark_file = input_base + "/" + dark_flat_scan_num + "/dark_00000.hdf"

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
    raise ValueError(f"Index is out of the range [0, {height - 1}]")
if start_center < 0 or start_center > width1 - 1:
    raise ValueError(f"Incorrect starting value, given image-width: {width1}")
if stop_center < 1 or stop_center > width1:
    raise ValueError(f"Incorrect stopping value, given image-width: {width1}")

sinogram = corr.flat_field_correction(proj_obj[:, slice_index, left:right],
                                      flat_field[slice_index, left:right],
                                      dark_field[slice_index, left:right])

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

if view == "sino":
    util.find_center_visual_sinograms(sinogram, output_base, start_center,
                                      stop_center, step=1, zoom=1.0)
else:
    util.find_center_visual_slices(sinogram, output_base, start_center,
                                   stop_center, 1.0, zoom=1.0, method=method,
                                   gpu=False, angles=angles, ratio=1.0,
                                   filter_name=None)

t_stop = timeit.default_timer()
print("====================================================================\n")
print("All done! Time cost {}".format(t_stop - t_start))
print("Output {}".format(output_base))
print("====================================================================\n")
