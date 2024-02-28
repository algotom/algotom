#!/users/conda/envs/2023-3.3-py310/bin/python

import argparse
import time
import timeit

import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.calculation as calc
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.rec.reconstruction as rec

usage = "This CLI script is used to reconstruct a few slices across " \
        "image height"

parser = argparse.ArgumentParser(description=usage)
parser.add_argument("-p", dest="proj_scan",
                    help="Scan number of tomographic data", type=str,
                    required=True)
parser.add_argument("-d", dest="df_scan", help="Scan number of dark-flat data",
                    type=str, required=True)
parser.add_argument("-c", dest="center", help="Center of rotation", type=float,
                    required=False, default=0.0)
parser.add_argument("-r", dest="ratio",
                    help="Ratio between delta and beta for phase filter",
                    type=float, required=False, default=0.0)
parser.add_argument("-o", dest="output_name",
                    help="Output folder if don't want to be overwritten",
                    type=str, required=False, default="")

parser.add_argument("--start", dest="start_slice", help="Start slice",
                    type=int, required=False, default=100)
parser.add_argument("--stop", dest="stop_slice", help="Stop slice", type=int,
                    required=False, default=-1)
parser.add_argument("--step", dest="step_slice", help="Step slice", type=int,
                    required=False, default=100)

parser.add_argument("--left", dest="crop_left", help="Crop left", type=int,
                    required=False, default=0)
parser.add_argument("--right", dest="crop_right", help="Crop right", type=int,
                    required=False, default=0)
parser.add_argument("--ring", dest="ring_removal",
                    help="Select ring removal: 'sort', 'norm', 'all', 'none'",
                    type=str, required=False, default='sort')
parser.add_argument("--method", dest="recon_method",
                    help="Select reconstruction method: "
                         "'fbp', 'gridrec', 'sirt'",
                    type=str, required=False, default='gridrec')
parser.add_argument("--iter", dest="num_iteration",
                    help="Select number of iterations for the SIRT method",
                    type=int, required=False, default=100)
args = parser.parse_args()

input_base = "/facl/data/beamline/proposals/2024/pass-123456/tomography/raw_data/"
output_base0 = "/facl/data/beamline/proposals/2024/pass-123456/tomography/processed/"

proj_scan_num = args.proj_scan
dark_flat_scan_num = args.df_scan
center = args.center
ratio = args.ratio
output_name = args.output_name
start_slice = args.start_slice
stop_slice = args.stop_slice
step_slice = args.step_slice
crop_left = args.crop_left
crop_right = args.crop_right
ring_removal = args.ring_removal
recon_method = args.recon_method
num_iteration = args.num_iteration

if output_name != "":
    output_base = output_base0 + "/" + proj_scan_num + "/" + output_name + "/"
else:
    output_base = output_base0 + "/" + proj_scan_num + "/"

print("====================================================================\n")
print("         Run the script for reconstructing a few slices")
print("         Time: {}".format(time.ctime(time.time())))
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


if stop_slice == -1 or stop_slice > height - 1:
    stop_slice = height - 1
if center == 0.0:
    # Extract a sinogram at the middle for calculating the center of rotation
    idx = (stop_slice - start_slice) // 2 + start_slice
    center_start = width1 // 2 - 150
    center_stop = width1 // 2 + 150
    sinogram = corr.flat_field_correction(proj_obj[:, idx, left:right],
                                          flat_field[idx, left:right],
                                          dark_field[idx, left:right])
    center = calc.find_center_vo(sinogram, start=center_start,
                                 stop=center_stop)
    print("---> Center of rotation: {}".format(center))

t_start = timeit.default_timer()
for idx in range(start_slice, stop_slice + 1, step_slice):
    # Get a sinogram and perform flat-field correction
    sinogram = corr.flat_field_correction(proj_obj[:, idx, left:right],
                                          flat_field[idx, left:right],
                                          dark_field[idx, left:right])
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

    # Perform reconstruction
    if recon_method == "fbp":
        rec_img = rec.fbp_reconstruction(sinogram, center, angles=angles,
                                         apply_log=True, gpu=True)
    elif recon_method == "sirt":
        # Using an iterative gpu-based method, available in Astra Toolbox
        rec_img = rec.astra_reconstruction(sinogram, center, angles=angles,
                                           method="SIRT_CUDA", apply_log=True,
                                           num_iter=num_iteration)
    else:
        # Using a fast cpu-based method, available in Tomopy.
        rec_img = rec.gridrec_reconstruction(sinogram, center, angles=angles,
                                             apply_log=True)
    out_file = output_base + "/rec_" + ("00000" + str(idx))[-5:] + ".tif"
    t1 = timeit.default_timer()
    print("-   Done slice: {0}. Time cost: {1}".format(idx, t1 - t_start))
    losa.save_image(out_file, rec_img)
t_stop = timeit.default_timer()
print("====================================================================\n")
print("All done! Time cost: {}".format(t_stop - t_start))
print("Output is at\n {}".format(output_base))
print("====================================================================\n")
