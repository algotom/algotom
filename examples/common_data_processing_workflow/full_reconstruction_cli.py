#!/users/conda/envs/2023-3.3-py310/bin/python

import sys
import time
import timeit
import argparse
import numpy as np
import multiprocessing as mp

import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.calculation as calc
import algotom.rec.reconstruction as rec
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.util.utility as util

usage = """
This CLI script is used for full reconstruction, editing the script to change 
default parameters of pre-processing methods (zinger removal, ring-artifact removal) 
or reconstruction methods.
"""

parser = argparse.ArgumentParser(description=usage)
parser.add_argument("-p", dest="proj_scan", help="Scan number of tomographic data",
                    type=str, required=True)
parser.add_argument("-d", dest="df_scan", help="Scan number of dark-flat data",
                    type=str, required=True)
parser.add_argument("-c", dest="center", help="Center of rotation", type=float,
                    required=False, default=0.0)
parser.add_argument("-r", dest="ratio", help="Ratio between delta and beta for phase filter",
                    type=float, required=False, default=0.0)
parser.add_argument("-o", dest="output_name", help="Output folder if don't want to be overwritten",
                    type=str, required=False, default="")
parser.add_argument("-f", dest="output_format", help="Output format: hdf or tif",
                    type=str, required=False, default="hdf")

parser.add_argument("--start", dest="start_slice", help="Start slice",
                    type=int, required=False, default=100)
parser.add_argument("--stop", dest="stop_slice", help="Stop slice", type=int,
                    required=False, default=-1)
parser.add_argument("--left", dest="crop_left", help="Crop left", type=int,
                    required=False, default=0)
parser.add_argument("--right", dest="crop_right", help="Crop right", type=int,
                    required=False, default=0)

parser.add_argument("--ring", dest="ring_removal", help="Select ring removal: 'sort', 'norm', 'all', 'none'",
                    type=str, required=False, default='all')
parser.add_argument("--zing", dest="zinger_removal", help="Enable/disable (1/0) zinger removal",
                    type=int, required=False, default=0)

parser.add_argument("--method", dest="method", help="Select a reconstruction method: 'fbp', 'gridrec', 'sirt'",
                    type=str, required=False, default='gridrec')
parser.add_argument("--ncore", dest="num_core", help="Select number of CPU cores",
                    type=int, required=False, default=None)
parser.add_argument("--iter", dest="num_iteration", help="Select number of iterations for the SIRT method",
                    type=int, required=False, default=100)
args = parser.parse_args()

input_base = "/facl/data/beamline/proposals/2024/pass-123456/tomography/raw_data/"
output_base0 = "/facl/data/beamline/proposals/2024/pass-123456/tomography/full_reconstruction/"

proj_scan_num = args.proj_scan
dark_flat_scan_num = args.df_scan
center = args.center
ratio = args.ratio
output_format = args.output_format

start_slice = args.start_slice
stop_slice = args.stop_slice
crop_left = args.crop_left
crop_right = args.crop_right
output_name = args.output_name

ring_removal = args.ring_removal
zinger_removal = args.zinger_removal
method = args.method
num_iteration = args.num_iteration
ncore = args.num_core

if output_name != "":
    output_base = output_base0 + "/" + proj_scan_num + "/" + output_name + "/"
else:
    output_base = output_base0 + "/" + proj_scan_num + "/"

print("\n====================================================================")
print("          Run the script for full reconstruction")
print("          Time: {}".format(time.ctime(time.time())))
print("====================================================================\n")

proj_file = input_base + "/" + proj_scan_num + "/" + proj_scan_num + ".nxs"
flat_file = input_base + "/" + dark_flat_scan_num + "/flat_00000.hdf"
dark_file = input_base + "/" + dark_flat_scan_num + "/dark_00000.hdf"

# Provide metadata for loading hdf file
proj_path = "entry/data/data"
flat_path = "entry/data/data"
dark_path = "entry/data/data"
angle_key = "entry/data/rotation_angle"

slice_chunk = 20  # Number of slices to be reconstructed in one go to reduce
                  # IO overhead (in loading a hdf file) and process in parallel
                  # (for CPU-based methods).
if ncore is None:
    ncore = np.clip(mp.cpu_count() - 2, 1, None)
else:
    if ncore > mp.cpu_count():
        ncore = np.clip(mp.cpu_count() - 2, 1, None)
        print("Number of available CPUs: {0}. Number of use: {1}".format(
            mp.cpu_count(), ncore))
if method == "fbp":
    rec_method = rec.fbp_reconstruction
elif method == "sirt":
    rec_method = rec.astra_reconstruction  # To use an iterative method. Must install Astra.
else:
    rec_method = rec.gridrec_reconstruction  # Fast cpu-method. Must install Tomopy.

proj_obj = losa.load_hdf(proj_file, proj_path)  # hdf object
(depth, height, width) = proj_obj.shape
left = crop_left
right = width - crop_right
width1 = right - left
angles = np.deg2rad(losa.load_hdf(proj_file, angle_key)[:])
print("1 -> Load dark-field and flat-field images, average each result")
flat_field = np.mean(np.asarray(losa.load_hdf(flat_file, flat_path)), axis=0)
dark_field = np.mean(np.asarray(losa.load_hdf(dark_file, dark_path)), axis=0)
time_start = timeit.default_timer()
print("     ...")
print("     ...", flush=True)
sys.stdout.flush()
print("     Done!\n", flush=True)
sys.stdout.flush()

# Load dark-field images and flat-field images, averaging each result.
print("2 -> Calculate the center-of-rotation")
if (stop_slice == -1) or (stop_slice > height):
    stop_slice = height

if center <= 0.0:
    idx = (stop_slice - start_slice) // 2 + start_slice
    cstart = width1 // 2 - 150
    cstop = width1 // 2 + 150
    sinogram = corr.flat_field_correction(proj_obj[:, idx, left:right],
                                          flat_field[idx, left:right],
                                          dark_field[idx, left:right])
    center = calc.find_center_vo(sinogram, cstart, cstop)
    print("     \nCenter-of-rotation is: {}\n".format(center), flush=True)
sys.stdout.flush()

total_slice = stop_slice - start_slice
offset = start_slice
if slice_chunk > total_slice:
    slice_chunk = total_slice
num_iter = total_slice // slice_chunk
num_rest = total_slice - num_iter * slice_chunk

if output_format == "hdf":
    recon_hdf = losa.open_hdf_stream(
        output_base + "/" + proj_scan_num + "_full_reconstruction.hdf",
        (total_slice, width1, width1),
        key_path='entry/data/data',
        data_type='float32', overwrite=True)

t_load_data_ffc = 0.0
t_recon = 0.0
t_save_data = 0.0

# Perform full reconstruction and save results
for i in range(num_iter):
    start_sino = i * slice_chunk + offset
    stop_sino = start_sino + slice_chunk

    # pre-processing
    t0 = timeit.default_timer()
    sinograms = corr.flat_field_correction(proj_obj[:, start_sino:stop_sino, left:right],
                                           flat_field[start_sino:stop_sino, left: right],
                                           dark_field[start_sino:stop_sino, left: right])
    if zinger_removal != 0:
        sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_zinger",
                                                            [0.08, 1], ncore=ncore, prefer="threads")
    if ring_removal != "none":
        if ring_removal == "sort":
            sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_stripe_based_sorting",
                                                                [21], ncore=ncore, prefer="threads")
        elif ring_removal == "norm":
            sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_stripe_based_normalization",
                                                                [15], ncore=ncore, prefer="threads")
        else:
            sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_all_stripe",
                                                                [2.5, 71, 31], ncore=ncore, prefer="threads")
    if ratio > 0.0:
        sinograms = util.apply_method_to_multiple_sinograms(sinograms, "fresnel_filter",
                                                            [ratio, 1], ncore=ncore, prefer="threads")
    t1 = timeit.default_timer()
    t_load_data_ffc += t1 - t0

    # Reconstruct a chunk of slices in parallel if using CPU-based method.
    t0 = timeit.default_timer()
    if method == "fbp":
        recon_img = rec_method(sinograms, center, angles=angles, ncore=ncore,
                               ratio=1.0, filter_name='hann', apply_log=True,
                               gpu=True, block=(16, 16))
    elif method == "sirt":
        recon_img = rec_method(sinograms, center, angles=angles, ratio=1.0,
                               method="SIRT_CUDA", num_iter=num_iteration,
                               filter_name="hann", pad=None, apply_log=True, ncore=1)
    else:
        recon_img = rec_method(sinograms, center, angles=None, ratio=1.0,
                               filter_name="shepp", apply_log=True, pad=100, ncore=1)
    t1 = timeit.default_timer()
    t_recon += t1 - t0

    # Save the results
    t0 = timeit.default_timer()
    if output_format == "hdf":
        recon_hdf[start_sino - start_slice:stop_sino - start_slice] = np.moveaxis(recon_img, 1, 0)
    else:
        for j in range(start_sino, stop_sino):
            name = "0000" + str(j)
            out_path = output_base + "/tifs/rec_" + name[-5:] + ".tif"
            losa.save_image(out_path, recon_img[:, j - start_sino, :])
    t1 = timeit.default_timer()
    t_save_data += t1 - t0
    t_stop = timeit.default_timer()
    print("     Done slice: {0} - {1} . Time {2}".format(start_sino, stop_sino, t_stop - time_start), flush=True)
    sys.stdout.flush()

if num_rest != 0:
    start_sino = num_iter * slice_chunk + offset
    stop_sino = start_sino + num_rest

    # pre-processing
    t0 = timeit.default_timer()
    sinograms = corr.flat_field_correction(proj_obj[:, start_sino:stop_sino, left:right],
                                           flat_field[start_sino:stop_sino, left: right],
                                           dark_field[start_sino:stop_sino, left: right])
    if zinger_removal != 0:
        sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_zinger",
                                                            [0.08, 1], ncore=ncore, prefer="threads")
    if ring_removal != "none":
        if ring_removal == "sort":
            sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_stripe_based_sorting",
                                                                [21], ncore=ncore, prefer="threads")
        elif ring_removal == "norm":
            sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_stripe_based_normalization",
                                                                [15], ncore=ncore, prefer="threads")
        else:
            sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_all_stripe",
                                                                [2.5, 71, 31], ncore=ncore, prefer="threads")
    if ratio > 0.0:
        sinograms = util.apply_method_to_multiple_sinograms(sinograms, "fresnel_filter",
                                                            [ratio, 1], ncore=ncore, prefer="threads")
    t1 = timeit.default_timer()
    t_load_data_ffc += t1 - t0

    # Reconstruct a chunk of slices in parallel if using CPU-based method.
    t0 = timeit.default_timer()
    if method == "fbp":
        recon_img = rec_method(sinograms, center, angles=angles, ncore=ncore,
                               ratio=1.0, filter_name='hann', apply_log=True,
                               gpu=True, block=(16, 16))
    elif method == "sirt":
        recon_img = rec_method(sinograms, center, angles=angles, ratio=1.0,
                               method="SIRT_CUDA", num_iter=num_iteration,
                               filter_name="hann", pad=None, apply_log=True, ncore=1)
    else:
        recon_img = rec_method(sinograms, center, angles=None, ratio=1.0,
                               filter_name="shepp", apply_log=True, pad=100, ncore=1)
    t1 = timeit.default_timer()
    t_recon += t1 - t0

    # Save the results
    t0 = timeit.default_timer()
    if output_format == "hdf":
        recon_hdf[start_sino - start_slice:stop_sino - start_slice] = np.moveaxis(recon_img, 1, 0)
    else:
        for j in range(start_sino, stop_sino):
            name = "0000" + str(j)
            out_path = output_base + "/tifs/rec_" + name[-5:] + ".tif"
            losa.save_image(out_path, recon_img[:, j - start_sino, :])
    t1 = timeit.default_timer()
    t_save_data += t1 - t0
    t_stop = timeit.default_timer()
    print("     Done slice: {0} - {1} . Time {2}".format(start_sino, stop_sino, t_stop - time_start), flush=True)
    sys.stdout.flush()

time_stop = timeit.default_timer()
t_total = time_stop - time_start

print("\n====================================================================")
print(" Time to load and pre-process data {}".format(t_load_data_ffc))
print(" Time to reconstruct data {}".format(t_recon))
print(" Time to save data {}".format(t_save_data))
print(" Output: {}".format(output_base))
print("!!! All Done. Total time cost {} !!!".format(t_total), flush=True)
print("\n====================================================================")
sys.stdout.flush()
