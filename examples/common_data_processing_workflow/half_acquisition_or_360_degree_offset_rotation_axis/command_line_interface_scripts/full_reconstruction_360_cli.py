#!/software/conda/hex_tomo/bin/python

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
import algotom.prep.conversion as conv
import algotom.prep.filtering as filt
import algotom.util.utility as util

proposal_id = "commissioning/pass-123456"

usage = """
This CLI script is used for full reconstruction, editing the script to change 
default parameters of pre-processing methods (zinger removal, ring-artifact removal) 
or reconstruction methods.
"""

parser = argparse.ArgumentParser(description=usage)
parser.add_argument("-p", dest="proj_scan",
                    help="Scan number of tomographic data",
                    type=int, required=True)
parser.add_argument("-d", dest="df_scan", help="Scan number of dark-flat data",
                    type=int, required=True)
parser.add_argument("-c", dest="center", help="Center of rotation", type=float,
                    required=False, default=0.0)
parser.add_argument("-r", dest="ratio",
                    help="Ratio between delta and beta for phase filter",
                    type=float, required=False, default=0.0)
parser.add_argument("-f", dest="output_format",
                    help="Output format: hdf or tif",
                    type=str, required=False, default="hdf")

parser.add_argument("--start", dest="start_slice", help="Start slice",
                    type=int, required=False, default=100)
parser.add_argument("--stop", dest="stop_slice", help="Stop slice", type=int,
                    required=False, default=-1)
parser.add_argument("--left", dest="crop_left", help="Crop left", type=int,
                    required=False, default=0)
parser.add_argument("--right", dest="crop_right", help="Crop right", type=int,
                    required=False, default=0)

parser.add_argument("--ring", dest="ring_removal",
                    help="Select ring removal: 'sort', 'norm', 'all', 'none'",
                    type=str, required=False, default='all')
parser.add_argument("--zing", dest="zinger_removal",
                    help="Enable/disable (1/0) zinger removal",
                    type=int, required=False, default=1)

parser.add_argument("--method", dest="method",
                    help="Select a reconstruction method: 'fbp', 'gridrec', "
                         "'sirt'", type=str, required=False, default='gridrec')
parser.add_argument("--ncore", dest="num_core",
                    help="Select number of CPU cores",
                    type=int, required=False, default=None)
parser.add_argument("--iter", dest="num_iteration",
                    help="Select number of iterations for the SIRT method",
                    type=int, required=False, default=100)
args = parser.parse_args()

input_base = "/nsls2/data/hex/proposals/" + proposal_id + "/tomography/raw_data/"
output_base0 = "/nsls2/data/hex/proposals/" + proposal_id + "/tomography/processed/full_reconstruction/"

proj_scan = args.proj_scan
df_scan = args.df_scan
center = args.center
ratio = args.ratio
output_format = args.output_format

start_slice = args.start_slice
stop_slice = args.stop_slice
crop_left = args.crop_left
crop_right = args.crop_right

ring_removal = args.ring_removal
zinger_removal = args.zinger_removal
method = args.method
num_iteration = args.num_iteration
ncore = args.num_core

proj_scan_num = "scan_" + ("0000" + str(proj_scan))[-5:]
dark_flat_scan_num = "scan_" + ("0000" + str(df_scan))[-5:]

output_base = output_base0 + "/" + proj_scan_num + "/"

if output_format != "hdf":
    output_name = losa.make_folder_name(output_base, name_prefix=proj_scan_num,
                                        zero_prefix=4)
    output_base = output_base + "/" + output_name + "/"

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

slice_chunk = 15  # Number of slices to be reconstructed in one go to reduce
# IO overhead (in loading a hdf file) and process in parallel
# (for CPU-based methods).
if ncore is None or ncore == 0:
    ncore = np.clip(mp.cpu_count() - 1, 1, None)
else:
    if ncore > mp.cpu_count():
        ncore = np.clip(mp.cpu_count() - 1, 1, None)
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
if (stop_slice == -1) or (stop_slice > height):
    stop_slice = height

if center <= 0.0:
    print("2 -> Calculate the center-of-rotation")
    idx = (stop_slice - start_slice) // 2 + start_slice
    cstart = width1 // 2 - 150
    cstop = width1 // 2 + 150
    sinogram = corr.flat_field_correction(proj_obj[:, idx, left:right],
                                          flat_field[idx, left:right],
                                          dark_field[idx, left:right])
    center = calc.find_center_vo(sinogram, cstart, cstop)
    print("\n     Center-of-rotation is: {}\n".format(center), flush=True)
sys.stdout.flush()

total_slice = stop_slice - start_slice
offset = start_slice
if slice_chunk > total_slice:
    slice_chunk = total_slice
num_iter = total_slice // slice_chunk
num_rest = total_slice - num_iter * slice_chunk

total_width = 2 * width1
if output_format == "hdf":
    output_path = output_base + "/" + proj_scan_num + "_full_recon.hdf"
    output_path = losa.make_file_name(output_path)
    recon_hdf = losa.open_hdf_stream(output_path,
                                     (total_slice, total_width, total_width),
                                     key_path='entry/data/data',
                                     data_type='float32', overwrite=True)

t_load_data_ffc = 0.0
t_recon = 0.0
t_save_data = 0.0

para_prefer = "threads"

# Perform full reconstruction and save results
for i in range(num_iter):
    start_sino = i * slice_chunk + offset
    stop_sino = start_sino + slice_chunk

    # pre-processing
    t0 = timeit.default_timer()
    sinograms = corr.flat_field_correction(
        proj_obj[:, start_sino:stop_sino, left:right],
        flat_field[start_sino:stop_sino, left: right],
        dark_field[start_sino:stop_sino, left: right])
    if zinger_removal != 0:
        # Algotom < 1.6.0
        # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
        #                                                     "remove_zinger",
        #                                                     [0.08, 1],
        #                                                     ncore=ncore,
        #                                                     prefer=para_prefer)
        sinograms = util.parallel_process_slices(sinograms, remo.remove_zinger,
                                                 [0.08, 1], ncore=ncore,
                                                 prefer=para_prefer)
    if ring_removal != "none":
        if ring_removal == "sort":
            # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
            #                                                     "remove_stripe_based_sorting",
            #                                                     [21],
            #                                                     ncore=ncore,
            #                                                     prefer=para_prefer)
            sinograms = util.parallel_process_slices(sinograms,
                                                     remo.remove_stripe_based_sorting,
                                                     [21], ncore=ncore, prefer="threads")
        elif ring_removal == "norm":
            # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
            #                                                     "remove_stripe_based_normalization",
            #                                                     [15],
            #                                                     ncore=ncore,
            #                                                     prefer=para_prefer)
            sinograms = util.parallel_process_slices(sinograms, remo.remove_stripe_based_normalization,
                                                     [15], ncore=ncore, prefer="threads")
        else:
            # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
            #                                                     "remove_all_stripe",
            #                                                     [2.5, 71, 31],
            #                                                     ncore=ncore,
            #                                                     prefer=para_prefer)
            sinograms = util.parallel_process_slices(sinograms, remo.remove_all_stripe,
                                                     [2.5, 71, 31], ncore=ncore, prefer="threads")
    if ratio > 0.0:
        # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
        #                                                     "fresnel_filter",
        #                                                     [ratio, 1],
        #                                                     ncore=ncore,
        #                                                     prefer=para_prefer)
        sinograms = util.parallel_process_slices(sinograms, filt.fresnel_filter,
                                                 [ratio, 1], ncore=ncore, prefer="threads")

    sinograms2 = []
    for jj in range(start_sino, stop_sino):
        sinogram = sinograms[:, jj - start_sino, :]
        sinogram = remo.remove_stripe_based_regularization(sinogram,
                                                           alpha=0.0005,
                                                           num_chunk=1,
                                                           apply_log=True,
                                                           sort=True)
        sinogram, center1 = conv.convert_sinogram_360_to_180(sinogram, center,
                                                             total_width=total_width)
        sinograms2.append(sinogram)
    sinograms = np.copy(np.moveaxis(np.float32(sinograms2), 0, 1))

    t1 = timeit.default_timer()
    t_load_data_ffc += t1 - t0

    # Reconstruct a chunk of slices in parallel if using CPU-based method.
    t0 = timeit.default_timer()
    if method == "fbp":
        # recon_img = rec_method(sinograms, center, angles=angles, ncore=ncore,
        #                     ratio=1.0, filter_name='hann', apply_log=True,
        #                     gpu=True, block=(16, 16))
        recon_img = rec_method(sinograms, center1, ratio=1.0,
                               method="FBP_CUDA", num_iter=num_iteration,
                               filter_name="hann", pad=None, apply_log=True,
                               ncore=ncore)
    elif method == "sirt":
        recon_img = rec_method(sinograms, center1, ratio=1.0,
                               method="SIRT_CUDA", num_iter=num_iteration,
                               filter_name="hann", pad=None, apply_log=True,
                               ncore=1)
    else:
        recon_img = rec_method(sinograms, center1, angles=None, ratio=1.0,
                               filter_name="hann", apply_log=True, pad=100,
                               ncore=ncore)
    t1 = timeit.default_timer()
    t_recon += t1 - t0

    # Save the results
    t0 = timeit.default_timer()
    if output_format == "hdf":
        recon_hdf[
        start_sino - start_slice:stop_sino - start_slice] = np.moveaxis(
            recon_img, 1, 0)
    else:
        for j in range(start_sino, stop_sino):
            name = "0000" + str(j)
            out_path = output_base + "/rec_" + name[-5:] + ".tif"
            losa.save_image(out_path, recon_img[:, j - start_sino, :])
    t1 = timeit.default_timer()
    t_save_data += t1 - t0
    t_stop = timeit.default_timer()
    print("     Done slice: {0} - {1} . Time {2}".format(start_sino, stop_sino,
                                                         t_stop - time_start),
          flush=True)
    sys.stdout.flush()

if num_rest != 0:
    start_sino = num_iter * slice_chunk + offset
    stop_sino = start_sino + num_rest

    # pre-processing
    t0 = timeit.default_timer()
    sinograms = corr.flat_field_correction(
        proj_obj[:, start_sino:stop_sino, left:right],
        flat_field[start_sino:stop_sino, left: right],
        dark_field[start_sino:stop_sino, left: right])
    if zinger_removal != 0:
        # Algotom < 1.6.0
        # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
        #                                                     "remove_zinger",
        #                                                     [0.08, 1],
        #                                                     ncore=ncore,
        #                                                     prefer=para_prefer)
        sinograms = util.parallel_process_slices(sinograms, remo.remove_zinger,
                                                 [0.08, 1], ncore=ncore,
                                                 prefer=para_prefer)
    if ring_removal != "none":
        if ring_removal == "sort":
            # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
            #                                                     "remove_stripe_based_sorting",
            #                                                     [21],
            #                                                     ncore=ncore,
            #                                                     prefer=para_prefer)
            sinograms = util.parallel_process_slices(sinograms,
                                                     remo.remove_stripe_based_sorting,
                                                     [21], ncore=ncore, prefer="threads")
        elif ring_removal == "norm":
            # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
            #                                                     "remove_stripe_based_normalization",
            #                                                     [15],
            #                                                     ncore=ncore,
            #                                                     prefer=para_prefer)
            sinograms = util.parallel_process_slices(sinograms, remo.remove_stripe_based_normalization,
                                                     [15], ncore=ncore, prefer="threads")
        else:
            # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
            #                                                     "remove_all_stripe",
            #                                                     [2.5, 71, 31],
            #                                                     ncore=ncore,
            #                                                     prefer=para_prefer)
            sinograms = util.parallel_process_slices(sinograms, remo.remove_all_stripe,
                                                     [2.5, 71, 31], ncore=ncore, prefer="threads")
    if ratio > 0.0:
        # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
        #                                                     "fresnel_filter",
        #                                                     [ratio, 1],
        #                                                     ncore=ncore,
        #                                                     prefer=para_prefer)
        sinograms = util.parallel_process_slices(sinograms, filt.fresnel_filter,
                                                 [ratio, 1], ncore=ncore, prefer="threads")

    sinograms2 = []
    for jj in range(start_sino, stop_sino):
        sinogram = sinograms[:, jj - start_sino, :]
        sinogram, center1 = conv.convert_sinogram_360_to_180(sinogram, center,
                                                             total_width=total_width)
        sinograms2.append(sinogram)
    sinograms = np.copy(np.moveaxis(np.float32(sinograms2), 0, 1))

    t1 = timeit.default_timer()
    t_load_data_ffc += t1 - t0

    # Reconstruct a chunk of slices in parallel if using CPU-based method.
    t0 = timeit.default_timer()
    if method == "fbp":
        recon_img = rec_method(sinograms, center1, angles=angles, ncore=ncore,
                               ratio=1.0, filter_name='hann', apply_log=True,
                               gpu=True, block=(16, 16))
    elif method == "sirt":
        recon_img = rec_method(sinograms, center1, angles=angles, ratio=1.0,
                               method="SIRT_CUDA", num_iter=num_iteration,
                               filter_name="hann", pad=None, apply_log=True,
                               ncore=1)
    else:
        recon_img = rec_method(sinograms, center1, angles=None, ratio=1.0,
                               filter_name="hann", apply_log=True, pad=100,
                               ncore=ncore)
    t1 = timeit.default_timer()
    t_recon += t1 - t0

    # Save the results
    t0 = timeit.default_timer()
    if output_format == "hdf":
        recon_hdf[
        start_sino - start_slice:stop_sino - start_slice] = np.moveaxis(
            recon_img, 1, 0)
    else:
        for j in range(start_sino, stop_sino):
            name = "0000" + str(j)
            out_path = output_base + "/rec_" + name[-5:] + ".tif"
            losa.save_image(out_path, recon_img[:, j - start_sino, :])
    t1 = timeit.default_timer()
    t_save_data += t1 - t0
    t_stop = timeit.default_timer()
    print("     Done slice: {0} - {1} . Time {2}".format(start_sino, stop_sino,
                                                         t_stop - time_start),
          flush=True)
    sys.stdout.flush()

time_stop = timeit.default_timer()
t_total = time_stop - time_start

print("\n====================================================================")
print(" Time to load and pre-process data {}".format(t_load_data_ffc))
print(" Time to reconstruct data {}".format(t_recon))
print(" Time to save data {}".format(t_save_data))
if output_format != "hdf":
    print(" Output: {}".format(output_base))
else:
    print(" Output: {}".format(output_path))
print("!!! All Done. Total time cost {} !!!".format(t_total), flush=True)
print("\n====================================================================")
sys.stdout.flush()