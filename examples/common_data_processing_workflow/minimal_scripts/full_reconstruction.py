import sys
import time
import timeit
import numpy as np
import multiprocessing as mp

import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.calculation as calc
import algotom.rec.reconstruction as rec
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.util.utility as util

"""
This script is used for full reconstruction, editing the script to change 
default parameters of pre-processing methods (zinger removal, ring-artifact removal)
or reconstruction methods.
"""


proj_file = "/tomography/raw_data/scan_00008/scan_00008.nxs"
flat_file = "/tomography/raw_data/scan_00009/flat_00000.hdf"
dark_file = "/tomography/raw_data/scan_00009/dark_00000.hdf"

output_base = "/tomography/tmp/scan_00008/full_reconstruction/"

start_slice = 0
stop_slice = -1

crop_left = 0
crop_right = 0
output_format = "tif"
center = 0.0  # For auto-determination
ncore = None
slice_chunk = 20  # Number of slices to be reconstructed in one go to reduce
                  # IO overhead (in loading a hdf file) and process in parallel
                  # (for CPU-based methods).

print("\n====================================================================")
print("          Run the script for full reconstruction")
print("          Time: {}".format(time.ctime(time.time())))
print("====================================================================\n")

# Provide metadata for loading hdf file
proj_path = "entry/data/data"
flat_path = "entry/data/data"
dark_path = "entry/data/data"
angle_key = "entry/data/rotation_angle"

if ncore is None or ncore == 0:
    ncore = np.clip(mp.cpu_count() - 2, 1, None)
else:
    if ncore > mp.cpu_count():
        ncore = np.clip(mp.cpu_count() - 2, 1, None)
        print("Number of available CPUs: {0}. Number of use: {1}".format(
            mp.cpu_count(), ncore))


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

# Load dark-field images and flat-field images, averaging each result.
print("2 -> Calculate the center-of-rotation")
if (stop_slice == -1) or (stop_slice > height):
    stop_slice = height

if center == 0.0:
    idx = (stop_slice - start_slice) // 2 + start_slice
    cstart = width1 // 2 - 150
    cstop = width1 // 2 + 150
    sinogram = corr.flat_field_correction(proj_obj[:, idx, left:right],
                                          flat_field[idx, left:right],
                                          dark_field[idx, left:right])
    center = calc.find_center_vo(sinogram, cstart, cstop)
    print("     \nCenter-of-rotation is: {}\n".format(center), flush=True)

total_slice = stop_slice - start_slice
offset = start_slice
if slice_chunk > total_slice:
    slice_chunk = total_slice
num_iter = total_slice // slice_chunk
num_rest = total_slice - num_iter * slice_chunk

if output_format == "hdf":
    recon_hdf = losa.open_hdf_stream(
        output_base + "/full_reconstruction.hdf", (total_slice, width1, width1),
        key_path='entry/data/data', data_type='float32', overwrite=True)

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
    # Algotom < 1.6
    # # # Apply zinger removal
    # # sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_zinger",
    # #                                                     [0.08, 1], ncore=ncore, prefer="threads")
    # # Apply ring removal (Check API Reference to get the name of other methods)
    # # sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_stripe_based_sorting",
    # #                                                     [21], ncore=ncore, prefer="threads")
    # # sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_stripe_based_normalization",
    # #                                                     [15], ncore=ncore, prefer="threads")
    # sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_all_stripe",
    #                                                     [2.5, 71, 31],
    #                                                     ncore=ncore, prefer="threads")
    # # Apply contrast enhancement/denoising
    # sinograms = util.apply_method_to_multiple_sinograms(sinograms, "fresnel_filter",
    #                                                     [100, 1],
    #                                                     ncore=ncore, prefer="threads")

    # Algotom >= 1.6
    # Apply zinger removal
    sinograms = util.parallel_process_slices(sinograms, remo.remove_zinger,
                                             [0.08, 1], ncore=ncore, prefer="threads")
    # Apply ring removal (Check API Reference to get the name of other methods)
    # sinograms = util.parallel_process_slices(sinograms, remo.remove_stripe_based_sorting,
    #                                          [21], ncore=ncore, prefer="threads")
    # sinograms = util.parallel_process_slices(sinograms, remo.remove_stripe_based_normalization,
    #                                          [15], ncore=ncore, prefer="threads")
    sinograms = util.parallel_process_slices(sinograms, remo.emove_all_stripe,
                                             [2.5, 71, 31],
                                             ncore=ncore, prefer="threads")
    # Apply contrast enhancement/denoising
    sinograms = util.parallel_process_slices(sinograms, filt.fresnel_filter,
                                             [100, 1], ncore=ncore, prefer="threads")
    t1 = timeit.default_timer()
    t_load_data_ffc += t1 - t0

    # Reconstruct a chunk of slices in parallel if using CPU-based method.
    t0 = timeit.default_timer()
    # recon_img = rec.fbp_reconstruction(sinograms, center, angles=angles, ncore=ncore,
    #                                    ratio=1.0, filter_name='hann', apply_log=True,
    #                                    gpu=True, block=(16, 16))
    # recon_img = rec.astra_reconstruction(sinograms, center, angles=angles, ratio=1.0,
    #                         method="SIRT_CUDA", num_iter=100,
    #                         filter_name="hann", pad=None, apply_log=True, ncore=1)
    recon_img = rec.gridrec_reconstruction(sinograms, center, angles=None,
                                           ratio=1.0, filter_name="shepp",
                                           apply_log=True, pad=100, ncore=1)
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
    # # # Apply zinger removal
    # # sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_zinger",
    # #                                                     [0.08, 1], ncore=ncore, prefer="threads")
    # # Apply ring removal (Check API Reference to get the name of other methods)
    # # sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_stripe_based_sorting",
    # #                                                     [21], ncore=ncore, prefer="threads")
    # # sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_stripe_based_normalization",
    # #                                                     [15], ncore=ncore, prefer="threads")
    # sinograms = util.apply_method_to_multiple_sinograms(sinograms, "remove_all_stripe",
    #                                                     [2.5, 71, 31],
    #                                                     ncore=ncore, prefer="threads")
    # # Apply contrast enhancement/denoising
    # sinograms = util.apply_method_to_multiple_sinograms(sinograms, "fresnel_filter",
    #                                                     [100, 1],
    #                                                     ncore=ncore, prefer="threads")
    # Algotom >= 1.6
    # Apply zinger removal
    sinograms = util.parallel_process_slices(sinograms, remo.remove_zinger,
                                             [0.08, 1], ncore=ncore, prefer="threads")
    # Apply ring removal (Check API Reference to get the name of other methods)
    # sinograms = util.parallel_process_slices(sinograms, remo.remove_stripe_based_sorting,
    #                                          [21], ncore=ncore, prefer="threads")
    # sinograms = util.parallel_process_slices(sinograms, remo.remove_stripe_based_normalization,
    #                                          [15], ncore=ncore, prefer="threads")
    sinograms = util.parallel_process_slices(sinograms, remo.emove_all_stripe,
                                             [2.5, 71, 31],
                                             ncore=ncore, prefer="threads")
    # Apply contrast enhancement/denoising
    sinograms = util.parallel_process_slices(sinograms, filt.fresnel_filter,
                                             [100, 1], ncore=ncore, prefer="threads")
    t1 = timeit.default_timer()
    t_load_data_ffc += t1 - t0

    # Reconstruct a chunk of slices in parallel if using CPU-based method.
    t0 = timeit.default_timer()
    # recon_img = rec.fbp_reconstruction(sinograms, center, angles=angles, ncore=ncore,
    #                                    ratio=1.0, filter_name='hann', apply_log=True,
    #                                    gpu=True, block=(16, 16))
    # recon_img = rec.astra_reconstruction(sinograms, center, angles=angles, ratio=1.0,
    #                         method="SIRT_CUDA", num_iter=100,
    #                         filter_name="hann", pad=None, apply_log=True, ncore=1)
    recon_img = rec.gridrec_reconstruction(sinograms, center, angles=None,
                                           ratio=1.0, filter_name="shepp",
                                           apply_log=True, pad=100, ncore=1)
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
