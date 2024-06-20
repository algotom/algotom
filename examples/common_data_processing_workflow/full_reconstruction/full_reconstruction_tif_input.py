"""
The following example shows how to reconstruct full size of a standard dataset
acquired as tif images.
"""

import os
import shutil
import timeit
import numpy as np
import algotom.io.converter as cvr
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.calculation as calc
import algotom.rec.reconstruction as rec
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.util.utility as util
import warnings
from numba import NumbaPerformanceWarning

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

proj_path = "E:/Tomo_data/68067_tif/projections/"
flat_path = "E:/Tomo_data/68067_tif/flats/"
dark_path = "E:/Tomo_data/68067_tif/darks/"

output_base0 = "E:/output/full_reconstruction/"
folder_name = losa.make_folder_name(output_base0, name_prefix="recon",
                                    zero_prefix=3)
output_base = output_base0 + "/" + folder_name + "/"

# Optional parameters
start_slice = 10
stop_slice = -1
chunk = 16  # Number of slices to be reconstructed in one go
ncore = None
output_format = "tif"  # "tif" or "hdf"
preprocessing = True

# Give alias to a reconstruction method which is convenient for later change
# recon_method = rec.dfi_reconstruction
recon_method = rec.fbp_reconstruction
# recon_method = rec.gridrec_reconstruction # Fast cpu-method. Must install Tomopy.
# recon_method = rec.astra_reconstruction # To use iterative methods. Must install Astra.

t_start = timeit.default_timer()
print("---------------------------------------------------------------")
print("-----------------------------Start-----------------------------\n")

# Load dark-field images and flat-field images.
print("1 -> Load dark-field and flat-field images, average each result")
flat_field = np.mean(np.asarray(
    [losa.load_image(file) for file in losa.find_file(flat_path + "/*tif*")]), axis=0)
dark_field = np.mean(np.asarray(
    [losa.load_image(file) for file in losa.find_file(dark_path + "/*tif*")]), axis=0)
proj_files = losa.find_file(proj_path + "/*tif*")

print("2 -> Save projections to a temp hdf-file for fast extracting sinograms")
# Save projections to a temp hdf file for fast extracting sinograms.
t0 = timeit.default_timer()
temp_hdf = output_base + "/tmp_/" + "tomo_data.hdf"
key_path = "entry/data"
cvr.convert_tif_to_hdf(proj_path, temp_hdf, key_path=key_path)
data, hdf_obj = losa.load_hdf(temp_hdf, key_path, return_file_obj=True)
(depth, height, width) = data.shape
remove_tmp_hdf = True # To delete temp hdf file at the end
t1 = timeit.default_timer()
print("  -> Done. Time {}".format(t1 - t0))

print("3 -> Calculate the center-of-rotation")
# Extract sinogram at the middle for calculating the center of rotation
index = height // 2
sinogram = corr.flat_field_correction(data[:, index, :],
                                      flat_field[index, :],
                                      dark_field[index, :])
center = calc.find_center_vo(sinogram)
print("Center-of-rotation is {}".format(center))

if (stop_slice == -1) or (stop_slice > height):
    stop_slice = height
total_slice = stop_slice - start_slice

if output_format == "hdf":
    # Note about the change of data-shape
    recon_hdf = losa.open_hdf_stream(output_base + "/recon_data.hdf",
                                     (total_slice, width, width),
                                     key_path='entry/data',
                                     data_type='float32', overwrite=True)
t_load = 0.0
t_prep = 0.0
t_rec = 0.0
t_save = 0.0
chunk = np.clip(chunk, 1, total_slice)
last_chunk = total_slice - chunk * (total_slice // chunk)
# Perform full reconstruction
for i in np.arange(start_slice, start_slice + total_slice - last_chunk, chunk):
    start_sino = i
    stop_sino = start_sino + chunk

    # Load data, perform flat-field correction
    t0 = timeit.default_timer()
    sinograms = corr.flat_field_correction(data[:, start_sino:stop_sino, :],
                                           flat_field[start_sino:stop_sino, :],
                                           dark_field[start_sino:stop_sino, :])
    t1 = timeit.default_timer()
    t_load = t_load + t1 - t0

    # Perform pre-processing
    if preprocessing:
        t0 = timeit.default_timer()
        # Algotom < 1.6
        # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
        #                                                     "remove_zinger",
        #                                                     [0.08, 1],
        #                                                     ncore=ncore,
        #                                                     prefer="threads")
        # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
        #                                                     "remove_all_stripe",
        #                                                     [3.0, 51, 21],
        #                                                     ncore=ncore,
        #                                                     prefer="threads")
        # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
        #                                                     "fresnel_filter",
        #                                                     [200, 1],
        #                                                     ncore=ncore,
        #                                                     prefer="threads")

        # Algotom >= 1.6
        sinograms = util.parallel_process_slices(sinograms, remo.remove_zinger,
                                                 [0.08, 1], ncore=ncore,
                                                 prefer="threads")
        sinograms = util.parallel_process_slices(sinograms,
                                                 remo.remove_all_stripe,
                                                 [3.0, 51, 21], ncore=ncore,
                                                 prefer="threads")
        sinograms = util.parallel_process_slices(sinograms,
                                                 filt.fresnel_filter,
                                                 [200, 1], ncore=ncore,
                                                 prefer="threads")

        t1 = timeit.default_timer()
        t_prep = t_prep + t1 - t0

    # Perform reconstruction
    t0 = timeit.default_timer()
    recon_imgs = recon_method(sinograms, center, ncore=ncore)
    t1 = timeit.default_timer()
    t_rec = t_rec + t1 - t0

    # Save output
    t0 = timeit.default_timer()
    if output_format == "hdf":
        recon_hdf[start_sino - start_slice:stop_sino - start_slice] = np.moveaxis(recon_imgs, 1, 0)
    else:
        for j in range(start_sino, stop_sino):
            out_file = output_base + "/rec_" + ("0000" + str(j))[-5:] + ".tif"
            losa.save_image(out_file, recon_imgs[:, j - start_sino, :])
    t1 = timeit.default_timer()
    t_save = t_save + t1 - t0
    t_stop = timeit.default_timer()
    print("Done slice: {0} - {1} . Time {2}".format(start_sino, stop_sino,
                                                    t_stop - t_start))
if last_chunk != 0:
    start_sino = start_slice + total_slice - last_chunk
    stop_sino = start_sino + last_chunk

    # Load data, perform flat-field correction
    t0 = timeit.default_timer()
    sinograms = corr.flat_field_correction(data[:, start_sino:stop_sino, :],
                                           flat_field[start_sino:stop_sino, :],
                                           dark_field[start_sino:stop_sino, :])
    t1 = timeit.default_timer()
    t_load = t_load + t1 - t0

    # Perform pre-processing
    if preprocessing:
        t0 = timeit.default_timer()
        # Algotom < 1.6
        # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
        #                                                     "remove_zinger",
        #                                                     [0.08, 1],
        #                                                     ncore=ncore,
        #                                                     prefer="threads")
        # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
        #                                                     "remove_all_stripe",
        #                                                     [3.0, 51, 21],
        #                                                     ncore=ncore,
        #                                                     prefer="threads")
        # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
        #                                                     "fresnel_filter",
        #                                                     [200, 1],
        #                                                     ncore=ncore)

        # Algotom >= 1.6
        sinograms = util.parallel_process_slices(sinograms, remo.remove_zinger,
                                                 [0.08, 1], ncore=ncore,
                                                 prefer="threads")
        sinograms = util.parallel_process_slices(sinograms,
                                                 remo.remove_all_stripe,
                                                 [3.0, 51, 21], ncore=ncore,
                                                 prefer="threads")
        sinograms = util.parallel_process_slices(sinograms,
                                                 filt.fresnel_filter,
                                                 [200, 1], ncore=ncore,
                                                 prefer="threads")

        t1 = timeit.default_timer()
        t_prep = t_prep + t1 - t0

    # Perform reconstruction
    t0 = timeit.default_timer()
    recon_imgs = recon_method(sinograms, center, ncore=ncore)
    t1 = timeit.default_timer()
    t_rec = t_rec + t1 - t0

    # Save output
    t0 = timeit.default_timer()
    if output_format == "hdf":
        recon_hdf[start_sino - start_slice:stop_sino - start_slice] = np.moveaxis(recon_imgs, 1, 0)
    else:
        for j in range(start_sino, stop_sino):
            out_file = output_base + "/rec_" + ("0000" + str(j))[-5:] + ".tif"
            losa.save_image(out_file, recon_imgs[:, j - start_sino, :])
    t1 = timeit.default_timer()
    t_save = t_save + t1 - t0
    t_stop = timeit.default_timer()
    print("Done slice: {0} - {1} . Time {2}".format(start_sino, stop_sino,
                                                    t_stop - t_start))

if remove_tmp_hdf:
    hdf_obj.close()
    print("4 -> Delete the temp hdf-file!")
    if os.path.isdir(output_base + "/tmp_/"):
        shutil.rmtree(output_base + "/tmp_/")

print("---------------------------------------------------------------")
print("-----------------------------Done-----------------------------")
print("Loading data cost: {0:0.2f}s".format(t_load))
print("Preprocessing cost: {0:0.2f}s".format(t_prep))
print("Reconstruction cost: {0:0.2f}s".format(t_rec))
print("Saving output cost: {0:0.2f}s".format(t_save))
print("Total time cost : {0:0.2f}s".format(t_stop - t_start))