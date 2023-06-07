"""
The following example shows how to reconstruct full size of a standard
dataset in h5/hdf/nxs format.
"""

import numpy as np
import timeit
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

file_path = "E:/Tomo_data/scan_68067.hdf"

output_base0 = "E:/output/full_reconstruction/"
folder_name = losa.make_folder_name(output_base0, name_prefix="recon", zero_prefix=3)
output_base = output_base0 + "/" + folder_name + "/"

# Optional parameters
start_slice = 10
stop_slice = 110
chunk = 100  # Number of slices to be reconstructed in one go. Adjust to suit RAM or GPU memory.
ncore = 16  # Number of cpu-core for parallel processing. Set to None for autoselecting.
output_format = "tif"  # "tif" or "hdf".
preprocessing = True  # Clean data before reconstruction.

# Give alias to a reconstruction method which is convenient for later change
# recon_method = rec.dfi_reconstruction
recon_method = rec.fbp_reconstruction
# recon_method = rec.gridrec_reconstruction # Fast cpu-method. Must install Tomopy.
# recon_method = rec.astra_reconstruction # To use iterative methods. Must install Astra.

# Provide metadata for loading hdf file
proj_path = "/entry/projections"
flat_path = "/entry/flats"
dark_path = "/entry/darks"
angle_path = "/entry/rotation_angle"

t_start = timeit.default_timer()
print("---------------------------------------------------------------")
print("-----------------------------Start-----------------------------\n")
print("1 -> Load dark-field and flat-field images, average each result")
# Load data, average flat and dark images, get data shape and rotation angles.
proj_obj = losa.load_hdf(file_path, proj_path)  # hdf object
(depth, height, width) = proj_obj.shape
flat_field = np.mean(np.asarray(losa.load_hdf(file_path, flat_path)), axis=0)
dark_field = np.mean(np.asarray(losa.load_hdf(file_path, dark_path)), axis=0)
angles = np.deg2rad(np.squeeze(np.asarray(losa.load_hdf(file_path, angle_path))))
(depth, height, width) = proj_obj.shape

print("2 -> Calculate the center-of-rotation")
# Extract sinogram at the middle for calculating the center of rotation
index = height // 2
sinogram = corr.flat_field_correction(proj_obj[:, index, :],
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
    sinograms = corr.flat_field_correction(
        proj_obj[:, start_sino:stop_sino, :],
        flat_field[start_sino:stop_sino, :],
        dark_field[start_sino:stop_sino, :])
    t1 = timeit.default_timer()
    t_load = t_load + t1 - t0

    # Perform pre-processing
    if preprocessing:
        t0 = timeit.default_timer()
        sinograms = util.apply_method_to_multiple_sinograms(sinograms,
                                                            "remove_zinger",
                                                            [0.08, 1],
                                                            ncore=ncore,
                                                            prefer="threads")
        sinograms = util.apply_method_to_multiple_sinograms(sinograms,
                                                            "remove_all_stripe",
                                                            [3.0, 51, 21],
                                                            ncore=ncore,
                                                            prefer="threads")
        sinograms = util.apply_method_to_multiple_sinograms(sinograms,
                                                            "fresnel_filter",
                                                            [200, 1],
                                                            ncore=ncore,
                                                            prefer="threads")
        t1 = timeit.default_timer()
        t_prep = t_prep + t1 - t0

    # Perform reconstruction
    t0 = timeit.default_timer()
    recon_imgs = recon_method(sinograms, center, angles=angles, ncore=ncore)
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
    sinograms = corr.flat_field_correction(
        proj_obj[:, start_sino:stop_sino, :],
        flat_field[start_sino:stop_sino, :],
        dark_field[start_sino:stop_sino, :])
    t1 = timeit.default_timer()
    t_load = t_load + t1 - t0

    # Perform pre-processing
    if preprocessing:
        t0 = timeit.default_timer()
        sinograms = util.apply_method_to_multiple_sinograms(sinograms,
                                                            "remove_zinger",
                                                            [0.08, 1],
                                                            ncore=ncore,
                                                            prefer="threads")
        sinograms = util.apply_method_to_multiple_sinograms(sinograms,
                                                            "remove_all_stripe",
                                                            [3.0, 51, 21],
                                                            ncore=ncore,
                                                            prefer="threads")
        sinograms = util.apply_method_to_multiple_sinograms(sinograms,
                                                            "fresnel_filter",
                                                            [200, 1],
                                                            ncore=ncore)
        t1 = timeit.default_timer()
        t_prep = t_prep + t1 - t0

    # Perform reconstruction
    t0 = timeit.default_timer()
    recon_imgs = recon_method(sinograms, center, angles=angles, ncore=ncore)
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
print("---------------------------------------------------------------")
print("-----------------------------Done-----------------------------")
print("Loading data cost: {0:0.2f}s".format(t_load))
print("Preprocessing cost: {0:0.2f}s".format(t_prep))
print("Reconstruction cost: {0:0.2f}s".format(t_rec))
print("Saving output cost: {0:0.2f}s".format(t_save))
print("Total time cost : {0:0.2f}s".format(t_stop - t_start))