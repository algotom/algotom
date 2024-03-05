import time
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.calculation as calc
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.rec.reconstruction as rec

"""This script is used to reconstruct a few slices across image height"""

proj_file = "/tomography/raw_data/scan_00008/scan_00008.nxs"
flat_file = "/tomography/raw_data/scan_00009/flat_00000.hdf"
dark_file = "/tomography/raw_data/scan_00009/dark_00000.hdf"

output_base = "/tomography/tmp/scan_00008/rec_few_slices/"

start_slice = 500
stop_slice = 2000
step_slice = 100
crop_left = 0
crop_right = 0

center = 0.0  # Auto-determination


print("====================================================================\n")
print("         Run the script for reconstructing a few slices               ")
print("         Time: {}".format(time.ctime(time.time())))
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

for idx in range(start_slice, stop_slice + 1, step_slice):
    # Get a sinogram and perform flat-field correction
    sinogram = corr.flat_field_correction(proj_obj[:, idx, left:right],
                                          flat_field[idx, left:right],
                                          dark_field[idx, left:right])
    # Apply zinger removal
    # sinogram = remo.remove_zinger(sinogram, 0.08)

    # Apply ring removal
    sinogram = remo.remove_stripe_based_normalization(sinogram, 15)
    # sinogram = remo.remove_stripe_based_sorting(sinogram, 21)
    # sinogram = remo.remove_all_stripe(sinogram, 2.0, 51, 21)

    # Apply contrast enhancement/denoising
    sinogram = filt.fresnel_filter(sinogram, 100)

    # # Perform reconstruction
    # rec_img = rec.fbp_reconstruction(sinogram, center, angles=angles,
    #                                     apply_log=True, gpu=True)
    # # Using an iterative gpu-based method, available in Astra Toolbox
    # rec_img = rec.astra_reconstruction(sinogram, center, angles=angles,
    #                                     method="SIRT_CUDA", apply_log=True,
    #                                     num_iter=100)

    # Using a fast cpu-based method, available in Tomopy.
    rec_img = rec.gridrec_reconstruction(sinogram, center, angles=angles,
                                            apply_log=True)
    out_file = output_base + "/rec_" + ("00000" + str(idx))[-5:] + ".tif"
    print("- Done slice {}".format(idx))
    losa.save_image(out_file, rec_img)
print("====================================================================\n")
print("All done! Output is at\n {}".format(output_base))
print("====================================================================\n")
