import time
import timeit
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.util.utility as util

"""
This script is used to find the center of rotation manually:
https://algotom.readthedocs.io/en/latest/toc/section4/section4_5.html#finding-the-center-of-rotation
"""

proj_file = "/tomography/raw_data/scan_00008/scan_00008.nxs"
flat_file = "/tomography/raw_data/scan_00009/flat_00000.hdf"
dark_file = "/tomography/raw_data/scan_00009/dark_00000.hdf"

output_base = "/tomography/tmp/scan_00008/find_center/"

slice_index = 1000

start_center = 1550
stop_center = 1650
step_center = 2
crop_left = 0
crop_right = 0

t_start = timeit.default_timer()
print("\n====================================================================")
print("            Run the script for finding center manually")
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
    raise ValueError(f"Index is out of the range [0, {height - 1}]")
if start_center < 0 or start_center > width1 - 1:
    raise ValueError(f"Incorrect starting value, given image-width: {width1}")
if stop_center < 1 or stop_center > width1:
    raise ValueError(f"Incorrect stopping value, given image-width: {width1}")

sinogram = corr.flat_field_correction(proj_obj[:, slice_index, left:right],
                                      flat_field[slice_index, left:right],
                                      dark_field[slice_index, left:right])
# Apply zinger removal
# sinogram = remo.remove_zinger(sinogram, 0.08)

# Apply ring removal
sinogram = remo.remove_stripe_based_normalization(sinogram, 15)
# sinogram = remo.remove_stripe_based_sorting(sinogram, 21)
# sinogram = remo.remove_all_stripe(sinogram, 2.0, 51, 21)

# Apply contrast enhancement
sinogram = filt.fresnel_filter(sinogram, 100)


## Visual using sinogram
# util.find_center_visual_sinograms(sinogram, output_base, start_center,
#                                     stop_center, step=step_center, zoom=1.0)

## Visual using reconstructed image
util.find_center_visual_slices(sinogram, output_base, start_center,
                               stop_center, step_center, zoom=1.0,
                               method="gridrec", gpu=False, angles=angles,
                               ratio=1.0, filter_name=None)

t_stop = timeit.default_timer()
print("====================================================================\n")
print("All done! Time cost {}".format(t_stop - t_start))
print("Output: {}".format(output_base))
print("====================================================================\n")
