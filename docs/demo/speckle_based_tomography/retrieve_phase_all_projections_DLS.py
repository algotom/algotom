# ===========================================================================
# Author: Nghia T. Vo
# E-mail:
# Description: Examples of how to use the Algotom package.
# ===========================================================================

"""
The following example shows how to retrieve phase-shift projections from
multi-position speckle-tracking tomographic datasets (hdf/nxs format).

This example is for tomographic files having conventional metadata used at
Diamond Light Source where sample-images and speckle-images are in the same
dataset distinguished by a metadata named "image_key".

Referring to "example_01_*.py" to know how to find key-paths and datasets
in a hdf/nxs file or using the function "get_hdf_tree" in the
loadersaver.py module
"""

import timeit
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.phase as ps

# Input examples for beamline I12, I13, B16, DLS
input_base = "/dls/beamline/data/2022/visit/rawdata/speckle_tomo_00009/"
output_base = "/dls/beamline/data/2022/visit/spool/speckle_tomo_00009/processed_projections/"

# Initial parameters
get_trans_dark_signal = True
num_use = 40  # Number of speckle positions used for phase retrieval.
gpu = True # Use GPU for computing
chunk_size = 100 # Process 100 rows in one go. Adjust to suit CPU/GPU memory.
dim = 2  # Use 1D/2D-searching for finding shifts
win_size = 7  # Size of window around each pixel
margin = 10  # Searching range for finding shifts

print("********************************")
print("*************Start**************")
print("********************************")

list_file = losa.find_file(input_base + "/*.nxs")
# Get keys to datasets
data_key = losa.find_hdf_key(list_file[0], "data/data")[0][0]
image_key = losa.find_hdf_key(list_file[0], "image_key")[0][-1]
# Get height, width of an image.
data_obj = losa.load_hdf(list_file[0], data_key)
(height, width) = data_obj.shape[-2:]
# Crop data if need to
crop_top, crop_bot, crop_left, crop_right = 10, 0, 0, 0
crop = (crop_top, crop_bot, crop_left, crop_right)
height1 = height - (crop_top + crop_bot)
width1 = width - (crop_left + crop_right)
# Get number of projections
num_proj = []
for file in list_file:
    int_keys = losa.load_hdf(file, image_key)[:]
    num_proj1 = len(np.squeeze(np.asarray(np.where(int_keys == 0.0))))
    num_proj.append(num_proj1)
num_proj = np.min(np.asarray(num_proj))

# Open hdf stream to save data
phase_hdf = losa.open_hdf_stream(output_base + "/phase.hdf",
                                 (num_proj, height1, width1),
                                 key_path="entry/data")
if get_trans_dark_signal:
    trans_hdf = losa.open_hdf_stream(output_base + "/transmission.hdf",
                                     (num_proj, height1, width1),
                                     key_path="entry/data")
    dark_hdf = losa.open_hdf_stream(output_base + "/dark_signal.hdf",
                                    (num_proj, height1, width1),
                                    key_path="entry/data")
# Assign aliases to functions for convenient use
f_alias1 = losa.get_reference_sample_stacks_dls
f_alias2 = ps.retrieve_phase_based_speckle_tracking
f_alias3 = ps.get_transmission_dark_field_signal
t0 = timeit.default_timer()
for i in range(num_proj):
    ref_stack, sam_stack = f_alias1(i, list_file, data_key=data_key,
                                    image_key=image_key, crop=crop,
                                    flat_field=None, dark_field=None,
                                    num_use=num_use, fix_zero_div=True)
    x_shifts, y_shifts, phase = f_alias2(ref_stack, sam_stack, dim=dim,
                                         win_size=win_size, margin=margin,
                                         method="diff", size=3, gpu=gpu,
                                         block=(16, 16), ncore=None, norm=True,
                                         norm_global=True, chunk_size=chunk_size,
                                         surf_method="SCS", return_shift=True)
    phase_hdf[i] = phase
    name = ("0000" + str(i))[-5:]
    losa.save_image(output_base + "/phase/img_" + name + ".tif", phase)
    if get_trans_dark_signal:
        trans, dark = f_alias3(ref_stack, sam_stack, x_shifts, y_shifts,
                               win_size, ncore=None)
        trans_hdf[i] = trans
        dark_hdf[i] = dark
        losa.save_image(output_base + "/transmission/img_" + name + ".tif",
                        trans)
        losa.save_image(output_base + "/dark/img_" + name + ".tif",
                        dark)
    t1 = timeit.default_timer()
    print("Done projection {0}. Time cost: {1} ".format(i, t1 - t0))
t1 = timeit.default_timer()
print("\n********************************")
print("All done!!!!!!!!! Total time: {}".format(t1 - t0))
print("********************************")
