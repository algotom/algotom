# ===========================================================================
# Author: Nghia T. Vo
# E-mail:
# Description: Examples of how to use the Algotom package.
# ===========================================================================

"""
The following example shows how to retrieve phase-shift projections from
multi-position speckle-tracking tomographic datasets (hdf/nxs format).

This example is for tomographic datasets and speckle datasets collected as
seperated hdf/nxs files.

Referring to "example_01_*.py" to know how to find key-paths and datasets
in a hdf/nxs file or using the function "get_hdf_tree" in the
loadersaver.py module
"""

import timeit
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.phase as ps


# Input examples for beamline K11, DLS
input_base = "/dls/k11/data/2022/cm31131-1/nexus/"
sam_num = np.arange(15533, 15665 + 3, 3)
ref_num = np.arange(15532, 15664 + 3, 3)
dark_field_num = 15528

# Output base
output_base = "/dls/k11/data/2022/cm31131-1/spool/speckle_phase/processed_projections/"

sam_path = []
for scan in sam_num:
    sam_path.append(losa.find_file(input_base + "/k11-" + str(scan) + "/" + "*imaging*")[0])
ref_path = []
for scan in ref_num:
    ref_path.append(losa.find_file(input_base + "/k11-" + str(scan) + "/" + "*imaging*")[0])
dark_field_path = losa.find_file(input_base + "/k11-" + str(dark_field_num) + "/" + "*imaging*")[0]
data_key = "entry/detector/detector"
# Get dark-field image (camera noise). Note that it's different to dark-signal image.
dark_field = np.mean(losa.load_hdf(dark_field_path, data_key)[:], axis=0)



# Initial parameters
get_trans_dark_signal = True
num_use = 40  # Number of speckle positions used for phase retrieval.
gpu = True # Use GPU for computing
chunk_size = 100 # Process 100 rows in one go. Adjust to suit CPU/GPU memory.
dim = 2  # Use 1D/2D-searching for finding shifts
win_size = 7  # Size of window around each pixel
margin = 10  # Searching range for finding shifts
align = True # Align if there're shifts between speckle-images and sample-images
             # Note to select ROIs without samples to calculate the shifts

print("********************************")
print("*************Start**************")
print("********************************")

# Get height, width of an image.
data_obj = losa.load_hdf(sam_path[0], data_key)
(height, width) = data_obj.shape[-2:]
# Crop data if need to
crop_top, crop_bot, crop_left, crop_right = 10, 0, 0, 0
crop = (crop_top, crop_bot, crop_left, crop_right)
height1 = height - (crop_top + crop_bot)
width1 = width - (crop_left + crop_right)
# Get number of projections
num_proj = []
for file in sam_path:
    data_obj = losa.load_hdf(file, data_key)
    num_proj.append(data_obj.shape[0])
num_proj = np.min(np.asarray(num_proj))

# Find shifts, if need to, between speckle-images and sample-images
# or between samples at the same rotation angle.

# Assign aliases to functions for convenient use
f_alias1 = losa.get_reference_sample_stacks
f_alias2 = ps.find_shift_between_image_stacks
f_alias3 = ps.find_shift_between_sample_images
f_alias4 = ps.align_image_stacks
if align:
    # Use projection 0 to align
    ref_stack, sam_stack = f_alias1(0, ref_path, sam_path, data_key, data_key,
                                crop=crop, flat_field=None,
                                dark_field=dark_field, num_use=num_use,
                                fix_zero_div=True)
    # Select a list of points in the no-sample areas for alignment
    # This needs to be changed for each experiment.
    list_ij = [np.random.randint(71, 700, size=20),
               np.random.randint(71, 150, size=20)]
    sr_shifts = f_alias2(ref_stack, sam_stack, 21, 20, gpu=True, list_ij=list_ij)
    print("Speckle-sample shifts: ")
    print(sr_shifts)

    # Align samples projections at the same rotation angle if need to.
    sam_shifts = -sr_shifts
    # Select a ROI to find the shifts between projections at the same angle
    # compared to the projection of the first dataset.
    # The ROI is pretty large (851 x 851) to detect global shifts. In such case
    # if using GPU the overhead for getting data from global memory can be high
    # list_ij2 = [height1//2, width1//2-250]
    # sam_shifts = f_alias3(ref_stack, sam_stack, sr_shifts, 851, 20, gpu=False,
    #                       list_ij=list_ij2)
    # print("Sample shifts: ")
    # print(sam_shifts)

    # Align image stacks
    (ref_stack_cr, sam_stack_cr) = f_alias4(ref_stack, sam_stack, sr_shifts, sam_shifts)
    # Check results to make sure
    for i in range(num_use):
        name = ("0000" + str(i))[-5:]
        losa.save_image(output_base + "/aligned/ref/img_" + name + ".tif", ref_stack_cr[i])
        losa.save_image(output_base + "/aligned/sam/img_" + name + ".tif", sam_stack_cr[i])


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
f_alias5 = losa.get_reference_sample_stacks
f_alias6 = ps.retrieve_phase_based_speckle_tracking
f_alias7 = ps.get_transmission_dark_field_signal
t0 = timeit.default_timer()
for i in range(num_proj):
    ref_stack, sam_stack = f_alias5(i, ref_path, sam_path, data_key, data_key,
                                    crop=crop, flat_field=None,
                                    dark_field=dark_field, num_use=num_use,
                                    fix_zero_div=True)
    if align:
        ref_stack, sam_stack = f_alias4(ref_stack, sam_stack, sr_shifts, sam_shifts)
    x_shifts, y_shifts, phase = f_alias6(ref_stack, sam_stack, dim=dim,
                                         win_size=win_size, margin=margin,
                                         method="diff", size=3, gpu=gpu,
                                         block=(16, 16), ncore=None, norm=True,
                                         norm_global=True, chunk_size=chunk_size,
                                         surf_method="SCS", return_shift=True)
    phase_hdf[i] = phase
    name = ("0000" + str(i))[-5:]
    losa.save_image(output_base + "/phase/img_" + name + ".tif", phase)
    if get_trans_dark_signal:
        trans, dark = f_alias7(ref_stack, sam_stack, x_shifts, y_shifts,
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
