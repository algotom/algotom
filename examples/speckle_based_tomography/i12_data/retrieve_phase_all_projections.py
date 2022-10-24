"""
The script used to retrieve phase-shifts of all projections
of speckle-based tomographic datasets demonstrated in the paper:
https://doi.org/10.1117/12.2636834
"""

import timeit
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.phase as ps

input_base = "/dls/i12/data/2022/cm31131-1/rawdata/"
sam_num = np.arange(81316, 81316 + 20, 1)
ref_num = np.arange(81337, 81337 + 20, 1)
dark_field_num = 81336
flat_field_num = 81357

output_base = "/dls/i12/data/2022/cm31131-1/processing/all_projections/"

sam_path = []
data_key = "entry/data/data"
for scan in sam_num:
    sam_path.append(
        losa.find_file(input_base + "/pco*" + str(scan) + "*.hdf")[0])
ref_path = []
for scan in ref_num:
    ref_path.append(input_base + "/" + str(scan) + "/projections/")
dark_field_path = input_base + "/" + str(dark_field_num) + "/projections/"
flat_field_path = input_base + "/" + str(flat_field_num) + "/projections/"
dark_field = losa.get_image_stack(None, dark_field_path, average=True)
flat_field = losa.get_image_stack(None, flat_field_path, average=True)

# Initial parameters
crop_top, crop_bot, crop_left, crop_right = 160, 260, 0, 0
dark_signal = False
num_use = None  # Number of speckle positions used for phase retrieval.
gpu = True  # Use GPU for computing
chunk_size = 100  # Process 100 rows in one go. Adjust to suit CPU/GPU memory.
win_size = 7  # Size of window around each pixel
margin = 10  # Searching range for finding shifts
align = True  # Align if there're shifts between speckle-images and sample-images
# Note to select ROIs without samples to calculate the shifts
ncore = None # Set number of core if using CPU.
find_shift = "umpa" # Using the UMPA method
dim = 2  # Specify 1D/2D-search if using a correlation-based method.

print("********************************")
print("*************Start**************")
print("********************************")

# Get height, width of an image.
data_obj = losa.load_hdf(sam_path[0], data_key)
(height, width) = data_obj.shape[-2:]
crop = (crop_top, crop_bot, crop_left, crop_right)
height1 = height - (crop_top + crop_bot)
width1 = width - (crop_left + crop_right)
# Get number of projections
num_proj = []
for file in sam_path:
    data_obj = losa.load_hdf(file, data_key)
    num_proj.append(data_obj.shape[0])
num_proj = np.min(np.asarray(num_proj))

ref_stack = []
for path in ref_path:
    mat = losa.get_image_stack(None, path, average=True, crop=crop,
                               flat_field=flat_field,
                               dark_field=dark_field)
    ref_stack.append(mat)
ref_stack = np.asarray(ref_stack)

if align:
    sam_stack = losa.get_image_stack(0, sam_path, data_key, average=False,
                                     crop=(crop_top, crop_bot, crop_left,
                                           crop_right), flat_field=flat_field,
                                     dark_field=dark_field,
                                     num_use=num_use, fix_zero_div=True)
    # Select a list of points in the no-sample areas for alignment
    # This needs to be changed for each experiment.
    list_ij = [np.random.randint(100, 1000, size=10),
               np.random.randint(70, 350, size=10)]
    sr_shifts = ps.find_shift_between_image_stacks(ref_stack, sam_stack, 61,
                                                   10, gpu=False,
                                                   list_ij=list_ij,
                                                   method="mixed")
    print("Speckle-sample shifts (x, y): ")
    print(sr_shifts)
    print("==============================")
    ref_stack = ps.align_image_stacks(ref_stack, sam_stack, sr_shifts, None)[0]

# Open hdf stream to save data
x_shifts_hdf = losa.open_hdf_stream(output_base + "/x_shifts.hdf",
                                    (num_proj, height1, width1),
                                    key_path="entry/data")
y_shifts_hdf = losa.open_hdf_stream(output_base + "/y_shifts.hdf",
                                    (num_proj, height1, width1),
                                    key_path="entry/data")
phase_hdf = losa.open_hdf_stream(output_base + "/phase.hdf",
                                 (num_proj, height1, width1),
                                 key_path="entry/data")
trans_hdf = losa.open_hdf_stream(output_base + "/transmission.hdf",
                                 (num_proj, height1, width1),
                                 key_path="entry/data")
dark_hdf = losa.open_hdf_stream(output_base + "/dark_signal.hdf",
                                (num_proj, height1, width1),
                                key_path="entry/data")

for proj in range(0, num_proj):
    sam_stack = losa.get_image_stack(proj, sam_path, data_key, average=False,
                                     crop=crop, flat_field=flat_field,
                                     dark_field=dark_field,
                                     num_use=num_use, fix_zero_div=True)
    t0 = timeit.default_timer()
    x_shifts, y_shifts, phase, trans, dark = ps.retrieve_phase_based_speckle_tracking(
                                                ref_stack, sam_stack,
                                                find_shift=find_shift,
                                                filter_name=None, dark_signal=True,
                                                dim=dim, win_size=win_size,
                                                margin=margin, method="diff", size=3,
                                                gpu=gpu, block=(16, 16),
                                                ncore=ncore, norm=True,
                                                norm_global=False, chunk_size=chunk_size,
                                                surf_method="SCS",
                                                correct_negative=True, pad=100,
                                                return_shift=True)

    phase_hdf[proj] = phase
    trans_hdf[proj] = trans
    dark_hdf[proj] = dark
    x_shifts_hdf[proj] = x_shifts
    y_shifts_hdf[proj] = y_shifts
    if proj % 50 == 0:
        name = ("0000" + str(proj))[-5:]
        losa.save_image(output_base + "/phase/img_" + name + ".tif", phase)
        losa.save_image(output_base + "/trans/img_" + name + ".tif", trans)
        losa.save_image(output_base + "/dark/img_" + name + ".tif", dark)
        losa.save_image(output_base + "/x_shifts/img_" + name + ".tif", x_shifts)
        losa.save_image(output_base + "/y_shifts/img_" + name + ".tif", y_shifts)
    t1 = timeit.default_timer()
    print("Done projection {0}. Time cost: {1} ".format(proj, t1 - t0))

t1 = timeit.default_timer()
print("********************************")
print("All done!!!!!!!!! Total time: {}".format(t1 - t0))
print("********************************")
