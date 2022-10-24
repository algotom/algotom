"""
The script used to reconstruct single slice of phase-shift image, transmission
image, and dark-signal image from raw data of speckle-based tomographic
datasets demonstrated in the paper:
https://doi.org/10.1117/12.2636834
"""

import timeit
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.phase as ps
import algotom.prep.calculation as calc
import algotom.prep.filtering as filt
import algotom.prep.removal as rem
import algotom.rec.reconstruction as reco


input_base = "/dls/i12/data/2022/cm31131-1/rawdata/"
sam_num = np.arange(81316, 81316 + 20, 1)
ref_num = np.arange(81337, 81337 + 20, 1)
dark_field_num = 81336
flat_field_num = 81357

# Output base
output_base = "/dls/i12/data/2022/cm31131-1/processing/single_slice_recon/"


sam_path = []
data_key = "entry/data/data"
for scan in sam_num:
    sam_path.append(losa.find_file(input_base + "/pco*" + str(scan)+"*.hdf")[0])
ref_path = []
for scan in ref_num:
    ref_path.append(input_base + "/" + str(scan) + "/projections/")
dark_field_path = input_base + "/" + str(dark_field_num) + "/projections/"
flat_field_path = input_base + "/" + str(flat_field_num) + "/projections/"
dark_field = losa.get_image_stack(None, dark_field_path, average=True)
flat_field = losa.get_image_stack(None, flat_field_path, average=True)

# Initial parameters
crop_top, crop_bot, crop_left, crop_right = 160, 260, 0, 0
dark_signal = True
num_use = None  # Number of speckle positions used for phase retrieval.
gpu = True  # Use GPU for computing
chunk_size = 100  # Process 100 rows in one go. Adjust to suit CPU/GPU memory.
win_size = 7  # Size of window around each pixel
margin = 10  # Searching range for finding shifts
align = True  # Align if there are shifts between speckle-images and sample-images
              # Note to select ROIs without samples to calculate the shifts
ncore = None  # Number of cpu
find_shift = "umpa"
# find_shift = "correl"
dim = 2  # Use 1D/2D-searching for finding shifts
slice_idx = 1200  # Slice to reconstruct, referred to original size image.

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

# Get reference images
ref_stack = []
for path in ref_path:
    mat = losa.get_image_stack(None, path, average=True, crop=crop,
                               flat_field=flat_field,
                               dark_field=dark_field)
    ref_stack.append(mat)
ref_stack = np.asarray(ref_stack)

if align:
    # Use projections at 0-degree to find shifts.
    sam_stack = losa.get_image_stack(0, sam_path, data_key, average=False,
                                     crop=crop, flat_field=flat_field,
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


# Define the ROI area to retrieve phase around the given slice (row) index
# Referring to the original size of image.
extra_edge = 2
crop_top1 = np.clip(slice_idx - 2 * (margin + extra_edge), crop_top, None)
crop_bot1 = np.clip(height - (slice_idx + 2 * (margin + extra_edge)), crop_bot, None)
crop_left1 = 0 + crop_left
crop_right1 = 0 + crop_right
crop1 = (crop_top1, crop_bot1, crop_left1, crop_right1)

if slice_idx >= (height1 - 2 * (margin + extra_edge)):
    raise ValueError("Selected slice index is out of range!!!")

# To crop reference-images again after loading, cropping, and alignment.
top1 = crop_top1 - crop_top
bot1 = height1 - (crop_bot1 - crop_bot)
left1 = crop_left1 - crop_left
rigth1 = width1 - (crop_right1 - crop_right)
ref_stack_roi = ref_stack[:, top1:bot1, left1:rigth1]

t0 = timeit.default_timer()
sino_phase = []
if dark_signal is True:
    sino_trans = []
    sino_dark = []
name = ("0000" + str(slice_idx))[-5:]

# Assign aliases to functions for convenient use

for i in range(0, num_proj):
    sam_stack_roi = losa.get_image_stack(i, sam_path, data_key, average=False,
                                     crop=crop1, flat_field=flat_field,
                                     dark_field=dark_field,
                                     num_use=num_use, fix_zero_div=True)
    phase, trans, dark = ps.retrieve_phase_based_speckle_tracking(
        ref_stack_roi, sam_stack_roi,
        find_shift=find_shift,
        filter_name=None,
        dark_signal=True, dim=dim, win_size=win_size,
        margin=margin, method="diff", size=3,
        gpu=gpu, block=(16, 16),
        ncore=ncore, norm=True,
        norm_global=False, chunk_size=chunk_size,
        surf_method="SCS",
        correct_negative=True, pad=100,
        return_shift=False)
    mid = phase.shape[0] // 2
    sino_phase.append(phase[mid])
    if dark_signal:
        sino_trans.append(trans[mid])
        sino_dark.append(dark[mid])
    t1 = timeit.default_timer()
    print("Done projection {0}. Time cost: {1} ".format(i, t1 - t0))

print("Done phase retrieval !!!")
sino_phase = np.asarray(sino_phase)
if dark_signal:
    sino_trans = np.asarray(sino_trans)
    sino_dark = np.asarray(sino_dark)
    losa.save_image(output_base + "/sinogram/trans_" + name + ".tif", sino_trans)
    losa.save_image(output_base + "/sinogram/dark_" + name + ".tif", sino_dark)
losa.save_image(output_base + "/sinogram/phase_" + name + ".tif", sino_phase)

if dark_signal:
    center = calc.find_center_vo(sino_trans)
else:
    center = calc.find_center_vo(sino_phase)
print("Center of rotation {}".format(center))

# Correct the fluctuation of the phase image.
sino_phase = filt.double_wedge_filter(sino_phase, center)

# Remove ring
sino_phase = rem.remove_stripe_based_wavelet_fft(sino_phase, 5, 1.0)
sino_trans = rem.remove_all_stripe(sino_trans, 1.5, 71, 31)
sino_dark = rem.remove_all_stripe(sino_dark, 1.5, 71, 21)

# Reconstruction
rec_phase = reco.fbp_reconstruction(sino_phase, center, apply_log=False, filter_name="hann")
rec_trans = reco.fbp_reconstruction(sino_trans, center, apply_log=True, filter_name="hann")
rec_dark = reco.fbp_reconstruction(sino_dark, center, apply_log=True, filter_name="hann")

# Save results
losa.save_image(output_base + "/reconstruction/phase.tif", rec_phase)
losa.save_image(output_base + "/reconstruction/trans.tif", rec_trans)
losa.save_image(output_base + "/reconstruction/dark.tif", rec_dark)

t1 = timeit.default_timer()
print("\n********************************")
print("All done!!!!!!!!! Total time: {}".format(t1 - t0))
print("********************************")
