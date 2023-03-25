# ===========================================================================
# Author: Nghia T. Vo
# E-mail:
# Description: Examples of how to use the Algotom package.
# ===========================================================================

"""
The following example shows how to reconstruct a single phase-sinogram
retrieved by using a chunk of projection-image rows from multi-position
speckle-tracking tomographic datasets (hdf/nxs format).

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
import algotom.prep.calculation as calc
import algotom.prep.filtering as filt
import algotom.prep.removal as rem
import algotom.rec.reconstruction as reco


# Input examples for beamline I12, I13, B16, DLS
input_base = "/dls/beamline/data/2022/visit/rawdata/speckle_tomo_00009/"
output_base = "/dls/beamline/data/2022/visit/spool/speckle_tomo_00009/single_slice/"

# Initial parameters
dark_signal = True
num_use = None  # Number of speckle positions used for phase retrieval.
gpu = True # Use GPU for computing
chunk_size = 100 # Process 100 rows in one go. Adjust to suit CPU/GPU memory.
win_size = 7  # Size of window around each pixel
margin = 10  # Searching range for finding shifts
ncore = None  # Number of cpu
# find_shift = "umpa"
find_shift = "correl"
dim = 1  # Use 1D/2D-searching for finding shifts
slice_idx = 1200 # Slice to reconstruct

print("********************************")
print("*************Start**************")
print("********************************")

list_file = losa.find_file(input_base + "/*.nxs")
# Get keys to datasets
data_key = losa.find_hdf_key(list_file[0], "data/data")[0][0]
image_key = losa.find_hdf_key(list_file[0], "image_key")[0][-1]
data_obj = losa.load_hdf(list_file[0], data_key)
(height, width) = data_obj.shape[-2:]

# Define the ROI area to retrieve phase around the given slice (row) index
crop_top = slice_idx - 2 * margin
crop_bot = height - (slice_idx + 2 * margin)
crop_left = 0
crop_right = 0
crop = (crop_top, crop_bot, crop_left, crop_right)

# Get number of projections.
num_proj = []
for file in list_file:
    int_keys = losa.load_hdf(file, image_key)[:]
    num_proj1 = len(np.squeeze(np.asarray(np.where(int_keys == 0.0))))
    num_proj.append(num_proj1)
num_proj = np.min(np.asarray(num_proj))


t0 = timeit.default_timer()
sino_phase = []
if dark_signal:
    sino_trans = []
    sino_dark = []
name = ("0000" + str(slice_idx))[-5:]
for i in range(num_proj):
    ref_stack, sam_stack = losa.get_reference_sample_stacks_dls(i, list_file, data_key=data_key,
                                                                image_key=image_key, crop=crop,
                                                                flat_field=None, dark_field=None,
                                                                num_use=num_use, fix_zero_div=True)
    if dark_signal:
        phase, trans, dark = ps.retrieve_phase_based_speckle_tracking(
            ref_stack, sam_stack,
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
    else:
        phase = ps.retrieve_phase_based_speckle_tracking(
            ref_stack, sam_stack,
            find_shift=find_shift,
            filter_name=None,
            dark_signal=False, dim=dim, win_size=win_size,
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
sino_phase1 = filt.double_wedge_filter(sino_phase, center)
losa.save_image(output_base + "/sinogram/phase_corr.tif", sino_phase1)

# Remove ring
sino_phase = rem.remove_stripe_based_wavelet_fft(sino_phase, 5, 1.0)
sino_trans = rem.remove_all_stripe(sino_trans, 1.5, 71, 31)
sino_dark = rem.remove_all_stripe(sino_dark, 1.5, 71, 21)

# Reconstruction
rec_phase = reco.fbp_reconstruction(sino_phase, center, apply_log=False, filter_name="hann")
rec_phase1 = reco.fbp_reconstruction(sino_phase1, center, apply_log=False, filter_name="hann")
rec_trans = reco.fbp_reconstruction(sino_trans, center, apply_log=True, filter_name="hann")
rec_dark = reco.fbp_reconstruction(sino_dark, center, apply_log=True, filter_name="hann")

# Save results
losa.save_image(output_base + "/reconstruction/phase.tif", rec_phase)
losa.save_image(output_base + "/reconstruction/phase_corr.tif", rec_phase1)
losa.save_image(output_base + "/reconstruction/trans.tif", rec_trans)
losa.save_image(output_base + "/reconstruction/dark.tif", rec_dark)

t1 = timeit.default_timer()
print("\n********************************")
print("All done!!!!!!!!! Total time: {}".format(t1 - t0))
print("********************************")
