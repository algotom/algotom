# ===========================================================================
# Author: Nghia T. Vo
# E-mail:
# Description: Examples of how to use the Algotom package.
# ===========================================================================

"""
The following example shows how to retrieve phase image from two stacks of
speckle-images and sample-images.

Referring to "example_01_*.py" to know how to find key-paths and datasets
in a hdf/nxs file or using the function "get_hdf_tree" in the
loadersaver.py module
"""

import timeit
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.phase as ps

# If data is hdf-format
speckle_stack = losa.load_hdf("C:/user/data/ref_stack.hdf", "entry/data")
sample_stack = losa.load_hdf("C:/user/data/sam_stack.hdf", "entry/data")

#  # If data is tif-format.
# speckle_stack = losa.get_tif_stack("C:/user/data/ref/")
# sample_stack = losa.get_tif_stack("C://user/data/sam/")

output_base = "C:/user/output/"

num_use = 20  # Number of speckle positions used for phase retrieval.
gpu = True # Use GPU for computing
chunk_size = 100 # Process 100 rows in one go. Adjust to suit CPU/GPU memory.
win_size = 7  # Size of window around each pixel
margin = 10  # Searching range for finding shifts
ncore = None  # Number of cpu
# find_shift = "umpa"
find_shift = "correl"
dim = 2  # Use 1D/2D-searching for finding shifts

speckle_stack = speckle_stack[:num_use, :, :]  # Data shape: 40 x 2560 x 2160
sample_stack = sample_stack[:num_use, :, :]


t0 = timeit.default_timer()
# dim=2 is slow (>45 mins) if running on CPU.
x_shifts, y_shifts, phase, trans, dark = ps.retrieve_phase_based_speckle_tracking(
    speckle_stack, sample_stack,
    find_shift=find_shift,
    filter_name=None,
    dark_signal=True, dim=dim, win_size=win_size,
    margin=margin, method="diff", size=3,
    gpu=gpu, block=(16, 16),
    ncore=ncore, norm=True,
    norm_global=False, chunk_size=chunk_size,
    surf_method="SCS",
    correct_negative=True, pad=100,
    return_shift=True)

t1 = timeit.default_timer()
print("Time cost for retrieve phase image:  {}".format(t1 - t0))

losa.save_image(output_base + "/x_shifts.tif", x_shifts)
losa.save_image(output_base + "/y_shifts.tif", y_shifts)
losa.save_image(output_base + "/phase.tif", phase)
losa.save_image(output_base + "/trans.tif", trans)
losa.save_image(output_base + "/dark.tif", dark)
print("Al done !!!")
