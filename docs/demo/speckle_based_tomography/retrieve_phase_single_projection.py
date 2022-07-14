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

num_use = 40  # Use 40 speckle-positions for calculation
speckle_stack = speckle_stack[:40, :, :]  # Data shape: 40 x 2560 x 2160
sample_stack = sample_stack[:40, :, :]

t0 = timeit.default_timer()
# dim=2 is slow (>45 mins) if running on CPU.
f_alias1 = ps.retrieve_phase_based_speckle_tracking  # For convenient use
x_shifts, y_shifts, phase = f_alias1(speckle_stack, sample_stack, dim=1,
                                     win_size=7, margin=10, method="diff",
                                     size=3, gpu=True, block=(16, 16),
                                     ncore=None, norm=False,
                                     norm_global=True, chunk_size=None,
                                     surf_method="SCS",
                                     correct_negative=True,
                                     return_shift=True)

t1 = timeit.default_timer()
print("Time cost for retrieve phase image:  {}".format(t1 - t0))

t0 = timeit.default_timer()
f_alias2 = ps.get_transmission_dark_field_signal  # For convenient use
trans, dark = f_alias2(speckle_stack, sample_stack,
                       x_shifts, y_shifts, 7, ncore=None)
t1 = timeit.default_timer()
print("Time cost for getting transmission + dark-signal:  {}".format(t1 - t0))
losa.save_image(output_base + "/x_shifts.tif", x_shifts)
losa.save_image(output_base + "/y_shifts.tif", y_shifts)
losa.save_image(output_base + "/phase.tif", phase)
losa.save_image(output_base + "/trans.tif", trans)
losa.save_image(output_base + "/dark.tif", -np.log(dark))
print("Al done !!!")
