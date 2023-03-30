# ===========================================================================
# Author: Nghia T. Vo
# E-mail:
# Description: Examples of how to use the Algotom package.
# ===========================================================================

"""
The following example shows how to reconstruct slices from phase-shift
projections retrieved by the previous step.

Referring to "example_01_*.py" to know how to find key-paths and datasets
in a hdf/nxs file, or using the function "get_hdf_tree" in the
loadersaver.py module
"""

import timeit
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.calculation as calc
import algotom.prep.filtering as filt
import algotom.prep.removal as rem
import algotom.rec.reconstruction as rec


phase_path = "C:/user/processed_data/phase.hdf"
dark_path = "C:/user/processed_data/dark_signal.hdf"
trans_path = "C:/user/processed_data/transmission.hdf"

output_base = "C:/user/processed_data/reconstruction/"

# Provide hdf-keys
phase_hdf = losa.load_hdf(phase_path, "entry/data")
dark_hdf = losa.load_hdf(dark_path, "entry/data")
trans_hdf = losa.load_hdf(trans_path, "entry/data")

(num_proj, height, width) = phase_hdf.shape

start_slice = 50
stop_slice = height - 1
step = 100

# Find the center of rotation using a transmission sinogram.
sino_center = trans_hdf[:, height // 2, :]
center = calc.find_center_vo(sino_center)
print("Center of rotation {}".format(center))

fluct_correct = False   # Using double-wedge filter to correct the
                        # fluctuation of phase sinograms
artifact_rem = False  # Remove ring artifacts
t0 = timeit.default_timer()
for i in np.arange(start_slice, stop_slice + 1, step):
    name = ("0000" + str(i))[-5:]
    sino_phase = phase_hdf[:, i, :]
    if fluct_correct:
        sino_phase = filt.double_wedge_filter(sino_phase, center)
    sino_trans = trans_hdf[:, i, :]
    sino_dark = dark_hdf[:, i, :]
    if artifact_rem:
        sino_phase = rem.remove_stripe_based_wavelet_fft(sino_phase, 5, 1.0)
        sino_trans = rem.remove_all_stripe(sino_trans, 2.0, 51, 17)
        sino_dark = rem.remove_all_stripe(sino_dark, 2.0, 51, 17)
    # Change to CPU methods (DFI or gridrec) if GPU not available
    rec_phase = rec.fbp_reconstruction(sino_phase, center, apply_log=False,
                                        filter_name="hann")
    rec_trans = rec.fbp_reconstruction(sino_trans, center, apply_log=True,
                                        filter_name="hann")
    rec_dark = rec.fbp_reconstruction(sino_dark, center, apply_log=True,
                                       filter_name="hann")
    losa.save_image(output_base + "/phase/rec_" + name + ".tif", rec_phase)
    losa.save_image(output_base + "/transmission/rec_" + name + ".tif",
                    rec_trans)
    losa.save_image(output_base + "/dark_signal/rec_" + name + ".tif",
                    rec_dark)
    print("Done slice: {}".format(i))
t1 = timeit.default_timer()
print("All done !!!. Time cost {}".format(t1 - t0))
