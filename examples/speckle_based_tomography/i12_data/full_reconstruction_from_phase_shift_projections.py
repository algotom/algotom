"""
The script was used to reconstruct the full size of tomographic data from
phase-shift projections, transmission projections, and dark-signal projection
of speckle-based tomographic datasets demonstrated in the paper:
https://doi.org/10.1117/12.2636834
"""
import timeit
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.calculation as calc
import algotom.prep.filtering as filt
import algotom.prep.removal as rem
import algotom.rec.reconstruction as rec


input_base = "/dls/i12/data/2022/cm31131-1/processing/all_projections/"
output_base = "/dls/i12/data/2022/cm31131-1/processing/reconstruction/"

phase_path = input_base + "/phase.hdf"
dark_path = input_base + "/dark_signal.hdf"
trans_path = input_base + "/transmission.hdf"

phase_hdf = losa.load_hdf(phase_path, "entry/data")
dark_hdf = losa.load_hdf(dark_path, "entry/data")
trans_hdf = losa.load_hdf(trans_path, "entry/data")

(num_proj, height, width) = phase_hdf.shape

start_slice = 0
stop_slice = height
slice_chunk = 32  # Number of sinograms loaded in one go
num_slice = stop_slice - start_slice

rec_phase_hdf = losa.open_hdf_stream(output_base + "/rec_phase.hdf",
                                     (num_slice, width, width),
                                     key_path="entry/data")
rec_trans_hdf = losa.open_hdf_stream(output_base + "/rec_transmission.hdf",
                                     (num_slice, width, width),
                                     key_path="entry/data")
rec_dark_hdf = losa.open_hdf_stream(output_base + "/rec_dark_signal.hdf",
                                    (num_slice, width, width),
                                    key_path="entry/data")

# Find the center of rotation using a transmission sinogram.
sino_center = trans_hdf[:, height // 2, :]
center = calc.find_center_vo(sino_center)
print("Center of rotation {}".format(center))

total_slice = stop_slice - start_slice
offset = start_slice
if slice_chunk > total_slice:
    slice_chunk = total_slice
num_iter = total_slice // slice_chunk
num_rest = total_slice - num_iter * slice_chunk

t0 = timeit.default_timer()
for i in range(num_iter):
    start_sino = i * slice_chunk + offset
    stop_sino = start_sino + slice_chunk
    sino_phase_stack = phase_hdf[:, start_sino:stop_sino, :]
    sino_trans_stack = trans_hdf[:, start_sino:stop_sino, :]
    sino_dark_stack = dark_hdf[:, start_sino:stop_sino, :]
    # Save the results to tif images
    for j in range(start_sino, stop_sino):
        sino_phase = sino_phase_stack[:, j - start_sino, :]
        sino_trans = sino_trans_stack[:, j - start_sino, :]
        sino_dark = sino_dark_stack[:, j - start_sino, :]

        sino_phase = filt.double_wedge_filter(sino_phase, center)
        sino_phase = rem.remove_stripe_based_wavelet_fft(sino_phase, 5, 1)
        sino_trans = rem.remove_all_stripe(sino_trans, 1.5, 71, 31)
        sino_dark = rem.remove_all_stripe(sino_dark, 1.5, 71, 21)

        rec_phase = rec.fbp_reconstruction(sino_phase, center,
                                            apply_log=False, filter_name="hann")
        rec_trans = rec.fbp_reconstruction(sino_trans, center,
                                            apply_log=True, filter_name="hann")
        rec_dark = rec.fbp_reconstruction(sino_dark, center,
                                           apply_log=True, filter_name="hann")
        rec_phase_hdf[j - start_slice] = rec_phase
        rec_trans_hdf[j - start_slice] = rec_trans
        rec_dark_hdf[j - start_slice] = rec_dark
        if (j - start_slice) % 30 == 0:
            name = ("0000" + str(j))[-5:]
            losa.save_image(output_base + "/phase/rec_" + name + ".tif", rec_phase)
            losa.save_image(output_base + "/transmission/rec_" + name + ".tif", rec_trans)
            losa.save_image(output_base + "/dark_signal/rec_" + name + ".tif", rec_dark)
    print("Done slice: {}".format((start_sino, stop_sino)))
if num_rest != 0:
    for i in range(num_rest):
        start_sino = num_iter * slice_chunk + offset
        stop_sino = start_sino + num_rest
        sino_phase_stack = phase_hdf[:, start_sino:stop_sino, :]
        sino_trans_stack = trans_hdf[:, start_sino:stop_sino, :]
        sino_dark_stack = dark_hdf[:, start_sino:stop_sino, :]
        # Save the results to tif images
        for j in range(start_sino, stop_sino):
            sino_phase = sino_phase_stack[:, j - start_sino, :]
            sino_trans = sino_trans_stack[:, j - start_sino, :]
            sino_dark = sino_dark_stack[:, j - start_sino, :]

            sino_phase = filt.double_wedge_filter(sino_phase, center)
            sino_phase = rem.remove_stripe_based_wavelet_fft(sino_phase, 5, 1)
            sino_trans = rem.remove_all_stripe(sino_trans, 1.5, 71, 31)
            sino_dark = rem.remove_all_stripe(sino_dark, 1.5, 71, 21)

            rec_phase = rec.fbp_reconstruction(sino_phase, center, apply_log=False,
                                                filter_name="hann")
            rec_trans = rec.fbp_reconstruction(sino_trans, center, apply_log=True,
                                                filter_name="hann")
            rec_dark = rec.fbp_reconstruction(sino_dark, center, apply_log=True,
                                               filter_name="hann")

            rec_phase_hdf[j - start_slice] = rec_phase
            rec_trans_hdf[j - start_slice] = rec_trans
            rec_dark_hdf[j - start_slice] = rec_dark
            if (j - start_slice) % 30 == 0:
                name = ("0000" + str(j))[-5:]
                losa.save_image(output_base + "/phase/rec_" + name + ".tif", rec_phase)
                losa.save_image(output_base + "/transmission/rec_" + name + ".tif", rec_trans)
                losa.save_image(output_base + "/dark_signal/rec_" + name + ".tif", rec_dark)
        print("Done slice: {}".format((start_sino, stop_sino)))
t1 = timeit.default_timer()
print("All done !!!. Time cost {}".format(t1 - t0))
