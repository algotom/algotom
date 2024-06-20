"""
The following examples show how to use Algotom to perform full reconstruction
of a standard tomographic data.

Raw data is at: https://zenodo.org/record/1443568
There're two files: "pco1-68067.hdf" contains flat-field, dark-field, and
projection images; "68067.nxs" contains metadata with a link to the
"pco1-68067.hdf" file.

Referring to "example_01_*.py" to know how to find key-paths and datasets
in a hdf/nxs file.

Referring to "example_06_*.py" to know how to include distortion correction.
"""

import numpy as np
import timeit
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.calculation as calc
import algotom.rec.reconstruction as rec
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.util.utility as util


file_path = "E:/Tomo_data/68067.nxs"
output_base = "E:/tmp/output4/"

# Optional parameters
start_slice = 10
stop_slice = -1
slice_chunk = 10 # Number of slices to be reconstructed in one go to reduce
                 # IO overhead (in loading a hdf file) and process in
                 # parallel (for CPU-based methods).

# Options to include artifact removal methods in the flat-field
# correction method.
opt1 = {"method": "remove_zinger", "para1": 0.08, "para2": 1}
opt2 = {"method": "remove_all_stripe", "para1": 3.0, "para2": 51, "para3": 17}
opt3 = None
# opt3 = {"method": "fresnel_filter", "para1": 100, "para2": 1} # Denoising

# Provide metadata for loading hdf file, get data shape and rotation angles.
data_key = "/entry1/flyScanDetector/data"
image_key = "/entry1/flyScanDetector/image_key"
angle_key = "/entry1/tomo_entry/data/rotation_angle"
ikey = np.squeeze(np.asarray(losa.load_hdf(file_path, image_key)))
angles = np.squeeze(np.asarray(losa.load_hdf(file_path, angle_key)))
data = losa.load_hdf(file_path, data_key)  # This is an object not ndarray.
(depth, height, width) = data.shape
# Get indices of projection images
proj_idx = np.squeeze(np.where(ikey == 0))
thetas = angles[proj_idx[0]:proj_idx[-1]] * np.pi / 180

time_start = timeit.default_timer()
print("---------------------------------------------------------------")
print("-----------------------------Start-----------------------------\n")
# Load dark-field images and flat-field images, averaging each result.
print("1 -> Load dark-field and flat-field images, average each result")
dark_field = np.mean(np.asarray(data[np.squeeze(np.where(ikey == 2.0)), :, :]),
                     axis=0)
flat_field = np.mean(np.asarray(data[np.squeeze(np.where(ikey == 1.0)), :, :]),
                     axis=0)
print("2 -> Calculate the center-of-rotation")
index = height // 2
sinogram = corr.flat_field_correction(data[proj_idx[0]:proj_idx[-1], index, :],
                                      flat_field[index, :],
                                      dark_field[index, :])
center = calc.find_center_vo(sinogram)
print("Center-of-rotation is {}".format(center))

if (stop_slice == -1) or (stop_slice > height):
    stop_slice = height
total_slice = stop_slice - start_slice
offset = start_slice
if slice_chunk > total_slice:
    slice_chunk = total_slice
num_iter = total_slice // slice_chunk
num_rest = total_slice - num_iter * slice_chunk

# Perform full reconstruction and save results as 32-bit tif images
for i in range(num_iter):
    start_sino = i * slice_chunk + offset
    stop_sino = start_sino + slice_chunk
    sinograms = corr.flat_field_correction(
        data[proj_idx[0]:proj_idx[-1], start_sino:stop_sino, :],
        flat_field[start_sino:stop_sino, :],
        dark_field[start_sino:stop_sino, :],
        option1=opt1, option2=opt2, option3=opt3)
    # Reconstruct a chunk of slices in parallel if using CPU-based method.
    # Algotom < 1.6
    # recon_img = util.apply_method_to_multiple_sinograms(sinograms,
    #                                                     "dfi_reconstruction",
    #                                                     [center])
    # Algotom >= 1.6
    recon_img = util.parallel_process_slices(sinograms, rec.dfi_reconstruction,
                                             [center])
    # Save the results to tif images
    for j in range(start_sino, stop_sino):
        name = "0000" + str(j)
        losa.save_image(output_base + "/rec_" + name[-5:] + ".tif",
                        recon_img[:, j - start_sino, :])

    # # Reconstruct the slices using a GPU-based method
    # for j in range(start_sino, stop_sino):
    #     recon_img = rec.fbp_reconstruction(sinograms[:, j - start_sino, :],
    #                                         center, angles=thetas)
    #     name = "0000" + str(j)
    #     losa.save_image(output_base + "/rec_" + name[-5:] + ".tif", recon_img)

    t_stop = timeit.default_timer()
    print("Done slice: {0} - {1} . Time {2}".format(start_sino, stop_sino,
                                                    t_stop - time_start))
if num_rest != 0:
    start_sino = num_iter * slice_chunk + offset
    stop_sino = start_sino + num_rest
    sinograms = corr.flat_field_correction(
        data[proj_idx[0]:proj_idx[-1], start_sino:stop_sino, :],
        flat_field[start_sino:stop_sino, :],
        dark_field[start_sino:stop_sino, :],
        option1=opt1, option2=opt2, option3=opt3)
    # Reconstruct a chunk of slices in parallel if using CPU-based method.

    # # Algotom < 1.6
    # recon_img = util.apply_method_to_multiple_sinograms(sinograms,
    #                                                     "dfi_reconstruction",
    #                                                     [center])
    # Algotom >= 1.6
    recon_img = util.parallel_process_slices(sinograms, rec.dfi_reconstruction,
                                             [center])
    # Save the results to tif images
    for j in range(start_sino, stop_sino):
        name = "0000" + str(j)
        losa.save_image(output_base + "/rec_" + name[-5:] + ".tif",
                        recon_img[:, j - start_sino, :])

    # # Reconstruct the slices using a GPU-based method
    # for j in range(start_sino, stop_sino):
    #     recon_img = rec.fbp_reconstruction(sinograms[:, j - start_sino, :],
    #                                         center, angles=thetas)
    #     name = "0000" + str(j)
    #     losa.save_image(output_base + "/rec_" + name[-5:] + ".tif",
    #                     recon_img)

    t_stop = timeit.default_timer()
    print("Done slice: {0} - {1} . Time {2}".format(start_sino, stop_sino,
                                                    t_stop - time_start))
time_stop = timeit.default_timer()
print("!!! All Done. Time cost {} !!!".format(time_stop - time_start))
