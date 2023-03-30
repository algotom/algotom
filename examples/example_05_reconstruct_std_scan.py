"""
The following examples show how to use Algotom to reconstruct a few slices
from a standard tomographic data.

Raw data is at: https://zenodo.org/record/1443568
There're two files: "pco1-68067.hdf" contains flat-field, dark-field, and
projection images; "68067.nxs" contains metadata with a link to the
"pco1-68067.hdf" file.

Referring to "example_01_*.py" to know how to find key-paths and datasets
in a hdf/nxs file.

Referring to "example_06_*.py" to know how to include distortion correction.
"""

import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.calculation as calc
import algotom.rec.reconstruction as rec
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.util.utility as util

file_path = "E:/Tomo_data/68067.nxs"
output_base = "E:/tmp/output3/"

# Provide path to datasets in the nxs file.
data_key = "/entry1/tomo_entry/data/data"
image_key = "/entry1/tomo_entry/instrument/detector/image_key"
angle_key = "/entry1/tomo_entry/data/rotation_angle"

ikey = np.squeeze(np.asarray(losa.load_hdf(file_path, image_key)))
angles = np.squeeze(np.asarray(losa.load_hdf(file_path, angle_key)))
data = losa.load_hdf(file_path, data_key) # This is an object not ndarray.
(depth, height, width) = data.shape

# Load dark-field images and flat-field images, averaging each result.
print("1 -> Load dark-field and flat-field images, average each result")
dark_field = np.mean(np.asarray(data[np.squeeze(np.where(ikey==2.0)), :, :]), axis=0)
flat_field = np.mean(np.asarray(data[np.squeeze(np.where(ikey==1.0)), :, :]), axis=0)

# Perform flat-field correction in the projection space and save the result.
# Note that in this data, there're time-stamps at the top-left of images with
# binary gray-scale (size ~ 10 x 80). This gives rise to the zero-division
# warning. Algotom replaces zeros by the mean value or 1. We also can crop 10
# pixels from the top to avoid this problem.
print("2 -> Save few projection images as tifs")
proj_idx = np.squeeze(np.where(ikey == 0))
proj_corr = corr.flat_field_correction(
    data[proj_idx[0], 10:,:], flat_field[10:], dark_field[10:])
losa.save_image(output_base + "/proj_corr/ff_corr_00000.tif", proj_corr)

# Perform flat-field correction in the sinogram space and save the result.
print("3 -> Generate a sinogram with flat-field correction and save the result")
index = height//2 # Index of a sinogram.
sinogram = corr.flat_field_correction(
    data[proj_idx[0]:proj_idx[-1], index,:],
    flat_field[index, :], dark_field[index, :])
losa.save_image(output_base + "/sinogram/sinogram_mid.tif", sinogram)

# Calculate the center-of-rotation by searching around the middle width of
# the sinogram (radius=50).
print("4 -> Calculate the center-of-rotation")
# center = calc.find_center_vo(sinogram, width//2-50, width//2+50)
center = calc.find_center_vo(sinogram)
print("Center-of-rotation is {}".format(center))
# Perform reconstruction and save the result.
# Users can choose CPU-based methods as follows
thetas = angles[proj_idx[0]:proj_idx[-1]]*np.pi/180
# # DFI method, a built-in function:
print("5 -> Perform reconstruction without artifact removal methods")
img_rec = rec.dfi_reconstruction(sinogram, center, angles=thetas, apply_log=True)

# # FBP-CPU method, a built-in function:
# img_rec = rec.fbp_reconstruction(sinogram, center, angles=thetas, apply_log=True, gpu=False)
#
# # Gridrec method in Tomopy (Tomopy must be installed before use):
# img_rec = rec.gridrec_reconstruction(sinogram, center, apply_log=True, ratio=1.0)
#
# # If GPU is available:
#
# # FBP-GPU method, a built-in function:
# img_rec = rec.fbp_reconstruction(sinogram, center, angles=thetas, apply_log=True, gpu=True)
#
# # FBP-GPU method in Astra (Astra must be installed before use):
# img_rec = rec.astra_reconstruction(sinogram, center, apply_log=True, ratio=1.0,method="FBP_CUDA")
losa.save_image(output_base + "/reconstruction/recon_mid.tif", img_rec)

# Pre-processing methods should be used to clean the data before reconstruction.
# Apply zinger-removal method
print("6 -> Apply methods of removing artifacts")
sinogram = remo.remove_zinger(sinogram, 0.08, 1)
# Apply ring-artifact removal methods. There're many methods available in algotom.
sinogram = remo.remove_all_stripe(sinogram, 3, 51, 17)
# # Apply a low-pass filter to improve the contrast of a reconstructed image.
# sinogram = filt.fresnel_filter(sinogram, 200, dim=1)
# Perform reconstruction and save result
print("7 -> Perform reconstruction with artifact removal methods")
img_rec = rec.dfi_reconstruction(sinogram, center, angles=thetas, apply_log=True)
losa.save_image(output_base + "/reconstruction/recon_mid_cleaned.tif", img_rec)

# Extracting sinograms one-by-one  and doing reconstruction is not efficient and slow due to
# the IO overhead. The following example shows how to process a chunk of sinograms in one go.
print("8 -> Load a chunk of 8 sinograms and clean artifacts to reduce IO time cost")
start_slice = 500
stop_slice = start_slice + 8
# Options to include removal methods in the flat-field correction step.
opt1 = {"method": "remove_zinger", "para1": 0.08, "para2": 1}
opt2 = {"method": "remove_all_stripe", "para1": 3.0, "para2": 51, "para3": 17}
# Load sinograms, and perform pre-processing.
sinograms = corr.flat_field_correction(
    data[proj_idx[0]:proj_idx[-1], start_slice:stop_slice,:],
    flat_field[start_slice:stop_slice, :], dark_field[start_slice:stop_slice, :],
    option1=opt1, option2=opt2)
# Perform reconstruction
print("9 -> Perform reconstruction on this chunk in parallel...")
recon_img = util.apply_method_to_multiple_sinograms(sinograms, "dfi_reconstruction",
                                                    [center])
for i in range(start_slice, stop_slice):
    #img_rec = rec.dfi_reconstruction(sinograms[:,i - start_slice, :], center, apply_log=True)
    name = "0000" + str(i)
    losa.save_image(output_base + "/reconstruction2/rec_" + name[-5:] \
                    + ".tif", recon_img[:, i - start_slice, :])
print("!!! Done !!!")
