"""
The following examples show how to use Algotom to reconstruct a sample slice
from a tomographic data acquired by using a helical scan with the offset
center-of-rotation.

Raw data is at: https://zenodo.org/record/4613047;
https://zenodo.org/record/4613224; https://zenodo.org/record/4613644;
There're 4 files: "projections_00000.hdf", "flats_00000.hdf",
"darks_00000.hdf", and "scan_00010.nxs" containing projection images,
flat-field images, dark-field images, and meta-data, respectively.

Referring to "example_01_*.py" to know how to find key-paths and datasets
in a hdf/nxs file.

Referring to "example_06_*.py" to know how to include distortion correction.
"""

import timeit
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.removal as remo
import algotom.prep.calculation as calc
import algotom.prep.conversion as conv
import algotom.prep.filtering as filt
import algotom.rec.reconstruction as rec

# Paths to data
proj_path = "D:/data/scan_00010/projections_00000.hdf"
flat_path = "D:/data/scan_00009/flats_00000.hdf"
dark_path = "D:/data/scan_00009/darks_00000.hdf"
meta_path = "D:/data/scan_00010/scan_00010.nxs"
key_path = "/entry/data/data"
pixel_size = 3.24297964149*10**(-3) # mm. Calibrated at the beamtime.

# Where to save the outputs
output_base = "D:/output/"

# Load data of projection images as an hdf object
proj_data = losa.load_hdf(proj_path, key_path)
(depth, height, width) = proj_data.shape
# Load flat-field images and dark-field images, average each of them
flat_field = np.mean(losa.load_hdf(flat_path, key_path)[:], axis=0)
dark_field = np.mean(losa.load_hdf(dark_path, key_path)[:], axis=0)
# Load metadata of the helical scan
pitch = np.float32(losa.load_hdf(meta_path, "/entry1/information/pitch")) #mm
angles = np.float32(losa.load_hdf(meta_path, "/entry1/tomo_entry/instrument/detector/rotation_angle")) #Degree
num_proj = np.int16(losa.load_hdf(meta_path, "/entry1/information/number_projections")) #mm
y_start = np.float32(losa.load_hdf(meta_path, "/entry1/information/y_start"))
y_stop = np.float32(losa.load_hdf(meta_path, "/entry1/information/y_stop"))

scan_type = "360"
(y_s, y_e) = calc.calculate_reconstructable_height(y_start, y_stop, pitch, scan_type)
max_index = calc.calculate_maximum_index(y_start, y_stop, pitch, pixel_size, scan_type)
print("1 -> Given y_start: {0}, y_stop: {1}, pitch: {2}, and scan_type: '{3}'".format(y_start, y_stop, pitch, scan_type))
print("Reconstructable height-range is: [{0}, {1}]".format(y_s, y_e))
print("Index-range of slices is: [0, {0}] using a pixel-size of {1}".format(max_index, pixel_size))

# Generate a blob map used for removing streak artifacts
blob_mask = remo.generate_blob_mask(flat_field, 71, 3)


# Generate a sinogram with flat-field correction and blob removal.
# Angles corresponding to this sinogram also is generated.
index = max_index // 2
print("2 -> Generate a circular sinogram from the helical data")
(sino_360, angle_sino) = conv.generate_sinogram_helical_scan(index, proj_data, num_proj, pixel_size,
                                               y_start, y_stop, pitch, scan_type=scan_type,
                                               angles=angles, flat=flat_field, dark=dark_field,
                                               mask=blob_mask, crop=(10,0,0,0))
print("3 -> Find the center of rotation, overlap-side, and overlap area between two halves of a 360-degree sinogram")
(center0, overlap, side,_) = calc.find_center_360(sino_360, 100)
print("Center-of-rotation: {0}. Side: {1} (0->'left', 1->'right'). Overlap: {2}".format(center0, side, overlap))
# Convert the 360-degree sinogram to the 180-degree sinogram.
sino_180, center1 = conv.convert_sinogram_360_to_180(sino_360, center0)
# Remove partial ring artifacts
sino_180 = remo.remove_stripe_based_sorting(sino_180, 15)
# Remove zingers
sino_180 = remo.remove_zinger(sino_180, 0.08)
# Denoising
sino_180 = filt.fresnel_filter(sino_180, 250, 1)
# Perform recosntruction
img_rec = rec.dfi_reconstruction(sino_180, center1, apply_log=True)

## Use gpu for fast reconstruction
# img_rec = rec.fbp_reconstruction(sino_180, center1, apply_log=True, gpu=True)

losa.save_image(output_base + "/reconstruction/sino_360.tif", sino_360)
losa.save_image(output_base + "/reconstruction/sino_180.tif", sino_180)
losa.save_image(output_base + "/reconstruction/recon_image.tif", img_rec)
print("!!! Done !!!")
