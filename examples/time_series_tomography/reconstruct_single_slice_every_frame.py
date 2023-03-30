# ===========================================================================
# Author: Nghia T. Vo
# Description: Examples of how to use the Algotom package.
# ===========================================================================

"""
The following examples show how to reconstruct a single slice for every frame
in a time-series tomography data. Output is a list of reconstructed slices in
a single folder. This is useful to see the progress of the experiment over
time.

The code written based on datasets collected at the beamline I12-DLS which
often have 4 files for each time-series scan:
- a hdf-file contains projection-images.
- a nxs-file contains metadata of an experiment: energy, number-of-projections
  per tomo, number-of-tomographs, detector-sample distance, detector
  pixel-size, exposure time,...
- a hdf-file contains flat-field images.
- a hdf-file contains dark-field images.

Referring to "example_01_*.py" to know how to find key-paths and datasets
in a hdf/nxs file.
"""

import timeit
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.calculation as calc
import algotom.rec.reconstruction as rec
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.util.utility as util

slice_idx = 1000  # Index of the slice to be reconstructed
output_base = "/home/user_id/reconstruction/"

proj_path = "/i12/data/projections.hdf"
flat_path = "/i12/data/flats.hdf"
dark_path = "/i12/data/darks.hdf"
metadata_path = "/i12/data/metadata.nxs"

scan_type = "continuous"  # stage is freely rotated
# scan_type = "swinging" # stage is restricted to rotate back-and-forward between 0 and 180 degree.

# Provide paths (keys) to datasets in the hdf/nxs files.
hdf_key = "/entry/data/data"
angle_key = "/entry1/tomo_entry/data/rotation_angle"
num_proj_key = "/entry1/information/number_projections"

# Crop images if need to.
crop_left = 0
crop_right = 0

data = losa.load_hdf(proj_path, hdf_key)  # This is an hdf-object not ndarray.
(depth, height, width) = data.shape
left = crop_left
right = width - crop_right

# Load metatdata
num_proj = int(np.asarray(losa.load_hdf(metadata_path, num_proj_key)))
num_tomo = depth // num_proj

angles = np.squeeze(np.asarray(losa.load_hdf(metadata_path, angle_key)))
# Sometime there's a mismatch between the number of acquired projections
# and number of angles due to technical reasons or early terminated scan.
# In such cases, we have to provide calculated angles.
if len(angles) < depth:
    if scan_type == "continuous":
        list_tmp = np.linspace(0, 180.0, num_proj)
        angles = np.ndarray.flatten(np.asarray([list_tmp for i in range(num_tomo)]))
    else:
        list_tmp1 = np.linspace(0, 180.0, num_proj)
        list_tmp2 = np.linspace(180.0, 0, num_proj)
        angles = np.ndarray.flatten(np.asarray(
            [list_tmp1 if i % 2 == 0 else list_tmp2 for i in range(num_tomo)]))
else:
    angles = angles[0:depth]

time_start = timeit.default_timer()
# Load flat-field images and dark-field images, average each of them
print("1 -> Load dark-field and flat-field images, average each result")
flat_field = np.mean(losa.load_hdf(flat_path, hdf_key)[:], axis=0)
dark_field = np.mean(losa.load_hdf(dark_path, hdf_key)[:], axis=0)

# Find the center of rotation using the sinogram of the first tomograph
print("2 -> Calculate the center-of-rotation...")
sinogram = corr.flat_field_correction(data[0: num_proj, slice_idx, left:right],
                                      flat_field[slice_idx, left:right],
                                      dark_field[slice_idx, left:right])
center = calc.find_center_vo(sinogram)
print("Center-of-rotation = {0}".format(center))

# Load sinogram of each tomograph, clean it, and do reconstruction.
folder_name = "slice_" + ("0000" + str(slice_idx))[-5:]
for i in range(num_tomo):
    sinogram = corr.flat_field_correction(data[i * num_proj: (i + 1) * num_proj, slice_idx, left:right],
                                          flat_field[slice_idx, left:right],
                                          dark_field[slice_idx, left:right])
    sinogram = remo.remove_zinger(sinogram, 0.05, 1)
    sinogram = remo.remove_all_stripe(sinogram, 3.0, 51, 17)
    sinogram = filt.fresnel_filter(sinogram, 100)
    thetas = np.deg2rad(angles[i * num_proj: (i + 1) * num_proj])
    # img_rec = rec.dfi_reconstruction(sinogram, center, angles=thetas, apply_log=True)
    # img_rec = rec.gridrec_reconstruction(sinogram, center, angles=thetas, apply_log=True)
    img_rec = rec.fbp_reconstruction(sinogram, center, angles=thetas, apply_log=True)
    file_name = "rec_tomo_" + ("0000" + str(i))[-5:] + ".tif"
    losa.save_image(output_base + "/" + folder_name + "/" + file_name, img_rec)
    print("Done tomograph {0}".format(i))

time_stop = timeit.default_timer()
print("All done!!! Total time cost: {}".format(time_stop - time_start))
