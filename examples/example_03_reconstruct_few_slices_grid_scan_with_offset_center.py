"""
The following examples show how to use Algotom to reconstruct a few slices
from a tomographic data acquired by using a grid scan (row-by-row scanning)
with the offset rotation-axis.

Raw data is at: https://zenodo.org/record/4614789 (searching on Zenodo to
download other parts: *Part01, ..., *Part24 ).
There're 24 scans (8 rows x 3 columns) of projection images (hdf files) under
folders named from "scan_00052" to "scan_00075". Dark-field and flat-field
images are under the folder named "scan_00051".

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
import algotom.util.utility as util
import algotom.rec.reconstruction as rec


input_base = "D:/data/"
key_path = "/entry/data/data"
# Where to save the outputs
output_base = "D:/output/"
# To get scan names.
proj_scan = np.arange(52,76)
df_scan = 51
prefix = "0000" + str(df_scan)
df_name = "scan_" + prefix[-5:]
proj_name = []
for i in proj_scan:
    prefix = "0000" + str(i)
    proj_name.append("scan_" + prefix[-5:])
# Separate scans to 8 rows x 3 columns
num_scan_total = len(proj_scan)
num_scan_col = 3
num_scan_row = num_scan_total//num_scan_col
overlap_window = 100 # Used to calculate the overlap-area and overlap-side.
# Load dark-field and flat-field images, average each of them.
flat_path = losa.find_file(input_base + "/" + df_name + "/*flat*")[0]
flat_field = np.mean(losa.load_hdf(flat_path, key_path)[:], axis = 0)
dark_path = losa.find_file(input_base + "/" + df_name + "/*dark*")[0]
dark_field = np.mean(losa.load_hdf(dark_path, key_path)[:], axis = 0)
# Load projection images of each scan as hdf objects
data_objects = []
list_depth = []
list_height = []
list_width = []
for i in range(num_scan_total):
    file_path = losa.find_file(input_base + "/" + proj_name[i] + "/*proj*")[0]
    hdf_object = losa.load_hdf(file_path, key_path)
    (depth1, height1, width1) = hdf_object.shape
    list_depth.append(depth1)
    list_height.append(height1)
    list_width.append(width1)
    data_objects.append(hdf_object)
# Number of projections may be different around 1 frame between scans caused
# by the synchronizer in the flat-scan mode.
depth = min(list_depth)
height = min(list_height)
width = min(list_width)

print("!!! Start !!!")
time_start = timeit.default_timer()
# Generate a sinogram at the scan-row of 2 and the row-index of 500
print("1 -> Generate a sinogram at the scan row of 2 and the row-index of 500")
row_idx = 2
slice_index = 500

# Options to remove artfacts
opt1 = {"method": "remove_zinger", "para1": 0.08, "para2": 1}
opt2 = {"method": "remove_all_stripe", "para1": 3.0, "para2": 51, "para3": 17}
list_sino = []
for i in range(num_scan_col):
    sinogram = corr.flat_field_correction(data_objects[i + row_idx * num_scan_col][:,slice_index,:],
                                        flat_field[slice_index], dark_field[slice_index],
                                        option1=opt1, option2=opt2)
    name = "0" + str(i)
    losa.save_image(output_base + "/reconstruction/sino_360_part_"+ name[-2:] + ".tif", sinogram)
    # Check if there's a part of sample in the sinogram.
    # check = util.detect_sample(sinogram)
    # if check:
    #     list_sino.append(sinogram)
    list_sino.append(sinogram)
# Calculate the overlap-sides and overlap-areas between sinograms.
print("2 -> Determine the overlap-side and overlap-area between sinograms and stitch them")
list_overlap = calc.find_overlap_multiple(list_sino, overlap_window)
print("  Results {}".format(list_overlap))
sino_360 = conv.stitch_image_multiple(list_sino, list_overlap, norm=True, total_width=None)
losa.save_image(output_base + "/reconstruction/sino_360_stitched.tif", sino_360)
print("3 -> Calculate the center of rotation and convert a 360-degree sinogram to a 180-degree sinogram")
(center0, overlap, side,_) = calc.find_center_360(sino_360, 100)
print("Center-of-rotation: {0}. Side: {1} (0->'left', 1->'right'). Overlap: {2}".format(center0, side, overlap))
# Convert the 360-degree sinogram to the 180-degree sinogram.
sino_180, center1 = conv.convert_sinogram_360_to_180(sino_360, center0)
losa.save_image(output_base + "/reconstruction/sino_180_converted.tif", sino_180)
# Apply the Fresnel filter
print("4 -> Apply a denoising method because the sinogram is undersampled")
sino_180 = filt.fresnel_filter(sino_180, 250, 1)
# Perform reconstruction
print("5 -> Perform reconstruction")
# img_rec = rec.dfi_reconstruction(sino_180, center1, apply_log=True)
## Use gpu for fast reconstruction
img_rec = rec.fbp_reconstruction(sino_180, center1, apply_log=True, gpu=True)
## Using gridrec code for faster CPU reconstruction if Tomopy installed
# img_rec = rec.gridrec_reconstruction(sino_180, center1, apply_log=True)
losa.save_image(output_base + "/reconstruction/recon_image.tif", img_rec)
time_stop = timeit.default_timer()
print("!!! Done !!! Time cost: {}".format(time_stop - time_start))
