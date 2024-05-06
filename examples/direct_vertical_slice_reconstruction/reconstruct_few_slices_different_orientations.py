import time
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.removal as remo
import algotom.prep.calculation as calc
import algotom.rec.vertrec as vrec

output_base = "F:/vertical_slice/scan_00081/multiple_angles/"

proj_file = "F:/Tomo_data/HEX/scan_00081/pro_00000.hdf"
flat_file = "F:/Tomo_data/HEX/scan_00085/flat_00000.hdf"
dark_file = "F:/Tomo_data/HEX/scan_00085/dark_00000.hdf"
key_path = "entry/data/data"

# Load projection data as a hdf object
proj_obj = losa.load_hdf(proj_file, key_path)
(depth, height, width) = proj_obj.shape
# Load dark-field and flat-field images, average each result
flat_field = np.mean(np.asarray(losa.load_hdf(flat_file, key_path)), axis=0)
dark_field = np.mean(np.asarray(losa.load_hdf(dark_file, key_path)), axis=0)

crop = (0, 0, 0, 0)  # (crop_top, crop_bottom, crop_left, crop_right)
(depth, height0, width0) = proj_obj.shape
top = crop[0]
bot = height0 - crop[1]
left = crop[2]
right = width0 - crop[3]
width = right - left
height = bot - top

t0 = time.time()
# Find center of rotation using a sinogram-based method
mid_slice = height // 2 + top
sinogram = corr.flat_field_correction(proj_obj[:, mid_slice, left:right],
                                      flat_field[mid_slice, left:right],
                                      dark_field[mid_slice, left:right])
sinogram = remo.remove_all_stripe(sinogram, 2.0, 51, 21)
center = calc.find_center_vo(sinogram)
print(f"Center-of-rotation is: {center}")


slice_indices = [width // 2, width // 3, width // 4, width // 2]
alphas = [0.0, 30.0, 60.0, 90.0]  # Orientation of the slices, in degree.
# Note that raw data is flat-field corrected and cropped if these parameters
# are provided. The center referred to cropped image.
ver_slices = vrec.vertical_reconstruction_different_angles(proj_obj,
                                                           slice_indices,
                                                           alphas, center,
                                                           flat_field=flat_field,
                                                           dark_field=dark_field,
                                                           angles=None,
                                                           crop=crop,
                                                           proj_start=0,
                                                           proj_stop=-1,
                                                           chunk_size=30,
                                                           ramp_filter='after',
                                                           filter_name='hann',
                                                           pad=None,
                                                           pad_mode='edge',
                                                           apply_log=True,
                                                           gpu=True,
                                                           block=(16, 16),
                                                           ncore=None,
                                                           prefer='threads',
                                                           show_progress=True,
                                                           masking=False)
print("Save output ...")
for i, idx in enumerate(slice_indices):
    losa.save_image(output_base + f"/slice_{idx:05}_angle_{alphas[i]:3.2f}.tif", ver_slices[i])
t1 = time.time()
print("All done !!!")
