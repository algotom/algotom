import time
import numpy as np
import algotom.io.loadersaver as losa
import algotom.io.converter as cvt
import algotom.prep.correction as corr
import algotom.prep.removal as remo
import algotom.prep.calculation as calc
import algotom.rec.vertrec as vrec


output_base = "F:/vertical_slice/few_slices_input_tif/"

proj_path = "F:/Tomo_data/HEX/scan_00081/tif/projections/"
flat_path = "F:/Tomo_data/HEX/scan_00081/tif/flats/"
dark_path = "F:/Tomo_data/HEX/scan_00081/tif/darks/"


# Load projection data as a hdf object
proj_obj = cvt.HdfEmulatorFromTif(proj_path)  # Create hdf-emulator
(depth, height, width) = proj_obj.shape
# Load dark-field and flat-field images, average each result
flat_field = np.mean(np.asarray(
    [losa.load_image(file) for file in losa.find_file(flat_path + "/*tif*")]), axis=0)
dark_field = np.mean(np.asarray(
    [losa.load_image(file) for file in losa.find_file(dark_path + "/*tif*")]), axis=0)
flat_dark = flat_field - dark_field
flat_dark[flat_dark == 0.0] = 1.0

crop = (0, 0, 0, 0)  # (crop_top, crop_bottom, crop_left, crop_right)
(depth, height0, width0) = proj_obj.shape
top = crop[0]
bot = height0 - crop[1]
left = crop[2]
right = width0 - crop[3]
width = right - left
height = bot - top

flat_crop = flat_field[top:bot, left:right]
dark_crop = dark_field[top:bot, left:right]
flat_dark_crop = flat_crop - dark_crop
flat_dark_crop[flat_dark_crop == 0.0] = 1.0

t0 = time.time()
# Find center of rotation using a sinogram-based method
mid_slice = height // 2 + top
sinogram = corr.flat_field_correction(proj_obj[:, mid_slice, left:right],
                                      flat_field[mid_slice, left:right],
                                      dark_field[mid_slice, left:right])
sinogram = remo.remove_all_stripe(sinogram, 2.0, 51, 21)
center = calc.find_center_vo(sinogram)
print(f"Center-of-rotation is: {center}")

start_index = width // 2 - 250
stop_index = width // 2 + 250
step_index = 20
alpha = 0.0  # Orientation of the slices, in degree.

# Note that raw data is flat-field corrected and cropped if these parameters
# are provided. The center referred to cropped image.
ver_slices = vrec.vertical_reconstruction_multiple(proj_obj, start_index,
                                                   stop_index, center,
                                                   alpha=alpha,
                                                   step_index=step_index,
                                                   flat_field=flat_field,
                                                   dark_field=dark_field,
                                                   angles=None,
                                                   crop=crop, proj_start=0,
                                                   proj_stop=-1,
                                                   chunk_size=30,
                                                   ramp_filter="after",
                                                   filter_name="hann",
                                                   apply_log=True,
                                                   gpu=True, block=(16, 16),
                                                   ncore=None,
                                                   prefer="threads",
                                                   show_progress=True,
                                                   masking=False)
print("Save output ...")
for i, idx in enumerate(np.arange(start_index, stop_index + 1, step_index)):
    losa.save_image(output_base + f"/slice_{idx:05}.tif", ver_slices[i])
t1 = time.time()
print("All done !!!")
