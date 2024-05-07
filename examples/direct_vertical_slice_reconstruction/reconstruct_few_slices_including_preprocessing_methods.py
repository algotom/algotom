import os
import time
import h5py
import shutil
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.prep.calculation as calc
import algotom.rec.vertrec as vrec
import algotom.util.utility as util

output_base = "F:/vertical_slice/scan_00081/preprocessing/"

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

crop = (1000, 1500, 500, 500)  # (crop_top, crop_bottom, crop_left, crop_right)
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

# Apply preprocessing methods along the sinogram direction and save intermediate
# results to disk
chunk_size = 30  # Number of sinograms to be processed at once
file_tmp = output_base + "/tmp_/preprocessed.hdf"
hdf_prep = losa.open_hdf_stream(file_tmp, (depth, height, width),
                                data_type="float32", key_path=key_path)
last_chunk = height - chunk_size * (height // chunk_size)
flat = flat_field[top:bot, left:right]
dark = dark_field[top:bot, left:right]
flat_dark = flat - dark
flat_dark[flat_dark == 0.0] = 1.0

# ring_removal_method = remo.remove_all_stripe
# ring_removal_paras = [2.5, 51, 21]
ring_removal_method = remo.remove_stripe_based_normalization
ring_removal_paras = [15, 1, False]

for i in range(0, height - last_chunk, chunk_size):
    start = i + top
    stop = start + chunk_size
    # Flat-field correction
    proj_chunk = (proj_obj[:, start: stop, left:right] -
                  dark[i:i + chunk_size]) / flat_dark[i:i + chunk_size]
    # Apply ring artifact removal
    proj_chunk = util.parallel_process_slices(proj_chunk, ring_removal_method,
                                              ring_removal_paras, axis=1,
                                              ncore=None, prefer="threads")
    # Apply contrast enhancement
    proj_chunk = util.parallel_process_slices(proj_chunk, filt.fresnel_filter,
                                              [300.0, 1], axis=1,
                                              ncore=None, prefer="threads")
    hdf_prep[:, i: i + chunk_size, :] = proj_chunk
    t1 = time.time()
    print(f" Done sinograms {i}-{i + chunk_size}. Time elapsed: {t1 - t0}")
if last_chunk != 0:
    start = height - last_chunk
    stop = height
    # Flat-field correction
    proj_chunk = (proj_obj[:, start: stop, left:right] -
                  dark[-last_chunk:]) / flat_dark[-last_chunk:]
    # Apply ring artifact removal
    proj_chunk = util.parallel_process_slices(proj_chunk, ring_removal_method,
                                              ring_removal_paras, axis=1,
                                              ncore=None, prefer="threads")
    # Apply contrast enhancement
    proj_chunk = util.parallel_process_slices(proj_chunk, filt.fresnel_filter,
                                              [300.0, 1], axis=1,
                                              ncore=None, prefer="threads")
    hdf_prep[:, -last_chunk:, :] = proj_chunk
    t1 = time.time()
    print(f" Done sinograms {start - top}-{stop - top}. Time elapsed: {t1 - t0}")
t1 = time.time()
print(f"\nDone preprocessing. Total time elapsed {t1 - t0}")

start_index = width // 2 - 250
stop_index = width // 2 + 250
step_index = 20
alpha = 0.0  # Orientation of the slices, in degree.
#  Load preprocessed projections and reconstruct
with h5py.File(file_tmp, 'r') as hdf_obj:
    preprocessed_projs = hdf_obj[key_path]
    ver_slices = vrec.vertical_reconstruction_multiple(preprocessed_projs, start_index,
                                                       stop_index, center, alpha=alpha,
                                                       step_index=step_index,
                                                       flat_field=None, dark_field=None,
                                                       angles=None, crop=(0, 0, 0, 0),
                                                       proj_start=0, proj_stop=-1,
                                                       chunk_size=30, ramp_filter="after",
                                                       filter_name="hann", apply_log=True,
                                                       gpu=True, block=(16, 16),
                                                       ncore=None, prefer="threads",
                                                       show_progress=True,
                                                       masking=False)
# Save output
print("Save output ...")
for i, idx in enumerate(np.arange(start_index, stop_index + 1, step_index)):
    losa.save_image(output_base + f"/slice_{idx:05}.tif", ver_slices[i])
t1 = time.time()
print(f"All done in {t1 - t0}s!")

# Delete the intermediate file
folder_tmp = os.path.dirname(file_tmp)
try:
    shutil.rmtree(folder_tmp)
except PermissionError as e:
    print(f"Error deleting the file in folder: {e}. It may still be in use.")