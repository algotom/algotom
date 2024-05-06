import time
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.removal as remo
import algotom.prep.calculation as calc
import algotom.rec.reconstruction as rec
import algotom.rec.vertrec as vrec
import matplotlib.pyplot as plt


output_base = "F:/vertical_slice/scan_00081/find_centers/"

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

# Crop data to fit the memory and for fast calculation
crop = (1000, 1000, 0, 0)  # (crop_top, crop_bottom, crop_left, crop_right)
(depth, height0, width0) = proj_obj.shape
top = crop[0]
bot = height0 - crop[1]
left = crop[2]
right = width0 - crop[3]
width = right - left
height = bot - top

t0 = time.time()
flat = flat_field[top:bot, left:right]
dark = dark_field[top:bot, left:right]
flat_dark = flat - dark
flat_dark[flat_dark == 0.0] = 1.0
# Load data to memory and perform flat-field correction
projs_corrected = (proj_obj[:, top:bot, left:right] - dark) / flat_dark

auto_finding = False
slice_use = width // 2 - 50  # Avoid the middle slice due to ring artifacts
start_center = width // 2 - 20
stop_center = width // 2 + 20
step = 1.0

if auto_finding:
    return_metric = True
    metric = "entropy"
    invert_metric = True  # Depending on samples, may need to invert the metrics.
    if return_metric:
        centers, metrics = vrec.find_center_vertical_slice(projs_corrected, slice_use,
                                                           start_center, stop_center,
                                                           step=step, metric=metric,
                                                           alpha=0.0, angles=None,
                                                           chunk_size=30, apply_log=True,
                                                           gpu=True, block=(32, 32),
                                                           ncore=None, prefer="threads",
                                                           show_progress=True,
                                                           invert_metric=invert_metric,
                                                           return_metric=return_metric)
        plt.xlabel("Center")
        plt.ylabel("Metric")
        plt.plot(centers, metrics, "-o")
        center = centers[np.argmin(metrics)]
    else:
        center = vrec.find_center_vertical_slice(projs_corrected, slice_use,
                                                 start_center, stop_center,
                                                 step=step, metric=metric,
                                                 alpha=0.0, angles=None,
                                                 chunk_size=30, apply_log=True,
                                                 gpu=True, block=(32, 32),
                                                 ncore=None, prefer="threads",
                                                 show_progress=True,
                                                 invert_metric=invert_metric,
                                                 return_metric=return_metric)
    print(f"Center of rotation {center}")
    if return_metric:
        plt.show()
else:
    vrec.find_center_visual_vertical_slices(projs_corrected, output_base,
                                            slice_use, start_center, stop_center,
                                            step=step, alpha=0.0, angles=None,
                                            chunk_size=30, apply_log=True, gpu=True,
                                            block=(16, 16), ncore=None,
                                            prefer="processes", display=True,
                                            masking=True)
t1 = time.time()
print(f"All done in {t1 - t0}s!")
