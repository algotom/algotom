import timeit
import multiprocessing as mp
from joblib import Parallel, delayed
import numpy as np
import algotom.io.loadersaver as losa

input_file = "E:/tmp/projections_preprocessed.hdf"
output_base = "E:/tmp/tif_projections/"

data = losa.load_hdf(input_file, key_path='entry/data')
# Note that the shape of data has been changed after the previous step
# where sinograms are arranged along 0-axis. Now we want to save the data
# as projections which are arranged along 1-axis.
(height, depth, width) = data.shape

t0 = timeit.default_timer()
# For parallel writing tif-images
ncore = mp.cpu_count()
chunk_size = np.clip(ncore - 1, 1, depth - 1)
last_chunk = depth - chunk_size * (depth // chunk_size)

for i in np.arange(0, depth - last_chunk, chunk_size):
    mat_stack = data[:, i: i + chunk_size, :]
    mat_stack = np.uint16(mat_stack)  # Convert to 16-bit data for tif-format
    file_names = [(output_base + "/proj_" + ("0000" + str(j))[-5:] + ".tif") for j in range(i, i + chunk_size)]
    # Save files in parallel
    Parallel(n_jobs=ncore, prefer="processes")(delayed(losa.save_image)(file_names[j], mat_stack[:, j, :]) for j in range(chunk_size))

if last_chunk != 0:
    mat_stack = data[:, depth - last_chunk:depth, :]
    mat_stack = np.uint16(mat_stack)  # Convert to 16-bit data for tif-format
    file_names = [(output_base + "/proj_" + ("0000" + str(j))[-5:] + ".tif") for j in range(depth - last_chunk, depth)]
    # Save files in parallel
    Parallel(n_jobs=ncore, prefer="processes")(delayed(losa.save_image)(file_names[j], mat_stack[:, j, :]) for j in range(last_chunk))

t1 = timeit.default_timer()
print("Done!!!. Total time cost: {}".format(t1 - t0))
