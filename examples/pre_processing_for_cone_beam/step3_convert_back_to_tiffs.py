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
    mat_stack = np.uint16(data[:, i: i + chunk_size, :])
    losa.save_image_multiple(output_base, mat_stack, axis=1,
                             overwrite=True, ncore=ncore,
                             prefer='processes', start_idx=i)
if last_chunk != 0:
    mat_stack = np.uint16(data[:, depth - last_chunk:depth,
                          :])  # Convert to 16-bit data for tif-format
    losa.save_image_multiple(output_base, mat_stack, axis=1,
                             overwrite=True, ncore=ncore,
                             prefer='processes', start_idx=depth - last_chunk)
t1 = timeit.default_timer()
print("Done!!!. Total time cost: {}".format(t1 - t0))
