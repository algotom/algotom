import timeit
import multiprocessing as mp
from joblib import Parallel, delayed
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.removal as rem
import algotom.prep.correction as corr

input_file = "E:/tmp/projections.hdf"
output_file = "E:/tmp/tmp/projections_preprocessed.hdf"

data = losa.load_hdf(input_file, key_path='entry/data')
(depth, height, width) = data.shape

# Note that the shape of output data is (height, depth, width)
# for faster writing to hdf file.
output = losa.open_hdf_stream(output_file, (height, depth, width), data_type="float32")

t0 = timeit.default_timer()
# For parallel processing
ncore = mp.cpu_count()
chunk_size = np.clip(ncore - 1, 1, height - 1)
last_chunk = height - chunk_size * (height // chunk_size)
for i in np.arange(0, height - last_chunk, chunk_size):
    sinograms = np.float32(data[:, i:i + chunk_size, :])
    # Note about the change of the shape of output_tmp (which is a list of processed sinogram)
    output_tmp = Parallel(n_jobs=ncore, prefer="threads")(delayed(rem.remove_all_stripe)(sinograms[:, j, :], 3.0, 51, 21) for j in range(chunk_size))

    # Apply beam hardening correction if need to
    # output_tmp = np.asarray(output_tmp)
    # output_tmp = Parallel(n_jobs=ncore, prefer="threads")(
    #     delayed(corr.beam_hardening_correction)(output_tmp[j], 40, 2.0, False) for j in range(chunk_size))

    output[i:i + chunk_size] = np.asarray(output_tmp, dtype=np.float32)
    t1 = timeit.default_timer()
    print("Done sinograms: {0}-{1}. Time {2}".format(i, i + chunk_size, t1 - t0))

if last_chunk != 0:
    sinograms = np.float32(data[:, height - last_chunk:height, :])
    output_tmp = Parallel(n_jobs=ncore, prefer="threads")(delayed(rem.remove_all_stripe)(sinograms[:, j, :], 3.0, 51, 21) for j in range(last_chunk))

    # Apply beam hardening correction if need to
    # output_tmp = np.asarray(output_tmp)
    # output_tmp = Parallel(n_jobs=ncore, prefer="threads")(
    #     delayed(corr.beam_hardening_correction)(output_tmp[j], 40, 2.0, False) for j in range(last_chunk))

    output[height - last_chunk:height] = np.asarray(output_tmp, dtype=np.float32)
    t1 = timeit.default_timer()
    print("Done sinograms: {0}-{1}. Time {2}".format(height - last_chunk, height - 1, t1 - t0))

t1 = timeit.default_timer()
print("Done!!!. Total time cost: {}".format(t1 - t0))