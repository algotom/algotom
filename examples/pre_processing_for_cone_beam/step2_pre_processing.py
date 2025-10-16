import timeit
import multiprocessing as mp
from joblib import Parallel, delayed
import numpy as np
import algotom.io.loadersaver as losa
import algotom.prep.removal as rem
import algotom.prep.correction as corr
import algotom.util.utility as util

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
    sinograms = util.parallel_process_slices(sinograms,
                                             rem.remove_all_stripe,
                                             [3.0, 51, 21],
                                             ncore=ncore, prefer="processes",
                                             axis=1)
    # Apply beam hardening correction if need to
    # sinograms = util.parallel_process_slices(sinograms,
    #                                          corr.beam_hardening_correction,
    #                                          [40, 2.0, False],
    #                                          ncore=ncore, prefer="processes",
    #                                          axis=1)
    output[i:i + chunk_size] = np.moveaxis(sinograms, 1, 0)
    t1 = timeit.default_timer()
    print("Done sinograms: {0}-{1}. Time {2}".format(i, i + chunk_size, t1 - t0))

if last_chunk != 0:
    sinograms = np.float32(data[:, height - last_chunk:height, :])
    sinograms = util.parallel_process_slices(sinograms,
                                             rem.remove_all_stripe,
                                             [3.0, 51, 21],
                                             ncore=ncore, prefer="processes",
                                             axis=1)

    # Apply beam hardening correction if need to
    # sinograms = util.parallel_process_slices(sinograms,
    #                                          corr.beam_hardening_correction,
    #                                          [40, 2.0, False],
    #                                          ncore=ncore, prefer="processes",
    #                                          axis=1)
    output[height - last_chunk:height] = np.moveaxis(sinograms, 1, 0)
    t1 = timeit.default_timer()
    print("Done sinograms: {0}-{1}. Time {2}".format(height - last_chunk, height - 1, t1 - t0))

t1 = timeit.default_timer()
print("Done!!!. Total time cost: {}".format(t1 - t0))