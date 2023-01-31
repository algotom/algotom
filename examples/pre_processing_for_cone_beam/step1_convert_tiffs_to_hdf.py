import timeit
import numpy as np
import algotom.io.converter as conv
import algotom.io.loadersaver as losa

input_base = "E:/cone_beam/rawdata/tif_projections/"
output_file = "E:/tmp/projections.hdf"

t0 = timeit.default_timer()
list_files = losa.find_file(input_base + "/*.tif*")
depth = len(list_files)
(height, width) = np.shape(losa.load_image(list_files[0]))
conv.convert_tif_to_hdf(input_base, output_file, key_path='entry/data', crop=(0, 0, 0, 0))
t1 = timeit.default_timer()
print("Done!!!. Total time cost: {}".format(t1 - t0))
