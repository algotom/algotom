Parallel processing in Python
=============================

Having a multicore CPU, certainly we want to make use of it for parallel processing. This is
easily done using the `Joblib <https://joblib.readthedocs.io/en/latest/>`__ library.
Explanation of the functions is as follow

    .. code-block:: python

        from joblib import Parallel, delayed

        # Note the use of parentheses
        results = Parallel(n_jobs=8, prefer="threads")(delayed(func_name)(func_para1, func_para2) for i in range(i_start, i_stop, i_step))

The first part of the code, :code:`Parallel(n_jobs=8, prefer="threads")` , is to select the number of cores and a `backend method <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html#examples-using-joblib-parallel>`__
for parallelization. The second part of the code, :code:`(delayed()() for ...)` has 3 sub-sections: the name of a function,
its parameters, and the loop. We can also use nested loops

    .. code-block:: python

        results = Parallel(n_jobs=8, prefer="threads")(delayed(func_name)(func_para1, func_para2) for i in range(i_start, i_stop, i_step) \
                                                                                                 for j in range(j_start, j_stop, j_step))

Note that :code:`results` is a list of the outputs of the function used. The order of the items in the list
corresponding to how the loops are defined. The following examples will make things more clear.

    -   Example to show the output order of nested loops:

        .. code-block:: python

            from joblib import Parallel, delayed

            def print_order(i, j):
                print("i = {0}; j = {1} \n".format(i, j))
                return i, j

            results = Parallel(n_jobs=4, prefer="threads")(delayed(print_order)(i, j) for i in range(0, 2, 1) \
                                                                                      for j in range(2, 4, 1))
            print("Output = ", results)

        .. code-block:: console

            >>>
            i = 0; j = 2
            i = 0; j = 3
            i = 1; j = 2
            i = 1; j = 3
            Output =  [(0, 2), (0, 3), (1, 2), (1, 3)]

    -   Example to show how to apply a smoothing filter to multiple images in parallel

        .. code-block:: python

            import timeit
            import multiprocessing as mp
            import numpy as np
            import scipy.ndimage as ndi
            from joblib import Parallel, delayed

            # Select number of cpu cores
            ncore = 16
            if ncore > mp.cpu_count():
                ncore = mp.cpu_count()

            # Create data for testing
            height, width = 3000, 5000
            image = np.zeros((height, width), dtype=np.float32)
            image[1000:2000, 1500:3500] = 1.0
            n_slice = 16
            data = np.moveaxis(np.asarray([i * image for i in range(n_slice)]), 0, 1)
            print(data.shape) # >>> (3000, 16, 5000)

            # Using sequential computing for comparison
            t0 = timeit.default_timer()
            results = []
            for i in range(n_slice):
                mat = ndi.gaussian_filter(data[:, i, :], (3, 5), 0)
                results.append(mat)
            t1 = timeit.default_timer()
            print("Time cost for sequential computing: ", t1 - t0) # >>> 8.831482099999999

            # Using parallel computing
            t0 = timeit.default_timer()
            results = Parallel(n_jobs=16, prefer="threads")(delayed(ndi.gaussian_filter)(data[:, i, :], (3, 5), 0) for i in range(n_slice))
            t1 = timeit.default_timer()
            print("Time cost for parallel computing: ", t1 - t0)   # >>> 0.8372323000000002

            # As the output is a list we have to convert it to a numpy array
            # and reshape to get back the original shape
            results = np.asarray(results)
            print(results.shape)  # >>> (16, 3000, 5000)
            results = np.moveaxis(results, 0, 1)
            print(results.shape)  # >>> (3000, 16, 5000)

        There are several options for choosing the `backend methods <https://joblib.readthedocs.io/en/latest/parallel.html#thread-based-parallelism-vs-process-based-parallelism>`__.
        Depending on the problem and how input data are used, their performance can be significantly different. In the above
        example, the "threads" option gives the best performance. Note that we can't use the above approaches for
        parallel reading or writing data from/to a hdf file. There is a `different way <https://docs.h5py.org/en/stable/mpi.html>`__ of doing these.

For convenient use, Algotom provides a wrapper for the Joblib library. Users can easily apply a method to multiple images or sinograms as follows:

.. code-block:: python

    import time
    import numpy as np
    import multiprocessing as mp
    from scipy.ndimage import gaussian_filter
    import algotom.prep.filtering as filt
    import algotom.util.utility as util

    num_img = 100
    height, width = 2000, 2000
    sim_data = np.random.rand(num_img, 2000, 2000)

    # Serial processing:
    t0 = time.time()
    smooth_data = []
    for i in range(num_img):
       smooth_data.append(gaussian_filter(sim_data[i], 5))
    smooth_data = np.asarray(smooth_data)
    t1 = time.time()
    print(f"Time elapsed for serial processing: {t1-t0}")

    # Parallel processing using Algotom utility module:
    num_core = mp.cpu_count()
    t0 = time.time()
    smooth_data = util.parallel_process_slices(sim_data, gaussian_filter, [5], axis=0, ncore=num_core, prefer="threads")
    t1 = time.time()
    print(f"Time elapsed for parallel processing using 'threads': {t1-t0}. Number of core: {num_core}")

    t0 = time.time()
    smooth_data = util.parallel_process_slices(sim_data, gaussian_filter, [5], axis=0, ncore=num_core, prefer="processes")
    t1 = time.time()
    print(f"Time elapsed for parallel processing using 'processes': {t1-t0}. Number of core: {num_core}")

    # We can pass any processing method and its parameters as follows; check Algotom document to know how to pass arguments.
    t0 = time.time()
    smooth_data = util.parallel_process_slices(sim_data, filt.fresnel_filter, [100.0, 1], axis=0, ncore=num_core, prefer="threads")
    t1 = time.time()
    print(f"Time elapsed for applying Fresnel filter: {t1-t0}. Number of core: {num_core}")

