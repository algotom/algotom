.. _section5_2:

Implementations of direct vertical-slice reconstruction for tomography
======================================================================

Introduction
------------

Traditionally, to obtain a vertical slice, we must reconstruct slice-by-slice to a full volume, then perform slicing
across the height of the reconstructed volume. However, this approach is inefficient for thin or elongated samples.
There are unused data volumes where no sample is present but are still retained on disk. It would be more effective to
directly reconstruct vertical slices only around the volume containing the sample.

.. figure:: section5_2/figs/fig_5_2_1.png
    :name: fig_5_2_1
    :figwidth: 100 %
    :align: center
    :figclass: align-center

    Demonstration of how to extract a vertical slice from a tomography dataset. Assume a tomography dataset consists
    of 1800 projections, each sized 2560 (W) x 2160 (H) in 16-bit format, totaling approximately 20 GB. The size of
    a full reconstruction in 32-bit format is about 52 GB. This volume needs to be stored temporarily before
    extracting a vertical slice.

Another important application of vertical slice reconstruction is for limited angle tomography, which is often the case
for tilt-series electron tomography or cryo-soft X-ray tomography. For reconstructed data from this acquisition,
artifacts make it difficult to identify the center of rotation or segment the image. However, if the volume is resliced
vertically, the sample features are complete, which simplifies segmentation or determining the center of rotation.

.. figure:: section5_2/figs/fig_5_2_2.png
    :name: fig_5_2_2
    :figwidth: 100 %
    :align: center
    :figclass: align-center

    Vertical slicing is crucial for analyzing data acquired by limited-angle tomography. (a) Conventionally
    reconstructed slice, showing artifacts caused by missing angles. (b) Same data, represented with a vertical slice.

Last but not least, for certain types of samples and their features, e.g., multilayer structures parallel to the beam,
it is challenging to find the center of rotation or preliminarily evaluate image quality using conventional reconstructed
slices. However, things are much easier when a vertical slice is used.

Given these reasons, it's important to implement this method and make it available to the community. Similar works have
been done elsewhere but have either been discontinued, are not implemented in pure Python, or lack practical features.
This section presents work done to enable vertical slice reconstruction. Methods can run on multi-core CPUs and GPUs
using Numba. Two reconstruction methods have been implemented: FBP (Filtered back-projection) and BPF (Back-projection filtering).
Data is processed chunk-by-chunk to fit available RAM or GPU memory. The methods allow the reconstruction of a single
vertical slice, a chunk of vertical slices with selectable gaps between slices, or multiple vertical slices at different
angles. Utilities for determining the center of rotation automatically and manually are provided.

.. figure:: section5_2/figs/fig_5_2_3.png
    :name: fig_5_2_3
    :figwidth: 100 %
    :align: center
    :figclass: align-center

    Demonstration of direct vertical reconstruction.

Implementation
--------------

Requirements
++++++++++++

-   Slice location and its angle (around the z-axis) can be chosen arbitrarily.
-   Users can choose to reconstruct a single slice or multiple slices.
-   Users don't need a high-specs computer to process data.
-   Methods can run on either multicore CPUs or a single GPU, depending on GPU availability..
-   Data can be read and processed chunk-by-chunk to fit available RAM or GPU memory.
-   Input is an hdf-object, numpy array, or emulated hdf-object; for a normal computer, input must be an hdf
    file from which data can be loaded or an extracted subset into memory. For other formats, it can be converted to hdf
    or wrapped into an hdf-emulator to extract a subset of data.
-   FBP method and BPF method are implemented as they are feasible and practical.
-   Users need methods to manually and automatically determine the center of rotation (rotation axis).

Geometry definition
+++++++++++++++++++

Given a reconstruction space with the dimensions of *Width (W) x Width (W)*, users will input the slice index as an
integer in the range of [0;  *W-1*], along with angle *alpha*. Based on this information, the coordinates of
pixels belonging to a vertical slice can be calculated, as shown in :numref:`fig_5_2_4`. Note that in the vertical
slice plane, the xy coordinates remain the same across the z-slice.

.. figure:: section5_2/figs/fig_5_2_4.png
    :name: fig_5_2_4
    :figwidth: 100 %
    :align: center
    :figclass: align-center

    XY-coordinates of pixels in a vertical slice at different orientations.

Back projection, the ramp filter, and reconstruction
++++++++++++++++++++++++++++++++++++++++++++++++++++

From the coordinates of data points on the slice (in pixel units), we can determine the contributions from different
sinograms to this slice, known as back-projection i.e, sinograms are projected onto the reconstructed line as
demonstrated in  :numref:`fig_5_2_5`

.. figure:: section5_2/figs/fig_5_2_5.png
    :name: fig_5_2_5
    :figwidth: 60 %
    :align: center
    :figclass: align-center

    Contributions of different sinograms to the reconstructed line.

The above routine is applied across the height of projection images.

.. figure:: section5_2/figs/fig_5_2_6.png
    :name: fig_5_2_6
    :figwidth: 60 %
    :align: center
    :figclass: align-center

    Contributions of different projections to the reconstructed slice.

In direct tomographic reconstruction methods, the ramp filter is used to compensate for the non-uniform sampling
rate of tomographic data. The closer a part of the sample is to the rotation axis, the higher the sampling rate; i.e.,
its contribution to projection-images is greater. The ramp filter can be applied to projection images before the back-projection, as shown in
:numref:`fig_5_2_7`. This is the  well-known `Filtered Back-Projection (FBP) method <http://engineering.purdue.edu/~malcolm/pct/CTI_Ch03.pdf>`__.

.. figure:: section5_2/figs/fig_5_2_7.png
    :name: fig_5_2_7
    :figwidth: 100 %
    :align: center
    :figclass: align-center

    Projection image is filtered by the ramp filter before the back-projection.

A problem with this approach is that the ramp filter is applied to every projection image, which means the
computational cost is high. A more practical approach is to apply the ramp filter after the back-projection, known as
the Back-Projection Filtering (BPF) method. In this method, the ramp filter is used only once after the back-projection
of all projection images is complete.

.. figure:: section5_2/figs/fig_5_2_8.png
    :name: fig_5_2_8
    :figwidth: 100 %
    :align: center
    :figclass: align-center

    Demonstration of the Back-Projection Filtering method.


The advantage of BPF over FBP is that a reconstructed slice is less noisy because the summation of projections
in the back-projection process cancels out random noise. In contrast, FBP enhances random noise
(by the ramp filter) before back-projection, which makes the reconstructed slice noisier. The disadvantage of BPF is
that it is not a quantifiable method (i.e., the reconstructed values are not linearly related to the attenuation
coefficients of the sample). Moreover, there are shadow artifacts around strongly absorbing areas, as can be seen by
comparing :numref:`fig_5_2_8` (b) and :numref:`fig_5_2_7` (d).

Despite these disadvantages, BPF is practical due to its lower computational cost and less noisy results. It can be
used for automatically finding the center of rotation. Most importantly, in real applications, users are more interested
in segmenting different features of reconstructed slices rather than measuring attenuation coefficients. For these
reasons, BPF is still considered useful in practice.

Center of rotation determination
++++++++++++++++++++++++++++++++

For a standard tomographic dataset, the center of rotation can be found using a sinogram, 0-degree and 180-degree
projection images, or reconstructed slices, as presented :ref:`here <find_center>`. However, for samples much larger
than the field of view, data with low signal-to-noise ratios, or limited-angle tomography, these methods cannot be
used or do not perform well. In such cases, using metrics from vertical reconstructed slices at different estimated
centers to find the optimal center is handy. In Algotom (version>=1.6.0), three metrics are provided:
`'entropy' <https://doi.org/10.1364/JOSAA.23.001048>`__ , 'sharpness', and 'autocorrelation'.

.. figure:: section5_2/figs/fig_5_2_9.png
    :name: fig_5_2_9
    :figwidth: 100 %
    :align: center
    :figclass: align-center

    Finding the center of rotation using metrics of reconstructed slices: (a) Entropy; (b) Sharpness.

The last two metrics make use of the double-edge artifacts in reconstructed vertical slices caused by an incorrect center
to find the optimal value. The efficiency of each metric can depend on the sample. Finding a robust metric that works for
most cases is still a work in progress. For cases where the provided metrics may not perform well, users have the option
to provide a custom metric function. If none of the automated methods work, a manual method is provided by generating a
series of reconstructed slices at different centers and saving them to disk for visual inspection.

.. figure:: section5_2/figs/fig_5_2_10.png
    :name: fig_5_2_10
    :figwidth: 100 %
    :align: center
    :figclass: align-center

    Finding the center of rotation by visual inspection: (a) Incorrect center; (b) optimal center

Demonstrations
--------------

Practical insights
++++++++++++++++++

**Loading data in chunks**

In vertical slice reconstruction, the entire dataset must be read and processed. To manage this without requiring a
high-spec computer, data must be processed in chunks. When the input is in HDF format, this process is straightforward
because subsets of the HDF file can be accessed directly. For other formats such as TIFF, TXRM, XRM, etc., we need
wrappers to simulate the behavior of HDF files,  allowing subset data to be loaded using `NumPy indexing syntax <https://numpy.org/doc/stable/user/basics.indexing.html>`__,
or by simply converting these file formats to HDF. As the I/O overhead for this reconstruction method is high, the
overall performance depends on the performance of the storage system. A faster I/O system yields faster results.
There is a significant difference in performance between SSD, HDD, and network storage systems.

**Finding the center of rotation**

If the tomographic data is complete, i.e., acquired over the full range of [0-180] degrees, other faster methods can
be used to find the center of rotation. In limited-angle tomography, or where the aforementioned methods do not
perform well, we can measure metrics of vertical slices at different centers. To reduce computational costs, it is
sufficient to process only a small height of projection images.

**Reconstructing multiple slices**

As the time cost of data loading is the same for reconstructing a single slice or multiple slices, it's more
efficient to reconstruct multiple slices at once. This feature is provided in Algotom, which allows users to
reconstruct multiple parallel slices with a selectable step (in pixel units) between slices. Alternatively, users
can choose to reconstruct different slices at different orientations around the z-axis.

**Selecting slice orientation**

Vertical slice reconstruction is most efficient for limited-angle tomography. To minimize artifacts from missing angles,
the optimal orientation for reconstructed vertical slices is perpendicular to the midpoint of the missing angle range.
For thin or rectangular-shaped samples, the slice should be parallel to the longest edge. To automate the determination
of the angle, we can identify the row in a sinogram image giving the minimum intensity (absorption-contrast tomography).

Workflows
+++++++++

The methods described in this technical note are implemented in the module *vertrec.py* within Algotom package. Details
of the API are provided :ref:`here <vertrec_module>`.

The following workflow reconstructs a few vertical slices from raw data under these conditions: the input consists of
hdf files; the center of rotation is calculated using a sinogram-based method; the BPF reconstruction method is used;
and the output is saved as tiff images.

    .. code-block:: python

        import time
        import numpy as np
        import algotom.io.loadersaver as losa
        import algotom.prep.correction as corr
        import algotom.prep.removal as remo
        import algotom.prep.calculation as calc
        import algotom.rec.vertrec as vrec

        output_base = "E:/vertical_slices/"

        proj_file = "E:/Tomo_data/projections.hdf"
        flat_file = "E:/Tomo_data/flats.hdf"
        dark_file = "E:/Tomo_data/darks.hdf"
        key_path = "entry/data/data"

        # Load projection data as a hdf object
        proj_obj = losa.load_hdf(proj_file, key_path)
        (depth, height, width) = proj_obj.shape
        # Load dark-field and flat-field images, average each result
        flat_field = np.mean(np.asarray(losa.load_hdf(flat_file, key_path)), axis=0)
        dark_field = np.mean(np.asarray(losa.load_hdf(dark_file, key_path)), axis=0)
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
        for inc, idx in enumerate(np.arange(start_index, stop_index + 1, step_index)):
            losa.save_image(output_base + f"/slice_{idx:05}.tif", ver_slices[inc])
        t1 = time.time()
        print("All done !!!")

