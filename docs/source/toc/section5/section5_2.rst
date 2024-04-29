.. _section5_2:

Implementation of direct vertical slice reconstruction for tomography
=====================================================================

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

    Figure 2: slice, vertical slice

Last but not least, for samples with multilayer structures parallel to the beam, it is very challenging to find the
center of rotation using conventional reconstructed slices. However, things are much easier when a vertical slice is used.

Given these reasons, it's important to implement this method and make it available to the community. Similar work was
done in Recast3D, but the project is no longer continued; furthermore, its core is in C++ code, making it difficult to
maintain. This section presents work done to enable vertical slice reconstruction. Methods can run on multi-core CPUs
and GPUs using Numba. Two reconstruction methods have been implemented: FBP (Filtered back-projection) and BPF
(Back-projection filtering). Data is processed chunk-by-chunk to fit available RAM or GPU memory. The methods allow
the reconstruction of a single vertical slice or a chunk of vertical slices with selectable gaps between slices.
Utilities for determining the center of rotation automatically and manually are provided.

   Figure 3: projection, vertical slice.

Implementation
--------------

Requirements
++++++++++++

- Methods can run across CPU and Gpu, small or large RAM, gpu memory.
- Slice location can angle can be chosen arbitrary.
- Users can choose to reconstruct single slice or multiple slices.
- Input can by a hdf file, loaded as numpy array or emulate hdf object where data can be
  loaded/ extracted subset into memory. For tif or orther format it can be converted to hdf
  or wrapper into an hdf-emulator to extract subset of data
- Filted backprojection and Backprojection filtering are implemented as they are standard
  and straightforward to implement.
- Users need methods to determine the center of rotation manually and automatically.

Step by step implementations
++++++++++++++++++++++++++++
