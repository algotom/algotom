.. _section2:

Features
========

Capabilities
------------

Algotom is a lightweight package. The software is built on top of a few core Python
libraries to ensure its ease-of-installation. Methods distributed in Algotom have
been developed and tested at synchrotron beamlines where massive datasets are produced.
This factor drives the methods developed to be easy-to-use, robust, and practical.
**Algotom can be used on a normal computer to process large tomographic data**.
Some featuring methods in Algotom are as follows:

-   Methods in a full data processing pipeline: reading-writing data,
    pre-processing, tomographic reconstruction, and post-processing.

.. _processing_pipeline:

    .. image:: section2/figs/fig_2_1.png
        :name: fig_2_1
        :width: 100 %
        :align: center

-   Methods for processing grid scans (or tiled scans) with the offset rotation-axis
    to multiply double the field-of-view (FOV) of a parallel-beam tomography system.

    .. image:: section2/figs/fig_2_2.jpg
       :width: 100%
       :name: fig_2_2
       :align: center

-   Methods for processing helical scans (with/without the offset rotation-axis).

    .. image:: section2/figs/fig_2_3.jpg
       :width: 100%
       :name: fig_2_3
       :align: center

-   Methods for determining the center-of-rotation (COR) and auto-stitching images
    in :ref:`half-acquisition scans <half_acquisition>` (360-degree acquisition with the offset COR).
-   Some practical methods developed and implemented for the package:
    zinger removal, tilted sinogram generation, sinogram distortion correction,
    beam hardening correction, DFI (direct Fourier inversion) reconstruction,
    FBP reconstruction, and double-wedge filter for removing sample parts larger
    than the FOV in a sinogram.

    .. image:: section2/figs/fig_2_4.jpg
       :width: 100%
       :name: fig_2_4
       :align: center

-   Utility methods for customizing :ref:`ring/stripe artifact removal methods <section4_3>` and
    parallelizing computational work.
-   Calibration methods for determining pixel-size in helical scans.
-   Methods for generating simulation data: phantom creation, sinogram calculation
    based on the Fourier slice theorem, and artifact generation.

    .. image:: section2/figs/fig_2_5.png
       :width: 100%
       :name: fig_2_5
       :align: center

-   Methods for phase-contrast imaging: phase unwrapping, :ref:`speckle-based phase retrieval <section5_1>`,
    image correlation, and image alignment.

    .. image:: section2/figs/fig_2_6.png
       :width: 100%
       :name: fig_2_6
       :align: center

-   Methods for downsampling, rescaling, and reslicing (+rotating, cropping)
    3D reconstructed image without large memory usage.

    .. image:: section2/figs/fig_2_7.jpg
       :width: 100%
       :name: fig_2_7
       :align: center

-   Direct vertical reconstruction for single slice, multiple slices, and multiple slices at
    different orientations.

    .. image:: section2/figs/fig_2_8.png
       :width: 100%
       :name: fig_2_8
       :align: center

    |

    .. image:: section2/figs/fig_2_9.png
       :width: 100%
       :name: fig_2_9
       :align: center

Development principles
----------------------

-   While Algotom offers a complete set of tools for tomographic data processing covering
    pre-processing, reconstruction, post-processing, data simulation, and calibration techniques;
    its development strongly focuses on pre-processing techniques. This distinction makes it a
    prominent feature among other tomographic software.

-   To ensure that the software can work across platforms and is easy-to-install; dependencies
    are minimized, and only well-maintained `Python libraries <https://github.com/algotom/algotom/blob/master/requirements.txt>`__
    are used.

-   To achieve high-performance computing and leverage GPU utilization while ensuring ease of
    understanding, usage, and software maintenance, Numba is used instead of Cupy or PyCuda.

-   Methods are structured into modules and functions rather than classes to enhance usability,
    debugging, and maintenance.

-   Algotom is highly practical as it can run on computers with or without a GPU, multicore CPUs;
    and accommodates both small and large memory capacities.
