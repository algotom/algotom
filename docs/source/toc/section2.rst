.. _section2:

Features and capabilities
=========================

Algotom is a lightweight package. The software is built on top of a few core Python
libraries to ensure its ease-of-installation. Methods distributed in Algotom have
been developed and tested at synchrotron beamlines where massive datasets are produced.
This factor drives the methods developed to be easy-to-use, robust, and practical.
Some featuring methods in Algotom are as follows:

- Methods in a full data processing pipeline: reading-writing data,
  pre-processing, tomographic reconstruction, and post-processing.

.. _processing_pipeline:

  .. figure:: section2/figs/fig_2_1.png
   :figwidth: 90%
   :name: fig_2_1
   :align: center
   :figclass: align-center

- Methods for processing grid scans (or tiled scans) with the offset rotation-axis
  to multiply double the field-of-view (FOV) of a parallel-beam tomography system.

  .. figure:: section2/figs/fig_2_2.jpg
   :figwidth: 90%
   :name: fig_2_2
   :align: center
   :figclass: align-center

- Methods for processing helical scans (with/without the offset rotation-axis).

  .. figure:: section2/figs/fig_2_3.jpg
   :figwidth: 90%
   :name: fig_2_3
   :align: center
   :figclass: align-center

- Methods for determining the center-of-rotation (COR) and auto-stitching images
  in :ref:`half-acquisition scans <half_acquisition>` (360-degree acquisition with the offset COR).
- Some practical methods developed and implemented for the package:
  zinger removal, tilted sinogram generation, sinogram distortion correction,
  beam hardening correction, DFI (direct Fourier inversion) reconstruction,
  FBP reconstruction, and double-wedge filter for removing sample parts larger
  than the FOV in a sinogram.

  .. figure:: section2/figs/fig_2_4.jpg
   :figwidth: 90%
   :name: fig_2_4
   :align: center
   :figclass: align-center

- Utility methods for customizing ring/stripe artifact removal methods and
  parallelizing computational work.
- Calibration methods for determining pixel-size in helical scans.
- Methods for generating simulation data: phantom creation, sinogram calculation
  based on the Fourier slice theorem, and artifact generation.

  .. figure:: section2/figs/fig_2_5.png
   :figwidth: 90%
   :name: fig_2_5
   :align: center
   :figclass: align-center

- Methods for speckle-based phase-contrast tomography, image correlation, and image alignment.

  .. figure:: section2/figs/fig_2_6.png
   :figwidth: 90%
   :name: fig_2_6
   :align: center
   :figclass: align-center