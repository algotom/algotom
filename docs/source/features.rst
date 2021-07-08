========
Features
========

Algotom is a lightweight package. The software is built on top of a few core
Python libraries to ensure its ease-of-installation. Methods distributed in 
Algotom have been developed and tested at a synchrotron beamline where massive
datasets are produced; image features can change significantly between 
experiments depending on X-ray energy and sample types which can be biological, 
medical, material science, or geological in origin. Users often don't have 
sufficient experience with image processing methods to know how to properly 
tune parameters. All these factors drive the methods developed to be 
easy-to-use, robust, and practical. Some featuring methods in Algotom are as 
follows:


- Methods for processing grid scans (or tiled scans) with the offset rotation-axis 
  to multiply double the field-of-view (FOV) of a parallel-beam tomography system.

  .. image:: img/grid_scan.jpg
   :width: 480px
   :alt: grid_scan

  .. image:: img/thumbnail.png
   :width: 480px
   :alt: grid_scan
 
  
- Methods for processing helical scans (with/without the offset rotation-axis).
  
  .. image:: img/helical_scan.jpg
   :width: 480px
   :alt: helical_scan

- Methods for determining the center-of-rotation (COR) and auto-stitching images 
  in half-acquisition scans (360-degree acquisition with the offset COR).
  
- Methods in a full data processing pipeline: reading-writing data, 
  pre-processing, tomographic reconstruction, and post-processing.
  
  .. image:: img/data_processing_space.png
   :width: 480px
   :alt: data_processing_space

- Some practical methods developed and implemented for the package:
  zinger removal, tilted sinogram generation, sinogram distortion correction, 
  beam hardening correction, DFI (direct Fourier inversion) reconstruction, 
  and double-wedge filter for removing sample parts larger than the FOV in
  a sinogram.
  
  .. image:: img/double_wedge_filter.jpg
   :width: 480px
   :alt: double_wedge_filter
  
- Utility methods for customizing ring/stripe artifact removal methods and 
  parallelizing computational work.

- Calibration methods for determining pixel-size in helical scans.
- Methods for generating simulation data: phantom creation, sinogram calculation
  based on the Fourier slice theorem, and artifact generation.

  .. image:: img/simulation.png
   :width: 480px
   :alt: simulation

.. contents:: Contents:
   :local:

