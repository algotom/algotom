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

  .. figure:: img/grid_scan.jpg
   :figwidth: 100 %
   :alt: grid_scan


  .. figure:: img/thumbnail.png
   :figwidth: 100%
   :alt: grid_scan_animation
   :target: https://www.youtube.com/watch?v=CNRGutasp0c
 
  
- Methods for processing helical scans (with/without the offset rotation-axis).
  
  .. figure:: img/helical_scan.jpg
   :figwidth: 100%
   :alt: helical_scan

- Methods for determining the center-of-rotation (COR) and auto-stitching images 
  in half-acquisition scans (360-degree acquisition with the offset COR).
  
- Methods in a full data processing pipeline: reading-writing data, 
  pre-processing, tomographic reconstruction, and post-processing.
  
  .. figure:: img/data_processing_space.png
   :figwidth: 100%
   :alt: data_processing_space

- Some practical methods developed and implemented for the package:
  zinger removal, tilted sinogram generation, sinogram distortion correction, 
  beam hardening correction, DFI (direct Fourier inversion) reconstruction, 
  and double-wedge filter for removing sample parts larger than the FOV in
  a sinogram.
  
  .. figure:: img/double_wedge_filter.jpg
   :figwidth: 100%
   :alt: double_wedge_filter
  
- Utility methods for customizing ring/stripe artifact removal methods and 
  parallelizing computational work.

- Calibration methods for determining pixel-size in helical scans.
- Methods for generating simulation data: phantom creation, sinogram calculation
  based on the Fourier slice theorem, and artifact generation.

  .. figure:: img/simulation.png
   :figwidth: 100%
   :alt: simulation

.. contents:: Contents:
   :local:

