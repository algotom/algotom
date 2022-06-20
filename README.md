# Algotom
### Data processing (**ALGO**)rithms for (**TOM**)ography.

![logo](https://github.com/algotom/algotom/raw/master/figs/readme/logo2.png)

**Algotom** is a Python package implementing methods for processing tomographic
data acquired by non-standard scanning techniques such as grid scans, helical 
scans, half-acquisition scans, or their combinations. Certainly, Algotom can 
also be used for standard scans. The software includes methods in a full 
pipeline of data processing: reading-writing data, pre-processing, tomographic 
reconstruction, post-processing, and data simulation. Many utility methods are 
provided to help users quickly develop prototype-methods or build a pipeline for
processing their own data. From version 1.1, methods for speckle-based phase-contrast
tomography were added to the package.

The software is made available for the paper; *"Data processing methods and data 
acquisition for samples larger than the field of view in parallel-beam tomography,"*
Nghia T. Vo, Robert C. Atwood, Michael Drakopoulos, and Thomas Connolley, Opt. 
Express 29, 17849-17874 (2021); https://doi.org/10.1364/OE.418448.      

Features
--------

Algotom is a lightweight package. The software is built on top of a few core
Python libraries to ensure its ease-of-installation. Methods distributed in 
Algotom have been developed and tested at synchrotron beamlines where massive
datasets are produced. This factor drives the methods developed to be easy-to-use, 
robust, and practical. Some featuring methods in Algotom are as follows:
- Methods for processing grid scans (or tiled scans) with the offset rotation-axis 
  to multiply double the field-of-view (FOV) of a parallel-beam tomography system.
  
  ![grid_scan](https://github.com/algotom/algotom/raw/master/figs/readme/grid_scan.jpg)
  
- Methods for processing helical scans (with/without the offset rotation-axis).
  
  ![helical_scan](https://github.com/algotom/algotom/raw/master/figs/readme/helical_scan.jpg)

- Methods for determining the center-of-rotation (COR) and auto-stitching images 
  in half-acquisition scans (360-degree acquisition with the offset COR).
  
- Methods in a full data processing pipeline: reading-writing data, 
  pre-processing, tomographic reconstruction, and post-processing.
  
  ![pipe_line](https://github.com/algotom/algotom/raw/master/figs/readme/data_processing_space.png) 

- Some practical methods developed and implemented for the package:
  zinger removal, tilted sinogram generation, sinogram distortion correction, 
  beam hardening correction, DFI (direct Fourier inversion) reconstruction, 
  and double-wedge filter for removing sample parts larger than the FOV in
  a sinogram.
  
  ![pipe_line](https://github.com/algotom/algotom/raw/master/figs/readme/double_wedge_filter.jpg)
  
- Utility methods for customizing ring/stripe artifact removal methods and 
  parallelizing computational work.
- Calibration methods for determining pixel-size in helical scans.
- Methods for generating simulation data: phantom creation, sinogram calculation
  based on the Fourier slice theorem, and artifact generation.
  
  ![simulation](https://github.com/algotom/algotom/raw/master/figs/readme/simulation.png)
- Methods for speckle-based phase-contrast tomography, image correlation, and image alignment.

Update notes
------------

- 13/05/2021: Publish codes
- 26/01/2022:
  + Add phase.py module 
  + Add phase-unwrapping methods
- 20/06/2022:
  + Add correlation.py module.
  + Add methods for speckle-based phase-contrast tomography.
  + Add methods for image alignment.
  + Release version 1.1

Author
------

- Nghia T. Vo - *Diamond Light Source, UK.*  

How to install
--------------

- https://algotom.readthedocs.io/en/latest/install.html
- If users install Algotom to an existing enviroment and Numba fails to install due to the requests of Numpy < 1.22:
  + Downgrade Numpy to 1.21 and install Algotom/Numba again.
  + Create a new environment and install Algotom first, then other packages.
  + Use conda instead of pip.
- Avoid to use the latest version of Python (e.g 3.10) or Numpy (1.22) as the Python ecosystem taking time to keep up with these twos.

How to use
----------

- https://algotom.readthedocs.io/en/latest/usage.html
 
Highlights
----------

Algotom was used for some experiments featured on media:
- Scanning [Moon rocks and Martian meteorites](https://www.diamond.ac.uk/Home/News/LatestNews/2019/17-07-2019.html) 
  using helical scans with offset rotation-axis. Featured on [Reuters](https://www.reuters.com/article/idUKKCN1UC16V?edition-redirect=uk).
- Scanning [Herculaneum Scrolls](https://www.diamond.ac.uk/Home/News/LatestNews/2019/03-10-2019.html) 
  using grid scans with offset rotation-axis respect to the grid's FOV. Featured on [BBC](https://www.bbc.co.uk/news/av/uk-england-oxfordshire-49926789).
- Scanning ['Little Foot' fossil](https://www.diamond.ac.uk/Home/News/LatestNews/2021/02-03-21.html) 
  using two-camera detector with offset rotation-axis. Featured on [BBC](https://www.bbc.co.uk/news/science-environment-56241509). 
