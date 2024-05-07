# Algotom
### Data processing (**ALGO**)rithms for (**TOM**)ography.

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/algotom/algotom/algotom_ga.yml) [![Downloads](https://static.pepy.tech/personalized-badge/algotom?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Pypi-downloads)](https://pepy.tech/project/algotom) ![Conda](https://img.shields.io/conda/dn/algotom/algotom?label=conda-downloads) ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/algotom/algotom) ![Conda](https://img.shields.io/conda/pn/algotom/algotom) ![GitHub issues](https://img.shields.io/github/issues-raw/algotom/algotom) ![Conda](https://img.shields.io/conda/dn/conda-forge/algotom?label=conda-forge%20downloads) ![Coverage](https://github.com/algotom/algotom/raw/doc/docs/coverage_report/coverage.svg)


![logo](https://github.com/algotom/algotom/raw/master/figs/readme/logo2.png)

**Algotom** is a Python package designed for tomography data processing. It 
offers a complete data processing pipeline; including reading and writing data, 
pre-processing, tomographic reconstruction, post-processing, data simulation, 
and calibration techniques. The package provides many utility methods to 
assist users in constructing a pipeline for processing their own data or 
developing new methods. Key features of Algotom include a wide range of 
processing methods such as artifact removal, distortion correction, 
speckle-based phase-contrast imaging, data reduction; and the capability of 
processing non-standard tomography acquisitions such as grid scans or helical scans. 
The software stands out for its readability, minimal dependencies, and rich documentation. 
Developed specifically for synchrotron-based tomographic beamlines, Algotom aims to 
maximize data quality, enhance workflow throughput, and exploit full beamline 
capabilities.

Features
--------

Algotom is a lightweight package. The software is built on top of a few core
Python libraries to ensure its ease-of-installation. Methods distributed in 
Algotom have been developed, used, and tested at synchrotron beamlines where massive
datasets are produced. This factor drives the methods developed to be easy-to-use, 
robust, and practical. Algotom can be used on a normal computer to process large 
tomographic data. Some featuring methods in Algotom are as follows:

- Methods in a full data processing pipeline: reading-writing data, 
  pre-processing, tomographic reconstruction, and post-processing.
  
  ![pipe_line](https://github.com/algotom/algotom/raw/master/figs/readme/data_processing_space.png) 
 
- Methods for processing grid scans (or tiled scans) with the offset rotation-axis 
  to multiply double the field-of-view (FOV) of a parallel-beam tomography system.
  These techniques enable high-resolution tomographic scanning of large samples.
  
  ![grid_scan](https://github.com/algotom/algotom/raw/master/figs/readme/grid_scan.jpg)
  
- Methods for processing helical scans (with/without the offset rotation-axis).
  
  ![helical_scan](https://github.com/algotom/algotom/raw/master/figs/readme/helical_scan.jpg)

- Methods for determining the center-of-rotation (COR) and auto-stitching images 
  in half-acquisition scans (360-degree acquisition with the offset COR).

- Practical methods developed and implemented for the package: zinger removal, 
  tilted sinogram generation, sinogram distortion correction, simplified form of Paganin's filter,
  beam hardening correction, DFI (direct Fourier inversion) reconstruction,
  FBP (filtered back-projection) reconstruction, BPF (back-projection filtering) reconstruction, 
  and double-wedge filter for removing sample parts larger than the FOV in a sinogram.
  
  ![pipe_line](https://github.com/algotom/algotom/raw/master/figs/readme/double_wedge_filter.jpg)
  
- Utility methods for [customizing ring/stripe artifact removal methods](https://algotom.readthedocs.io/en/latest/toc/section4/section4_3.html) 
  and parallelizing computational work.

- Calibration methods for helical scans and tomography alignment.

- Methods for generating simulation data: phantom creation, sinogram calculation
  based on the Fourier slice theorem, and artifact generation.
  
  ![simulation](https://github.com/algotom/algotom/raw/master/figs/readme/simulation.png)

- Methods for phase-contrast imaging: phase unwrapping, speckle-based phase retrieval, image correlation, and image alignment.

  ![speckle](https://github.com/algotom/algotom/raw/master/figs/readme/speckle_based_tomography.png)

- Methods for downsampling, rescaling, and reslicing (+rotating, cropping) 
  3D reconstructed image without large memory usage.

  ![reslicing](https://github.com/algotom/algotom/raw/master/figs/readme/reslicing.jpg)

- Direct vertical reconstruction for single slice, multiple slices, and multiple slices at 
  different orientations.
  
  ![vertical_slice1](https://github.com/algotom/algotom/raw/master/figs/readme/direct_vertical_reconstruction.png)

  ![vertical_slice1](https://github.com/algotom/algotom/raw/master/figs/readme/limited_angle_tomography.png)
  
   

Installation
------------

- https://algotom.readthedocs.io/en/latest/toc/section3.html
- If users install Algotom to an existing environment and Numba fails to install due to the latest Numpy:
  + Downgrade Numpy and install Algotom/Numba again.
  + Create a new environment and install Algotom first, then other packages.
  + Use conda instead of pip.
- Avoid using the latest version of Python or Numpy as the Python ecosystem taking time to keep up with these twos.

Usage
-----
- https://algotom.readthedocs.io/en/latest/toc/section4/section4_5.html
- https://algotom.readthedocs.io/en/latest/toc/section1/section1_4.html
- https://algotom.readthedocs.io/en/latest/toc/section4.html
- https://github.com/algotom/algotom/tree/master/examples

Development principles
----------------------

- While Algotom offers a complete set of tools for tomographic data processing covering 
  pre-processing, reconstruction, post-processing, data simulation, and calibration techniques;
  its development strongly focuses on pre-processing techniques. This distinction makes it a
  prominent feature among other tomographic software.   

- To ensure that the software can work across platforms and is easy-to-install; dependencies are minimized, and only 
  well-maintained [Python libraries](https://github.com/algotom/algotom/blob/master/requirements.txt) are used.

- To achieve high-performance computing and leverage GPU utilization while ensuring ease of understanding, usage, and software 
  maintenance, Numba is used instead of Cupy or PyCuda.

- Methods are structured into modules and functions rather than classes to enhance usability, debugging, and maintenance.

- Algotom is highly practical as it can run on computers with or without a GPU, multicore CPUs; and accommodates both small 
  and large memory capacities.

Update notes
------------

- https://algotom.readthedocs.io/en/latest/toc/section6.html

Author
------

- Nghia T. Vo - *NSLS-II, Brookhaven National Lab, USA*; *Diamond Light Source, UK.*  
 
Highlights
----------

Algotom was used for some experiments featured on media:

- Scanning [Moon rocks and Martian meteorites](https://www.diamond.ac.uk/Home/News/LatestNews/2019/17-07-2019.html) 
  using helical scans with offset rotation-axis. Featured on [Reuters](https://www.reuters.com/article/idUKKCN1UC16V?edition-redirect=uk).
 
  ![moon_rock](https://github.com/algotom/algotom/raw/master/figs/readme/Moon_rock_Mars_meteorite.jpg)

- Scanning [Herculaneum Scrolls](https://www.diamond.ac.uk/Home/News/LatestNews/2019/03-10-2019.html) 
  using grid scans with offset rotation-axis respect to the grid's FOV (pixel size of 7.9 micron; 
  total size of 11.3 TB). Featured on [BBC](https://www.bbc.co.uk/news/av/uk-england-oxfordshire-49926789).
  The latest updates on the scroll's reading progress are [here](https://www.nature.com/articles/d41586-023-03212-1).

  ![herculaneum_scroll](https://github.com/algotom/algotom/raw/master/figs/readme/Herculaneum_scroll.jpg)

- Scanning ['Little Foot' fossil](https://www.diamond.ac.uk/Home/News/LatestNews/2021/02-03-21.html) 
  using two-camera detector with offset rotation-axis. Featured on [BBC](https://www.bbc.co.uk/news/science-environment-56241509).
    
  ![little_foot](https://github.com/algotom/algotom/raw/master/figs/readme/Little_foot.jpg)  
