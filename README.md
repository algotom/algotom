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
processing their own data.

The software is made available for the paper; *"Data processing methods and data 
acquisition for samples larger than the field of view in parallel-beam tomography,"*
Nghia T. Vo, Robert C. Atwood, Michael Drakopoulos, and Thomas Connolley, Opt. 
Express 29, 17849-17874 (2021); https://doi.org/10.1364/OE.418448. Selected
answers to technical questions of anonymous reviewers about methods in the paper
is [here](https://www.researchgate.net/profile/Nghia-T-Vo/publication/351559034_Selected_replies_to_technical_questions_from_reviewerspdf/data/609d2c69a6fdcc9aa7e697ea/Selected-replies-to-technical-questions-from-reviewers.pdf).     

> "... Algotom development was started at the I12-JEEP beamline in 2014 as Python
> codes to process data acquired by the beamline's large field-of-view (FOV) detector, 
> which uses two imaging sensors to cover a rectangular FOV. Images from these 
> cameras must be stitched before tomographic reconstruction can take place. 
> Data processing methods for improving the quality of tomographic data; 
> removing artifacts caused by imperfections of hardware components; 
> making use the beamline capabilities; processing data acquired by non-traditional
> scanning techniques; and automating data processing pipeline have been actively
> developed at I12 over the years. These methods have been used internally by I12's
> users and refined further for publication and sharing with the research community
> through open-source software such as Tomopy and Savu ...
> 
> ... In contrast to Savu and Tomopy which are optimized for speed, Algotom is a 
> package of data processing algorithms and tools which are designed to be 
> easy-to-use and easy-to-deploy prototype methods. The development of Algotom 
> has focused on pre-processing methods which work in the sinogram space to 
> reduce computational cost. Methods working in the projection space such as 
> phase filter, distortion correction, or rotation correction have been adapted 
> to work in the sinogram space..." 

Features
--------
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
  
  ![grid_scan](https://github.com/algotom/algotom/raw/master/figs/readme/grid_scan.jpg)
 
  [![animation](https://github.com/algotom/algotom/raw/master/figs/readme/thumbnail.png)](https://www.youtube.com/watch?v=CNRGutasp0c)
  
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

Author
------

Nghia T. Vo - *Diamond Light Source, UK.*  

How to install
--------------
Algotom is installable across operating systems (Windows, Ubuntu, Mac) and 
works with Python >=3.7. To install:
- From source code:
  + Download or `git clone` the [source code](https://github.com/algotom/algotom) to a local folder.
  + Download and install Miniconda software: https://docs.conda.io/en/latest/miniconda.html
  + Open Linux terminal or Windows command prompt and run the following commands:
      
      `conda create -n algotom python=3.7`
      
      `conda activate algotom`
      
      `cd <path-to-source-code>`
      
      `python setup.py install`
- Using conda:
  + Install Miniconda as instructed above.
  + Open terminal or command prompt and run the following commands:
    * If install to an existing environment:
    
      `conda install -c algotom algotom`
    * If install to a new environment:
      ```commandline
      conda create -n algotom python=3.7
      conda activate algotom
      conda install -c algotom algotom
      ```

- Using pip:
  + Install Miniconda as instructed above.
  + Open terminal or command prompt and run the following commands:
    * If install to an existing environment:  
      
      `pip install algotom`
    * If install to a new environment:
      ```commandline
      conda create -n algotom python=3.7
      conda activate algotom
      pip install algotom
      ```

 How to use
----------
- Documentation: https://algotom.readthedocs.io/en/latest/
- Examples of how to use the package are in the "examples/" folder on [github](https://github.com/algotom/algotom). 
  They cover most of use-cases which users can adapt to process their own data.
- Real tomographic data for testing methods can be downloaded from [zenodo.org](https://www.zenodo.org/search?page=1&size=20&q=tomographic%20data%20nghia%20vo&type=dataset)
- Methods can also be tested using simulation data as shown in "examples/example_08*.py"
- Users can use Algotom to re-process some old data collected at synchrotron facilities suffering from:
  + Various types of [ring artifacts](https://sarepy.readthedocs.io/). 
  + Cupping artifacts (also known as beam hardening artifacts) which 
    are caused by using: FFT-based reconstruction methods without proper padding; 
    polychromatic X-ray sources; or low-dynamic-range detectors to record 
    high-dynamic-range projection-images.       
  
  Methods distributed in Algotom can run on a normal computer which enable users
  to process these data locally. 
- There are tools and [methods](https://sarepy.readthedocs.io/toc/section5.html) users can use to customize their own algorithms:
  + Methods to transform images back-and-forth between the polar coordinate 
    system and the Cartesian coordinate system.
  + Methods to separate stripe artifacts.
  + Methods to transform back-and-forth between reconstruction images and 
    sinogram images.
 
Highlights
-----------

Algotom was used for some experiments featured on media:
- Scanning [Moon rocks and Martian meteorites](https://www.diamond.ac.uk/Home/News/LatestNews/2019/17-07-2019.html) 
  using helical scans with offset rotation-axis. Featured on [Reuters](https://www.reuters.com/article/idUKKCN1UC16V?edition-redirect=uk).
- Scanning [Herculaneum Scrolls](https://www.diamond.ac.uk/Home/News/LatestNews/2019/03-10-2019.html) 
  using grid scans with offset rotation-axis respect to the grid's FOV. Featured on [BBC](https://www.bbc.co.uk/news/av/uk-england-oxfordshire-49926789).
- Scanning ['Little Foot' fossil](https://www.diamond.ac.uk/Home/News/LatestNews/2021/02-03-21.html) 
  using two-camera detector with offset rotation-axis. Featured on [BBC](https://www.bbc.co.uk/news/science-environment-56241509). 
