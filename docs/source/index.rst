=======
Algotom
=======

Data processing (**ALGO**)rithms for (**TOM**)ography

.. image:: img/logo.png
   :width: 50%
   :alt: logo

**Algotom** is a Python package implementing methods for processing tomographic
data acquired by non-standard scanning techniques such as grid scans, helical 
scans, half-acquisition scans, or their combinations. Certainly, Algotom can 
also be used for standard scans. The software includes methods in a full 
pipeline of data processing: reading-writing data, pre-processing, tomographic 
reconstruction, post-processing, and data simulation. Many utility methods are 
provided to help users quickly develop prototype-methods or build a pipeline for
processing their own data.

The software is made available for :cite:`Vo:21`. Selected
answers to technical questions of anonymous reviewers about methods in the paper
is `here <https://www.researchgate.net/profile/Nghia-T-Vo/publication/351559034_Selected_replies_to_technical_questions_from_reviewerspdf/data/609d2c69a6fdcc9aa7e697ea/Selected-replies-to-technical-questions-from-reviewers.pdf>`_.

*... Algotom development was started at the I12-JEEP beamline in 2014 as Python
codes to process data acquired by the beamline's large field-of-view (FOV) detector, 
which uses two imaging sensors to cover a rectangular FOV. Images from these 
cameras must be stitched before tomographic reconstruction can take place. 
Data processing methods for improving the quality of tomographic data; 
removing artifacts caused by imperfections of hardware components; 
making use the beamline capabilities; processing data acquired by non-traditional
scanning techniques; and automating data processing pipeline have been actively
developed at I12 over the years. These methods have been used internally by I12's
users and refined further for publication and sharing with the research community
through open-source software such as Tomopy and Savu ...*

*... In contrast to Savu and Tomopy which are optimized for speed, Algotom is a 
package of data processing algorithms and tools which are designed to be 
easy-to-use and easy-to-deploy prototype methods. The development of Algotom 
has focused on pre-processing methods which work in the sinogram space to 
reduce computational cost. Methods working in the projection space such as 
phase filter, distortion correction, or rotation correction have been adapted 
to work in the sinogram space...*


Content
-------

.. toctree::
   :maxdepth: 1

   features
   install
   usage
   api
   highlights
   credits
