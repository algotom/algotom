Usage
-----
 
* Examples of how to use the package are in the examples folder of `Algotom <https://github.com/algotom/algotom>`_. They cover most of use-cases which users can adapt to process their own data.

* Real tomographic data for testing methods can be downloaded from `zenodo <https://www.zenodo.org/search?page=1&size=20&q=tomographic%20data%20nghia%20vo&type=dataset>`_. 

* Methods can also be tested using simulation data as shown in "examples/example_08*.py"

* Users can use `Algotom <https://github.com/algotom/algotom>`_ to re-process some old data collected at synchrotron facilities suffering from:

	* Various types of `ring artifacts <https://sarepy.readthedocs.io/>`_.
	* Cupping artifacts (also known as beam hardening artifacts) which are caused by using: FFT-based reconstruction methods without proper padding; polychromatic X-ray sources; or low-dynamic-range detectors to record high-dynamic-range projection-images.       
  
Methods distributed in Algotom can run on a normal computer which enable users to process these data locally. 

* There are tools and `methods <https://sarepy.readthedocs.io/toc/section5.html>`_ users can use to customize their own algorithms:

	* Methods to transform images back-and-forth between the polar coordinate system and the Cartesian coordinate system.
	* Methods to separate stripe artifacts.
	* Methods to transform back-and-forth between reconstruction images and sinogram images.
