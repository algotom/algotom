Demonstrations
==============

.. toctree::

    section4/section4_1
    section4/section4_2
    section4/section4_3
    section4/section4_4
    section4/section4_5

Examples of how to use the package are under the example folder of `Algotom <https://github.com/algotom/algotom/tree/master/examples>`_.
They cover most of use-cases which users can adapt to process their own data. Examples
of how to process speckle-based phase-contrast tomography is at `here <https://github.com/algotom/algotom/tree/master/examples/speckle_based_tomography>`__.

Users can use Algotom to re-process some old data collected at synchrotron facilities suffering from:

    +   Various types of `ring artifacts <https://sarepy.readthedocs.io/toc/section2.html/>`__.
    +   Cupping artifacts (also known as beam hardening artifacts) which are caused by using:
        FFT-based reconstruction methods without proper padding; polychromatic X-ray sources;
        or low-dynamic-range detectors to record high-dynamic-range projection-images.

There are tools and :ref:`methods <section4_3>` users can use to customize their own algorithms:

	+   Methods to transform images between the polar coordinate system and the Cartesian coordinate system.
	+   Methods to separate stripe artifacts.
	+   Methods to transform an image between the reconstruction space and the sinogram space.

Tomographic data for testing or developing methods can be downloaded from `Zenodo.org <https://zenodo.org/search?page=1&size=20&q=%22tomographic%20data%22%20%26%20%22nghia%20t.%20vo%22&type=dataset#>`__
or `TomoBank <https://tomobank.readthedocs.io/en/latest/>`__. Methods can also be tested using
simulation data as demonstrated `here <https://github.com/algotom/algotom/blob/master/examples/example_08_generate_simulation_data.py>`__.
