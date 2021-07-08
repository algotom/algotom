=======
Install
=======

Algotom is installable across operating systems (Windows, Ubuntu, Mac) and 
works with Python >=3.7. To install:

From source
-----------

Clone the `Algotom <https://github.com/algotom/algotom>`_  
from `GitHub <https://github.com>`_ repository::

    git clone https://github.com/algotom/algotom.git algotom

Download and install `Miniconda  <https://docs.conda.io/en/latest/miniconda.html>`_ software::

    conda create -n algotom python=3.7
    conda activate algotom
    cd algotom
    python setup.py install


Using conda
-----------

Install Miniconda as instructed above, then:

If install to an existing environment::

    conda install -c algotom algotom

If install to a new environment::

    conda create -n algotom python=3.7
    conda activate algotom
    conda install -c algotom algotom


Using pip
---------

Install Miniconda as instructed above.

If install to an existing environment:: 
      
    pip install algotom`


If install to a new environment::

    conda create -n algotom python=3.7
    conda activate algotom
    pip install algotom
