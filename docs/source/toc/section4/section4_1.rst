Setting up a Python workspace
=============================

This section demonstrates step-by-step how to install Python
libraries, software, and tools; i.e. setting up a workspace for coding; on WinOS
to write Python codes and process tomographic data. There are many ways to set up
a Python workspace. However, we only show approaches which are easy-to-follow
and less troublesome for beginners.

**1. Install Conda, a package manager, to install Python libraries**

    Download Miniconda from `here <https://docs.conda.io/en/latest/miniconda.html>`__
    and install it. After that, run Anaconda Powershell Prompt. This Powershell is
    a command-line interface where users can run commands to install/manage Python
    environments and packages.

        .. image:: section4_1/figs/fig_4_1_1.png
            :name: fig_4_1_1
            :width: 100 %
            :align: center

    There is a list of commands in Conda, but we just need a few of them. The first
    command is to create a new environment. An environment is a collection of Python
    packages. We should create different environments for different usages (such as
    to process tomographic data, write sphinx documentation, or develop a specific Python
    software...) to avoid the conflict between Python libraries. The following
    command will create an environment named *myspace*

        .. code-block:: console

            conda create -n myspace

    Then we must activate this environment before installing Python packages into it.

        .. code-block:: console

            conda activate myspace

    Name of the activated environment with be shown in the command line as below

        .. image:: section4_1/figs/fig_4_1_2.png
            :name: fig_4_1_2
            :width: 100 %
            :align: center

    First things first, we install Python. Here we specify Python 3.9, not the
    latest one, as the Python ecosystem taking time to keep up.

        .. code-block:: console

            conda install python=3.9

    Then we install tomographic packages. A Python package can be distributed
    through its `own channel <https://anaconda.org/algotom>`__,
    the `conda-forge <https://anaconda.org/conda-forge>`__ channel (a huge collection of Python packages),
    `Pypi <https://pypi.org/project/algotom/>`__, or users can download the `source
    codes <https://github.com/algotom/algotom>`__ and install themselves using *setup.py*.
    The order of priority should be: conda-forge, own channel, Pypi, then source codes.
    Let install the Algotom package first using the instruction shown on its
    documentation page.

        .. code-block:: console

            conda install -c conda-forge algotom

    Because Algotom relies on `dependencies <https://github.com/algotom/algotom/blob/master/requirements.txt>`__,
    e.g. Numpy, Numba, Scipy, H5py,... they are also installed at the same time.
    The Python environment and its packages are at *C:/Users/user_ID/miniconda3/envs/myspace*.
    Users can run a Python script, in the activated environment, by

        .. code-block:: console

            python C:/my_project/my_script.py

    or in the Window Command Prompt by providing the absolute path to *python.exe* of the enviroment

        .. code-block:: console

            C:/Users/user_ID/miniconda3/envs/myspace/python C:/my_project/my_script.py

    Other conda commands are often used:

    -   *conda list* : list packages installed in an activated environment.
    -   *conda uninstall <package>* : to uninstall a package.
    -   *conda deactivate* : to deactivate a current environment
    -   *conda remove -n myspace --all* : delete an environment.
    -   *conda info -e* : list environments created.


**2. Install tomography-related, image-processing packages**

    There are a few of tomography packages which users should install along with
    Algotom: `Astra Toolbox <https://www.astra-toolbox.com/docs/install.html>`__
    and `Tomopy <https://tomopy.readthedocs.io/en/stable/install.html#installing-from-conda>`__

        .. code-block:: console

            conda install -c astra-toolbox astra-toolbox

            conda install -c conda-forge tomopy

    For packages using Nvidia GPUs, making sure to install the `CUDA toolkit <https://developer.nvidia.com/cuda-toolkit-archive>`__
    as well. A popular visualization package, `Matplotlib <https://matplotlib.org/stable/api/index>`__, is
    important to check or save results of a workflow.

        .. code-block:: console

            conda install -c conda-forge matplotlib

    If users need to calculate distortion coefficients of a lens-based detector
    of a tomography system, using `Discorpy <https://discorpy.readthedocs.io/en/latest/index.html>`__

        .. code-block:: console

            conda install -c conda-forge discorpy

**3. Install Pycharm for writing and debugging Python codes**

    Pycharm is one of the most favorite IDE software for Python programming. It has
    many features which make it easy for coding such as syntax highlight,
    auto-completion, auto-format, auto-suggestion, typo check, version control,
    or change history. `Pycharm (Community edition) <https://www.jetbrains.com/pycharm/download/>`__
    is free software. After installing, users needs to configure the Python
    interpreter (File->Settings->Project->Python interpreter-> Add ->Conda environment)
    pointing to the created conda environment, *C:/Users/user_ID/miniconda3/envs/myspace*,
    as demonstrated in :ref:`section 1.1 <section1_1>`. It's very easy to create a python file,
    write codes, and run them as shown below.

        .. image:: section4_1/figs/fig_4_1_3.png
            :name: fig_4_1_3
            :width: 100 %
            :align: center

**4. Write and run codes interactively using Jupyter notebook (optional)**

    Using Python scripts is efficient and practical for processing multiple datasets.
    However, if users want to work with data interactively to define a workflow,
    `Jupyter Notebook <https://jupyter-notebook.readthedocs.io/en/latest/>`__ is
    a good choice.

    Install Jupyter in the activated environment

        .. code-block:: console

            conda install -c conda-forge jupyter

    Run the following command to enable the current environment in notebook

        .. code-block:: console

            ipython kernel install --user --name="myspace"

    Then run Jupyter notebook by

        .. code-block:: console

            jupyter notebook

    Select the kernel as shown below

        .. image:: section4_1/figs/fig_4_1_4.png
            :name: fig_4_1_4
            :width: 100 %
            :align: center

    It will create a new tab for inputting codes

        .. image:: section4_1/figs/fig_4_1_5.png
            :name: fig_4_1_5
            :width: 100 %
            :align: center
