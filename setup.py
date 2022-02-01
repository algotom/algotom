import pathlib
import setuptools
import sys

py_ver = sys.version.split(".")[:2]
python_version = py_ver[0] + "." + py_ver[1]
if python_version == "3.7":
    dependencies = [
        "numpy>=1.17,<1.21",
        "scipy<=1.7",
        "numba>=0.50.1",
        "pywavelets",
        "pillow",
        "h5py",
        "joblib"]
else:
    dependencies = [
        "numpy>=1.17,<1.21",
        "scipy",
        "numba>=0.50.1",
        "pywavelets",
        "pillow",
        "h5py",
        "joblib"]

current_folder = pathlib.Path(__file__).parent
readme = (current_folder / "README.md").read_text()

setuptools.setup(
    name="algotom",
    version="1.0.3",
    author="Nghia Vo",
    author_email="nghia.vo@diamond.ac.uk",
    description="Data processing algorithms for tomography",
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords=["Parallel-beam Computed Tomography", "Image Processing",
              "Tomography", "X-ray Imaging"],
    url="https://github.com/algotom/algotom",
    license="Apache 2.0",
    platforms="Any",
    packages=setuptools.find_packages(include=["algotom", "algotom.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Image Processing"
    ],
    install_requires=dependencies,
    python_requires='>=3.7',
    scripts=['bin/algotomcli.py'],
    entry_points={'console_scripts': ['algotom = algotomcli:main'], },
)
