import pathlib
import setuptools

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setuptools.setup(
    name="algotom",
    version=open('VERSION').read().strip(),
    author="Nghia Vo",
    author_email="nghia.vo@diamond.ac.uk",
    description="Data processing algorithms for tomography",
    long_description=README,
    long_description_content_type="text/markdown",
    keywords=["Parallel-beam Computed Tomography", "Image Processing",
              "Tomography", "X-ray Imaging"],
    url="https://github.com/algotom/algotom",
    license="Apache 2.0",
    platforms="Any",
    packages=setuptools.find_packages(include=["algotom","algotom.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Image Processing"
    ],
    install_requires=[
        "numpy>=1.15",
        "scipy",
        "numba>=0.50.1",
        "pywavelets",
        "pillow",
        "h5py",
        "joblib"
    ],
    python_requires='>=3.7',
    scripts=['bin/algotom'],
)
