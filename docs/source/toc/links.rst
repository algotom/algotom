Quick links
===========

-   How to set up Python workspace for coding and installing libraries:

    +   `Section 4.1. <https://algotom.readthedocs.io/en/latest/toc/section4/section4_1.html>`__
    +   `Section 1.1. <https://algotom.readthedocs.io/en/latest/toc/section1/section1_1.html>`__

-   How to read/write or explore hdf/nxs/h5 files:

    +   `Section 4.2.1. <https://algotom.readthedocs.io/en/latest/toc/section4/section4_2.html#nxs-hdf-files>`__
    +   `Section 1.2.1 <https://algotom.readthedocs.io/en/latest/toc/section1/section1_2.html#hdf-format>`__
    +   `Broh5 software. <https://github.com/algotom/broh5>`__
    +   `API of loading a hdf file. <https://algotom.readthedocs.io/en/latest/toc/api/algotom.io.loadersaver.html#algotom.io.loadersaver.load_hdf>`__
    +   `API of writing to a hdf file. <https://algotom.readthedocs.io/en/latest/toc/api/algotom.io.loadersaver.html#algotom.io.loadersaver.open_hdf_stream>`__
    +   `Script for exploring a hdf file. <https://github.com/algotom/algotom/blob/master/examples/example_01_explore_hdf_tomo_data.py>`__

-   How to read/write tiff images:

    +   `Section 1.2.2. <https://algotom.readthedocs.io/en/latest/toc/section1/section1_2.html#tiff-format>`__
    +   `Section 4.2.2. <https://algotom.readthedocs.io/en/latest/toc/section4/section4_2.html#tiff-files>`__
    +   `Loader/saver API. <https://algotom.readthedocs.io/en/latest/toc/api/algotom.io.loadersaver.html>`__

-   How to process standard tomography data:

    +   Workflow:

        *   `Section 1.4. <https://algotom.readthedocs.io/en/latest/toc/section1/section1_4.html>`__
        *   `Section 4.5. <https://algotom.readthedocs.io/en/latest/toc/section4/section4_5.html>`__

    +   Command line interface scripts: few-slices reconstruction, full reconstruction, data reduction:

        *   `Common data processing workflow. <https://github.com/algotom/algotom/tree/master/examples/common_data_processing_workflow>`__

    +   Scripts:

        *   `Reconstructing a few slices of a standard scan. <https://github.com/algotom/algotom/blob/master/examples/example_05_reconstruct_std_scan.py>`__
        *   `Full size reconstruction of a standard scan. <https://github.com/algotom/algotom/blob/master/examples/example_05_reconstruct_std_scan_full_size.py>`__

-   How to process half-acquisition tomography (360-degree scanning with offset center) data:

    +   `Demo script. <https://github.com/algotom/algotom/blob/master/examples/example_02_reconstruct_360_degree_scan_with_offset_center.py>`__
    +   `Section 1.4.8.2. <https://algotom.readthedocs.io/en/latest/toc/section1/section1_4.html#sinogram-stitching-for-a-half-acquisition-scan>`__

-   How to apply distortion correction:

    +   `Reconstruct a scan with distortion correction. <https://github.com/algotom/algotom/blob/master/examples/example_06_reconstruct_std_scan_with_distortion_correction.py>`__
    +   `Use Discorpy for finding distortion coefficients. <https://github.com/DiamondLightSource/discorpy?tab=readme-ov-file#demonstrations>`__

-   How to choose ring artifact removal methods:

    +   `Section 4.4. <https://algotom.readthedocs.io/en/latest/toc/section4/section4_4.html>`__
    +   `Section 4.5.4. <https://algotom.readthedocs.io/en/latest/toc/section4/section4_5.html#tweaking-parameters-of-preprocessing-methods>`__
    +   `Sarepy documentation about ring artifacts in tomography. <https://sarepy.readthedocs.io/toc/section3.html>`__

-   How to find center of rotation (rotation axis):

    +   `Section 4.5.3. <https://algotom.readthedocs.io/en/latest/toc/section4/section4_5.html#finding-the-center-of-rotation>`__

-   How to process time-series tomography data:

    +   `Demo scripts <https://github.com/algotom/algotom/tree/master/examples/time_series_tomography>`__

-   How to process grid-scanning tomography data (tiled scans):

    +   `Reconstruct a few slices. <https://github.com/algotom/algotom/blob/master/examples/example_03_reconstruct_few_slices_grid_scan_with_offset_center.py>`__
    +   `Full reconstruction: step 1. <https://github.com/algotom/algotom/blob/master/examples/example_07_full_reconstruction_a_grid_scan_step_01.py>`__
    +   `Full reconstruction: step 2. <https://github.com/algotom/algotom/blob/master/examples/example_07_full_reconstruction_a_grid_scan_step_02.py>`__
    +   `Full reconstruction: step 3. <https://github.com/algotom/algotom/blob/master/examples/example_07_full_reconstruction_a_grid_scan_step_03_downsample.py>`__

-   How to process helical tomography data:

    +   `Demo script. <https://github.com/algotom/algotom/blob/master/examples/example_04_reconstruct_helical_scan_with_offset_center.py>`__

-   How to perform data reduction of reconstructed volume (cropping, rescaling, downsampling, reslicing,...):

    +   `Section 4.5.8. <https://algotom.readthedocs.io/en/latest/toc/section4/section4_5.html#downsampling-rescaling-and-reslicing-reconstructed-volume>`__
    +   `Command line interface script. <https://github.com/algotom/algotom/blob/master/examples/common_data_processing_workflow/data_reduction_cli.py>`__

-   How to process speckle-based phase-contrast tomography data:

    +   `Section 5.1. <https://algotom.readthedocs.io/en/latest/toc/section5/section5_1.html>`__
    +   `Demo scripts. <https://github.com/algotom/algotom/tree/master/examples/speckle_based_tomography>`__

-   How to correct tilted tomography data:

    +   `Demo script 1. <https://github.com/tomopy/tomopy/issues/602#issuecomment-1440808547>`__
    +   `Demo script 2. <https://github.com/algotom/algotom/blob/master/examples/example_09_generate_tilted_sinogram.py>`__
    +   `Tomography alignment tutorial. <https://algotom.readthedocs.io/en/latest/toc/section1/section1_6.html>`__

-   How to automate the workflow:

    +   `Section 4.5.7. <https://algotom.readthedocs.io/en/latest/toc/section4/section4_5.html#automating-the-workflow>`__
    +   `Utility scripts. <https://github.com/algotom/algotom/tree/master/examples/utilities>`__

-   How tomography works:

    +   `Section 1.3. <https://algotom.readthedocs.io/en/latest/toc/section1/section1_3.html>`__
    +   `Section 1.6. <https://algotom.readthedocs.io/en/latest/toc/section1/section1_6.html>`__

-   How to generate simulated data:

    +   `Simulation module. <https://algotom.readthedocs.io/en/latest/toc/api/algotom.util.simulation.html>`__
    +   `Demo script. <https://github.com/algotom/algotom/blob/master/examples/example_08_generate_simulation_data.py>`__

-   How to customize ring-artifact removal methods:

    +   `Section 4.3. <https://algotom.readthedocs.io/en/latest/toc/section4/section4_3.html>`__

-   Tools for finding image shift, stitching images:

    +   `Correlation module. <https://algotom.readthedocs.io/en/latest/toc/api/algotom.util.correlation.html>`__
    +   `Conversion module. <https://algotom.readthedocs.io/en/latest/toc/api/algotom.prep.conversion.html>`__

-   Tools for phase unwrapping, phase retrieval:

    +   `Phase module. <https://algotom.readthedocs.io/en/latest/toc/api/algotom.prep.phase.html>`__
    +   `Section 5.1.3. <https://algotom.readthedocs.io/en/latest/toc/section5/section5_1.html#data-processing>`__

-   Datasets for testing algorithms:

    +   `Zenodo. <https://zenodo.org/search?q=nghia%20t.%20vo&f=resource_type%3Adataset&l=list&p=1&s=10&sort=bestmatch>`__
    +   `Tomobank. <https://tomobank.readthedocs.io/en/latest/source/data.html>`__

-   Parallel processing, GPU programming, and high-performance computing with Numba:

    +   `Section 1.5. <https://algotom.readthedocs.io/en/latest/toc/section1/section1_5.html>`__
    +   `GPU programming. <https://github.com/algotom/algotom/blob/f096bf2d202efe1261d0a5e14823efba35a2b542/algotom/rec/reconstruction.py#L153>`__
    +   `Compiling python code. <https://github.com/algotom/algotom/blob/f096bf2d202efe1261d0a5e14823efba35a2b542/algotom/rec/reconstruction.py#L265>`__

-   Common mistakes and useful tips:

    +   `Section 4.5.9. <https://algotom.readthedocs.io/en/latest/toc/section4/section4_5.html#common-mistakes-and-useful-tips>`__
