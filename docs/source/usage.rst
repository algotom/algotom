=====
Usage
=====
 
- Examples of how to use the package are in the examples folder of `Algotom <https://github.com/algotom/algotom/tree/master/examples>`_. They cover most of use-cases which users can adapt to process their own data.
    + :download:`explore_hdf_tomo_data.py <../../docs/demo/example_01_explore_hdf_tomo_data.py>`
    + :download:`reconstruct_360_degree_scan_with_offset_center.py <../../docs/demo/example_02_reconstruct_360_degree_scan_with_offset_center.py>`
    + :download:`reconstruct_few_slices_grid_scan_with_offset_center.py <../../docs/demo/example_03_reconstruct_few_slices_grid_scan_with_offset_center.py>`
    + :download:`reconstruct_helical_scan_with_offset_center.py <../../docs/demo/example_04_reconstruct_helical_scan_with_offset_center.py>`
    + :download:`reconstruct_std_scan_full_size.py <../../docs/demo/example_05_reconstruct_std_scan_full_size.py>`
    + :download:`reconstruct_std_scan.py <../../docs/demo/example_05_reconstruct_std_scan.py>`
    + :download:`reconstruct_std_scan_with_distortion_correction.py <../../docs/demo/example_06_reconstruct_std_scan_with_distortion_correction.py>`
    + :download:`full_reconstruction_a_grid_scan_step_01.py <../../docs/demo/example_07_full_reconstruction_a_grid_scan_step_01.py>`
    + :download:`full_reconstruction_a_grid_scan_step_02.py <../../docs/demo/example_07_full_reconstruction_a_grid_scan_step_02.py>`
    + :download:`full_reconstruction_a_grid_scan_step_03_downsample.py <../../docs/demo/example_07_full_reconstruction_a_grid_scan_step_03_downsample.py>`
    + :download:`generate_simulation_data.py <../../docs/demo/example_08_generate_simulation_data.py>`
    + :download:`generate_tilted_sinogram.py <../../docs/demo/example_09_generate_tilted_sinogram.py>`
    + :download:`pre_process_data_in_the_projection_space.py <../../docs/demo/example_10_pre_process_data_in_the_projection_space.py>`
    + :download:`reconstruct_few_slices_each_frame_in_time_series_tomography <../../docs/demo/time_series_tomography/reconstruct_few_slices_each_frame.py>`
    + :download:`reconstruct_full_each_frame_in_time_series_tomography <../../docs/demo/time_series_tomography/reconstruct_full_each_frame.py>`
    + :download:`reconstruct_single_slice_each_frame_in_time_series_tomography <../../docs/demo/time_series_tomography/reconstruct_single_slice_every_frame.py>`

- Examples of how to use Algotom for processing speckle-based phase-contrast tomography:
	+ :download:`retrieve_phase_single_projection <../../docs/demo/speckle_based_tomography/retrieve_phase_single_projection.py>`
	+ :download:`retrieve_phase_all_projections <../../docs/demo/speckle_based_tomography/retrieve_phase_all_projections.py>`
	+ :download:`reconstruct_single_slice_from_raw_data <../../docs/demo/speckle_based_tomography/reconstruct_single_slice_from_raw_data.py>`
	+ :download:`reconstruct_from_phase_projections <../../docs/demo/speckle_based_tomography/reconstruct_from_phase_projections.py>`

- Real tomographic data for testing methods can be downloaded from `zenodo <https://www.zenodo.org/search?page=1&size=20&q=tomographic%20data%20nghia%20vo&type=dataset>`_.

- Methods can also be tested using simulation data as shown in "examples/example_08*.py"

- Users can use Algotom to re-process some old data collected at synchrotron facilities suffering from:
    +   Various types of `ring artifacts <https://sarepy.readthedocs.io/>`_
    +   Cupping artifacts (also known as beam hardening artifacts) which are caused by using:
        FFT-based reconstruction methods without proper padding; polychromatic X-ray sources;
        or low-dynamic-range detectors to record high-dynamic-range projection-images.

- Methods distributed in Algotom can run on a normal computer which enable users to process these data locally.

- There are tools and `methods <https://sarepy.readthedocs.io/toc/section5.html>`_ users can use to customize their own algorithms:
	+   Methods to transform images back-and-forth between the polar coordinate system and the Cartesian coordinate system.
	+   Methods to separate stripe artifacts.
	+   Methods to transform back-and-forth between reconstruction images and sinogram images.
