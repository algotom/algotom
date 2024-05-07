Update notes
============

- 13/05/2021:

	+ Publish codes.

- 26/01/2022:

    + Add phase.py module.
    + Add phase-unwrapping methods.

- 20/06/2022:

	+ Add correlation.py module.
	+ Add methods for speckle-based phase-contrast tomography.
	+ Add methods for image alignment.
	+ Release version 1.1.0

- 27/06/2022:

	+ Publish https://algotom.github.io/

- 20/10/2022:

    + Publish implementation of the UMPA method.

- 24/10/2022:

    + Release version 1.2.0

- 03/02/2023:

    + Add reslicing 3D-data method. Increase code coverage.

- 25/03/2023:

    + Add upsampling sinogram method.
    + Add method for finding the center of rotation (COR) using the entropy-based metric.
    + Add utility methods for visually finding the COR using: converted 360-degree sinograms and reconstructed slices.
    + Improve reconstruction methods to process multiple-sinograms.

- 30/03/2023:

    + Improve the performance of the reslicing method.
    + Release version 1.3.0

- 19/11/2023:

    + Add methods for loading and saving multiple tiff images in parallel.
    + Release version 1.4.0

- 24/03/2024:

    + Add calibration methods for tomography alignment.
    + Release version 1.5.0

- 26/04/2024:

    + Add back-projection filtering (BPF) method.
    + Update method for processing slices of 3D array in parallel.
    + Move "find_center_based_slice_metric" and "find_center_based_slice_metric" from utility.py
      to reconstruction.py

- 06/05/2024:

    + Add module **vertrec.py** for direct vertical slice reconstructions: single slice, multiple slices, and multiple
      slices at different orientations; and methods for finding the center of rotation using vertical slices.
    + Add demos for vertical reconstruction.
    + Add technical note on implementations of direct vertical-slice reconstruction for tomography.
    + Enable parallel computing for image-stitching-related methods in calculation.py module
    + Add "sharpness" metric to "find_center_based_slice_metric" method.
    + Release version 1.6.0
