# ===========================================================================
# Author: Nghia T. Vo
# Description: Examples of how to generate simulation data
# ===========================================================================


import numpy as np
import algotom.io.loadersaver as losa
import algotom.util.simulation as sim
import algotom.prep.calculation as calc
import algotom.prep.removal as rem
import algotom.prep.filtering as filt
import algotom.rec.reconstruction as rec


# Where to save the outputs
output_base = "E:/tmp/output/"

size = 1024
# Generate a built-in phantom
phantom = sim.make_face_phantom(size)
losa.save_image(output_base + "/face_phantom.tif", phantom)
angles = np.linspace(0.0, 180.0, size) * np.pi / 180.0

# Generate sinogram
sinogram = sim.make_sinogram(phantom, angles)
losa.save_image(output_base + "/sinogram.tif", sinogram)
# Find center of rotation
center = calc.find_center_vo(sinogram)
# Reconstruct using the DFI method
rec_image = rec.dfi_reconstruction(sinogram, center, apply_log=False)
losa.save_image(output_base + "/recon_dfi.tif", rec_image)
# Reconstruct using the FBP (GPU) method
rec_image = rec.fbp_reconstruction(sinogram, center, apply_log=False)
losa.save_image(output_base + "/recon_fbp.tif", rec_image)

# Convert to X-ray image
sinogram = sim.convert_to_Xray_image(sinogram)
# Add noise
sinogram = sim.add_noise(sinogram, noise_ratio=0.1)
# Add stripe artifacts
sinogram = sim.add_stripe_artifact(sinogram, 2, size // 4, strength_ratio=0.3,
                                   stripe_type="partial")
sinogram = sim.add_stripe_artifact(sinogram, 1, size // 3, strength_ratio=0.3,
                                   stripe_type="full")
sinogram = sim.add_stripe_artifact(sinogram, 2, size // 2 + size // 3,
                                   strength_ratio=0.7, stripe_type="dead")
sinogram = sim.add_stripe_artifact(sinogram, 1, size // 2 - size // 4,
                                   strength_ratio=0.2,
                                   stripe_type="fluctuating")
losa.save_image(output_base + "/sinogram_with_artifacts.tif", sinogram)
# Reconstruct
rec_image = rec.dfi_reconstruction(sinogram, center)
losa.save_image(output_base + "/recon_with_artifacs.tif", rec_image)

# Remove stripe artifacts
sinogram = rem.remove_all_stripe(sinogram, 2.0, 9, 5)
# Denoise
sinogram = filt.fresnel_filter(sinogram, 50)
losa.save_image(output_base + "/sinogram_after_artifact_removed.tif", sinogram)
# Reconstruct
rec_image = rec.dfi_reconstruction(sinogram, center)
losa.save_image(output_base + "/recon_after_artifacs_removed.tif", rec_image)
