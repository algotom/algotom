#!/user/conda/envs/2023-3.3-py310/bin/python

# ============================================================================
# Author: Nghia T. Vo
# E-mail: nvo@bnl.gov
# Description: Script for checking the alignment of a parallel-beam
# tomography system given projections of a dense sphere scanned in the
# range of [0; 360] degrees.
# ============================================================================
# Contributors:
# ============================================================================

import os
import argparse
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage import measure, segmentation
import algotom.io.loadersaver as losa
import algotom.util.calibration as calib

usage = """
This script is used for checking the alignment of a parallel-beam tomography 
system given projections of a dense sphere scanned in the range of [0; 360] 
degrees. Acceptable file formats: tif, hdf, or nxs.
"""

parser = argparse.ArgumentParser(description=usage)
parser.add_argument("-p", dest="proj_path", help="Path to projection files",
                    required=True)
parser.add_argument("-f", dest="flat_path", help="Path to flat-field files",
                    required=False,
                    default=None)
parser.add_argument("-k", dest="hdf_key",
                    help="Key to the dataset if inputs are in hdf/nxs/h5 format",
                    required=False, default=None)
parser.add_argument("-l", "--cleft", dest="crop_left",
                    help="Crop left (from the image border)",
                    default=0, type=int, required=False)
parser.add_argument("-r", "--cright", dest="crop_right", help="Crop right",
                    default=0, type=int, required=False)
parser.add_argument("-t", "--ctop", dest="crop_top", help="Crop top",
                    default=0, type=int, required=False)
parser.add_argument("-b", "--cbottom", dest="crop_bottom", help="Crop bottom",
                    default=0, type=int, required=False)
parser.add_argument("-m", "--method", dest="method",
                    help="Select method: 'ellipse' or 'linear'",
                    default="ellipse", type=str, required=False)
parser.add_argument("--ratio", dest="ratio",
                    help="To adjust the threshold for binarization",
                    default=1.0, type=float, required=False)

args = parser.parse_args()
proj_path = args.proj_path
flat_path = args.flat_path
hdf_key = args.hdf_key
crop_left = args.crop_left
crop_right = args.crop_right
crop_top = args.crop_top
crop_bottom = args.crop_bottom
method = args.method
ratio = args.ratio

print("\n*********************************************\n")
print("          Start the script!!!\n                ")
print("*********************************************")


def check_path(input_path):
    file_format = None
    path_exist = os.path.exists(input_path)
    if not path_exist:
        raise ValueError(f"The path '{input_path}' does not exist.")
    if os.path.isfile(input_path) and input_path.lower().endswith(
            ('.hdf', '.nxs', '.h5')):
        file_format = "hdf"
        if hdf_key is None:
            raise ValueError(
                "Please provide key to the dataset in the hdf file!!!")
    elif os.path.isfile(input_path) and input_path.lower().endswith('.tif',
                                                                    '.tiff'):
        file_format = "tif"
    elif os.path.isdir(input_path):
        file_format = "folder"
        if len(losa.find_file(input_path + "/*.tif*")) == 0:
            raise ValueError(f"No tif files in the folder: '{input_path}'")
    else:
        raise ValueError(f"The path '{input_path}' does not match the "
                         f"specified file formats or a folder.")
    return file_format


check_proj = check_path(proj_path)
if check_proj == "hdf":
    proj_data = losa.load_hdf(proj_path, hdf_key)
elif check_proj == "folder":
    proj_files = losa.find_file(proj_path + "/*.tif*")
    proj_data = np.asarray([losa.load_image(file) for file in proj_files])
else:
    raise ValueError("Input must be a hdf file or a folder of tif files!!!")

have_flat = False
if flat_path is not None:
    check_flat = check_path(flat_path)
    if check_flat == "hdf":
        flat = np.mean(np.asarray(losa.load_hdf(flat_path, hdf_key)[:]),
                       axis=0)
    elif check_proj == "folder":
        flat_files = losa.find_file(flat_path + "/*.tif*")
        flat = np.mean(
            np.asarray([losa.load_image(file) for file in flat_files]), axis=0)
    else:
        flat = losa.load_image(flat_path)
    flat[flat == 0.0] = np.mean(flat)
    have_flat = True

if method == "ellipse":
    fit_ellipse = True
else:
    fit_ellipse = False

figsize = (15, 7)
(depth, height, width) = proj_data.shape
left = crop_left
right = width - crop_right
top = crop_top
bottom = height - crop_bottom
width_cr = right - left
height_cr = bottom - top

x_centers = []
y_centers = []
img_list = []
print("\n=============================================")
print("Extract the sphere and get its center-of-mass\n")


def remove_non_round_objects(binary_image, ratio_threshold=0.9):
    """
    To clean binary image and remove non-round objects
    """
    binary_image = segmentation.clear_border(binary_image)
    binary_image = ndi.binary_fill_holes(binary_image)
    label_image = measure.label(binary_image)
    properties = measure.regionprops(label_image)
    mask = np.zeros_like(binary_image, dtype=bool)
    # Filter objects based on the axis ratio
    for prop in properties:
        if prop.major_axis_length > 0:
            axis_ratio = prop.minor_axis_length / prop.major_axis_length
            if axis_ratio > ratio_threshold:
                mask[label_image == prop.label] = True
    # Apply mask to keep only round objects
    filtered_image = np.logical_and(binary_image, mask)
    return filtered_image


for i, img in enumerate(proj_data):
    # Crop image and perform flat-field correction
    if have_flat:
        mat = img[top: bottom, left:right] / flat[top: bottom, left:right]
    else:
        mat = img[top: bottom, left:right]
    # Denoise
    mat = ndi.gaussian_filter(mat, 2)
    # Normalize the background.
    # Optional, should be used if there's no flat-field.
    mat = calib.normalize_background_based_fft(mat, 5)
    threshold = calib.calculate_threshold(mat, bgr='bright')
    # Binarize the image
    mat_bin0 = calib.binarize_image(mat, threshold=ratio * threshold,
                                    bgr='bright')
    # Optional, clean the image.
    mat_bin0 = remove_non_round_objects(mat_bin0)
    nmean = np.sum(mat_bin0)
    if nmean == 0.0:
        print("\n******************************************************")
        print("Adjust threshold or the field of view to get the sphere!")
        print("Current threshold used: {}".format(threshold))
        print("********************************************************")
        plt.figure(figsize=figsize)
        plt.imshow(mat, cmap="gray")
        plt.show()
        raise ValueError("No binary object selected!")
    # Keep the sphere only
    sphere_size = calib.get_dot_size(mat_bin0, size_opt="max")
    mat_bin = calib.select_dot_based_size(mat_bin0, sphere_size)
    (y_cen, x_cen) = ndi.center_of_mass(mat_bin)
    x_centers.append(x_cen)
    y_centers.append(height_cr - y_cen)
    img_list.append(mat)
    print("  ---> Done image: {}".format(i))
x = np.float32(x_centers)
y = np.float32(y_centers)
img_list = np.asarray(img_list)
img_overlay = np.min(img_list, axis=0)


def fit_points_to_ellipse(x, y):
    if len(x) != len(y):
        raise ValueError("x and y must have the same length!!!")
    A = np.array([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)]).T
    vh = np.linalg.svd(A, full_matrices=False)[-1]
    a0, b0, c0, d0, e0, f0 = vh.T[:, -1]
    denom = b0 ** 2 - 4 * a0 * c0
    msg = "Can't fit to an ellipse!!!"
    if denom == 0:
        raise ValueError(msg)
    xc = (2 * c0 * d0 - b0 * e0) / denom
    yc = (2 * a0 * e0 - b0 * d0) / denom
    roll_angle = np.rad2deg(
        np.arctan2(c0 - a0 - np.sqrt((a0 - c0) ** 2 + b0 ** 2), b0))
    if roll_angle > 90.0:
        roll_angle = - (180 - roll_angle)
    if roll_angle < -90.0:
        roll_angle = (180 + roll_angle)
    a_term = 2 * (a0 * e0 ** 2 + c0 * d0 ** 2 - b0 * d0 * e0 + denom * f0) * (
            a0 + c0 + np.sqrt((a0 - c0) ** 2 + b0 ** 2))
    if a_term < 0.0:
        raise ValueError(msg)
    a_major = -2 * np.sqrt(a_term) / denom
    b_term = 2 * (a0 * e0 ** 2 + c0 * d0 ** 2 - b0 * d0 * e0 + denom * f0) * (
            a0 + c0 - np.sqrt((a0 - c0) ** 2 + b0 ** 2))
    if b_term < 0.0:
        raise ValueError(msg)
    b_minor = -2 * np.sqrt(b_term) / denom
    if a_major < b_minor:
        a_major, b_minor = b_minor, a_major
        if roll_angle < 0.0:
            roll_angle = 90 + roll_angle
        else:
            roll_angle = -90 + roll_angle
    return roll_angle, a_major, b_minor, xc, yc


# Calculate the tilt and roll using an ellipse-fit or a linear-fit method

if fit_ellipse is True:
    (a, b) = np.polyfit(x, y, 1)[:2]
    dist_list = np.abs(a * x - y + b) / np.sqrt(a ** 2 + 1)
    dist_list = ndi.gaussian_filter1d(dist_list, 2)
    if np.max(dist_list) < 1.0:
        fit_ellipse = False
        print("\nDistances of points to a fitted line is small, "
              "Use a linear-fit method instead!\n")

if fit_ellipse is True:
    try:
        result = fit_points_to_ellipse(x, y)
        roll_angle, major_axis, minor_axis, xc, yc = result
        tilt_angle = np.rad2deg(np.arctan2(minor_axis, major_axis))
    except ValueError:
        # If can't fit to an ellipse, using a linear-fit method instead
        fit_ellipse = False
        print("\nCan't fit points to an ellipse, using a linear-fit "
              "method instead!\n")

if fit_ellipse is False:
    (a, b) = np.polyfit(x, y, 1)[:2]
    dist_list = np.abs(a * x - y + b) / np.sqrt(a ** 2 + 1)
    appr_major = np.max(np.asarray(
        [np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) for i in
         range(len(x)) for j in range(i + 1, len(x))]))
    dist_list = ndi.gaussian_filter1d(dist_list, 2)
    appr_minor = 2.0 * np.max(dist_list)
    tilt_angle = np.rad2deg(np.arctan2(appr_minor, appr_major))
    roll_angle = np.rad2deg(np.arctan(a))

print("=============================================")
print("Roll angle: {} degree".format(roll_angle))
print("Tilt angle: {} degree".format(tilt_angle))
print("=============================================\n")

# Show the results
plt.figure(1, figsize=figsize)
plt.imshow(img_overlay, cmap="gray", extent=(0, width_cr, 0, height_cr))
plt.tight_layout(rect=[0, 0, 1, 1])

plt.figure(0, figsize=figsize)
plt.plot(x, y, marker="o", color="blue")
plt.title(
    "Roll : {0:2.4f}; Tilt : {1:2.4f} (degree)".format(roll_angle, tilt_angle))
if fit_ellipse is True:
    # Use parametric form for plotting the ellipse
    angle = np.radians(roll_angle)
    theta = np.linspace(0, 2 * np.pi, 100)
    x_fit = (xc + 0.5 * major_axis * np.cos(theta) * np.cos(
        angle) - 0.5 * minor_axis * np.sin(theta) * np.sin(angle))
    y_fit = (yc + 0.5 * major_axis * np.cos(theta) * np.sin(
        angle) + 0.5 * minor_axis * np.sin(theta) * np.cos(angle))
    plt.plot(x_fit, y_fit, color="red")
else:
    plt.plot(x, a * x + b, color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
