# ============================================================================
# ============================================================================
# Copyright (c) 2021 Nghia T. Vo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Author: Nghia T. Vo
# E-mail:  
# Description: Tests for the Algotom package.
# Contributors:
# ============================================================================
"""
Tests for methods in util/calibration.py
"""

import unittest
import numpy as np
import scipy.ndimage as ndi
import algotom.util.calibration as calib


class CalibrationMethods(unittest.TestCase):

    def setUp(self):
        self.eps = 10 ** (-6)
        self.var = 0.05
        sigma = 30
        (self.hei, self.wid) = (64, 64)
        (ycen, xcen) = (self.hei // 2, self.wid // 2)
        y, x = np.ogrid[-ycen:self.hei - ycen, -xcen:self.wid - xcen]
        num = 2.0 * sigma * sigma
        self.bck = np.exp(-(x * x / num + y * y / num))
        mat = np.zeros((self.hei, self.wid), dtype=np.float32)
        self.num_dots = 1
        mat[ycen - 3:ycen + 3, xcen - 3:xcen + 3] = 1
        self.mat_dots = np.float32(ndi.binary_dilation(mat, iterations=2))

    def test_normalize_background(self):
        mat_nor = calib.normalize_background(self.bck, 3)
        std_val = np.std(mat_nor)
        self.assertTrue(std_val <= self.var)

        bck_zero = np.copy(self.bck)
        bck_zero[6, 5:15] = 0.0
        mat_nor = calib.normalize_background(bck_zero, 3)
        std_val = np.std(mat_nor)
        self.assertTrue(std_val <= self.var)

    def test_normalize_background_based_fft(self):
        mat_nor = calib.normalize_background_based_fft(self.bck, sigma=5,
                                                       pad=10)
        std_val = np.std(mat_nor)
        self.assertTrue(std_val <= self.var)

        bck_zero = np.copy(self.bck)
        bck_zero[6, 5:15] = 0.0
        mat_nor = calib.normalize_background_based_fft(bck_zero, sigma=5,
                                                       pad=10)
        std_val = np.std(mat_nor)
        self.assertTrue(std_val <= self.var)

        bck_zero = np.pad(bck_zero, ((2, 2), (0, 0)), mode="edge")
        mat_nor = calib.normalize_background_based_fft(bck_zero, sigma=5,
                                                       pad=10)
        std_val = np.std(mat_nor)
        self.assertTrue(std_val <= self.var)

    def test_binarize_image(self):
        bck = 0.5 * np.random.rand(self.hei, self.wid)
        mat_bin = calib.binarize_image(self.mat_dots + bck, bgr="dark",
                                       denoise=False)
        num_dots = ndi.label(mat_bin)[-1]
        self.assertTrue(self.num_dots == num_dots)

        mat_bin = calib.binarize_image(1.5 - self.mat_dots + bck, bgr="bright",
                                       denoise=True, norm=True)
        num_dots = ndi.label(mat_bin)[-1]
        self.assertTrue(self.num_dots == num_dots)

        mat_bin = calib.binarize_image(self.mat_dots + bck, threshold=0.85,
                                       bgr="dark")
        num_dots = ndi.label(mat_bin)[-1]
        self.assertTrue(self.num_dots == num_dots)

        self.assertRaises(ValueError, calib.binarize_image,
                          self.mat_dots + bck,
                          threshold=1.5, denoise=True, bgr="bright")

    def test_calculate_distance(self):
        mat1 = np.zeros((self.hei, self.wid), dtype=np.float32)
        mat2 = np.zeros_like(mat1)
        bck = 0.5 * np.random.rand(self.hei, self.wid)
        mat1[5, 10] = 1.0
        mat1 = np.float32(ndi.binary_dilation(mat1, iterations=3))
        mat2[5, 20] = 1.0
        mat2 = np.float32(ndi.binary_dilation(mat2, iterations=3))
        dis = calib.calculate_distance(mat1 + bck, mat2 + bck, bgr="dark",
                                       denoise=False)
        self.assertTrue(np.abs(dis - 10.0) <= self.eps)

        dis = calib.calculate_distance(mat1 + bck, mat2 + bck, bgr="dark",
                                       size_opt="median", denoise=False)
        self.assertTrue(np.abs(dis - 10.0) <= self.eps)

        dis = calib.calculate_distance(mat1 + bck, mat2 + bck, bgr="dark",
                                       size_opt="mean", denoise=False)
        self.assertTrue(np.abs(dis - 10.0) <= self.eps)

        dis = calib.calculate_distance(mat1 + bck, mat2 + bck, bgr="dark",
                                       size_opt="min", denoise=False)
        self.assertTrue(np.abs(dis - 10.0) <= self.eps)

    def test_find_tilt_roll(self):
        def __generate_ellipse_points(roll, tilt, a_major, noise=0.1):
            roll = np.deg2rad(roll)
            b_minor = np.abs(np.tan(np.deg2rad(tilt)) * a_major)
            theta = np.linspace(0, 2 * np.pi, 91)
            x = 0.5 * (a_major * np.cos(theta) * np.cos(
                roll) - b_minor * np.sin(theta) * np.sin(roll))
            y = 0.5 * (a_major * np.cos(theta) * np.sin(
                roll) + b_minor * np.sin(theta) * np.cos(roll))
            if noise > 0.0:
                x = x + noise * np.random.rand(len(x))
                y = y + noise * np.random.rand(len(x))
            return x, y

        np.random.seed(1)
        eps = 0.005

        roll1 = 0.05
        tilt1 = 0.08
        x1, y1 = __generate_ellipse_points(roll1, tilt1, 1500, noise=0.0)
        tilt1a, roll1a = calib.find_tilt_roll(x1, y1, method="ellipse")
        tilt1b, roll1b = calib.find_tilt_roll(x1, y1, method="linear")
        self.assertTrue(
            (abs(tilt1 - tilt1a) <= eps) and (abs(roll1 - roll1a) <= eps))
        self.assertTrue(
            (abs(tilt1 - tilt1b) <= eps) and (abs(roll1 - roll1b) <= eps))

        roll1 = -0.05
        tilt1 = 0.08
        x1, y1 = __generate_ellipse_points(roll1, tilt1, 1500, noise=0.0)
        tilt1a, roll1a = calib.find_tilt_roll(x1, y1, method="ellipse")
        tilt1b, roll1b = calib.find_tilt_roll(x1, y1, method="linear")
        self.assertTrue(
            (abs(tilt1 - tilt1a) <= eps) and (abs(roll1 - roll1a) <= eps))
        self.assertTrue(
            (abs(tilt1 - tilt1b) <= eps) and (abs(roll1 - roll1b) <= eps))

        roll1 = -89.8
        tilt1 = 0.5
        x1, y1 = __generate_ellipse_points(roll1, tilt1, 1500, noise=0.0)
        tilt1a, roll1a = calib.find_tilt_roll(x1, y1, method="ellipse")
        self.assertTrue(
            (abs(tilt1 - tilt1a) <= eps) and (abs(roll1 - roll1a) <= eps))

        roll1 = 89.8
        tilt1 = 0.1
        x1, y1 = __generate_ellipse_points(roll1, tilt1, 1500, noise=0.0)
        tilt1a, roll1a = calib.find_tilt_roll(x1, y1, method="ellipse")
        self.assertTrue(
            (abs(tilt1 - tilt1a) <= eps) and (abs(roll1 - roll1a) <= eps))

        roll1 = 0.05
        tilt1 = 0.01
        x2, y2 = __generate_ellipse_points(roll1, tilt1, 1500, noise=0.8)

        with self.assertWarns(UserWarning):
            calib.find_tilt_roll(x2, y2, method="ellipse")
