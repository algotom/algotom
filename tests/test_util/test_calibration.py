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
import algotom.util.calibration as cali


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
        mat_nor = cali.normalize_background(self.bck, 3)
        std_val = np.std(mat_nor)
        self.assertTrue(std_val <= self.var)

        bck_zero = np.copy(self.bck)
        bck_zero[6, 5:15] = 0.0
        mat_nor = cali.normalize_background(bck_zero, 3)
        std_val = np.std(mat_nor)
        self.assertTrue(std_val <= self.var)

    def test_normalize_background_based_fft(self):
        mat_nor = cali.normalize_background_based_fft(self.bck, sigma=5,
                                                      pad=10)
        std_val = np.std(mat_nor)
        self.assertTrue(std_val <= self.var)

        bck_zero = np.copy(self.bck)
        bck_zero[6, 5:15] = 0.0
        mat_nor = cali.normalize_background_based_fft(bck_zero, sigma=5,
                                                      pad=10)
        std_val = np.std(mat_nor)
        self.assertTrue(std_val <= self.var)

        bck_zero = np.pad(bck_zero, ((2, 2), (0, 0)), mode="edge")
        mat_nor = cali.normalize_background_based_fft(bck_zero, sigma=5,
                                                      pad=10)
        std_val = np.std(mat_nor)
        self.assertTrue(std_val <= self.var)

    def test_binarize_image(self):
        bck = 0.5 * np.random.rand(self.hei, self.wid)
        mat_bin = cali.binarize_image(self.mat_dots + bck, bgr="dark",
                                      denoise=False)
        num_dots = ndi.label(mat_bin)[-1]
        self.assertTrue(self.num_dots == num_dots)

        mat_bin = cali.binarize_image(1.5 - self.mat_dots + bck, bgr="bright",
                                      denoise=True, norm=True)
        num_dots = ndi.label(mat_bin)[-1]
        self.assertTrue(self.num_dots == num_dots)

        mat_bin = cali.binarize_image(self.mat_dots + bck, threshold=0.85,
                                      bgr="dark")
        num_dots = ndi.label(mat_bin)[-1]
        self.assertTrue(self.num_dots == num_dots)

        self.assertRaises(ValueError, cali.binarize_image, self.mat_dots + bck,
                          threshold=1.5, denoise=True, bgr="bright")

    def test_calculate_distance(self):
        mat1 = np.zeros((self.hei, self.wid), dtype=np.float32)
        mat2 = np.zeros_like(mat1)
        bck = 0.5 * np.random.rand(self.hei, self.wid)
        mat1[5, 10] = 1.0
        mat1 = np.float32(ndi.binary_dilation(mat1, iterations=3))
        mat2[5, 20] = 1.0
        mat2 = np.float32(ndi.binary_dilation(mat2, iterations=3))
        dis = cali.calculate_distance(mat1 + bck, mat2 + bck, bgr="dark",
                                      denoise=False)
        self.assertTrue(np.abs(dis - 10.0) <= self.eps)

        dis = cali.calculate_distance(mat1 + bck, mat2 + bck, bgr="dark",
                                      size_opt="median", denoise=False)
        self.assertTrue(np.abs(dis - 10.0) <= self.eps)

        dis = cali.calculate_distance(mat1 + bck, mat2 + bck, bgr="dark",
                                      size_opt="mean", denoise=False)
        self.assertTrue(np.abs(dis - 10.0) <= self.eps)

        dis = cali.calculate_distance(mat1 + bck, mat2 + bck, bgr="dark",
                                      size_opt="min", denoise=False)
        self.assertTrue(np.abs(dis - 10.0) <= self.eps)
