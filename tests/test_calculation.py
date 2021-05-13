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
# E-mail: algotomography@gmail.com
# Description: Tests for the Algotom package.
# Contributors:
# ============================================================================
"""
Tests for methods in prep/calculation.py
"""

import unittest
import numpy as np
import scipy.ndimage as ndi
import algotom.prep.calculation as calc


class CalculationMethods(unittest.TestCase):

    def setUp(self):
        self.error = 0.15
        self.size = 64

    def test_find_center_vo(self):
        center0 = 31.0
        mat = np.zeros((self.size, self.size), dtype=np.float32)
        mat[20:30, 35:42] = np.float32(1.0)
        sinogram = np.zeros((37, self.size), dtype=np.float32)
        angles = np.linspace(0.0, 180.0, len(sinogram), dtype=np.float32)
        for i, angle in enumerate(angles):
            sinogram[i] = np.sum(ndi.rotate(mat, angle, reshape=False), axis=0)
        center = calc.find_center_vo(sinogram)
        self.assertTrue(np.abs(center - center0) < self.error)

    def test_find_overlap(self):
        overlap = 20
        side = 1
        win_width = 10
        mat1 = np.zeros((self.size, self.size), dtype=np.float32)
        mat2 = np.copy(mat1)
        noise1 = np.float32(0.1 * np.random.rand(self.size, self.size))
        noise2 = np.float32(0.1 * np.random.rand(self.size, self.size))
        mat1 = mat1 + noise1
        mat2 = mat2 + noise2
        mat_cor = np.tile(np.sin(np.arange(overlap) / 3.0), (self.size, 1))
        mat1[:, self.size - overlap:] = np.float32(0.2) + mat_cor
        mat2[:, :overlap] = np.float32(0.2) + mat_cor
        (overlap1, side1, _) = calc.find_overlap(mat1, mat2, win_width)
        self.assertTrue(
            np.abs(overlap1 - overlap) < self.error and side1 == side)

    def test_find_overlap_multiple(self):
        overlap = 20
        side = 1
        win_width = 10
        mat1 = np.zeros((self.size, self.size), dtype=np.float32)
        mat2 = np.copy(mat1)
        mat3 = np.copy(mat1)
        noise1 = np.float32(0.1 * np.random.rand(self.size, self.size))
        noise2 = np.float32(0.1 * np.random.rand(self.size, self.size))
        noise3 = np.float32(0.1 * np.random.rand(self.size, self.size))
        mat1 = mat1 + noise1
        mat2 = mat2 + noise2
        mat3 = mat3 + noise3
        mat_cor1 = np.tile(np.sin(np.arange(overlap) / 3.0), (self.size, 1))
        mat_cor2 = np.tile(np.sin(np.arange(overlap, 0, -1) / 3.0),
                           (self.size, 1))
        mat1[:, self.size - overlap:] = np.float32(0.2) + mat_cor1
        mat2[:, :overlap] = np.float32(0.2) + mat_cor1
        mat2[:, self.size - overlap:] = np.float32(0.2) + mat_cor2
        mat3[:, :overlap] = np.float32(0.2) + mat_cor2
        results = calc.find_overlap_multiple([mat1, mat2, mat3], win_width)
        num1 = np.abs(results[0][0] - overlap)
        num2 = np.abs(results[1][0] - overlap)
        side1 = results[0][1]
        side2 = results[1][1]
        self.assertTrue((num1 < self.error and side1 == side)
                        and (num2 < self.error and side2 == side))

    def test_find_center_360(self):
        mat = np.zeros((self.size, self.size), dtype=np.float32)
        mat[20:30, 35:42] = np.float32(1.0)
        sinogram = np.zeros((73, self.size), dtype=np.float32)
        angles = np.linspace(0.0, 360.0, len(sinogram), dtype=np.float32)
        for i, angle in enumerate(angles):
            sinogram[i] = np.sum(ndi.rotate(mat, angle, reshape=False), axis=0)
        sinogram = sinogram / np.max(sinogram)
        noise = 0.1 * np.random.rand(73, self.size)
        sinogram = np.pad(sinogram[:, 22:], ((0, 0), (0, 22)), mode='constant')
        sinogram = sinogram + noise
        (cor, _, side) = calc.find_center_360(sinogram, 6)[0:3]
        self.assertTrue(np.abs(cor - 9.0) < self.error and side == 0)

    def test_find_shift_based_phase_correlation(self):
        mat1 = np.zeros((self.size, self.size), dtype=np.float32)
        mat1[25:36, 25:36] = np.float32(1.0)
        xshift0 = 9
        yshift0 = -5
        mat2 = ndi.shift(mat1, (-yshift0, -xshift0))
        (yshift, xshift) = calc.find_shift_based_phase_correlation(mat1, mat2)
        num1 = np.abs(yshift - yshift0)
        num2 = np.abs(xshift - xshift0)
        self.assertTrue(num1 < self.error and num2 < self.error)

    def test_center_based_phase_correlation(self):
        mat1 = np.zeros((self.size, self.size), dtype=np.float32)
        mat1[25:36, 25:36] = np.float32(1.0)
        mat2 = np.fliplr(mat1)
        shift = -5
        mat1 = ndi.shift(mat1, (0, shift))
        mat2 = ndi.shift(mat2, (0, shift))
        center0 = (self.size - 1) / 2.0 + shift
        cor = calc.find_center_based_phase_correlation(mat1, mat2)
        self.assertTrue(np.abs(cor - center0) < self.error)

    def test_find_center_projection(self):
        mat1 = np.zeros((self.size, self.size), dtype=np.float32)
        mat1[26:36, 26:36] = np.tile(np.sin(np.arange(10) / 3.0), (10, 1))
        mat2 = np.fliplr(mat1)
        noise1 = np.float32(0.1 * np.random.rand(self.size, self.size))
        noise2 = np.float32(0.1 * np.random.rand(self.size, self.size))
        mat1 = mat1 + noise1
        mat2 = mat2 + noise2
        shift = -5
        mat1 = ndi.shift(mat1, (0, shift))
        mat2 = ndi.shift(mat2, (0, shift))
        center0 = (self.size - 1) / 2.0 + shift
        cor = calc.find_center_projection(mat1, mat2)
        self.assertTrue(np.abs(cor - center0) < self.error)

    def test_calculate_reconstructable_height(self):
        (y_s1, y_e1) = calc.calculate_reconstructable_height(5.0, 20.0, 4.0,
                                                             scan_type="180")
        (y_s2, y_e2) = calc.calculate_reconstructable_height(5.0, 20.0, 4.0,
                                                             scan_type="360")
        num1 = np.abs(y_s1 - 7.0)
        num2 = np.abs(y_e1 - 18.0)
        num3 = np.abs(y_s2 - 9.0)
        num4 = np.abs(y_e2 - 16.0)
        self.assertTrue((num1 < self.error) and (num2 < self.error)
                        and (num3 < self.error) and (num4 < self.error))

    def test_calculate_maximum_index(self):
        idx1 = calc.calculate_maximum_index(5.0, 10.0, 2.0, 0.001, "180")
        idx2 = calc.calculate_maximum_index(5.0, 10.0, 2.0, 0.001, "360")
        self.assertTrue(idx1 == 3001 and idx2 == 1001)
