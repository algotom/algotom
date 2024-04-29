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
Tests for the methods in prep/filtering.py

"""

import unittest
import numpy as np
import scipy.ndimage as ndi
import algotom.prep.filtering as filt


class FilteringMethods(unittest.TestCase):

    def setUp(self):
        self.eps = 10 ** (-6)

    def test_fresnel_filter(self):
        mat = np.random.rand(64, 64)
        mat1 = filt.fresnel_filter(mat, 10, dim=1)
        mat2 = filt.fresnel_filter(mat, 10, dim=2)
        num1 = np.sum(np.abs(mat - mat1))
        num2 = np.sum(np.abs(mat - mat2))
        num3 = np.sum(np.abs(mat1 - mat2))
        self.assertTrue(num1 > self.eps and num2 > self.eps
                        and num3 > self.eps)

    def test_double_wedge_filter(self):
        size = 129
        idx1 = size // 2
        rad = size // 4
        num_proj = 73
        # Create a phantom and its sinogram.
        mat = np.zeros((size, size), dtype=np.float32)
        mat[idx1 - 10:idx1 + 5, idx1 + 10:idx1 + 20] = np.float32(1.0)
        mat = ndi.gaussian_filter(mat, 1.0)
        sino_360_std = np.zeros((num_proj, size), dtype=np.float32)
        angles = np.linspace(0.0, 360.0, len(sino_360_std), dtype=np.float32)
        for i, angle in enumerate(angles):
            sino_360_std[i] = np.sum(ndi.rotate(mat, angle, reshape=False),
                                     axis=0)
        sino_360_std = sino_360_std / size
        # Create a phantom with a feature larger than the crop FOV.
        mat = np.zeros((size, size), dtype=np.float32)
        mat[idx1 - 10:idx1 + 5, idx1 + 10:idx1 + 20] = np.float32(1.0)
        mat[5:25, 10:25] = np.float32(1.5)
        mat = ndi.gaussian_filter(mat, 1.0)
        sino_360 = np.zeros((num_proj, size), dtype=np.float32)
        angles = np.linspace(0.0, 360.0, len(sino_360), dtype=np.float32)
        for i, angle in enumerate(angles):
            sino_360[i] = np.sum(ndi.rotate(mat, angle, reshape=False), axis=0)
        sino_360 = sino_360 / size
        sino_360_crop0 = sino_360_std[:, idx1 - rad: idx1 + rad]
        sino_360_crop = sino_360[:, idx1 - rad: idx1 + rad]
        sino_180_crop0 = sino_360_crop0[:num_proj // 2 + 1]
        sino_180_crop = sino_360_crop[:num_proj // 2 + 1]
        sino_360_filt = filt.double_wedge_filter(sino_360_crop,
                                                 sino_type="360", iteration=10)
        sino_360_filt = sino_360_filt * (
                np.mean(sino_360_crop0) / np.mean(np.abs(sino_360_filt)))
        num1 = np.max(np.abs(sino_360_filt - sino_360_crop0))
        sino_180_filt = filt.double_wedge_filter(sino_180_crop, center=32.0,
                                                 sino_type="180", iteration=10)
        sino_180_filt = sino_180_filt * (
                np.mean(sino_180_crop0) / np.mean(np.abs(sino_180_filt)))
        num2 = np.max(np.abs(sino_180_filt - sino_180_crop0))
        self.assertTrue(num1 <= 0.1 and num2 <= 0.1)

        self.assertRaises(ValueError, filt.double_wedge_filter, sino_180_crop,
                          center=32.0, sino_type="18")

        self.assertRaises(ValueError, filt.double_wedge_filter, sino_180_crop,
                          center=0.0, sino_type="180")

        mask = filt.make_double_wedge_mask(num_proj, size, size / 2.0)
        self.assertRaises(ValueError, filt.double_wedge_filter, sino_360,
                          center=0.0, sino_type="360", pad=1, mask=mask)
