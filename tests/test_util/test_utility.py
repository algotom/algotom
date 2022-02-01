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
Tests for the methods in util/utility.py

"""

import unittest
import numpy as np
import scipy.ndimage as ndi
import algotom.util.utility as util


class UtilityMethods(unittest.TestCase):

    def setUp(self):
        self.eps = 10 ** (-6)
        size = 65
        mask1 = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        radius1 = center - 6
        y, x = np.ogrid[-center:size - center, -center:size - center]
        mask_check = x * x + y * y <= radius1 * radius1
        mask1[mask_check] = 1.0
        radius2 = center - 7
        mask2 = np.zeros((size, size), dtype=np.float32)
        mask_check = x * x + y * y <= radius2 * radius2
        mask2[mask_check] = 1.0
        self.mat_rec = mask1 - mask2
        list_mask = np.zeros(size)
        list_mask[2 * radius2: 2 * radius1] = 1.0
        self.mat_sino = np.tile(list_mask, (size, 1))
        self.size = size - 1

    def test_apply_method_to_multiple_sinograms(self):
        data = np.random.rand(32, self.size, self.size)
        method = 'remove_stripe_based_sorting'
        data_after = util.apply_method_to_multiple_sinograms(data, method,
                                                             [11, 1])
        num = np.mean(np.abs(data - data_after))
        self.assertTrue(num > self.eps and data_after.shape == data.shape)

    def test_sort_forward(self):
        mat = np.transpose(np.tile(np.arange(self.size - 1, -1, -1),
                                   (self.size, 1)))
        mat1 = np.transpose(np.tile(np.arange(self.size), (self.size, 1)))
        mat2, mat_idx = util.sort_forward(mat, axis=0)
        num1 = np.mean(np.abs(mat2 - mat1))
        num2 = np.mean(np.abs(mat - mat_idx))
        self.assertTrue(num1 < self.eps and num2 < self.eps)

    def test_sort_backward(self):
        mat = np.transpose(np.tile(np.arange(self.size - 1, -1, -1),
                                   (self.size, 1)))
        mat1, mat_idx = util.sort_forward(mat, axis=0)
        mat2 = util.sort_backward(mat1, mat_idx, axis=0)
        num = np.mean(np.abs(mat2 - mat))
        self.assertTrue(num < self.eps)

    def test_separate_frequency_component(self):
        mat = np.random.rand(self.size, self.size)
        mat_smth, mat_shrp = util.separate_frequency_component(mat)
        num1 = np.abs(np.mean(mat_smth) - 0.5)
        num2 = np.abs(np.mean(mat_shrp))
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

    def test_generate_fitted_image(self):
        mat = np.random.rand(self.size, self.size)
        mat_fit = util.generate_fitted_image(mat, 1)
        num1 = np.abs(np.mean(mat_fit) - 0.5)
        num2 = np.abs(np.mean(mat - mat_fit))
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

    def test_detect_stripe(self):
        np.random.seed(1)
        lis = np.random.rand(self.size)
        lis_off = np.linspace(0, 1, len(lis))
        lis = lis + lis_off
        lis[self.size // 2:self.size // 2 + 1] = 6.0
        lis_bin = util.detect_stripe(lis, 1.5)
        pos = np.where(lis_bin == 1.0)
        self.assertTrue((len(pos) > 0) and (pos[0] == self.size // 2))

    def test_apply_filter_to_wavelet_component(self):
        mat = np.random.rand(2 * self.size, 2 * self.size)
        data = util.apply_wavelet_decomposition(mat, 'db5', 3)
        data1 = util.apply_filter_to_wavelet_component(data, 2, order=1)
        num = np.mean(np.abs(data[2][1] - data1[2][1]))
        self.assertTrue(num > self.eps)

    def test_interpolate_inside_stripe(self):
        mat = np.ones((self.size, self.size), dtype=np.float32)
        list_mask = np.zeros(self.size)
        list_mask[self.size // 2:self.size // 2 + 2] = 0.0
        mat_corr = util.interpolate_inside_stripe(mat, list_mask)
        num = np.abs(np.mean(
            mat_corr[:, self.size // 2:self.size // 2 + 2]) - 1.0)
        self.assertTrue(num < self.eps)

    def test_transform_slice_forward(self):
        mat_sino1 = util.transform_slice_forward(self.mat_rec)
        num = np.sum((1 - ndi.binary_dilation(self.mat_sino))
                     * np.round(mat_sino1))
        self.assertTrue(num < self.eps)

    def test_transform_slice_backward(self):
        mat_rec1 = util.transform_slice_backward(self.mat_sino)
        num = np.sum((1 - ndi.binary_dilation(self.mat_rec))
                     * np.round(mat_rec1))
        self.assertTrue(num < self.eps)

    def test_apply_gaussian_filter(self):
        mat = np.random.rand(self.size, self.size)
        mat_smth = util.apply_gaussian_filter(mat, 5, 5, 10)
        num1 = np.abs(np.mean(mat_smth) - 0.5)
        num2 = np.abs(np.mean(mat - mat_smth))
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

    def test_apply_regularization_filter(self):
        mat = np.random.rand(self.size, self.size)
        mat_smth = util.apply_regularization_filter(mat, 0.01)
        num1 = np.abs(np.mean(mat_smth) - 0.5)
        num2 = np.abs(np.mean(mat - mat_smth))
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

    def test_detect_sample(self):
        mat = np.zeros((self.size + 1, self.size + 1), dtype=np.float32)
        mat[20:30, 35:42] = np.float32(1.0)
        mat = ndi.gaussian_filter(mat, 2.0)
        sino_360 = np.zeros((73, self.size + 1), dtype=np.float32)
        angles = np.linspace(0.0, 360.0, len(sino_360), dtype=np.float32)
        for i, angle in enumerate(angles):
            sino_360[i] = np.sum(ndi.rotate(mat, angle, reshape=False), axis=0)
        sino_360 = sino_360 / np.max(sino_360)
        sino_180 = sino_360[:37]
        check1 = util.detect_sample(sino_180)
        check2 = util.detect_sample(sino_360, sino_type="360")
        check3 = util.detect_sample(mat)
        self.assertTrue(check1 and check2 and (not check3))

    def test_transform_1d_window_to_2d(self):
        list1 = np.zeros(self.size, dtype=np.float32)
        list1[self.size // 2 - 5: self.size // 2 + 6] = np.float32(1.0)
        mat1 = util.transform_1d_window_to_2d(list1)
        (width, height) = mat1.shape
        self.assertTrue(width == self.size and height == self.size)
