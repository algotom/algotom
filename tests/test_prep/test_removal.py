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
Tests for the methods in prep/removal.py

"""

import unittest
import numpy as np
import algotom.prep.removal as remo


class RemovalMethods(unittest.TestCase):

    def setUp(self):
        self.eps = 10 ** (-6)
        self.size = 64
        self.mat = np.random.rand(self.size, self.size)
        (self.b, self.e) = (30, 31)
        self.mat[:, self.b:self.e] = 0.0
        self.idx1 = self.size // 2 - 2
        self.idx2 = self.size // 2 + 2

    def test_remove_stripe_based_sorting(self):
        f_alias = remo.remove_stripe_based_sorting
        mat_corr = f_alias(self.mat, 3, dim=1)
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(self.mat, 3, dim=2)
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(self.mat, 3, dim=1,
                           options={"method": "gaussian_filter",
                                    "para1": (1, 21)})
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(self.mat, 3, dim=1,
                           options={"method": "apply_gaussian_filter",
                                    "para1": 5, "para2": 20})
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

        self.assertRaises(ValueError, f_alias, self.mat, 3, dim=1, options=3)
        self.assertRaises(ValueError, f_alias, self.mat, 3, dim=1,
                          options={"method": "gaussian_filter",
                                   "para1": (1, 2, 1)})
        self.assertRaises(ValueError, f_alias, self.mat, 3, dim=1,
                          options={"method": "test_filter",
                                   "para1": (1, 2)})
        self.assertRaises(ValueError, f_alias, self.mat, 3, dim=1,
                          options={"method": "apply_gaussian_filter",
                                   "para1": (2, 6)})

    def test_remove_stripe_based_filtering(self):
        f_alias = remo.remove_stripe_based_filtering
        mat_corr = f_alias(self.mat, 3, 3, dim=1)
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(self.mat, 3, 3, dim=2)
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(self.mat, 3, 3,
                           options={"method": "gaussian_filter",
                                    "para1": (1, 21)})
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(self.mat, 3, 3,
                           options={"method": "apply_gaussian_filter",
                                    "para1": 5, "para2": 20})
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

        self.assertRaises(ValueError, f_alias, self.mat, 3, 3, options=3)
        self.assertRaises(ValueError, f_alias, self.mat, 3, 3,
                          options={"method": "gaussian_filter",
                                   "para1": (1, 2, 1)})
        self.assertRaises(ValueError, f_alias, self.mat, 3, 3,
                          options={"method": "test_filter",
                                   "para1": (1, 2)})
        self.assertRaises(ValueError, f_alias, self.mat, 3, 3,
                          options={"method": "apply_gaussian_filter",
                                   "para1": (2, 6)})

    def test_remove_stripe_based_fitting(self):
        mat = np.random.rand(self.size, self.size)
        mat[:, self.b:self.e] = 1.0
        f_alias = remo.remove_stripe_based_fitting
        mat_corr = f_alias(mat, 1, 5, sort=True)
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 1.0)
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(mat, 1, 5, options={"method": "gaussian_filter",
                                               "para1": (1, 21)})
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 1.0)
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(mat, 1, 5,
                           options={"method": "apply_gaussian_filter",
                                    "para1": 5, "para2": 20})
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 1.0)
        self.assertTrue(num > self.eps)

        self.assertRaises(ValueError, f_alias, self.mat, 1, 5, options=3)
        self.assertRaises(ValueError, f_alias, self.mat, 1, 5,
                          options={"method": "gaussian_filter",
                                   "para1": (1, 2, 1)})
        self.assertRaises(ValueError, f_alias, self.mat, 1, 5,
                          options={"method": "test_filter",
                                   "para1": (1, 2)})
        self.assertRaises(ValueError, f_alias, self.mat, 1, 5,
                          options={"method": "apply_gaussian_filter",
                                   "para1": (2, 6)})

    def test_remove_large_stripe(self):
        mat = np.random.rand(self.size, self.size)
        mat[:, self.b:self.e] = 6.0
        f_alias = remo.remove_large_stripe
        mat_corr = f_alias(mat, 1.5, 5, norm=False)
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 6.0)
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(mat, 1.5, 5, norm=False,
                           options={"method": "gaussian_filter",
                                    "para1": (1, 21)})
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 6.0)
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(mat, 1.5, 5, norm=False,
                           options={"method": "apply_gaussian_filter",
                                    "para1": 5, "para2": 20})
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 6.0)
        self.assertTrue(num > self.eps)

        self.assertRaises(ValueError, f_alias, mat, 1.5, 5, options=3)
        self.assertRaises(ValueError, f_alias, mat, 1.5, 5,
                          options={"method": "gaussian_filter",
                                   "para1": (1, 2, 1)})
        self.assertRaises(ValueError, f_alias, mat, 1.5, 5,
                          options={"method": "test_filter",
                                   "para1": (1, 2)})
        self.assertRaises(ValueError, f_alias, mat, 1.5, 5,
                          options={"method": "apply_gaussian_filter",
                                   "para1": (2, 6)})

    def test_remove_dead_stripe(self):
        mat = np.random.rand(self.size, self.size)
        mat[:, self.b:self.e] = 6.0
        mat_corr = remo.remove_dead_stripe(mat, 1.5, 5)
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 6.0)
        self.assertTrue(num > self.eps)

    def test_remove_all_stripe(self):
        mat = np.random.rand(self.size, self.size)
        mat[:, self.b:self.e] = 6.0
        mat_corr = remo.remove_all_stripe(mat, 1.5, 5, 3)
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 6.0)
        self.assertTrue(num > self.eps)

    def test_remove_stripe_based_2d_filtering_sorting(self):
        mat_corr = remo.remove_stripe_based_2d_filtering_sorting(self.mat, 3,
                                                                 3, dim=1)
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

    def test_remove_stripe_based_normalization(self):
        f_alias = remo.remove_stripe_based_normalization
        mat_corr = f_alias(self.mat, 5, 1)
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(self.mat, 5, 1,
                           options={"method": "median_filter",
                                    "para1": 3})
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

        self.assertRaises(ValueError, f_alias, self.mat, 5, 1, options=3)
        self.assertRaises(ValueError, f_alias, self.mat, 5, 1,
                          options={"method": "gaussian_filter",
                                   "para1": (1, 2, 1)})
        self.assertRaises(ValueError, f_alias, self.mat, 5, 1,
                          options={"method": "test_filter",
                                   "para1": (1, 2)})
        self.assertRaises(ValueError, f_alias, self.mat, 5, 1,
                          options={"method": "apply_gaussian_filter",
                                   "para1": (2, 6)})

    def test_remove_stripe_based_regularization(self):
        mat = np.random.rand(self.size, self.size)
        mat[:, self.b:self.e] = 1.0
        mat_corr = remo.remove_stripe_based_regularization(mat, 0.01, 1,
                                                           apply_log=False)
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 1.0)
        self.assertTrue(num > self.eps)

        mat_corr = remo.remove_stripe_based_regularization(mat, 0.01, 1,
                                                           apply_log=True)
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 1.0)
        self.assertTrue(num > self.eps)

        mat[0, 0] = 0.0
        mat_corr = remo.remove_stripe_based_regularization(mat, 0.01, 1,
                                                           apply_log=True)
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 1.0)
        self.assertTrue(num > self.eps)

    def test_remove_stripe_based_fft(self):
        mat = np.random.rand(self.size, self.size)
        mat[:, self.b:self.e] = 1.0
        mat_corr = remo.remove_stripe_based_fft(mat, 5, 4, 1)
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 1.0)
        self.assertTrue(num > self.eps)

        mat_corr = remo.remove_stripe_based_fft(mat, 5, 4, 1, sort=True)
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 1.0)
        self.assertTrue(num > self.eps)

    def test_remove_stripe_based_wavelet_fft(self):
        mat = np.random.rand(self.size, self.size)
        mat[:, self.b:self.e] = 1.0
        f_alias = remo.remove_stripe_based_wavelet_fft
        mat_corr = f_alias(mat, 5, 1)
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 1.0)
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(mat, 5, 1, options={"method": "gaussian_filter",
                                               "para1": (1, 7)}, sort=True)
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 1.0)
        self.assertTrue(num > self.eps)

        self.assertRaises(ValueError, f_alias, mat, 5, 1, options=3)
        self.assertRaises(ValueError, f_alias, mat, 5, 1,
                          options={"method": "gaussian_filter",
                                   "para1": (1, 2, 1)})
        self.assertRaises(ValueError, f_alias, mat, 5, 1,
                          options={"method": "test_filter",
                                   "para1": (1, 2)})
        self.assertRaises(ValueError, f_alias, mat, 5, 1,
                          options={"method": "apply_gaussian_filter",
                                   "para1": (2, 6)})

    def test_remove_stripe_based_interpolation(self):
        f_alias = remo.remove_stripe_based_interpolation
        mat_corr = f_alias(self.mat, 1.5, 5, norm=False)
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(self.mat, 1.5, 5, kind="cubic")
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(self.mat, 1.5, 5, kind="quintic")
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(self.mat, 1.5, 5, norm=True,
                           options={"method": "gaussian_filter",
                                    "para1": (1, 7)})
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

        mat_corr = f_alias(self.mat, 1.5, 5, norm=True,
                           options={"method": "apply_gaussian_filter",
                                    "para1": 5, "para2": 10})
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

        self.assertRaises(ValueError, f_alias, self.mat, 1.5, 5, options=3)
        self.assertRaises(ValueError, f_alias, self.mat, 1.5, 5,
                          options={"method": "gaussian_filter",
                                   "para1": (1, 2, 1)})
        self.assertRaises(ValueError, f_alias, self.mat, 1.5, 5,
                          options={"method": "test_filter",
                                   "para1": (1, 2)})
        self.assertRaises(ValueError, f_alias, self.mat, 1.5, 5,
                          options={"method": "apply_gaussian_filter",
                                   "para1": (2, 6)})

    def test_remove_zinger(self):
        mat = 0.5 * np.ones((self.size, self.size))
        mat[self.idx1, self.idx1] = 1.0
        mat_corr = remo.remove_zinger(mat, 0.08)
        num = np.abs(mat_corr[self.idx1, self.idx1] - 0.5)
        self.assertTrue(num < self.eps)

    def test_generate_blob_mask(self):
        mat = 0.5 * np.ones((self.size, self.size))

        mat[self.idx1:self.idx2, self.idx1:self.idx2] = 1.0
        mat = mat + 0.05 * np.random.rand(self.size, self.size)
        mask = remo.generate_blob_mask(mat, 12, 2.0)
        nmean = np.mean(mask[self.idx1:self.idx2, self.idx1:self.idx2])
        num = np.abs(nmean - 1.0)
        self.assertTrue(num < self.eps)

    def test_remove_blob_1d(self):
        list_1d = 0.5 * np.ones(self.size)
        list_1d[self.idx1:self.idx2] = 1.0
        mask_1d = np.zeros_like(list_1d)
        mask_1d[self.idx1:self.idx2] = 1.0
        list_corr = remo.remove_blob_1d(list_1d, mask_1d)
        nmean = np.mean(list_corr[self.idx1:self.idx2])
        num = np.abs(nmean - 0.5)
        self.assertTrue(num < self.eps)

    def test_remove_blob(self):
        mat = 0.5 * np.ones((self.size, self.size))
        mat[:, self.idx1:self.idx2] = 1.0
        mask = np.zeros_like(mat)
        mask[:, self.idx1:self.idx2] = 1.0
        mat_corr = remo.remove_blob(mat, mask)
        nmean = np.mean(mat_corr[:, self.idx1:self.idx2])
        num = np.abs(nmean - 0.5)
        self.assertTrue(num < self.eps)
        self.assertRaises(ValueError, remo.remove_blob, mat, mask[1:])
