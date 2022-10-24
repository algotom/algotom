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
Tests for the methods in post/postprocessing.py

"""

import os
import glob
import shutil
import unittest
import numpy as np
import algotom.post.postprocessing as post


class UtilityMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir("data"):
            os.makedirs("data")

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir("data"):
            shutil.rmtree("data")

    def setUp(self):
        size = 129
        mask1 = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        radius1 = center - 5
        y, x = np.ogrid[-center:size - center, -center:size - center]
        mask_check = x * x + y * y <= radius1 * radius1
        mask1[mask_check] = 1.0
        radius2 = center - 6
        mask2 = np.zeros((size, size), dtype=np.float32)
        mask_check = x * x + y * y <= radius2 * radius2
        mask2[mask_check] = 1.0
        self.mask = mask1 - mask2
        self.mat = np.ones_like(mask1) - np.ceil(self.mask)
        self.size = size

    def test_get_statistical_information(self):
        mat = np.random.rand(64, 64)
        results = post.get_statistical_information(mat)
        num = np.abs(0.5 - results[5])
        self.assertTrue(len(results) == 7 and num < 0.1)

    def test_get_statistical_information_dataset(self):
        mat = np.random.rand(32, 64, 64)
        results =post.get_statistical_information_dataset(mat)
        num = np.abs(0.5 - results[5])
        self.assertTrue(len(results) == 7 and num < 0.1)

    def test_downsample(self):
        mat = np.random.rand(64, 64)
        mat_dsp = post.downsample(mat, (2, 2))
        self.assertTrue(mat_dsp.shape == (32, 32))

    def test_downsample_dataset(self):
        mat = np.random.rand(16, 64, 64)
        mat_dsp = post.downsample_dataset(mat, None, (2, 2, 2))
        self.assertTrue(mat_dsp.shape == (8, 32, 32))
        post.downsample_dataset(mat, "data/dsp/", (2, 2, 2))
        files = glob.glob("data/dsp/*tif*")
        self.assertTrue(len(files) == 8)
        post.downsample_dataset(mat, "data/dsp2/file.hdf", (2, 2, 2))
        self.assertTrue(os.path.isfile("data/dsp2/file.hdf"))

    def test_rescale(self):
        mat = np.random.rand(64, 64)
        mat_res = post.rescale(mat, nbit=16)
        self.assertTrue(65536 > mat_res[32, 32] >= 0)

    def test_rescale_dataset(self):
        mat = np.random.rand(16, 64, 64)
        mat_res = post.rescale_dataset(mat, None, nbit=8)
        self.assertTrue(256 > mat_res[0, 32, 32] >= 0)
        post.rescale_dataset(mat, "data/rescale/", nbit=8)
        files = glob.glob("data/rescale/*tif*")
        self.assertTrue(len(files) == 16)
        post.rescale_dataset(mat, "data/rescale2/file.hdf", nbit=8)
        self.assertTrue(os.path.isfile("data/rescale2/file.hdf"))

    def test_remove_ring_based_wavelet_fft(self):
        mat_corr = post.remove_ring_based_wavelet_fft(self.mat, 3, 1)
        num = np.sum(mat_corr * self.mask) / np.sum(self.mask)
        self.assertTrue(num > 0.5)

    def test_remove_ring_based_fft(self):
        mat_corr = post.remove_ring_based_fft(self.mat, 10, 4, 1)
        num = np.sum(mat_corr * self.mask) / np.sum(self.mask)
        self.assertTrue(num > 0.5)
