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
Tests for the methods in prep/correction.py

"""

import unittest
import numpy as np
import scipy.ndimage as ndi
import algotom.prep.correction as corr


class CorrectionMethods(unittest.TestCase):

    def setUp(self):
        self.eps = 10 ** (-6)
        self.size = 64
        self.mat = np.zeros((self.size, self.size), dtype=np.float32)
        self.mat[self.size // 3: self.size // 3 + 3] = 1.0
        self.x = self.size / 2.0 + 1
        self.y = self.size / 2.0 - 2
        self.ffacts = [1.0, 0.03, 1.0 * 10 ** (-4)]
        pad = 20
        mat_pad = np.pad(self.mat, pad, mode='edge')
        mat_for = corr.unwarp_projection(mat_pad, self.x + pad,
                                         self.y + pad, self.ffacts)
        self.mat_for = mat_for[pad:pad + self.size, pad:pad + self.size]
        self.bfacts = [9.34082475e-01, -1.39192784e-02, 1.18758023e-04]
        mat_tilt = ndi.rotate(mat_pad, 5.0, reshape=False)
        self.mat_tilt = mat_tilt[pad:pad + self.size, pad:pad + self.size]

    def test_flat_field_correction(self):
        proj = np.random.rand(32, self.size, self.size)
        flat = 0.5 * np.ones((self.size, self.size), dtype=np.float32)
        dark = np.zeros((self.size, self.size), dtype=np.float32)
        m1 = corr.flat_field_correction(proj[:, 0:5, :], flat[0:5], dark[0:5])
        m2 = corr.flat_field_correction(proj[0], flat, dark)
        m3 = corr.flat_field_correction(proj[0:5], flat, dark)
        opt1 = {"method": "remove_zinger", "para1": 0.1, "para2": 1}
        m4 = corr.flat_field_correction(proj[:, 0, :], flat[0], dark[0],
                                        option=opt1)
        m5 = corr.flat_field_correction(proj, flat, dark, option=opt1)
        num1 = np.sum(np.abs(m4 - proj[:, 0, :]))
        num2 = np.sum(np.abs(m5 - proj))
        self.assertTrue(m1.shape == (32, 5, self.size) and
                        m2.shape == (self.size, self.size) and
                        m3.shape == (5, self.size, self.size)
                        and num1 > self.eps and num2 > self.eps)

    def test_unwarp_projection(self):
        mat_corr = corr.unwarp_projection(self.mat_for, self.x,
                                          self.y, self.bfacts)
        mat_corr = np.round(mat_corr)
        num = np.sum(np.abs(mat_corr - self.mat))
        self.assertTrue(num < self.eps)

    def test_unwarp_sinogram(self):
        proj = np.random.rand(32, self.size, self.size)
        proj[:] = self.mat_for
        sino_corr = corr.unwarp_sinogram(proj, self.size // 3,
                                         self.x, self.y, self.bfacts)
        sino_corr = np.round(sino_corr)
        num = np.abs(np.mean(sino_corr) - 1.0)
        self.assertTrue(num < self.eps)

    def test_unwarp_sinogram_chunk(self):
        proj = np.random.rand(32, self.size, self.size)
        proj[:] = self.mat_for
        sino_corr = corr.unwarp_sinogram_chunk(proj, self.size // 3,
                                         self.size // 3 + 3, self.x, self.y,
                                         self.bfacts)
        sino_corr = np.round(sino_corr)
        num = np.abs(np.mean(sino_corr) - 1.0)
        self.assertTrue(num < self.eps)

    def test_mtf_deconvolution(self):
        proj = np.random.rand(self.size, self.size)
        window = 0.5 * np.ones((self.size, self.size))
        window[self.size // 2, self.size // 2] = 1.0
        proj_corr = corr.mtf_deconvolution(proj, window, 10)
        num = np.sum(np.abs(proj - proj_corr))
        self.assertTrue(num > self.eps and isinstance(proj_corr[0,0], float))

    def test_generate_tilted_sinogram(self):
        proj = np.random.rand(32, self.size, self.size)
        proj[:] = self.mat_tilt
        sino_corr = corr.generate_tilted_sinogram(proj, self.size // 3, -5.0)
        sino_corr = np.round(sino_corr)
        num = np.abs(np.mean(sino_corr) - 1.0)
        self.assertTrue(num < self.eps)

    def test_generate_tilted_sinogram_chunk(self):
        proj = np.random.rand(32, self.size, self.size)
        proj[:] = self.mat_tilt
        sino_corr = corr.generate_tilted_sinogram_chunk(proj, self.size // 3,
                                                        self.size // 3 + 2,
                                                        -5.0)
        sino_corr = np.round(sino_corr)
        num = np.abs(np.mean(sino_corr) - 1.0)
        self.assertTrue(num < self.eps)

    def test_generate_tilted_profile_line(self):
        line = corr.generate_tilted_profile_line(self.mat_tilt, self.size // 3,
                                                 -5.0)
        line = np.round(line)
        num = np.abs(np.mean(line) - 1.0)
        self.assertTrue(num < self.eps)

    def test_generate_tilted_profile_chunk(self):
        line = corr.generate_tilted_profile_chunk(self.mat_tilt, self.size // 3,
                                                 self.size // 3 + 2, -5.0)
        line = np.round(line)
        num = np.abs(np.mean(line) - 1.0)
        self.assertTrue(num < self.eps)

    def test_beam_hardening_correction(self):
        line = np.linspace(0.0, 1.0, self.size)
        mat = np.tile(line, (self.size, 1))
        mat_corr1 = corr.beam_hardening_correction(mat, 0.01, 3, opt=False)
        num1 = np.sum(mat_corr1[0] - mat[0])
        mat_corr1 = corr.beam_hardening_correction(mat, 0.01, 3, opt=True)
        num2 = np.sum(mat_corr1[0] - mat[0])
        self.assertTrue(num1 > self.eps and num2 < self.eps)