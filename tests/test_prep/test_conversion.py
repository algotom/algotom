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
Tests for methods in prep/calculation.py
"""

import unittest
import numpy as np
import scipy.ndimage as ndi
import algotom.prep.conversion as conv


class ConversionMethods(unittest.TestCase):

    def setUp(self):
        self.error = 0.15
        self.eps = 10 ** (-6)
        self.size = 64
        mat = np.zeros((self.size + 1, self.size + 1), dtype=np.float32)
        mat[20:30, 35:42] = np.float32(1.0)
        mat = ndi.gaussian_filter(mat, 2.0)
        sino_360 = np.zeros((73, self.size + 1), dtype=np.float32)
        angles = np.linspace(0.0, 360.0, len(sino_360), dtype=np.float32)
        for i, angle in enumerate(angles):
            sino_360[i] = np.sum(ndi.rotate(mat, angle, reshape=False), axis=0)
        self.sino_360 = sino_360 / np.max(sino_360)

    def test_stitch_image(self):
        width = int(0.8 * 2 * self.size)
        overlap = 2 * self.size - width
        mat0 = np.tile(1.1 + np.sin(0.5 * np.arange(width)), (self.size, 1))
        mat1 = mat0[:, :self.size]
        mat2 = mat0[:, -self.size:]

        mat3 = conv.stitch_image(mat1, mat2, overlap, 1)
        num1 = np.max(np.abs(mat3 - mat0))
        self.assertTrue(num1 < self.eps)

        mat4 = conv.stitch_image(mat1, mat2, overlap + 0.5, 1)
        self.assertTrue(mat3.shape[-1] == mat4.shape[-1])

        mat5 = conv.stitch_image(mat1, mat2, overlap - 0.5, 0)
        self.assertTrue(mat3.shape[-1] == (mat5.shape[-1] - 1))

    def test_join_image(self):
        mat1 = np.tile(1.1 + np.sin(0.5 * np.arange(self.size)),
                       (self.size, 1))
        mat2 = np.copy(mat1)
        mat1[:, -5:] = 1.0
        mat2[:, :5] = 1.0
        mat_join = conv.join_image(mat1, mat2, 10, 1)
        num = np.mean(mat_join[:, self.size:self.size + 10])
        self.assertTrue(np.abs(num - 1.0) < self.eps)

        mat_join2 = conv.join_image(mat1, mat2, 10.5, 1)
        self.assertTrue(mat_join.shape[-1] == mat_join2.shape[-1])

        mat_join3 = conv.join_image(mat1, mat2, 11.5, 0)
        self.assertTrue(mat_join.shape[-1] == (mat_join3.shape[-1] - 1))

    def test_stitch_image_multiple(self):
        width = int(0.9 * 3 * self.size)
        overlap = int((3 * self.size - width) / 2.0)
        mat0 = np.tile(1.1 + np.sin(0.5 * np.arange(width)), (self.size, 1))
        mat1 = mat0[:, :self.size]
        mat2 = mat0[:, self.size - overlap: 2 * self.size - overlap]
        mat3 = mat0[:, -self.size:]
        list_mat = [mat1, mat2, mat3]
        list_overlap = [[overlap, 1], [overlap, 1]]
        mat_stitch = conv.stitch_image_multiple(list_mat, list_overlap)
        num1 = np.max(np.abs(mat_stitch - mat0))
        self.assertTrue(num1 < self.eps)

    def test_join_image_multiple(self):
        mat1 = np.tile(1.1 + np.sin(0.5 * np.arange(self.size)),
                       (self.size, 1))
        mat2 = np.copy(mat1)
        mat3 = np.copy(mat1)
        mat1[:, -5:] = 1.0
        mat2[:, :5] = 1.0
        mat2[:, -5:] = 1.0
        mat3[:, :5] = 1.0
        list_mat = [mat1, mat2, mat3]
        list_join = [[10, 1], [10, 1]]
        mat_join = conv.join_image_multiple(list_mat, list_join)
        num1 = np.mean(mat_join[:, self.size:self.size + 10])
        num2 = np.mean(mat_join[:, -self.size - 10:-self.size])
        self.assertTrue(np.abs(num1 - 1.0) < self.eps and
                        np.abs(num2 - 1.0) < self.eps)

    def test_convert_sinogram_360_to_180(self):
        sino_180 = self.sino_360[0:37]
        sino_360 = np.pad(self.sino_360[:, 22:], ((0, 0), (0, 22)),
                          mode='constant')
        sino_conv, center = conv.convert_sinogram_360_to_180(sino_360, 10.0)
        center = int(np.floor(center))
        radi = self.size // 2
        sino_conv = sino_conv[:, center - radi: center + self.size + 1 - radi]
        num = np.max(np.abs(sino_conv - sino_180))
        self.assertTrue(num < self.eps)

    def test_convert_sinogram_180_to_360(self):
        sino_180 = self.sino_360[0:37]
        sino_conv = conv.convert_sinogram_180_to_360(sino_180, 32.0)
        num = np.max(np.abs(sino_conv - self.sino_360))
        self.assertTrue(num < self.eps)

    def test_extend_sinogram(self):
        sino_360 = np.pad(self.sino_360[:, 22:], ((0, 0), (0, 22)),
                          mode='constant')
        sino_ext, _ = conv.extend_sinogram(sino_360, 10, apply_log=False)
        nrow = sino_ext.shape[0] // 2 + 1
        sino1 = sino_ext[:nrow] + np.fliplr(sino_ext[-nrow:])
        sino2, _ = conv.convert_sinogram_360_to_180(sino_360, 10, norm=False)
        num = np.max(np.abs(sino1 - sino2))
        self.assertTrue(num < self.eps)

        sino_360 = np.max(self.sino_360) - self.sino_360 + 0.05
        overlap, side = (10.5, 1)
        sino_ext = conv.extend_sinogram(sino_360,
                                        (overlap, side), apply_log=True)[0]
        width = int(sino_360.shape[-1] * 2 - np.floor(overlap))
        self.assertTrue(width == sino_ext.shape[-1])

        overlap, side = (9.5, 0)
        sino_ext = conv.extend_sinogram(sino_360,
                                        (overlap, side), apply_log=True)[0]
        width = int(sino_360.shape[-1] * 2 - np.floor(overlap))
        self.assertTrue(width == sino_ext.shape[-1])

    def test_generate_sinogram_helical_scan(self):
        size = self.size + 1
        proj = np.zeros((size, size), dtype=np.float32)
        for i in range(1, self.size):
            mask = np.copy(proj[i])
            center = size // 2
            radius = i // 2
            x = np.ogrid[-center: size - center]
            mask_check = np.floor(np.abs(x)) <= radius
            mask[mask_check] = 1.0
            proj[i] = mask
        proj = np.pad(proj, ((32, 32), (0, 0)), mode='constant')
        y_start = 5.0
        y_stop = 45.0
        pitch = 30.0
        num_proj = 37
        pixel_size = 1.0
        helix_data = []
        y_step = pitch / (2 * (num_proj - 1.0))
        total_proj = int(np.floor((y_stop - y_start) / y_step))
        for i in range(total_proj):
            proj_tmp = ndi.shift(proj, (-i * y_step / pixel_size, 0),
                                 mode='nearest', order=1)[:size]
            helix_data.append(proj_tmp)
        helix_data = np.asarray(helix_data)
        sinogram, _ = conv.generate_sinogram_helical_scan(11, helix_data,
                                                          num_proj, pixel_size,
                                                          y_start, y_stop,
                                                          pitch)
        sinogram = np.floor(sinogram)
        num1 = np.sum(sinogram[0])
        num2 = np.sum(sinogram[-1])
        self.assertTrue(np.abs(num1 - num2) < self.eps)

    def test_generate_full_sinogram_helical_scan(self):
        size = self.size + 1
        proj = np.zeros((size, size), dtype=np.float32)
        for i in range(1, self.size):
            mask = np.copy(proj[i])
            center = size // 2
            radius = i // 2
            x = np.ogrid[-center: size - center]
            mask_check = np.floor(np.abs(x)) <= radius
            mask[mask_check] = 1.0
            proj[i] = mask
        proj = np.pad(proj, ((32, 32), (0, 0)), mode='constant')
        y_start = 5.0
        y_stop = 45.0
        pitch = 30.0
        num_proj = 37
        helix_data = []
        y_step = pitch / (2 * (num_proj - 1.0))
        pixel_size = 0.5
        total_proj = int(np.floor((y_stop - y_start) / y_step))
        for i in range(total_proj):
            proj_tmp = ndi.shift(proj, (-i * y_step / pixel_size, 0),
                                 mode='nearest', order=1)[:size]
            helix_data.append(proj_tmp)
        helix_data = np.asarray(helix_data)
        sinogram, _ = conv.generate_full_sinogram_helical_scan(12, helix_data,
                                                               num_proj,
                                                               pixel_size,
                                                               y_start, y_stop,
                                                               pitch)
        sinogram = np.floor(sinogram)
        num1 = np.sum(sinogram[0])
        num2 = np.sum(sinogram[-1])
        self.assertTrue(np.abs(num1 - num2) < self.eps)
