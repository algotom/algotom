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

import os
import shutil
import unittest
import numpy as np
import scipy.ndimage as ndi
import algotom.util.utility as util
import algotom.io.loadersaver as losa


class UtilityMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir("./tmp"):
            os.makedirs("./tmp")

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir("./tmp"):
            shutil.rmtree("./tmp")

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
        self.overlaps0 = np.asarray(
            [[[300, 1], [300, 1], [300, 1]], [[301, 1], [301, 1], [301, 1]],
             [[299, 1], [299, 1], [299, 1]]])
        self.overlaps1 = np.asarray(
            [[[300, 0], [300, 0], [300, 0]], [[301, 0], [301, 0], [301, 0]],
             [[299, 0], [299, 0], [299, 0]]])

    def test_apply_method_to_multiple_sinograms(self):
        f_alias = util.apply_method_to_multiple_sinograms
        data = np.random.rand(32, self.size, self.size)
        method = 'remove_stripe_based_sorting'
        data_after = f_alias(data, method, [11, 1], ncore=1)
        num = np.mean(np.abs(data - data_after))
        self.assertTrue(num > self.eps and data_after.shape == data.shape)

        method = 'remove_stripe_based_sorting'
        data_after = f_alias(data, method, 11, ncore=None)
        num = np.mean(np.abs(data - data_after))
        self.assertTrue(num > self.eps and data_after.shape == data.shape)

        method = 'fresnel_filter'
        data_after = f_alias(data, method, [30, 1])
        num = np.mean(np.abs(data - data_after))
        self.assertTrue(num > self.eps and data_after.shape == data.shape)

        method = 'dfi_reconstruction'
        data = np.random.rand(self.size, self.size, self.size)
        data_after = f_alias(data, method, [self.size // 2, None, None,
                                            "hann", 0.1, "edge", False])
        num = np.mean(np.abs(data - data_after))
        self.assertTrue(num > self.eps and data_after.shape == data.shape)

        self.assertRaises(ValueError, f_alias, data,
                          'move_stripe_based_sorting', [11, 1])

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
        f_alias = util.separate_frequency_component
        mat = np.random.rand(self.size, self.size)
        mat_smth, mat_shrp = f_alias(mat)
        num1 = np.abs(np.mean(mat_smth) - 0.5)
        num2 = np.abs(np.mean(mat_shrp))
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

        window = np.ones(self.size)
        mat_smth = f_alias(mat, window=window)[0]
        num1 = np.abs(np.mean(mat_smth) - 0.5)
        self.assertTrue(num1 < 0.05)

        self.assertRaises(ValueError, f_alias, mat,
                          window=np.ones(self.size + 1))

    def test_generate_fitted_image(self):
        mat = np.random.rand(self.size, self.size)
        mat_fit = util.generate_fitted_image(mat, 1, axis=0)
        num1 = np.abs(np.mean(mat_fit) - 0.5)
        num2 = np.abs(np.mean(mat - mat_fit))
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

        mat_fit = util.generate_fitted_image(mat, 1, axis=1)
        num1 = np.abs(np.mean(mat_fit) - 0.5)
        num2 = np.abs(np.mean(mat - mat_fit))
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

    def test_detect_stripe(self):
        f_alias = util.detect_stripe
        np.random.seed(1)
        lis = np.random.rand(self.size)
        lis_off = np.linspace(0, 1, len(lis))
        lis = lis + lis_off
        lis[self.size // 2:self.size // 2 + 1] = 6.0
        lis_bin = f_alias(lis, 1.5)
        pos = np.where(lis_bin == 1.0)
        self.assertTrue((len(pos) > 0) and (pos[0] == self.size // 2))
        self.assertRaises(ValueError, f_alias, np.ones(self.size), 1.5)

    def test_make_2d_damping_window(self):
        window1 = util.make_2d_damping_window(self.size, self.size, 10)
        window2 = util.make_2d_damping_window(self.size, self.size, 10,
                                              window_name="butter")
        window = np.ones((self.size, self.size))
        num1 = np.mean(np.abs(window - window1))
        num2 = np.mean(np.abs(window - window2))
        num3 = np.mean(np.abs(window1 - window2))
        self.assertTrue(
            num1 > self.eps and num2 > self.eps and num3 > self.eps)

    def test_apply_wavelet_reconstruction(self):
        f_alias = util.apply_wavelet_reconstruction
        mat = np.random.rand(2 * self.size, 2 * self.size)
        data_dec = util.apply_wavelet_decomposition(mat, 'db5', 3)

        data_rec = f_alias(data_dec, 'db5')
        self.assertTrue(data_rec.shape == mat.shape)

        data_rec = f_alias(data_dec, 'db5', ignore_level=2)
        self.assertTrue(data_rec.shape == mat.shape)

    def test_apply_filter_to_wavelet_component(self):
        f_alias = util.apply_filter_to_wavelet_component
        mat = np.random.rand(2 * self.size, 2 * self.size)
        data = util.apply_wavelet_decomposition(mat, 'db5', 3)
        data1 = f_alias(data, 2, order=1)
        num = np.mean(np.abs(data[2][1] - data1[2][1]))
        self.assertTrue(num > self.eps)

        data1 = f_alias(data, level=None, order=1)
        num = np.mean(np.abs(data[2][1] - data1[2][1]))
        self.assertTrue(num > self.eps)

        data1 = f_alias(data, level=[1, 2], order=1)
        num = np.mean(np.abs(data[2][1] - data1[2][1]))
        self.assertTrue(num > self.eps)

        data1 = f_alias(data, level=(1, 2), order=1)
        num = np.mean(np.abs(data[2][1] - data1[2][1]))
        self.assertTrue(num > self.eps)

        data1 = f_alias(data, level=None, order=1,
                        method="apply_gaussian_filter", para=[5, 5])
        num = np.mean(np.abs(data[2][1] - data1[2][1]))
        self.assertTrue(num > self.eps)

        data1 = f_alias(data, level=None, order=1, method="gaussian_filter",
                        para=5)
        num = np.mean(np.abs(data[2][1] - data1[2][1]))
        self.assertTrue(num > self.eps)

        self.assertRaises(ValueError, f_alias, data, None, 1,
                          method="ssian_filter", para=[5, 5])

    def test_interpolate_inside_stripe(self):
        f_alias = util.interpolate_inside_stripe
        mat = np.ones((self.size, self.size), dtype=np.float32)
        list_mask = np.zeros(self.size)
        begin, end = self.size // 2, self.size // 2 + 2
        list_mask[begin:end] = 1.0

        mat_corr = f_alias(mat, list_mask)
        num = np.abs(np.mean(mat_corr[:, begin:end]) - 1.0)
        self.assertTrue(num < self.eps)

        mat_corr = f_alias(mat, list_mask, kind="cubic")
        num = np.abs(np.mean(mat_corr[:, begin:end]) - 1.0)
        self.assertTrue(num < self.eps)

        mat_corr = f_alias(mat, list_mask, kind="quintic")
        num = np.abs(np.mean(mat_corr[:, begin:end]) - 1.0)
        self.assertTrue(num < self.eps)

        self.assertRaises(ValueError, f_alias, mat, np.zeros(self.size + 1))

    def test_transform_slice_forward(self):
        f_alias = util.transform_slice_forward
        mat_sino1 = f_alias(self.mat_rec)
        num = np.sum((1 - ndi.binary_dilation(self.mat_sino))
                     * np.round(mat_sino1))
        self.assertTrue(num < self.eps)

        ncol = self.mat_rec.shape[-1]
        coord_mat = util.rectangular_from_polar(ncol, ncol, ncol, ncol)
        mat_sino1 = f_alias(self.mat_rec, coord_mat=coord_mat)
        num = np.sum((1 - ndi.binary_dilation(self.mat_sino))
                     * np.round(mat_sino1))
        self.assertTrue(num < self.eps)

        self.assertRaises(ValueError, f_alias, self.mat_rec[:, 2:])

        (x_mat, y_mat) = coord_mat
        self.assertRaises(ValueError, f_alias, self.mat_rec,
                          (x_mat, y_mat[1:]))

    def test_transform_slice_backward(self):
        f_alias = util.transform_slice_backward
        mat_rec1 = f_alias(self.mat_sino)
        num = np.sum((1 - ndi.binary_dilation(self.mat_rec))
                     * np.round(mat_rec1))
        self.assertTrue(num < self.eps)

        ncol = self.mat_rec.shape[-1]
        coord_mat = util.polar_from_rectangular(ncol, ncol, ncol, ncol)
        mat_rec1 = f_alias(self.mat_sino, coord_mat=coord_mat)
        num = np.sum((1 - ndi.binary_dilation(self.mat_rec))
                     * np.round(mat_rec1))
        self.assertTrue(num < self.eps)

        self.assertRaises(ValueError, f_alias, self.mat_rec[:, 2:])

        (x_mat, y_mat) = coord_mat
        self.assertRaises(ValueError, f_alias, self.mat_rec,
                          (x_mat, y_mat[1:]))

    def test_apply_gaussian_filter(self):
        mat = np.random.rand(self.size, self.size)

        mat_smth = util.apply_gaussian_filter(mat, 5, 5, 10)
        num1 = np.abs(np.mean(mat_smth) - 0.5)
        num2 = np.abs(np.mean(mat - mat_smth))
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

        mat_smth = util.apply_gaussian_filter(mat, 5, 5, 10,
                                              mode=["mean", "mean"])
        num1 = np.abs(np.mean(mat_smth) - 0.5)
        num2 = np.abs(np.mean(mat - mat_smth))
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

    def test_apply_regularization_filter(self):
        mat = np.random.rand(self.size, self.size)
        mat_smth = util.apply_regularization_filter(mat, 0.01)
        num1 = np.abs(np.mean(mat_smth) - 0.5)
        num2 = np.abs(np.mean(mat - mat_smth))
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

        mat_smth = util.apply_regularization_filter(mat, 0.01, axis=0)
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

        size1 = self.size + 1
        list1 = np.zeros(size1, dtype=np.float32)
        list1[size1 // 2 - 5: size1 // 2 + 6] = np.float32(1.0)
        mat1 = util.transform_1d_window_to_2d(list1)
        (width, height) = mat1.shape
        self.assertTrue(width == size1 and height == size1)

    def test_fix_non_sample_areas(self):
        overlaps = np.copy(self.overlaps0)
        overlaps[1, 1] = [0, 0]
        fixed_overlaps = util.fix_non_sample_areas(overlaps,
                                                   direction="vertical")
        self.assertTrue(overlaps[1, 0, 0] == fixed_overlaps[1, 1, 0] and
                        overlaps[1, 0, 1] == fixed_overlaps[1, 1, 1])

        overlaps = np.copy(self.overlaps0)
        overlaps[1, 1] = [0, 0]
        overlaps[1, 2] = [0, 0]
        fixed_overlaps = util.fix_non_sample_areas(overlaps,
                                                   direction="vertical")
        self.assertTrue(
            fixed_overlaps[1, 1, 0] != 0 and fixed_overlaps[1, 1, 1] == 1)

        overlaps = np.copy(self.overlaps0)
        overlaps[2, 1] = [0, 0]
        overlaps[2, 2] = [0, 0]
        overlaps[1, 2] = [0, 0]
        overlaps[1, 1] = [0, 0]
        overlaps[1, 0] = [0, 0]
        overlaps[0, 2] = [0, 0]
        overlaps[0, 1] = [0, 0]
        fixed_overlaps = util.fix_non_sample_areas(overlaps,
                                                   direction="vertical")
        self.assertTrue(
            fixed_overlaps[1, 1, 0] != 0 and fixed_overlaps[1, 1, 1] == 1)

        overlaps = np.copy(self.overlaps0)
        overlaps[0, 0] = [0, 0]
        overlaps[0, 1] = [0, 0]
        overlaps[0, 2] = [0, 0]
        overlaps[1, 2] = [0, 0]
        overlaps[2, 0] = [0, 0]
        fixed_overlaps = util.fix_non_sample_areas(overlaps,
                                                   direction="vertical")
        self.assertTrue(
            fixed_overlaps[1, 1, 0] != 0 and fixed_overlaps[1, 1, 1] == 1)

        overlaps = np.copy(self.overlaps0)
        overlaps[0, 0] = [0, 0]
        overlaps[1, 0] = [0, 0]
        overlaps[2, 0] = [0, 0]
        overlaps[1, 1] = [0, 0]
        overlaps[1, 2] = [0, 0]
        fixed_overlaps = util.fix_non_sample_areas(overlaps,
                                                   direction="vertical")
        self.assertTrue(
            fixed_overlaps[1, 1, 0] != 0 and fixed_overlaps[1, 1, 1] == 1)

        overlaps = np.copy(self.overlaps0)
        overlaps[1, 1] = [0, 0]
        fixed_overlaps = util.fix_non_sample_areas(overlaps,
                                                   direction="horizontal")
        self.assertTrue(overlaps[0, 1, 0] == fixed_overlaps[1, 1, 0] and
                        overlaps[0, 1, 1] == fixed_overlaps[1, 1, 1])

        overlaps = np.copy(self.overlaps0)
        overlaps[1, 1] = [0, 0]
        overlaps[1, 2] = [0, 0]
        fixed_overlaps = util.fix_non_sample_areas(overlaps,
                                                   direction="horizontal")
        self.assertTrue(
            fixed_overlaps[1, 1, 0] != 0 and fixed_overlaps[1, 1, 1] == 1)

        overlaps = np.copy(self.overlaps0)
        overlaps[2, 1] = [0, 0]
        overlaps[2, 2] = [0, 0]
        overlaps[1, 2] = [0, 0]
        overlaps[1, 1] = [0, 0]
        overlaps[1, 0] = [0, 0]
        overlaps[0, 2] = [0, 0]
        overlaps[0, 1] = [0, 0]
        fixed_overlaps = util.fix_non_sample_areas(overlaps,
                                                   direction="horizontal")
        self.assertTrue(
            fixed_overlaps[1, 1, 0] != 0 and fixed_overlaps[1, 1, 1] == 1)

        overlaps = np.copy(self.overlaps0)
        overlaps[0, 0] = [0, 0]
        overlaps[0, 1] = [0, 0]
        overlaps[0, 2] = [0, 0]
        overlaps[1, 2] = [0, 0]
        overlaps[2, 0] = [0, 0]
        fixed_overlaps = util.fix_non_sample_areas(overlaps,
                                                   direction="horizontal")
        self.assertTrue(
            fixed_overlaps[1, 1, 0] != 0 and fixed_overlaps[1, 1, 1] == 1)

        overlaps = np.copy(self.overlaps0)
        overlaps[0, 0] = [0, 0]
        overlaps[1, 0] = [0, 0]
        overlaps[2, 0] = [0, 0]
        overlaps[1, 1] = [0, 0]
        overlaps[1, 2] = [0, 0]
        fixed_overlaps = util.fix_non_sample_areas(overlaps,
                                                   direction="horizontal")
        self.assertTrue(
            fixed_overlaps[1, 1, 0] != 0 and fixed_overlaps[1, 1, 1] == 1)

    def test_locate_slice(self):
        results = util.locate_slice(100, 2160, self.overlaps0)
        self.assertTrue(len(results) == 1)

        results = util.locate_slice(2100, 2160, self.overlaps0)
        self.assertTrue(len(results) == 2)

        results = util.locate_slice(100, 2160, self.overlaps1)
        self.assertTrue(len(results) == 1)

        results = util.locate_slice(2100, 2160, self.overlaps1)
        self.assertTrue(len(results) == 2)

    def test_locate_slice_chunk(self):
        results = util.locate_slice_chunk(100, 120, 2160, self.overlaps0)
        self.assertTrue(len(results[0]) == 20)

        results = util.locate_slice_chunk(2100, 2120, 2160, self.overlaps0)
        self.assertTrue(len(results[0]) == 20 and len(results[1]) == 20)

        results = util.locate_slice_chunk(1690, 1710, 2000, self.overlaps0)
        self.assertTrue(results[0][0][-1] == 1.0 and results[1][-1][-1] < 1.0)

        results = util.locate_slice_chunk(100, 120, 2160, self.overlaps1)
        self.assertTrue(len(results[0]) == 20)

        results = util.locate_slice_chunk(2100, 2120, 2160, self.overlaps1)
        self.assertTrue(len(results[0]) == 20 and len(results[1]) == 20)

        results = util.locate_slice_chunk(1690, 1710, 2000, self.overlaps1)
        self.assertTrue(results[0][0][-1] == 1.0 and results[1][-1][-1] < 1.0)

    def test_generate_spiral_positions(self):
        f_alias = util.generate_spiral_positions
        xy_list = f_alias(40, 40, 2160, 2560, spiral_shape=1.0)
        self.assertTrue(len(xy_list) == 40)

        self.assertRaises(ValueError, f_alias, 200, 40, 2000, 2000)

    def test_find_center_visual_sinograms(self):
        output_base = "./tmp"
        (hei, wid) = self.mat_sino.shape
        start, stop = wid // 2 - 3, wid // 2 + 3
        num_img = stop - start + 1
        output_folder = util.find_center_visual_sinograms(self.mat_sino,
                                                          output_base, start,
                                                          stop, step=1,
                                                          zoom=1.0)
        files = losa.find_file(output_folder + "/*.tif*")
        img = losa.load_image(files[0])
        hei2 = img.shape[0]
        self.assertTrue(len(files) == num_img and hei2 == 2 * hei)

    def test_find_center_visual_slices(self):
        output_base = "./tmp"
        (hei, wid) = self.mat_sino.shape
        start, stop = wid // 2 - 3, wid // 2 + 3
        num_img = stop - start + 1
        output_folder = util.find_center_visual_slices(self.mat_sino,
                                                       output_base, start,
                                                       stop, step=1, zoom=0.5,
                                                       apply_log=False)
        files = losa.find_file(output_folder + "/*.tif*")
        img = losa.load_image(files[0])
        hei2 = img.shape[0]
        self.assertTrue(len(files) == num_img and hei2 == hei // 2)
