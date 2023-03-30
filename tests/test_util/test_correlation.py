# ============================================================================
# ============================================================================
# Copyright (c) 2022 Nghia T. Vo. All rights reserved.
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
Tests for the methods in util/correlation.py

"""

import unittest
import warnings
import numba
import numpy as np
from numba import cuda
import scipy.ndimage as ndi
import algotom.util.correlation as corl


class UtilityMethods(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore',
                                category=numba.NumbaPerformanceWarning)
        self.eps = 10 ** (-6)
        self.size = (65, 65)
        speckle_size = 2
        np.random.seed(1)
        mat_tmp1 = ndi.gaussian_filter(np.random.normal(
            0.5, scale=0.2, size=self.size), speckle_size)
        np.random.seed(11)
        mat_tmp2 = ndi.gaussian_filter(np.random.normal(
            0.5, scale=0.2, size=self.size), speckle_size)
        speckle = np.abs(mat_tmp1 + 1j * mat_tmp2) ** 2
        self.shift = 1.5
        sample = ndi.shift(speckle, (self.shift, self.shift), mode="nearest")
        self.ref_stack = np.asarray([speckle for _ in range(3)])
        self.sam_stack = np.asarray([sample for _ in range(3)])

    def tearDown(self):
        warnings.filterwarnings("default",
                                category=numba.NumbaPerformanceWarning)

    def test_normalize_image(self):
        mat = np.random.normal(0.5, 0.6, (64, 64))
        mat1 = corl.normalize_image(mat)
        num1 = np.mean(mat1)
        num2 = np.abs(np.std(mat1) - 1.0)
        self.assertTrue(num1 < self.eps and num2 < self.eps)
        mat = np.random.normal(0.5, 0.6, (6, 64, 64))
        mat1 = corl.normalize_image(mat)
        num1 = np.min(mat1)
        self.assertTrue(num1 >= 0.0)

    def test_generate_correlation_map(self):
        f_alias = corl.generate_correlation_map
        drop = 10
        size = 2 * drop + 1
        coef_mat = f_alias(self.ref_stack[0],
                           self.sam_stack[0][drop:-drop, drop:-drop],
                           gpu=False)
        num1 = np.percentile(coef_mat, 90) / np.max(coef_mat)
        self.assertTrue(coef_mat.shape == (size, size) and num1 < 0.5)

        coef_mat = f_alias(self.ref_stack,
                           self.sam_stack[:, drop:-drop, drop:-drop],
                           gpu=False)
        num1 = np.percentile(coef_mat, 90) / np.max(coef_mat)
        self.assertTrue(coef_mat.shape == (size, size) and num1 < 0.5)

        coef_mat = f_alias(self.ref_stack,
                           self.sam_stack[:, drop:-drop, drop:-drop],
                           gpu=True)
        num1 = np.percentile(coef_mat, 90) / np.max(coef_mat)
        self.assertTrue(coef_mat.shape == (size, size) and num1 < 0.5)

        coef_mat = f_alias(self.ref_stack[0],
                           self.sam_stack[0, drop:-drop, drop:-drop],
                           gpu=True)
        num1 = np.percentile(coef_mat, 90) / np.max(coef_mat)
        self.assertTrue(coef_mat.shape == (size, size) and num1 < 0.5)

        self.assertRaises(ValueError, f_alias, self.ref_stack[0, 1],
                          self.sam_stack[0, 1], gpu=False)
        self.assertRaises(ValueError, f_alias,
                          self.ref_stack[0, drop:-drop, drop:-drop],
                          self.sam_stack[0], gpu=False)
        self.assertRaises(ValueError, f_alias, self.ref_stack[0:1],
                          self.sam_stack[:, drop:-drop, drop:-drop], gpu=False)

    def test_locate_peak(self):
        f_alias = corl.generate_correlation_map
        drop = 10
        mat = f_alias(self.ref_stack[0],
                      self.sam_stack[0][drop:-drop, drop:-drop],
                      gpu=False)
        x_pos, y_pos = corl.locate_peak(mat, sub_pixel=True, method="diff",
                                        dim=2, size=3, max_peak=True)
        x_sh, y_sh = x_pos - drop, y_pos - drop
        num1, num2 = np.abs(x_sh + self.shift), np.abs(y_sh + self.shift)
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

        x_pos, y_pos = corl.locate_peak(mat, sub_pixel=True, method="diff",
                                        dim=1, size=3, max_peak=True)
        x_sh, y_sh = x_pos - drop, y_pos - drop
        num1, num2 = np.abs(x_sh + self.shift), np.abs(y_sh + self.shift)
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

        x_pos, y_pos = corl.locate_peak(mat, sub_pixel=True, method="poly_fit",
                                        dim=2, size=3, max_peak=True)
        x_sh, y_sh = x_pos - drop, y_pos - drop
        num1, num2 = np.abs(x_sh + self.shift), np.abs(y_sh + self.shift)
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

        x_pos, y_pos = corl.locate_peak(mat, sub_pixel=True, method="poly_fit",
                                        dim=1, size=3, max_peak=True)
        x_sh, y_sh = x_pos - drop, y_pos - drop
        num1, num2 = np.abs(x_sh + self.shift), np.abs(y_sh + self.shift)
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

        x_pos, y_pos = corl.locate_peak(mat, sub_pixel=True, method="poly_fit",
                                        dim=2, size=5, max_peak=True)
        x_sh, y_sh = x_pos - drop, y_pos - drop
        num1, num2 = np.abs(x_sh + self.shift), np.abs(y_sh + self.shift)
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

        x_pos, y_pos = corl.locate_peak(mat, sub_pixel=True, method="poly_fit",
                                        dim=1, size=5, max_peak=True)
        x_sh, y_sh = x_pos - drop, y_pos - drop
        num1, num2 = np.abs(x_sh + self.shift), np.abs(y_sh + self.shift)
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

    def test_find_shift_based_correlation_map(self):
        f_alias = corl.find_shift_based_correlation_map
        x_sh, y_sh = f_alias(self.ref_stack[0], self.sam_stack[0], margin=10,
                             axis=None, sub_pixel=True, method="diff", dim=2,
                             size=3, gpu=False)
        num1, num2 = np.abs(x_sh + self.shift), np.abs(y_sh + self.shift)
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

        x_sh, y_sh = f_alias(self.ref_stack, self.sam_stack, margin=10,
                             axis=None, sub_pixel=True, method="diff", dim=2,
                             size=3, gpu=False)
        num1, num2 = np.abs(x_sh + self.shift), np.abs(y_sh + self.shift)
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

        x_sh, y_sh = f_alias(self.ref_stack, self.sam_stack, margin=10,
                             axis=1, sub_pixel=True, method="poly_fit", dim=2,
                             size=3, gpu=False)
        num1 = np.abs(x_sh + self.shift)
        self.assertTrue(num1 < 0.1)

        x_sh, y_sh = f_alias(self.ref_stack, self.sam_stack, margin=10,
                             axis=0, sub_pixel=True, method="poly_fit", dim=2,
                             size=3, gpu=False)
        num1 = np.abs(y_sh + self.shift)
        self.assertTrue(num1 < 0.1)

        x_sh, y_sh = f_alias(self.ref_stack[0], self.sam_stack[0], margin=10,
                             axis=1, sub_pixel=True, method="poly_fit", dim=2,
                             size=3, gpu=False)
        num1 = np.abs(x_sh + self.shift)
        self.assertTrue(num1 < 0.1)

        x_sh, y_sh = f_alias(self.ref_stack[0], self.sam_stack[0], margin=10,
                             axis=0, sub_pixel=True, method="poly_fit", dim=2,
                             size=3, gpu=False)
        num1 = np.abs(y_sh + self.shift)
        self.assertTrue(num1 < 0.1)

        self.assertRaises(ValueError, f_alias, self.ref_stack[0],
                          self.sam_stack[0, 16:-16], margin=20, axis=None,
                          sub_pixel=True, method="poly_fit", dim=2, size=5,
                          gpu=False)

    def test_find_local_shifts(self):
        f_alias = corl.find_local_shifts
        margin = 5
        edge = margin + 2
        x_shifts, y_shifts = f_alias(self.ref_stack[0], self.sam_stack[0],
                                     dim=1, win_size=5, margin=margin,
                                     method="diff", size=3, gpu=False,
                                     block=(16, 16), ncore=1, norm=True,
                                     norm_global=True, chunk_size=None)
        num1 = np.abs(np.mean(x_shifts[edge:-edge, edge:-edge]) + self.shift)
        num2 = np.abs(np.mean(y_shifts[edge:-edge, edge:-edge]) + self.shift)
        self.assertTrue(num1 < 0.1 and num2 < 0.1)

        x_shifts, y_shifts = f_alias(self.ref_stack[0], self.sam_stack[0],
                                     dim=1, win_size=5, margin=margin,
                                     method="poly_fit", size=3, gpu=False,
                                     block=(16, 16), ncore=1, norm=True,
                                     norm_global=True, chunk_size=20)
        num1 = np.abs(np.mean(x_shifts[edge:-edge, edge:-edge]) + self.shift)
        num2 = np.abs(np.mean(y_shifts[edge:-edge, edge:-edge]) + self.shift)
        self.assertTrue(num1 < 0.1 and num2 < 0.1)

        x_shifts, y_shifts = f_alias(self.ref_stack[0], self.sam_stack[0],
                                     dim=1, win_size=5, margin=margin,
                                     method="poly_fit", size=3, gpu=False,
                                     block=(16, 16), ncore=None, norm=True,
                                     norm_global=False, chunk_size=20)
        num1 = np.abs(np.mean(x_shifts[edge:-edge, edge:-edge]) + self.shift)
        num2 = np.abs(np.mean(y_shifts[edge:-edge, edge:-edge]) + self.shift)
        self.assertTrue(num1 < 0.1 and num2 < 0.1)

        x_shifts, y_shifts = f_alias(self.ref_stack[0], self.sam_stack[0],
                                     dim=2, win_size=5, margin=margin,
                                     method="diff", size=3, gpu=False,
                                     block=(16, 16), ncore=1, norm=True,
                                     norm_global=True, chunk_size=None)
        num1 = np.abs(np.mean(x_shifts[edge:-edge, edge:-edge]) + self.shift)
        num2 = np.abs(np.mean(y_shifts[edge:-edge, edge:-edge]) + self.shift)
        self.assertTrue(num1 < 0.1 and num2 < 0.1)

        x_shifts, y_shifts = f_alias(self.ref_stack[0], self.sam_stack[0],
                                     dim=2, win_size=5, margin=margin,
                                     method="poly_fit", size=3, gpu=False,
                                     block=(16, 16), ncore=1, norm=True,
                                     norm_global=True, chunk_size=20)
        num1 = np.abs(np.mean(x_shifts[edge:-edge, edge:-edge]) + self.shift)
        num2 = np.abs(np.mean(y_shifts[edge:-edge, edge:-edge]) + self.shift)
        self.assertTrue(num1 < 0.1 and num2 < 0.1)

        x_shifts, y_shifts = f_alias(self.ref_stack, self.sam_stack,
                                     dim=2, win_size=5, margin=margin,
                                     method="diff", size=3, gpu=False,
                                     block=(16, 16), ncore=1, norm=True,
                                     norm_global=True, chunk_size=None)
        num1 = np.abs(np.mean(x_shifts[edge:-edge, edge:-edge]) + self.shift)
        num2 = np.abs(np.mean(y_shifts[edge:-edge, edge:-edge]) + self.shift)
        self.assertTrue(num1 < 0.1 and num2 < 0.1)

        x_shifts, y_shifts = f_alias(self.ref_stack, self.sam_stack,
                                     dim=2, win_size=5, margin=margin,
                                     method="poly_fit", size=3, gpu=False,
                                     block=(16, 16), ncore=1, norm=True,
                                     norm_global=True, chunk_size=20)
        num1 = np.abs(np.mean(x_shifts[edge:-edge, edge:-edge]) + self.shift)
        num2 = np.abs(np.mean(y_shifts[edge:-edge, edge:-edge]) + self.shift)
        self.assertTrue(num1 < 0.1 and num2 < 0.1)

        if cuda.is_available():
            x_shifts, y_shifts = f_alias(self.ref_stack[0], self.sam_stack[0],
                                         dim=1, win_size=5, margin=margin,
                                         method="diff", size=3, gpu=True,
                                         block=(16, 16), ncore=1, norm=True,
                                         norm_global=True, chunk_size=None)
            num1 = np.abs(
                np.mean(x_shifts[edge:-edge, edge:-edge]) + self.shift)
            num2 = np.abs(
                np.mean(y_shifts[edge:-edge, edge:-edge]) + self.shift)
            self.assertTrue(num1 < 0.1 and num2 < 0.1)

            x_shifts, y_shifts = f_alias(self.ref_stack, self.sam_stack,
                                         dim=1, win_size=5, margin=margin,
                                         method="diff", size=3, gpu=True,
                                         block=(16, 16), ncore=1, norm=False,
                                         norm_global=False, chunk_size=10)
            num1 = np.abs(
                np.mean(x_shifts[edge:-edge, edge:-edge]) + self.shift)
            num2 = np.abs(
                np.mean(y_shifts[edge:-edge, edge:-edge]) + self.shift)
            self.assertTrue(num1 < 1.0 and num2 < 1.0)

            x_shifts, y_shifts = f_alias(self.ref_stack, self.sam_stack,
                                         dim=2, win_size=5, margin=margin,
                                         method="diff", size=3, gpu=True,
                                         block=(16, 16), ncore=1, norm=True,
                                         norm_global=False, chunk_size=None)
            num1 = np.abs(
                np.mean(x_shifts[edge:-edge, edge:-edge]) + self.shift)
            num2 = np.abs(
                np.mean(y_shifts[edge:-edge, edge:-edge]) + self.shift)
            self.assertTrue(num1 < 0.1 and num2 < 0.1)

            x_shifts, y_shifts = f_alias(self.ref_stack, self.sam_stack,
                                         dim=2, win_size=5, margin=margin,
                                         method="diff", size=3, gpu="hybrid",
                                         block=(16, 16), ncore=1, norm=True,
                                         norm_global=True, chunk_size=None)
            num1 = np.abs(
                np.mean(x_shifts[edge:-edge, edge:-edge]) + self.shift)
            num2 = np.abs(
                np.mean(y_shifts[edge:-edge, edge:-edge]) + self.shift)
            self.assertTrue(num1 < 0.1 and num2 < 0.1)

    def test_find_global_shift_based_local_shifts(self):
        f_alias = corl.find_global_shift_based_local_shifts
        list_ij = [[25, 30, 35], [26, 31, 36]]
        x_shift, y_shift = f_alias(self.ref_stack[0], self.sam_stack[0], 17,
                                   5, list_ij=list_ij, num_point=None,
                                   global_value="median", gpu=False,
                                   block=32, sub_pixel=True, method="diff",
                                   size=3, ncore=None, norm=True,
                                   return_list=False)
        num1 = np.abs(x_shift + self.shift)
        num2 = np.abs(y_shift + self.shift)
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

        x_shift, y_shift = f_alias(self.ref_stack[0], self.sam_stack[0], 17,
                                   5, list_ij=list_ij, num_point=None,
                                   global_value="mean", gpu=False,
                                   block=32, sub_pixel=True, method="poly_fit",
                                   size=3, ncore=None, norm=True,
                                   return_list=False)
        num1 = np.abs(x_shift + self.shift)
        num2 = np.abs(y_shift + self.shift)
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

        x_shifts, y_shifts = f_alias(self.ref_stack[0], self.sam_stack[0], 17,
                                     5, list_ij=list_ij, num_point=None,
                                     global_value="mixed", gpu=False,
                                     block=32, sub_pixel=True, method="diff",
                                     size=3, ncore=None, norm=True,
                                     return_list=True)
        num1 = np.abs(np.mean(x_shifts) + self.shift)
        num2 = np.abs(np.mean(x_shifts) + self.shift)
        self.assertTrue(num1 < 0.05 and num2 < 0.05 and len(x_shifts) == 3)

        x_shift, y_shift = f_alias(self.ref_stack[0], self.sam_stack[0], 17,
                                   5, list_ij=None, num_point=10,
                                   global_value="mixed", gpu=False,
                                   block=32, sub_pixel=True, method="diff",
                                   size=3, ncore=None, norm=False,
                                   return_list=False)
        num1 = np.abs(x_shift + self.shift)
        num2 = np.abs(y_shift + self.shift)
        self.assertTrue(num1 < 0.05 and num2 < 0.05)

        if cuda.is_available():
            x_shift, y_shift = f_alias(self.ref_stack[0], self.sam_stack[0],
                                       17, 5, list_ij=list_ij, num_point=None,
                                       global_value="median", gpu=True,
                                       block=32, sub_pixel=True, method="diff",
                                       size=3, ncore=None, norm=True,
                                       return_list=False)
            num1 = np.abs(x_shift + self.shift)
            num2 = np.abs(y_shift + self.shift)
            self.assertTrue(num1 < 0.05 and num2 < 0.05)

            x_shift, y_shift = f_alias(self.ref_stack[0], self.sam_stack[0],
                                       17, 5, list_ij=list_ij, num_point=None,
                                       global_value="mean", gpu=True,
                                       block=32, sub_pixel=True,
                                       method="poly_fit", size=3, ncore=None,
                                       norm=True, return_list=False)
            num1 = np.abs(x_shift + self.shift)
            num2 = np.abs(y_shift + self.shift)
            self.assertTrue(num1 < 0.05 and num2 < 0.05)

            x_shift, y_shift = f_alias(self.ref_stack[0], self.sam_stack[0],
                                       17, 5, list_ij=None,
                                       num_point=None, global_value="mixed",
                                       gpu=True, block=32, sub_pixel=True,
                                       method="diff", size=3, ncore=None,
                                       norm=True, return_list=False)
            num1 = np.abs(x_shift + self.shift)
            num2 = np.abs(y_shift + self.shift)
            self.assertTrue(num1 < 0.05 and num2 < 0.05)

    def test_find_local_shifts_umpa(self):
        f_alias = corl.find_local_shifts_umpa
        margin = 5
        edge = margin + 2
        x_shifts, y_shifts = f_alias(self.ref_stack, self.sam_stack,
                                     win_size=5, margin=margin,
                                     method="diff", size=3, gpu=False,
                                     block=(16, 16), ncore=None,
                                     chunk_size=None, filter_name="hamming",
                                     dark_signal=False)
        num1 = np.abs(np.mean(x_shifts[edge:-edge, edge:-edge]) + self.shift)
        num2 = np.abs(np.mean(y_shifts[edge:-edge, edge:-edge]) + self.shift)
        check1 = True if (num1 < 0.1 and num2 < 0.1) else False

        x_shifts, y_shifts = f_alias(self.ref_stack, self.sam_stack,
                                     win_size=5, margin=margin,
                                     method="diff", size=3, gpu=False,
                                     block=(16, 16), ncore=None,
                                     chunk_size=None, filter_name=None,
                                     dark_signal=False)
        num1 = np.abs(np.mean(x_shifts[edge:-edge, edge:-edge]) + self.shift)
        num2 = np.abs(np.mean(y_shifts[edge:-edge, edge:-edge]) + self.shift)
        check2 = True if (num1 < 0.1 and num2 < 0.1) else False
        check3, check4 = True, True
        if cuda.is_available() is True:
            x_shifts, y_shifts = f_alias(self.ref_stack, self.sam_stack,
                                         win_size=5, margin=margin,
                                         method="diff", size=3, gpu=True,
                                         block=(16, 16), ncore=None,
                                         chunk_size=None,
                                         filter_name="hamming",
                                         dark_signal=False)
            num1 = np.abs(
                np.mean(x_shifts[edge:-edge, edge:-edge]) + self.shift)
            num2 = np.abs(
                np.mean(y_shifts[edge:-edge, edge:-edge]) + self.shift)
            check3 = True if (num1 < 0.1 and num2 < 0.1) else False

            x_shifts, y_shifts = f_alias(self.ref_stack, self.sam_stack,
                                         win_size=5, margin=margin,
                                         method="diff", size=3, gpu=True,
                                         block=(16, 16), ncore=None,
                                         chunk_size=None, filter_name=None,
                                         dark_signal=False)
            num1 = np.abs(
                np.mean(x_shifts[edge:-edge, edge:-edge]) + self.shift)
            num2 = np.abs(
                np.mean(y_shifts[edge:-edge, edge:-edge]) + self.shift)
            check4 = True if (num1 < 0.1 and num2 < 0.1) else False
        self.assertTrue(check1 and check2 and check3 and check4)
