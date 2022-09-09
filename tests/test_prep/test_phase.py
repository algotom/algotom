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
Tests for the methods in prep/phase.py

"""

import unittest
import numpy as np
import scipy.ndimage as ndi
import algotom.prep.phase as ps
import algotom.util.simulation as sim


class PhaseMethods(unittest.TestCase):

    def peak_function(self, x, y):
        z = 3 * ((1 - x) ** 2) * np.exp(-(x ** 2) - (y + 1) ** 2) \
            - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(-x ** 2 - y ** 2) \
            - 1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)
        return z

    def make_phase_phantom(self, size):
        xlist = (np.arange(size) - 0.5 * size) / (0.15 * size)
        ylist = 1.0 * xlist
        x_mat, y_mat = np.meshgrid(xlist, ylist)
        z_mat = 4.0 * np.asarray(
            [self.peak_function(x, y) for x in xlist for y in ylist])
        z_mat = z_mat.reshape(size, size)
        z_mat = z_mat + 0.035 * x_mat * size + 0.003 * y_mat * size + 27
        nmean = np.mean(z_mat)
        return z_mat - nmean

    def setUp(self):
        self.eps = 0.01
        phantom = self.make_phase_phantom(128)
        self.phase_wrapped = np.arctan2(np.sin(phantom), np.cos(phantom))
        self.phase_image = phantom
        size = (65, 65)
        speckle_size = 2
        np.random.seed(1)
        mat_tmp1 = ndi.gaussian_filter(np.random.normal(
            0.5, scale=0.2, size=size), speckle_size)
        np.random.seed(11)
        mat_tmp2 = ndi.gaussian_filter(np.random.normal(
            0.5, scale=0.2, size=size), speckle_size)
        speckle = np.abs(mat_tmp1 + 1j * mat_tmp2) ** 2
        self.shift = 1.5
        sample1 = ndi.shift(speckle, (self.shift, self.shift), mode="nearest")
        sample2 = 0.1 * sim.make_elliptic_mask(size[0], 0.0, 20, 0.0)
        self.ref_stack = np.asarray([speckle for _ in range(3)])
        self.sam_stack1 = np.asarray([sample1 for _ in range(3)])
        self.sam_stack2 = np.asarray(
            [ndi.shift(sample2, (i, i), mode="nearest") for i in range(3)])
        self.sam_stack2 = self.sam_stack2 + self.sam_stack1

    def test_unwrap_phase_based_cosine_transform(self):
        phase_unwrapped1 = ps.unwrap_phase_based_cosine_transform(
            self.phase_wrapped)
        (height, width) = phase_unwrapped1.shape
        window = ps._make_cosine_window(height, width)
        phase_unwrapped2 = ps.unwrap_phase_based_cosine_transform(
            self.phase_wrapped, window=window)
        num1 = np.median(np.abs(phase_unwrapped1 - self.phase_image))
        num2 = np.median(np.abs(phase_unwrapped2 - self.phase_image))
        self.assertTrue(num1 < self.eps and num2 < self.eps)

    def test_unwrap_phase_based_fft(self):
        phase_unwrapped1 = ps.unwrap_phase_based_fft(self.phase_wrapped)
        (height, width) = phase_unwrapped1.shape
        win_for = ps._make_window(2 * height, 2 * width, direction="forward")
        win_back = ps._make_window(2 * height, 2 * width, direction="backward")
        phase_unwrapped2 = ps.unwrap_phase_based_fft(self.phase_wrapped,
                                                     win_for=win_for,
                                                     win_back=win_back)
        num1 = np.median(np.abs(phase_unwrapped1 - self.phase_image))
        num2 = np.median(np.abs(phase_unwrapped2 - self.phase_image))
        self.assertTrue(num1 < self.eps and num2 < self.eps)

    def test_unwrap_phase_iterative_fft(self):
        phase_unwrapped1 = ps.unwrap_phase_iterative_fft(self.phase_wrapped,
                                                         iteration=3)
        (height, width) = phase_unwrapped1.shape
        win_for = ps._make_window(2 * height, 2 * width, direction="forward")
        win_back = ps._make_window(2 * height, 2 * width, direction="backward")
        phase_unwrapped2 = ps.unwrap_phase_iterative_fft(self.phase_wrapped,
                                                         iteration=3,
                                                         win_for=win_for,
                                                         win_back=win_back)
        num1 = np.median(np.abs(phase_unwrapped1 - self.phase_image))
        num2 = np.median(np.abs(phase_unwrapped2 - self.phase_image))
        self.assertTrue(num1 < self.eps and num2 < self.eps)

    def test_reconstruct_surface_from_gradient_FC_method(self):
        pad = 50
        mat_tmp = np.pad(self.phase_image, pad, mode="linear_ramp")
        (grad_y, grad_x) = np.gradient(mat_tmp)
        f_alias = ps.reconstruct_surface_from_gradient_FC_method
        mat = f_alias(grad_x, grad_y, correct_negative=False)
        mat = mat[pad:-pad, pad:-pad]
        num = np.mean(np.abs(mat - self.phase_image))
        self.assertTrue(num < 2.0)

    def test_reconstruct_surface_from_gradient_SCS_method(self):
        pad = 50
        mat_tmp = np.pad(self.phase_image, pad, mode="linear_ramp")
        (grad_y, grad_x) = np.gradient(mat_tmp)
        f_alias = ps.reconstruct_surface_from_gradient_SCS_method
        mat = f_alias(grad_x, grad_y, correct_negative=False)
        mat = mat[pad:-pad, pad:-pad]
        num = np.mean(np.abs(mat - self.phase_image))
        self.assertTrue(num < 2.0)

    def test_find_shift_between_image_stacks(self):
        f_alias = ps.find_shift_between_image_stacks
        list_ij = [[20, 30, 40], [21, 31, 41]]
        xy_shifts = f_alias(self.ref_stack, self.sam_stack1, 5, 8,
                            list_ij, global_value="median", gpu=False,
                            block=32, sub_pixel=True, method="poly_fit",
                            size=3, ncore=1, norm=False)
        num1 = 0.5 * (np.mean(np.abs(xy_shifts[:, 0] + self.shift)) + np.mean(
            np.abs(xy_shifts[:, 1] + self.shift)))
        xy_shifts = f_alias(self.ref_stack, self.sam_stack1, 5, 8,
                            list_ij, global_value="median", gpu=False,
                            block=32, sub_pixel=True, method="diff",
                            size=3, ncore=1, norm=False)
        num2 = 0.5 * (np.mean(np.abs(xy_shifts[:, 0] + self.shift)) + np.mean(
            np.abs(xy_shifts[:, 1] + self.shift)))
        self.assertTrue(num1 < 0.1 and num2 < 0.1)

    def test_find_shift_between_sample_images(self):
        list_ij = [[20, 30, 40], [21, 31, 41]]
        f_alias1 = ps.find_shift_between_image_stacks
        sr_shifts = f_alias1(self.ref_stack, self.sam_stack1, 5, 8,
                             list_ij, global_value="median", gpu=False,
                             block=32, sub_pixel=True, method="poly_fit",
                             size=3, ncore=1, norm=False)
        f_alias2 = ps.find_shift_between_sample_images
        list_ij = [32, 32]
        sam_shifts = f_alias2(self.ref_stack, self.sam_stack2, sr_shifts, 41,
                              8, list_ij, global_value="median", gpu=False,
                              block=32, sub_pixel=True, method="diff", size=3,
                              ncore=1, norm=False)
        num1 = np.mean(np.abs(sam_shifts[:, 0] + np.arange(3)))
        num2 = np.mean(np.abs(sam_shifts[:, 1] + np.arange(3)))
        self.assertTrue(num1 < 0.1 and num2 < 0.1)

    def test_align_image_stacks(self):
        list_ij = [[20, 30, 40], [21, 31, 41]]
        f_alias1 = ps.find_shift_between_image_stacks
        sr_shifts = f_alias1(self.ref_stack, self.sam_stack1, 5, 8,
                             list_ij, global_value="median", gpu=False,
                             block=32, sub_pixel=True, method="diff",
                             size=3, ncore=1, norm=False)
        f_alias2 = ps.find_shift_between_sample_images
        list_ij = [32, 32]
        sam_shifts = f_alias2(self.ref_stack, self.sam_stack2, sr_shifts, 41,
                              8, list_ij, global_value="median", gpu=False,
                              block=32, sub_pixel=True, method="diff", size=3,
                              ncore=1, norm=False)
        f_alias3 = ps.align_image_stacks
        ref_stack, sam_stack = f_alias3(self.ref_stack, self.sam_stack2,
                                        sr_shifts, sam_shifts=sam_shifts,
                                        mode="reflect")
        mat1 = sam_stack[0] - ref_stack[0]
        mat2 = sam_stack[1] - ref_stack[1]
        mat3 = sam_stack[2] - ref_stack[2]
        num1 = np.mean(np.abs(mat1[5:-5, 5:-5] - mat2[5:-5, 5:-5]))
        num2 = np.mean(np.abs(mat1[5:-5, 5:-5] - mat3[5:-5, 5:-5]))
        self.assertTrue(num1 < 0.01 and num2 < 0.01)

    def test_retrieve_phase_based_speckle_tracking(self):
        f_alias = ps.retrieve_phase_based_speckle_tracking
        margin = 5
        x_shifts, y_shifts, phase = f_alias(self.ref_stack, self.sam_stack1,
                                            find_shift="correl",
                                            filter_name="hamming",
                                            dark_signal=False,
                                            dim=2, win_size=5, margin=margin,
                                            method="diff", size=3, gpu=False,
                                            block=(16, 16), ncore=1,
                                            norm=False, norm_global=True,
                                            chunk_size=None, surf_method="FC",
                                            correct_negative=True, window=None,
                                            pad=0, pad_mode="linear_ramp",
                                            return_shift=True)
        num1 = np.abs(np.mean(
            np.abs(x_shifts[margin:-margin, margin:-margin])) - self.shift)
        num2 = np.abs(np.mean(
            np.abs(y_shifts[margin:-margin, margin:-margin])) - self.shift)
        num3 = np.std(phase)
        check1 = True if (num1 < 0.1 and num2 < 0.1 and num3 < 1.0) else False
        x_shifts, y_shifts, phase = f_alias(self.ref_stack, self.sam_stack1,
                                            find_shift="correl",
                                            filter_name="hamming",
                                            dark_signal=False,
                                            dim=2, win_size=5, margin=margin,
                                            method="poly_fit", size=3,
                                            gpu=False, block=(16, 16), ncore=1,
                                            norm=False, norm_global=True,
                                            chunk_size=None, surf_method="SCS",
                                            correct_negative=True, window=None,
                                            pad=0, pad_mode="linear_ramp",
                                            return_shift=True)
        num1 = np.abs(np.mean(
            np.abs(x_shifts[margin:-margin, margin:-margin])) - self.shift)
        num2 = np.abs(np.mean(
            np.abs(y_shifts[margin:-margin, margin:-margin])) - self.shift)
        num3 = np.std(phase)
        check2 = True if (num1 < 0.1 and num2 < 0.1 and num3 < 1.0) else False
        x_shifts, y_shifts, phase = f_alias(self.ref_stack, self.sam_stack1,
                                            find_shift="umpa",
                                            filter_name="hamming",
                                            dark_signal=False,
                                            dim=2, win_size=5, margin=margin,
                                            method="diff", size=3,
                                            gpu=False, block=(16, 16), ncore=1,
                                            norm=False, norm_global=True,
                                            chunk_size=None, surf_method="SCS",
                                            correct_negative=True, window=None,
                                            pad=0, pad_mode="linear_ramp",
                                            return_shift=True)
        num1 = np.abs(np.mean(
            np.abs(x_shifts[margin:-margin, margin:-margin])) - self.shift)
        num2 = np.abs(np.mean(
            np.abs(y_shifts[margin:-margin, margin:-margin])) - self.shift)
        num3 = np.std(phase)
        check3 = True if (num1 < 0.1 and num2 < 0.1 and num3 < 1.0) else False
        self.assertTrue(check1 and check2 and check3)

    def test_get_transmission_dark_field_signal(self):
        f_alias1 = ps.retrieve_phase_based_speckle_tracking
        margin = 5
        x_shifts, y_shifts = f_alias1(self.ref_stack, self.sam_stack1,
                                      dim=2, win_size=5, margin=margin,
                                      method="diff", size=3, gpu=False,
                                      block=(16, 16), ncore=1,
                                      norm=False, norm_global=True,
                                      chunk_size=None, surf_method="FC",
                                      correct_negative=True,
                                      window=None, pad=0,
                                      pad_mode="linear_ramp",
                                      return_shift=True)[0:2]
        f_alias2 = ps.get_transmission_dark_field_signal
        trans, dark = f_alias2(self.ref_stack, self.sam_stack1, x_shifts,
                               y_shifts, 5, ncore=1)
        num1 = np.abs(
            np.mean(np.abs(trans[margin:-margin, margin:-margin])) - 1.0)
        num2 = np.std(np.abs(dark[margin:-margin, margin:-margin]))
        self.assertTrue(num1 < 0.1 and num2 < 1.0)
