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
import algotom.prep.phase as ps


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

    def test_phase_unwrap_based_cosine_transform(self):
        phase_unwrapped1 = ps.phase_unwrap_based_cosine_transform(
            self.phase_wrapped)
        (height, width) = phase_unwrapped1.shape
        window = ps._make_cosine_window(height, width)
        phase_unwrapped2 = ps.phase_unwrap_based_cosine_transform(
            self.phase_wrapped, window=window)
        num1 = np.median(np.abs(phase_unwrapped1 - self.phase_image))
        num2 = np.median(np.abs(phase_unwrapped2 - self.phase_image))
        self.assertTrue(num1 < self.eps and num2 < self.eps)

    def test_phase_unwrap_based_fft(self):
        phase_unwrapped1 = ps.phase_unwrap_based_fft(self.phase_wrapped)
        (height, width) = phase_unwrapped1.shape
        win_for = ps._make_window(2 * height, 2 * width, direction="forward")
        win_back = ps._make_window(2 * height, 2 * width, direction="backward")
        phase_unwrapped2 = ps.phase_unwrap_based_fft(self.phase_wrapped,
                                                     win_for=win_for,
                                                     win_back=win_back)
        num1 = np.median(np.abs(phase_unwrapped1 - self.phase_image))
        num2 = np.median(np.abs(phase_unwrapped2 - self.phase_image))
        self.assertTrue(num1 < self.eps and num2 < self.eps)

    def test_phase_unwrap_iterative_fft(self):
        phase_unwrapped1 = ps.phase_unwrap_iterative_fft(self.phase_wrapped,
                                                         iteration=3)
        (height, width) = phase_unwrapped1.shape
        win_for = ps._make_window(2 * height, 2 * width, direction="forward")
        win_back = ps._make_window(2 * height, 2 * width, direction="backward")
        phase_unwrapped2 = ps.phase_unwrap_iterative_fft(self.phase_wrapped,
                                                         iteration=3,
                                                         win_for=win_for,
                                                         win_back=win_back)
        num1 = np.median(np.abs(phase_unwrapped1 - self.phase_image))
        num2 = np.median(np.abs(phase_unwrapped2 - self.phase_image))
        self.assertTrue(num1 < self.eps and num2 < self.eps)
