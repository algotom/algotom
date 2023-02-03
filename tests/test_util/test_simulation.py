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
Tests for methods in util/simulation.py
"""

import unittest
import numpy as np
import algotom.util.simulation as sim
import algotom.rec.reconstruction as reco


class SimulationMethods(unittest.TestCase):

    def setUp(self):
        self.error = 0.15
        self.eps = 10 ** (-6)
        self.size = 64
        self.phantom = sim.make_face_phantom(self.size)
        angles = np.linspace(0.0, 180.0, self.size + 1) * np.pi / 180.0
        self.sinogram = sim.make_sinogram(self.phantom, angles)

    def test_make_triangular_mask(self):
        x_len = self.size // 4
        y_len = self.size // 8
        mask = sim.make_triangular_mask(self.size, 0.0, (x_len, y_len), 0.0)
        list1 = np.sum(mask, axis=0)
        list_pos = np.where(list1 > 0.0)[0]
        self.assertTrue(list1[list_pos[0]] < list1[list_pos[-1]])

    def test_make_line_target(self):
        mask = 1.0 - sim.make_line_target(self.size)
        list1 = mask[self.size // 2]
        self.assertTrue(np.sum(list1) > 3)

        size2 = 2 * self.size
        mask = 1.0 - sim.make_line_target(size2)
        list2 = mask[size2 // 2]
        self.assertTrue(np.sum(list2) > size2 / 4)

    def test_make_face_phantom(self):
        mat = sim.make_face_phantom(self.size)
        list1 = np.mean(mat, axis=1)
        ratio = 1.0 - 1.0 * len(np.where(list1 == 0.0)) / self.size
        self.assertTrue(0.96 < ratio < 0.985)

    def test_make_sinogram(self):
        rec_image = reco.dfi_reconstruction(self.sinogram, self.size // 2,
                                            apply_log=False)
        mat_com = np.abs(self.phantom - rec_image)
        num1 = np.mean(mat_com[mat_com > 0.0])
        self.assertTrue(num1 < self.error)

    def test_add_noise(self):
        mat = np.ones((self.size, self.size))
        mat_noise = sim.add_noise(mat, noise_ratio=0.1)
        num1 = (0.1 - np.mean(np.abs(mat - mat_noise))) / 0.1
        self.assertTrue(num1 < 0.05)

    def test_add_stripe_artifact(self):
        size = 3
        sinogram1 = sim.add_stripe_artifact(self.sinogram, size,
                                            self.size // 4, strength_ratio=0.5,
                                            stripe_type="full")
        mat1 = np.abs(sinogram1 - self.sinogram)
        mat1[mat1 > self.eps] = 1.0
        list1 = np.mean(mat1, axis=0)
        num1 = np.sum(list1)
        self.assertTrue(int(num1) == size)

        sinogram1 = sim.add_stripe_artifact(self.sinogram, size,
                                            self.size // 4, strength_ratio=0.5,
                                            stripe_type="partial")
        mat1 = np.abs(sinogram1 - self.sinogram)
        mat1[mat1 > self.eps] = 1.0
        list1 = np.mean(mat1, axis=0)
        num1 = np.sum(list1)
        self.assertTrue(size > num1 > 0)

        sinogram1 = sim.add_stripe_artifact(self.sinogram, size,
                                            self.size // 4, strength_ratio=0.5,
                                            stripe_type="dead")
        mat1 = np.abs(sinogram1 - self.sinogram)
        mat1[mat1 > self.eps] = 1.0
        list1 = np.mean(mat1, axis=0)
        num1 = np.sum(list1)
        self.assertTrue(int(num1) == size)

        sinogram1 = sim.add_stripe_artifact(self.sinogram, size,
                                            self.size // 4, strength_ratio=0.5,
                                            stripe_type="fluctuating")
        mat1 = np.abs(sinogram1 - self.sinogram)
        mat1[mat1 > self.eps] = 1.0
        list1 = np.mean(mat1, axis=0)
        num1 = np.sum(list1)
        self.assertTrue(int(num1) == size)

    def test_convert_to_Xray_image(self):
        sinogram1 = sim.convert_to_Xray_image(self.sinogram)
        num1 = np.abs((np.max(sinogram1) - 1.0))
        self.assertTrue(num1 < 0.1)

    def test_add_background_fluctuation(self):
        sinogram1 = sim.convert_to_Xray_image(self.sinogram)
        sinogram2 = sim.add_background_fluctuation(sinogram1,
                                                   strength_ratio=0.2)
        mat1 = np.abs(sinogram1 - sinogram2)
        num1 = np.mean(mat1[mat1 > 0.0])
        self.assertTrue(num1 > 0.1)
