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
# E-mail: algotomography@gmail.com
# Description: Tests for the Algotom package.
# Contributors:
# ============================================================================

"""
Tests for the methods in rec/reconstruction.py

"""

import unittest
import numpy as np
from numba import cuda
import scipy.ndimage as ndi
import algotom.rec.reconstruction as reco


class ReconstructionMethods(unittest.TestCase):

    def setUp(self):
        self.size = 64
        mat = np.zeros((self.size + 1, self.size + 1), dtype=np.float32)
        mat[20:40, 30:45] = np.float32(1.0)
        self.mat = ndi.gaussian_filter(mat, 2.0)
        self.sino_360 = np.zeros((73, self.size + 1), dtype=np.float32)
        self.angles = np.linspace(0.0, 360.0, len(self.sino_360),
                                  dtype=np.float32)
        for i, angle in enumerate(self.angles):
            self.sino_360[i] = np.sum(ndi.rotate(self.mat, -angle,
                                                 reshape=False), axis=0)
        self.sino_180 = self.sino_360[:37]
        self.center = self.size // 2

    def test_fbp_reconstruction(self):
        mat_rec1 = reco.fbp_reconstruction(self.sino_180, self.center,
                                          apply_log=False, gpu=False)
        num1 = np.max(np.abs(self.mat - mat_rec1))
        mat_rec2 = reco.fbp_reconstruction(self.sino_360, self.center,
                                          angles=np.deg2rad(self.angles),
                                          apply_log=False, gpu=False)
        num2 = np.max(np.abs(self.mat - mat_rec2))
        check = True
        if cuda.is_available() is True:
            mat_rec1 = reco.fbp_reconstruction(self.sino_180, self.center,
                                               apply_log=False, gpu=True)
            num3 = np.max(np.abs(self.mat - mat_rec1))
            mat_rec2 = reco.fbp_reconstruction(self.sino_360, self.center,
                                               angles=np.deg2rad(self.angles),
                                               apply_log=False, gpu=True)
            num4 = np.max(np.abs(self.mat - mat_rec2))
            if num3 > 0.1 or num4 > 0.1:
                check = False
        self.assertTrue(num1 <= 0.1 and num2 <= 0.1 and check)

    def test_dfi_reconstruction(self):
        mat_rec1 = reco.dfi_reconstruction(self.sino_180, self.center,
                                          apply_log=False)
        num1 = np.max(np.abs(self.mat - mat_rec1))
        mat_rec2 = reco.dfi_reconstruction(self.sino_360, self.center,
                                          angles=np.deg2rad(self.angles),
                                          apply_log=False)
        num2 = np.max(np.abs(self.mat - mat_rec2))
        self.assertTrue(num1 <= 0.1 and num2 <= 0.1)
