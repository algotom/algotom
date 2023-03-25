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
Tests for the methods in rec/reconstruction.py

"""

import unittest
import warnings
import numba
import numpy as np
from numba import cuda
import scipy.ndimage as ndi
import algotom.rec.reconstruction as reco

warnings.filterwarnings('ignore', category=numba.NumbaPerformanceWarning)


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
        self.angles2 = self.angles[:37]
        self.center = self.size // 2

    def test_make_smoothing_window(self):
        win1 = reco.make_smoothing_window(None, self.size)
        win2 = reco.make_smoothing_window("hann", self.size)
        self.assertTrue(np.mean(np.abs(win1 - win2)) > 0.0)
        win2 = reco.make_smoothing_window("bartlett", self.size)
        self.assertTrue(np.mean(np.abs(win1 - win2)) > 0.0)
        win2 = reco.make_smoothing_window("blackman", self.size)
        self.assertTrue(np.mean(np.abs(win1 - win2)) > 0.0)
        win2 = reco.make_smoothing_window("hamming", self.size)
        self.assertTrue(np.mean(np.abs(win1 - win2)) > 0.0)
        win2 = reco.make_smoothing_window("nuttall", self.size)
        self.assertTrue(np.mean(np.abs(win1 - win2)) > 0.0)
        win2 = reco.make_smoothing_window("parzen", self.size)
        self.assertTrue(np.mean(np.abs(win1 - win2)) > 0.0)
        win2 = reco.make_smoothing_window("triang", self.size)
        self.assertTrue(np.mean(np.abs(win1 - win2)) > 0.0)

    def test_fbp_reconstruction(self):
        f_alias = reco.fbp_reconstruction
        mat_rec1 = f_alias(self.sino_180, self.center, apply_log=False,
                           gpu=True, ratio=0.0)
        num1 = np.max(np.abs(self.mat - mat_rec1))
        mat_rec2 = f_alias(self.sino_360, self.center,
                           angles=np.deg2rad(self.angles), apply_log=False,
                           gpu=False, ratio=None)
        num2 = np.max(np.abs(self.mat - mat_rec2))
        check = True
        if cuda.is_available() is True:
            mat_rec1 = f_alias(self.sino_180, self.center, apply_log=False,
                               gpu=True)
            num3 = np.max(np.abs(self.mat - mat_rec1))
            mat_rec2 = f_alias(self.sino_360, self.center,
                               angles=np.deg2rad(self.angles), apply_log=False,
                               gpu=True)
            num4 = np.max(np.abs(self.mat - mat_rec2))
            if num3 > 0.1 or num4 > 0.1:
                check = False
        self.assertTrue(num1 <= 0.2 and num2 <= 0.2 and check)

        gpu = False
        sino_stack = np.pad(np.expand_dims(self.sino_180, 1), ((0, 0), (2, 2),
                            (0, 0)), mode="edge")
        rec_stack = f_alias(sino_stack, self.center, apply_log=False,
                            gpu=gpu, angles=np.deg2rad(self.angles2),
                            ratio=0.0, ncore=1)
        num = np.max(np.abs(self.mat - rec_stack[:, 0, :]))
        n_slice = rec_stack.shape[1]
        self.assertTrue(num <= 0.2 and n_slice == 5)

        rec_stack = f_alias(sino_stack, self.center, apply_log=False,
                            gpu=gpu, ratio=0.0, ncore=None)
        num = np.max(np.abs(self.mat - rec_stack[:, 0, :]))
        self.assertTrue(num <= 0.2)

        if cuda.is_available() is True:
            gpu = True
            sino_stack = np.pad(np.expand_dims(self.sino_180, 1),
                                ((0, 0), (2, 2), (0, 0)), mode="edge")
            rec_stack = f_alias(sino_stack, self.center, apply_log=False,
                                gpu=gpu, ratio=0.0, ncore=1)
            num = np.max(np.abs(self.mat - rec_stack[:, 0, :]))
            n_slice = rec_stack.shape[1]
            self.assertTrue(num <= 0.1 and n_slice == 5)

            rec_stack = f_alias(sino_stack, self.center, apply_log=False,
                                gpu=gpu, ratio=1.0, ncore=None)
            num = np.max(np.abs(self.mat - rec_stack[:, 0, :]))
            self.assertTrue(num <= 0.1)

        self.assertWarns(Warning, f_alias, self.sino_180, self.center,
                         apply_log=True)

        self.assertRaises(ValueError, f_alias, self.sino_180, self.center,
                          np.deg2rad(self.angles), apply_log=False)

        self.assertRaises(ValueError, f_alias, self.sino_180, -1,
                          apply_log=False)

    def test_dfi_reconstruction(self):
        f_alias = reco.dfi_reconstruction
        mat_rec1 = f_alias(self.sino_180, self.center, ratio=0.0,
                           apply_log=False)
        num1 = np.max(np.abs(self.mat - mat_rec1))
        mat_rec2 = f_alias(self.sino_360, self.center,
                           angles=np.deg2rad(self.angles),
                           apply_log=False, ratio=None)
        num2 = np.max(np.abs(self.mat - mat_rec2))
        self.assertTrue(num1 <= 0.1 and num2 <= 0.1)

        sino_stack = np.pad(np.expand_dims(
            self.sino_180, 1), ((0, 0), (2, 2), (0, 0)), mode="edge")
        rec_stack = f_alias(sino_stack, self.center, apply_log=False,
                            ratio=0.0, angles=np.deg2rad(self.angles2),
                            ncore=1)
        num = np.max(np.abs(self.mat - rec_stack[:, 0, :]))
        n_slice = rec_stack.shape[1]
        self.assertTrue(num <= 0.1 and n_slice == 5)

        rec_stack = f_alias(sino_stack, self.center, apply_log=False,
                            ratio=0.0, ncore=None)
        num = np.max(np.abs(self.mat - rec_stack[:, 0, :]))
        self.assertTrue(num <= 0.1)

        self.assertRaises(ValueError, f_alias, self.sino_180, self.center,
                          np.deg2rad(self.angles), apply_log=False)

        self.assertWarns(Warning, f_alias, self.sino_180, self.center,
                         apply_log=True)

        self.assertRaises(ValueError, f_alias, self.sino_180, -1,
                          apply_log=False)

    def get_negative(self, mat, n=2):
        metric = np.abs(np.mean(mat[mat < 0.0])) ** n
        return metric

    def test_find_center_based_slice_metric(self):
        f_alias = reco.find_center_based_slice_metric
        center_cal = f_alias(self.sino_180, self.center - 2, self.center + 2,
                             step=0.25, radius=2, zoom=0.8, method="dfi",
                             gpu=False, angles=self.angles2, ratio=1.0,
                             filter_name="hann", apply_log=False, ncore=1,
                             sigma=1, metric_function=None)
        self.assertTrue(np.abs(center_cal - self.center) < 0.1)

        if cuda.is_available() is True:
            center_cal = f_alias(self.sino_180, self.center - 2,
                                 self.center + 2, step=0.25, radius=2,
                                 zoom=1.0, method="fbp", gpu=True,
                                 angles=None, ratio=1.0, filter_name="hann",
                                 apply_log=False, ncore=None,
                                 sigma=1, metric_function=None)
            self.assertTrue(np.abs(center_cal - self.center) < 0.1)

        center_cal = f_alias(self.sino_180, self.center - 2, self.center + 2,
                             step=0.25, radius=2, zoom=1.0, method="dfi",
                             gpu=False, angles=None, ratio=1.0,
                             filter_name="hann", apply_log=False, ncore=None,
                             sigma=1, metric_function=self.get_negative, n=2)
        self.assertTrue(np.abs(center_cal - self.center) < 0.1)
