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
Tests for methods in io/loadersaver.py
"""

import os
import unittest
import shutil
import h5py
import numpy as np
import algotom.io.loadersaver as losa


class LoaderSaverMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir("data"):
            os.makedirs("data")

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir("data"):
            shutil.rmtree("data")

    def test_load_image(self):
        file_path = "data/img.tif"
        losa.save_image(file_path, np.random.rand(64, 64))
        mat = losa.load_image(file_path)
        self.assertTrue(len(mat.shape) == 2)

    def test_get_hdf_information(self):
        file_path = "data/data.hdf"
        ifile = h5py.File(file_path, "w")
        ifile.create_dataset("entry/data", data=np.random.rand(64, 64))
        ifile.create_dataset("entry/energy", data=25.0)
        ifile.close()
        results = losa.get_hdf_information(file_path)
        self.assertTrue(len(results) == 3 and isinstance(results[0][0], str))

    def test_find_hdf_key(self):
        file_path = "data/data.hdf"
        ifile = h5py.File(file_path, "w")
        key = "/energy"
        ifile.create_dataset(key, data=25.0)
        ifile.close()
        results = losa.find_hdf_key(file_path, key)
        self.assertTrue(len(results) == 3 and results[0][0] == key)

    def test_load_hdf(self):
        file_path = "data/data.hdf"
        ifile = h5py.File(file_path, "w")
        ifile.create_dataset("entry/data", data=np.random.rand(64, 64))
        ifile.close()
        data = losa.load_hdf(file_path, "entry/data")[:]
        self.assertTrue(isinstance(data, np.ndarray))

    def test_save_image(self):
        file_path = "data/img.tif"
        losa.save_image(file_path, np.random.rand(64, 64))
        self.assertTrue(os.path.isfile(file_path))

    def test_open_hdf_stream(self):
        data_out = losa.open_hdf_stream("data/data.hdf", (64, 64))
        self.assertTrue(isinstance(data_out, object))

    def test_save_distortion_coefficient(self):
        file_path = "data/coef.txt"
        losa.save_distortion_coefficient(file_path, 32, 32, (1.0, 0.0, 0.0))
        self.assertTrue(os.path.isfile(file_path))

    def test_load_distortion_coefficient(self):
        file_path = "data/coef.txt"
        losa.save_distortion_coefficient(file_path, 31.0, 32.0, [1.0, 0.0])
        (x, y, facts) = losa.load_distortion_coefficient(file_path)
        self.assertTrue(((x == 31.0) and (y == 32.0)) and facts == [1.0, 0.0])
