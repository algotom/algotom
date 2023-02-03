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
Tests for methods in io/converter.py
"""

import os
import unittest
import shutil
import h5py
import numpy as np
import algotom.io.converter as con
import algotom.io.loadersaver as losa


class ConverterMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir("data"):
            os.makedirs("data")

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir("data"):
            shutil.rmtree("data")

    def test_convert_tif_to_hdf(self):
        losa.save_image("data/tif/image_01.tif",
                        np.float32(np.random.rand(64, 64)))
        losa.save_image("data/tif/image_02.tif",
                        np.float32(np.random.rand(64, 64)))

        file_path = "data/hdf/data.hdf"
        con.convert_tif_to_hdf("data/tif/", file_path)
        self.assertTrue(os.path.isfile(file_path))

        file_path2 = "data/hdf/data2.hdf"
        con.convert_tif_to_hdf("data/tif/", file_path2, pattern="im")
        self.assertTrue(os.path.isfile(file_path2))

        file_path3 = "data/hdf/data3"
        con.convert_tif_to_hdf("data/tif/", file_path3)
        self.assertTrue(os.path.isfile(file_path3 + ".hdf"))

        file_path5 = "data/hdf/data5.hdf"
        self.assertRaises(ValueError, con.convert_tif_to_hdf, "data/tif/",
                          file_path5, crop=(0, 0, 33, 35))

    def test_extract_tif_from_hdf(self):
        file_path = "data/data.hdf"
        ifile = h5py.File(file_path, "w")
        ifile.create_dataset("entry/data",
                             data=np.float32(np.random.rand(2, 64, 64)))
        ifile.close()
        out_base = "data/extract_tif/"
        con.extract_tif_from_hdf(file_path, out_base, "entry/data")
        list_file = losa.find_file(out_base + "/*.tif")
        self.assertTrue(len(list_file) == 2)

        out_base = "data/extract_tif2/"
        con.extract_tif_from_hdf(file_path, out_base, "entry/data", axis=1)
        list_file = losa.find_file(out_base + "/*.tif")
        self.assertTrue(len(list_file) == 64)

        out_base = "data/extract_tif3/"
        con.extract_tif_from_hdf(file_path, out_base, "entry/data", axis=2)
        list_file = losa.find_file(out_base + "/*.tif")
        self.assertTrue(len(list_file) == 64)

        out_base = "data/extract_tif4/"
        con.extract_tif_from_hdf(file_path, out_base, "entry/data", axis=1,
                                 index=10)
        list_file = losa.find_file(out_base + "/*.tif")
        self.assertTrue(len(list_file) == 1)
