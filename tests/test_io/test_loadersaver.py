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
        losa.save_image(file_path, np.float32(np.random.rand(64, 64)))
        mat = losa.load_image(file_path)
        self.assertTrue(len(mat.shape) == 2)

        file_path = "data/img.png"
        losa.save_image(file_path, np.uint8(255 * np.random.rand(64, 64, 3)))
        mat = losa.load_image(file_path)
        self.assertTrue(len(mat.shape) == 2)
        self.assertRaises(Exception, losa.load_image, "data1/")
        self.assertRaises(ValueError, losa.load_image, "data1\\")

    def test_get_hdf_information(self):
        file_path = "data/data.hdf"
        ifile = h5py.File(file_path, "w")
        ifile.create_dataset("entry/data",
                             data=np.float32(np.random.rand(64, 64)))
        ifile.create_dataset("entry/energy", data=25.0)
        ifile.close()
        results = losa.get_hdf_information(file_path)
        self.assertTrue(len(results) == 3 and isinstance(results[0][0], str))

    def test_find_hdf_key(self):
        file_path = "data/data.hdf"
        ifile = h5py.File(file_path, "w")
        key = "/entry/data/energy"
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
        self.assertRaises(Exception, losa.load_hdf, "data/data2.hdf",
                          "entry/data")
        self.assertRaises(ValueError, losa.load_hdf, "data/data.hdf",
                          "entry/data2")

    def test_make_folder_name(self):
        folder = "data/new_folder/"
        folder_name = losa.make_folder_name(folder, name_prefix="Check",
                                            zero_prefix=5)
        self.assertTrue(folder_name == "Check_00001")
        losa.make_folder(folder + "/" + folder_name + "/")
        folder_name = losa.make_folder_name(folder, name_prefix="Check",
                                            zero_prefix=5)
        self.assertTrue(folder_name == "Check_00002")

    def test_save_image(self):
        file_path = "data/test_overwrite/img.tif"
        losa.save_image(file_path, np.float32(np.random.rand(64, 64)))
        self.assertTrue(os.path.isfile(file_path))

        file_path = "data/test_overwrite/img.tif"
        losa.save_image(file_path, np.float32(np.random.rand(64, 64)),
                        overwrite=False)
        num_file = len(losa.find_file("data/test_overwrite/*tif"))
        self.assertTrue(num_file == 2)

        file_path = "data\\img.tif"
        self.assertRaises(ValueError, losa.save_image, file_path, np.float32(
            np.random.rand(64, 64)))

        file_path = "data/img.tif"
        self.assertRaises(ValueError, losa.save_image, file_path,
                          1j * np.random.rand(64, 64))

    def test_open_hdf_stream(self):
        data_out = losa.open_hdf_stream("data/data.hdf", (64, 64))
        self.assertTrue(isinstance(data_out, object))

        data_out2 = losa.open_hdf_stream("data/data2", (64, 64))
        self.assertTrue(isinstance(data_out2, object))

        data_out3 = losa.open_hdf_stream("data/data3.hdf", (64, 64),
                                         options={"entry/energy": 25.0})
        self.assertTrue(isinstance(data_out3, object))

        self.assertRaises(ValueError, losa.open_hdf_stream, "data/data4.hdf",
                          (64, 64), options={"energy/entry/data": 25.0})

    def test_save_distortion_coefficient(self):
        file_path = "data/coef.txt"
        losa.save_distortion_coefficient(file_path, 32, 32, (1.0, 0.0, 0.0))
        self.assertTrue(os.path.isfile(file_path))
        file_path = "data/coef2"
        losa.save_distortion_coefficient(file_path, 32, 32, (1.0, 0.0, 0.0),
                                         overwrite=False)
        self.assertTrue(os.path.isfile(file_path + ".txt"))

    def test_load_distortion_coefficient(self):
        file_path = "data/coef.txt"
        losa.save_distortion_coefficient(file_path, 31.0, 32.0, [1.0, 0.0])
        (x, y, facts) = losa.load_distortion_coefficient(file_path)
        self.assertTrue(((x == 31.0) and (y == 32.0)) and facts == [1.0, 0.0])
        self.assertRaises(ValueError, losa.load_distortion_coefficient,
                          "data\\coef.text")

    def test_get_hdf_tree(self):
        file_path = "data/data.hdf"
        ifile = h5py.File(file_path, "w")
        ifile.create_dataset("entry/data", data=np.random.rand(64, 64))
        ifile.create_dataset("entry/energy", data=25.0)
        ifile.close()
        entries = losa.get_hdf_tree(file_path, output="data/tree.txt",
                                    display=False)
        self.assertTrue(
            os.path.isfile("data/tree.txt") and ("data" in entries[2]))

    def test_get_reference_sample_stacks_dls(self):
        num_stack = 3
        for i in range(num_stack):
            file_path = "data/speck_tomo_" + str(i) + ".hdf"
            ifile = h5py.File(file_path, "w")
            data = np.concatenate((np.zeros((2, 64, 64)),
                                   np.ones((3, 64, 64)),
                                   2 * np.ones((4, 64, 64))),
                                  axis=0)
            ifile.create_dataset("entry/data/data", data=np.float32(data))
            keys = np.concatenate((2.0 * np.ones(2), np.ones(3), np.zeros(4)))
            ifile.create_dataset("entry/data/image_key", data=np.float32(keys))
            ifile.close()
        list_path = losa.find_file("data/speck_tomo*")
        f_alias = losa.get_reference_sample_stacks_dls
        ref_stack, sam_stack = f_alias(0, list_path, data_key=None,
                                       image_key=None, crop=(0, 0, 0, 0),
                                       flat_field=None, dark_field=None,
                                       num_use=None, fix_zero_div=False)
        check1 = True if (ref_stack.shape[0] == num_stack and
                          sam_stack.shape[0] == num_stack) else False
        check2 = True if (np.mean(ref_stack) == 1.0 and
                          np.mean(sam_stack) == 2.0) else False
        self.assertTrue(check1 and check2)

        flat_field = np.ones((64, 64))
        dark_field = np.zeros_like(flat_field)
        ref_stack, sam_stack = f_alias(0, list_path, data_key=None,
                                       image_key=None, crop=(0, 0, 0, 0),
                                       flat_field=flat_field,
                                       dark_field=dark_field,
                                       num_use=4, fix_zero_div=True)
        check1 = True if (ref_stack.shape[0] == num_stack and
                          sam_stack.shape[0] == num_stack) else False
        check2 = True if (np.mean(ref_stack) == 1.0 and
                          np.mean(sam_stack) == 2.0) else False
        self.assertTrue(check1 and check2)

        self.assertRaises(ValueError, f_alias, 0, list_path, data_key=None,
                          image_key=1.0, crop=(0, 0, 0, 0),
                          flat_field=flat_field,
                          dark_field=dark_field,
                          num_use=1, fix_zero_div=True)

        self.assertRaises(ValueError, f_alias, 0, list_path, data_key=None,
                          image_key=[0.0, 1.0], crop=(0, 0, 0, 0),
                          flat_field=flat_field,
                          dark_field=dark_field,
                          num_use=None, fix_zero_div=False)

        self.assertRaises(ValueError, f_alias, 4, list_path, data_key=None,
                          image_key=None, crop=(0, 0, 0, 0),
                          flat_field=None, dark_field=None,
                          num_use=None, fix_zero_div=False)

    def test_get_reference_sample_stacks(self):
        num_stack = 3
        for i in range(num_stack):
            file_path = "data/speck_" + str(i) + ".hdf"
            ifile = h5py.File(file_path, "w")
            data = np.ones((2, 64, 64))
            ifile.create_dataset("entry/speck", data=np.float32(data))
            ifile.close()

            file_path = "data/tomo_" + str(i) + ".hdf"
            ifile = h5py.File(file_path, "w")
            data = 2 * np.ones((4, 64, 64))
            ifile.create_dataset("entry/tomo", data=np.float32(data))
            ifile.close()
        ref_path = losa.find_file("data/speck*")
        sam_path = losa.find_file("data/tomo*")
        ref_key, sam_key = "entry/speck", "entry/tomo"
        f_alias = losa.get_reference_sample_stacks
        ref_stack, sam_stack = f_alias(0, ref_path, sam_path, ref_key, sam_key,
                                       crop=(0, 0, 0, 0), flat_field=None,
                                       dark_field=None, num_use=None,
                                       fix_zero_div=False)
        check1 = True if (ref_stack.shape[0] == num_stack and
                          sam_stack.shape[0] == num_stack) else False
        check2 = True if (np.mean(ref_stack) == 1.0 and
                          np.mean(sam_stack) == 2.0) else False
        self.assertTrue(check1 and check2)

        flat_field = np.ones((64, 64))
        dark_field = np.zeros_like(flat_field)
        ref_stack, sam_stack = f_alias(0, ref_path, sam_path, ref_key, sam_key,
                                       crop=(0, 0, 0, 0),
                                       flat_field=flat_field,
                                       dark_field=dark_field, num_use=None,
                                       fix_zero_div=True)
        check1 = True if (ref_stack.shape[0] == num_stack and
                          sam_stack.shape[0] == num_stack) else False
        check2 = True if (np.mean(ref_stack) == 1.0 and
                          np.mean(sam_stack) == 2.0) else False
        self.assertTrue(check1 and check2)

        self.assertRaises(ValueError, f_alias, 0, ref_path, sam_path, ref_key,
                          sam_key, crop=(32, 35, 0, 0), flat_field=flat_field,
                          dark_field=dark_field, num_use=None,
                          fix_zero_div=True)

    def test_get_image_stack(self):
        num_stack = 3
        for i in range(num_stack):
            file_path = "data/img_stk_" + str(i) + ".hdf"
            ifile = h5py.File(file_path, "w")
            data = np.ones((3, 64, 64))
            ifile.create_dataset("entry/data", data=np.float32(data))
            ifile.close()
        list_path = losa.find_file("data/img_stk*")
        f_alias = losa.get_image_stack

        img_stack = f_alias(0, list_path, data_key="entry/data", average=False,
                            crop=(0, 0, 0, 0), flat_field=None,
                            dark_field=None, num_use=None, fix_zero_div=False)
        check1 = True if (img_stack.shape[0] == num_stack and
                          np.mean(img_stack) == 1.0) else False

        img_stack = f_alias(4, list_path, data_key="entry/data", average=True,
                            crop=(0, 0, 0, 0), flat_field=None,
                            dark_field=None, num_use=None, fix_zero_div=False)
        check2 = True if (img_stack.shape[0] == num_stack and
                          np.mean(img_stack) == 1.0) else False

        num_img = 4
        for i in range(num_stack):
            folder_path = "data/img_stk_tif_" + str(i) + "/"
            for j in range(num_img):
                file_path = folder_path + "/img_" + str(j) + ".tif"
                losa.save_image(file_path, np.ones((64, 64), dtype=np.float32))
        list_path = losa.find_file("data/img_stk_tif*")

        img_stack = f_alias(0, list_path, data_key=None, average=False,
                            crop=(0, 0, 0, 0), flat_field=None,
                            dark_field=None, num_use=None, fix_zero_div=False)
        check3 = True if (img_stack.shape[0] == num_stack and
                          np.mean(img_stack) == 1.0) else False

        img_stack = f_alias(4, list_path, data_key=None, average=True,
                            crop=(0, 0, 0, 0), flat_field=None,
                            dark_field=None, num_use=None, fix_zero_div=False)
        check4 = True if (img_stack.shape[0] == num_stack and
                          np.mean(img_stack) == 1.0) else False

        flat_field = np.ones((64, 64))
        dark_field = np.zeros_like(flat_field)
        img_stack = f_alias(None, list_path[0], data_key=None, average=False,
                            crop=(0, 0, 0, 0), flat_field=flat_field,
                            dark_field=dark_field, num_use=None,
                            fix_zero_div=True)
        check5 = True if (img_stack.shape[0] == num_img and
                          np.mean(img_stack) == 1.0) else False

        flat_field = np.ones((64, 64))
        dark_field = np.zeros_like(flat_field)
        img_stack = f_alias(None, list_path[0], data_key=None, average=False,
                            crop=(0, 0, 0, 0), flat_field=flat_field,
                            dark_field=dark_field, num_use=None,
                            fix_zero_div=True)
        check6 = True if (img_stack.shape[0] == num_img) else False

        img_stack = f_alias(0, list_path[0], data_key=None, average=False,
                            crop=(0, 0, 0, 0), flat_field=None,
                            dark_field=None, num_use=4,
                            fix_zero_div=False)
        check7 = True if (img_stack.shape[0] == 1) else False

        self.assertTrue((check1 and check2 and check3 and check4) and (
                check5 and check6 and check7))
