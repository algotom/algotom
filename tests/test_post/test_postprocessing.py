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
Tests for the methods in post/postprocessing.py

"""

import os
import glob
import shutil
import unittest
import h5py
import numpy as np
from PIL import Image
import algotom.io.loadersaver as losa
import algotom.post.postprocessing as post


class UtilityMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir("data"):
            os.makedirs("data")

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir("data"):
            shutil.rmtree("data")

    def setUp(self):
        size = 129
        mask1 = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        radius1 = center - 5
        y, x = np.ogrid[-center:size - center, -center:size - center]
        mask_check = x * x + y * y <= radius1 * radius1
        mask1[mask_check] = 1.0
        radius2 = center - 6
        mask2 = np.zeros((size, size), dtype=np.float32)
        mask_check = x * x + y * y <= radius2 * radius2
        mask2[mask_check] = 1.0
        self.mask = mask1 - mask2
        self.mat = np.ones_like(mask1) - np.ceil(self.mask)
        (self.dep, self.hei, self.wid) = (30, 50, 60)
        self.mat3d = np.float32(np.random.rand(self.dep, self.hei, self.wid))
        self.tif_folder = "data/tif/"
        for i in range(self.dep):
            name = "0000" + str(i)
            losa.save_image(self.tif_folder + "/img_" + name[-3:] + ".tif",
                            self.mat3d[i, :, :])
        self.hdf_file = "data/data_test.hdf"
        ifile = h5py.File(self.hdf_file, "w")
        self.key_path = "entry/data"
        ifile.create_dataset(self.key_path, data=self.mat3d)
        ifile.close()

    def test_get_statistical_information(self):
        mat = np.random.rand(64, 64)
        results = post.get_statistical_information(mat)
        num = np.abs(0.5 - results[5])
        self.assertTrue(len(results) == 7 and num < 0.1)

    def test_get_statistical_information_dataset(self):
        f_alias = post.get_statistical_information_dataset
        results1 = f_alias(self.mat3d)
        num1 = np.abs(0.5 - results1[5])
        self.assertTrue(len(results1) == 7 and num1 < 0.1)
        results2 = f_alias(self.tif_folder)
        num2 = np.abs(0.5 - results2[5])
        self.assertTrue(num2 < 0.1)
        results3 = f_alias(self.hdf_file, key_path=self.key_path)
        num3 = np.abs(0.5 - results3[5])
        self.assertTrue(num3 < 0.1)

    def test_downsample(self):
        mat = np.random.rand(64, 64)
        mat_dsp = post.downsample(mat, (2, 2))
        self.assertTrue(mat_dsp.shape == (32, 32))

    def test_downsample_dataset(self):
        mat = np.float32(np.random.rand(self.dep, self.hei, self.wid))
        (dep1, hei1, wid1) = (self.dep // 2, self.hei // 2, self.wid // 2)

        mat_dsp = post.downsample_dataset(mat, None, (2, 2, 2))
        self.assertTrue(mat_dsp.shape == (dep1, hei1, wid1))

        post.downsample_dataset(mat, "data/dsp1/", (2, 2, 2))
        files = glob.glob("data/dsp1/*tif*")
        self.assertTrue(len(files) == self.dep // 2)

        post.downsample_dataset(self.tif_folder, "data/dsp2/", (2, 2, 2))
        files = glob.glob("data/dsp2/*tif*")
        self.assertTrue(len(files) == self.dep // 2)

        post.downsample_dataset(self.hdf_file, "data/dsp3/file.hdf", (2, 2, 2),
                                crop=(0, 0, 0, 0, 0, 0), rescaling=True,
                                nbit=16, key_path=self.key_path)
        self.assertTrue(os.path.isfile("data/dsp3/file.hdf"))

        output = post.downsample_dataset(mat, None, (2, 2, 2),
                                         crop=(4, 4, 5, 5, 6, 6),
                                         rescaling=True, nbit=8)
        (dep2, hei2, wid2) = (
            (self.dep - 8) // 2, (self.hei - 10) // 2, (self.wid - 12) // 2)
        self.assertTrue(output.shape == (dep2, hei2, wid2))
        self.assertTrue(str(output.dtype) == "uint8")

    def test_rescale(self):
        mat = np.float32(np.random.rand(64, 64))
        mat_res = post.rescale(mat, nbit=16)
        self.assertTrue(65536 > mat_res[32, 32] >= 0)
        mat_res = post.rescale(mat, nbit=8)
        self.assertTrue(256 > mat_res[32, 32] >= 0)

    def test_rescale_dataset(self):
        mat = np.float32(np.random.rand(16, 64, 64))
        mat_res = post.rescale_dataset(mat, None, nbit=8)
        self.assertTrue(256 > mat_res[0, 32, 32] >= 0)
        post.rescale_dataset(self.tif_folder, "data/rescale/", nbit=8,
                             crop=(2, 2, 3, 3, 4, 4))
        files = glob.glob("data/rescale/*tif*")
        mat_tmp = np.asarray(Image.open(files[0]))
        data_type = str(mat_tmp.dtype)
        data_shape = (len(files), mat_tmp.shape[0], mat_tmp.shape[1])
        crop_shape = (self.dep - 4, self.hei - 6, self.wid - 8)
        self.assertTrue(data_shape == crop_shape and data_type == "uint8")

        output = "data/rescale2/file.hdf"
        post.rescale_dataset(self.hdf_file, output, nbit=16,
                             key_path=self.key_path)
        key_path = losa.get_hdf_information(output)[0][0]
        data = losa.load_hdf(output, key_path)
        data_type = str(data.dtype)
        self.assertTrue(data_type == "uint16")
        self.assertTrue(os.path.isfile(output))
        self.assertTrue(data.shape == (self.dep, self.hei, self.wid))

    def test_reslice_dataset(self):
        post.reslice_dataset(self.tif_folder, "data/reslice/", axis=1,
                             rescaling=True, nbit=8, crop=(2, 2, 3, 3, 4, 4))
        files = glob.glob("data/reslice/*tif*")
        mat_tmp = np.asarray(Image.open(files[0]))
        data_type = str(mat_tmp.dtype)
        data_shape = (len(files), mat_tmp.shape[0], mat_tmp.shape[1])
        crop_shape = (self.hei - 6, self.dep - 4, self.wid - 8)
        self.assertTrue(data_shape == crop_shape and data_type == "uint8")

        post.reslice_dataset(self.tif_folder, "data/reslice2/", axis=2,
                             rescaling=False, crop=(2, 2, 3, 3, 4, 4))
        files = glob.glob("data/reslice2/*tif*")
        mat_tmp = np.asarray(Image.open(files[0]))
        data_shape = (len(files), mat_tmp.shape[0], mat_tmp.shape[1])
        crop_shape = (self.wid - 8, self.dep - 4, self.hei - 6)
        self.assertTrue(data_shape == crop_shape)

        output = "data/reslice3/file.hdf"
        post.reslice_dataset(self.hdf_file, output, axis=1, rescaling=True,
                             nbit=16, key_path=self.key_path,
                             crop=(2, 2, 3, 3, 4, 4))
        key_path = losa.get_hdf_information(output)[0][0]
        data = losa.load_hdf(output, key_path)
        data_type = str(data.dtype)
        self.assertTrue(os.path.isfile(output))
        crop_shape = (self.hei - 6, self.dep - 4, self.wid - 8)
        self.assertTrue(data.shape == crop_shape and data_type == "uint16")

        output = "data/reslice4/file.hdf"
        post.reslice_dataset(self.hdf_file, output, axis=2, rescaling=True,
                             nbit=8, key_path=self.key_path,
                             crop=(2, 2, 3, 3, 4, 4))
        key_path = losa.get_hdf_information(output)[0][0]
        data = losa.load_hdf(output, key_path)
        data_type = str(data.dtype)
        self.assertTrue(os.path.isfile(output))
        crop_shape = (self.wid - 8, self.dep - 4, self.hei - 6)
        self.assertTrue(data.shape == crop_shape and data_type == "uint8")

        output = "data/reslice5/file.hdf"
        post.reslice_dataset(self.hdf_file, output, axis=2,
                             key_path=self.key_path)
        key_path = losa.get_hdf_information(output)[0][0]
        data = losa.load_hdf(output, key_path)
        data_type = str(data.dtype)
        self.assertTrue(os.path.isfile(output))
        exp_shape = (self.wid, self.dep, self.hei)
        self.assertTrue(data.shape == exp_shape and data_type == "float32")

    def test_remove_ring_based_wavelet_fft(self):
        mat_corr = post.remove_ring_based_wavelet_fft(self.mat, 3, 1)
        num = np.sum(mat_corr * self.mask) / np.sum(self.mask)
        self.assertTrue(num > 0.5)

    def test_remove_ring_based_fft(self):
        mat_corr = post.remove_ring_based_fft(self.mat, 10, 4, 1)
        num = np.sum(mat_corr * self.mask) / np.sum(self.mask)
        self.assertTrue(num > 0.5)
