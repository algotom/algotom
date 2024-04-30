# ============================================================================
# ============================================================================
# Copyright (c) 2024 Nghia T. Vo. All rights reserved.
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
Tests for the methods in rec/vertrec.py

"""
import os
import shutil
import unittest
import warnings
import numba
import numpy as np
from numba import cuda
import scipy.ndimage as ndi
import algotom.io.loadersaver as losa
import algotom.util.utility as util
import algotom.rec.vertrec as vrec


class VerticalReconstructionMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir("./tmp"):
            os.makedirs("./tmp")

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir("./tmp"):
            shutil.rmtree("./tmp")

    def setUp(self):
        warnings.filterwarnings('ignore',
                                category=numba.NumbaPerformanceWarning)
        self.width = 71
        self.height = 50
        mat = util.make_circle_mask(self.width, 0.8)
        rad, i_s, j_s = 10, 30, 20
        mat[i_s:i_s + rad, j_s:j_s + rad] = 0
        rad, i_s, j_s = 5, 20, 30
        mat[i_s:i_s + rad, j_s:j_s + rad] = 0
        rad, i_s, j_s = 5, 20, 50
        mat[i_s:i_s + rad, j_s:j_s + rad] = 0
        mat = ndi.gaussian_filter(mat, 1)
        self.phantom_3d = np.zeros((self.height, self.width, self.width),
                                   dtype=np.float32)
        self.phantom_3d[1:] = mat
        self.num_proj_360 = 91
        self.projs_360 = np.zeros((self.num_proj_360, self.height, self.width),
                                  dtype=np.float32)
        self.angles_360 = np.linspace(0.0, 360.0, self.num_proj_360,
                                      dtype=np.float32)
        for i, angle in enumerate(self.angles_360):
            self.projs_360[i] = np.sum(ndi.rotate(self.phantom_3d, -angle,
                                                  axes=(1, 2), order=1,
                                                  reshape=False), axis=1)
        self.angles_360 = np.deg2rad(self.angles_360)
        self.num_proj = self.num_proj_360 // 2 + 1
        self.projs = self.projs_360[:self.num_proj]
        self.angles = self.angles_360[:self.num_proj]
        self.center = self.width // 2

    def tearDown(self):
        warnings.filterwarnings("default",
                                category=numba.NumbaPerformanceWarning)

    def test_vertical_back_projection(self):
        f_alias = vrec.vertical_back_projection_cpu
        alpha = 0.0
        num_slice = 5
        slice_index = self.width // 2 - 20
        xlist, ylist = vrec._get_points_single_line(slice_index, alpha,
                                                    self.width)
        self.assertTrue(len(xlist) == self.width and len(ylist) == self.width)

        bp_img1 = f_alias(self.projs + 1, self.angles, xlist, ylist,
                          self.center, edge_pad=False)
        bp_img2 = f_alias(self.projs + 1, self.angles, xlist, ylist,
                          self.center, edge_pad=True)
        self.assertTrue(bp_img1.shape == (self.height, self.width)
                        and np.min(bp_img1[self.height // 2]) > 1.0
                        and np.mean(bp_img2) > np.mean(bp_img1))

        start_index = slice_index
        stop_index = start_index + num_slice - 1
        x_mat, y_mat = vrec._get_points_multiple_lines(start_index, stop_index,
                                                       alpha, self.width,
                                                       step_index=1)
        self.assertTrue(len(x_mat) == num_slice and len(y_mat) == num_slice)

        f_alias1 = vrec.vertical_back_projection_cpu_chunk
        bp_img3 = f_alias1(self.projs + 1, self.angles, x_mat, y_mat,
                           self.center, edge_pad=False)
        bp_img4 = f_alias1(self.projs + 1, self.angles, x_mat, y_mat,
                           self.center, edge_pad=True)
        self.assertTrue(bp_img3.shape == (num_slice, self.height, self.width)
                        and np.min(bp_img3[0, self.height // 2]) > 1.0
                        and np.mean(bp_img4) > np.mean(bp_img3))

        if cuda.is_available() is True:
            f_alias2 = vrec.vertical_back_projection_gpu
            bp_img5 = f_alias2(self.projs + 1, self.angles, xlist, ylist,
                               self.center, edge_pad=False)
            bp_img6 = f_alias2(self.projs + 1, self.angles, xlist, ylist,
                               self.center, edge_pad=True)
            self.assertTrue(bp_img5.shape == (self.height, self.width)
                            and np.min(bp_img5[self.height // 2]) > 1.0
                            and np.mean(bp_img6) > np.mean(bp_img5))

            f_alias3 = vrec.vertical_back_projection_gpu_chunk
            bp_img7 = f_alias3(self.projs + 1, self.angles, x_mat, y_mat,
                               self.center, edge_pad=False)
            bp_img8 = f_alias3(self.projs + 1, self.angles, x_mat, y_mat,
                               self.center, edge_pad=True)
            self.assertTrue(
                bp_img7.shape == (num_slice, self.height, self.width)
                and np.min(bp_img7[0, self.height // 2]) > 1.0
                and np.mean(bp_img8) > np.mean(bp_img7))

            eps = 0.05
            self.assertTrue(abs(np.mean(bp_img1) - np.mean(bp_img5)) < eps
                            and abs(np.mean(bp_img2) - np.mean(bp_img6)) < eps
                            and abs(np.mean(bp_img3) - np.mean(bp_img7)) < eps
                            and abs(np.mean(bp_img4) - np.mean(bp_img8)) < eps)

    def test_vertical_reconstruction(self):
        f_alias = vrec.vertical_reconstruction
        alpha = 0.0
        gpu = False
        fbp_eps, bpf_eps = 0.05, 0.3
        slice_index = self.width // 2
        ref_slice = self.phantom_3d[:, slice_index, :]
        ver_slice1 = f_alias(self.projs, slice_index, self.center, alpha=alpha,
                             chunk_size=10, ramp_filter="before",
                             filter_name="hann", apply_log=False, gpu=gpu,
                             ncore=None, prefer="threads", show_progress=False)
        self.assertTrue(np.mean(np.abs(ref_slice - ver_slice1)) < fbp_eps)

        ver_slice2 = f_alias(self.projs, slice_index, self.center, alpha=alpha,
                             chunk_size=10, ramp_filter="after",
                             filter_name="hann", apply_log=False, gpu=gpu,
                             ncore=None, prefer="threads", show_progress=False)
        self.assertTrue(np.mean(np.abs(ref_slice - ver_slice2)) < bpf_eps)

        ver_slice3 = f_alias(self.projs, slice_index, self.center, alpha=alpha,
                             chunk_size=10, ramp_filter="before",
                             filter_name=None, apply_log=False, gpu=gpu,
                             ncore=None, prefer="threads", show_progress=False)
        self.assertTrue(np.mean(np.abs(ver_slice3 - ver_slice1)) > 1.0e-6)

        xshift = 4
        ver_slice4 = f_alias(self.projs, slice_index, self.center - xshift,
                             alpha=alpha, chunk_size=10, ramp_filter="after",
                             filter_name="hann", apply_log=False, gpu=gpu,
                             ncore=None, prefer="threads", show_progress=False,
                             masking=True)
        ver_list4 = np.mean(ver_slice4, axis=0)
        num_zero = len(np.where(ver_list4 == 0.0)[0])
        self.assertTrue(num_zero == (2 * xshift + 1))

        ver_slice5 = f_alias(self.projs, slice_index, self.center - xshift,
                             alpha=alpha, angles=None, crop=(10, 5, 6, 4),
                             chunk_size=10, ramp_filter="after",
                             filter_name="hann", apply_log=False, gpu=gpu,
                             ncore=1, prefer="threads", show_progress=False,
                             masking=True)
        height1, width1 = ver_slice5.shape
        self.assertTrue((height1 + 15) == self.height
                        and (width1 + 10) == self.width)

        ver_slice6 = f_alias(self.projs, slice_index, self.center, alpha=alpha,
                             chunk_size=10, ramp_filter=None,
                             filter_name="hann", apply_log=False, gpu=gpu,
                             ncore=1, prefer="threads", show_progress=False)
        xlist, ylist = vrec._get_points_single_line(slice_index, alpha,
                                                    self.width)
        bp_img6 = vrec.vertical_back_projection_cpu(self.projs, self.angles,
                                                    xlist, ylist, self.center,
                                                    edge_pad=False)
        self.assertTrue(abs(np.mean(ver_slice6) - np.mean(bp_img6) * np.pi / (
                    self.num_proj - 1)) < 1.0e-4)

        ver_slice7 = f_alias(self.projs_360, slice_index, self.center,
                             alpha=alpha, angles=self.angles_360,
                             chunk_size=None, ramp_filter="before",
                             filter_name="hann", apply_log=False, gpu=gpu,
                             ncore=None, prefer="threads", show_progress=False)
        self.assertTrue(np.mean(np.abs(ref_slice - ver_slice7)) < fbp_eps)

        ver_slice8 = f_alias(self.projs_360, slice_index, self.center,
                             alpha=alpha, angles=self.angles_360,
                             chunk_size=-1, ramp_filter="after",
                             filter_name="hann",  apply_log=False, gpu=gpu,
                             ncore=None, prefer="threads", show_progress=False)
        self.assertTrue(np.mean(np.abs(ref_slice - ver_slice8)) < bpf_eps)

        ver_slice9 = f_alias(self.projs_360, slice_index, self.center,
                             alpha=alpha, angles=self.angles,
                             proj_start=0, proj_stop=self.num_proj,
                             chunk_size=10, ramp_filter="before",
                             filter_name="hann", apply_log=False, pad=50,
                             pad_mode="linear_ramp", gpu=gpu, ncore=None,
                             prefer="threads", show_progress=False)
        self.assertTrue(np.mean(np.abs(ref_slice - ver_slice9)) < fbp_eps)

        ver_slice10 = f_alias(self.projs_360, slice_index, self.center,
                              alpha=alpha,
                              angles=self.angles_360[-self.num_proj:],
                              proj_start=(self.num_proj - 1), proj_stop=-1,
                              filter_name="hann", chunk_size=10,
                              ramp_filter="before", apply_log=False, gpu=gpu,
                              show_progress=False)
        self.assertTrue(np.mean(np.abs(ref_slice - ver_slice10)) < fbp_eps)

        flat_field = np.ones((self.height, self.width))
        dark_field = 0.1 * np.ones((self.height, self.width))
        ver_slice11 = f_alias(self.projs, slice_index, self.center,
                              alpha=alpha, flat_field=flat_field,
                              dark_field=dark_field, angles=self.angles,
                              chunk_size=10, ramp_filter="before",
                              apply_log=False, gpu=gpu, show_progress=False)
        self.assertTrue(np.mean(np.abs(ref_slice - ver_slice11)) > 0.05)

        dark_field = np.zeros((self.height, self.width))
        ver_slice12 = f_alias(self.projs, slice_index, self.center,
                              alpha=alpha, flat_field=flat_field,
                              dark_field=dark_field, angles=self.angles,
                              crop=(10, 5, 6, 4), ramp_filter="before",
                              apply_log=False, gpu=gpu, show_progress=False)
        height1, width1 = ver_slice12.shape
        self.assertTrue((height1 + 15) == self.height
                        and (width1 + 10) == self.width)

        alpha = 90.0
        slice_index = 30
        ver_slice13 = f_alias(self.projs, slice_index, self.center,
                              alpha=alpha, flat_field=None, dark_field=None,
                              angles=self.angles, ramp_filter="before",
                              apply_log=False, gpu=gpu, show_progress=False)
        ref_slice = np.fliplr(self.phantom_3d[:, :, slice_index])
        self.assertTrue(np.mean(np.abs(ref_slice - ver_slice13)) < fbp_eps)

        alpha = 40.0
        slice_index = 27
        ver_slice14 = f_alias(self.projs, slice_index, self.center,
                              alpha=alpha, flat_field=None, dark_field=None,
                              angles=self.angles, ramp_filter="before",
                              apply_log=False, gpu=gpu, show_progress=False)
        phantom_rot = ndi.rotate(self.phantom_3d, -alpha, axes=(1, 2), order=1,
                                 reshape=False)
        ref_slice = phantom_rot[:, slice_index, :]
        self.assertTrue(np.mean(np.abs(ref_slice - ver_slice14)) < fbp_eps)

        alpha = 60.0
        slice_index = 24
        ver_slice15 = f_alias(self.projs, slice_index, self.center,
                              alpha=alpha, flat_field=None, dark_field=None,
                              angles=self.angles, ramp_filter="before",
                              apply_log=False, gpu=gpu, show_progress=False)
        phantom_rot = ndi.rotate(self.phantom_3d, -alpha, axes=(1, 2), order=1,
                                 reshape=False)
        ref_slice = phantom_rot[:, slice_index, :]
        self.assertTrue(np.mean(np.abs(ref_slice - ver_slice15)) < fbp_eps)

        alpha = 110.0
        slice_index = 24
        ver_slice16 = f_alias(self.projs, slice_index, self.center,
                              alpha=alpha, flat_field=None, dark_field=None,
                              angles=self.angles, ramp_filter="before",
                              apply_log=False, gpu=gpu, show_progress=False)
        phantom_rot = ndi.rotate(self.phantom_3d, -alpha, axes=(1, 2), order=1,
                                 reshape=False)
        ref_slice = phantom_rot[:, slice_index, :]
        self.assertTrue(np.mean(np.abs(ref_slice - ver_slice16)) < fbp_eps)

        if cuda.is_available() is True:
            gpu = True
            eps = 1.0e-5
            alpha = 0.0
            slice_index = self.width // 2
            gver_slice1 = f_alias(self.projs, slice_index, self.center,
                                  alpha=alpha, chunk_size=10,
                                  ramp_filter="before", filter_name="hann",
                                  apply_log=False, gpu=gpu, ncore=None,
                                  prefer="threads", show_progress=False)
            self.assertTrue(np.mean(np.abs(ver_slice1 - gver_slice1)) < eps)

            gver_slice2 = f_alias(self.projs, slice_index, self.center,
                                  alpha=alpha, chunk_size=10,
                                  ramp_filter="after", filter_name="hann",
                                  apply_log=False, gpu=gpu, ncore=None,
                                  prefer="threads", show_progress=False)
            self.assertTrue(np.mean(np.abs(gver_slice2 - ver_slice2)) < eps)

            gver_slice6 = f_alias(self.projs, slice_index, self.center,
                                  alpha=alpha, chunk_size=10, ramp_filter=None,
                                  filter_name="hann", apply_log=False, gpu=gpu,
                                  show_progress=False)
            self.assertTrue(np.mean(np.abs(gver_slice6 - ver_slice6)) < eps)

    def test_vertical_reconstruction_multiple(self):
        f_alias = vrec.vertical_reconstruction_multiple
        alpha = 0.0
        gpu = False
        fbp_eps, bpf_eps = 0.05, 0.35
        eps = 1.0e-4
        start_index = self.width // 2 - 10
        stop_index = start_index + 5
        ref_slices = np.copy(np.moveaxis(
            self.phantom_3d[:, start_index:stop_index + 1, :], 1, 0))
        ver_slices1 = f_alias(self.projs, start_index, stop_index, self.center,
                              alpha=alpha, chunk_size=10, ramp_filter="before",
                              filter_name="hann", apply_log=False, gpu=gpu,
                              ncore=None, prefer="threads",
                              show_progress=False)
        self.assertTrue(ref_slices.shape == ver_slices1.shape and
                        np.mean(np.abs(ref_slices - ver_slices1)) < fbp_eps)

        ver_slices2 = f_alias(self.projs, start_index, stop_index, self.center,
                              alpha=alpha, chunk_size=10, ramp_filter="after",
                              filter_name="hann", apply_log=False, gpu=gpu,
                              show_progress=False)
        self.assertTrue(ref_slices.shape == ver_slices2.shape and
                        np.mean(np.abs(ref_slices - ver_slices2)) < bpf_eps)

        ver_slices3 = f_alias(self.projs, start_index, stop_index, self.center,
                              alpha=alpha, chunk_size=10, ramp_filter=None,
                              filter_name="hann", apply_log=False, gpu=gpu,
                              ncore=1, show_progress=False)

        x_mat, y_mat = vrec._get_points_multiple_lines(start_index, stop_index,
                                                       alpha, self.width, 1)
        bp_imgs3 = vrec.vertical_back_projection_cpu_chunk(self.projs,
                                                           self.angles,
                                                           x_mat, y_mat,
                                                           self.center,
                                                           edge_pad=False)
        bp_imgs3 = bp_imgs3 * np.pi / (self.num_proj - 1)
        self.assertTrue(ref_slices.shape == ver_slices3.shape and
                        np.mean(np.abs(ver_slices3 - bp_imgs3)) < eps)

        start_index = self.width // 2 - 10
        stop_index = start_index + 12
        step_index = 4
        ref_slices = np.copy(
            np.moveaxis(self.phantom_3d[:,
                        start_index:stop_index + 1: step_index, :], 1, 0))
        ver_slices4 = f_alias(self.projs, start_index, stop_index, self.center,
                              alpha=alpha, step_index=step_index,
                              chunk_size=10, ramp_filter="before",
                              filter_name="hann", apply_log=False, gpu=gpu,
                              show_progress=False)
        self.assertTrue(ref_slices.shape == ver_slices4.shape and
                        np.mean(np.abs(ref_slices - ver_slices4)) < fbp_eps)

        alpha = 40
        ver_slices5 = f_alias(self.projs, start_index, stop_index, self.center,
                              alpha=alpha, step_index=step_index,
                              chunk_size=10, ramp_filter="before",
                              filter_name="hann", apply_log=False, gpu=gpu,
                              show_progress=False)
        phantom_rot = ndi.rotate(self.phantom_3d, -alpha, axes=(1, 2), order=1,
                                 reshape=False)
        ref_slices = np.copy(
            np.moveaxis(phantom_rot[:,
                        start_index:stop_index + 1: step_index, :], 1, 0))
        self.assertTrue(ref_slices.shape == ver_slices5.shape and
                        np.mean(np.abs(ref_slices - ver_slices5)) < fbp_eps)

        if cuda.is_available() is True:
            gpu = True
            alpha = 0.0
            start_index = self.width // 2 - 10
            stop_index = start_index + 5
            gver_slices1 = f_alias(self.projs, start_index, stop_index,
                                   self.center, alpha=alpha, chunk_size=10,
                                   ramp_filter="before", filter_name="hann",
                                   apply_log=False, gpu=gpu,
                                   show_progress=False)
            self.assertTrue(gver_slices1.shape == ver_slices1.shape and
                            np.mean(np.abs(gver_slices1 - ver_slices1)) < eps)

            gver_slices2 = f_alias(self.projs, start_index, stop_index,
                                   self.center, alpha=alpha, chunk_size=10,
                                   ramp_filter="after", filter_name="hann",
                                   apply_log=False, gpu=gpu,
                                   show_progress=False)
            self.assertTrue(gver_slices2.shape == ver_slices2.shape and
                            np.mean(np.abs(gver_slices2 - ver_slices2)) < eps)

            gver_slices3 = f_alias(self.projs, start_index, stop_index,
                                   self.center, alpha=alpha, chunk_size=10,
                                   ramp_filter=None, filter_name="hann",
                                   apply_log=False, gpu=gpu,
                                   show_progress=False)
            self.assertTrue(gver_slices3.shape == ver_slices3.shape and
                            np.mean(np.abs(gver_slices3 - ver_slices3)) < eps)

            start_index = self.width // 2 - 10
            stop_index = start_index + 12
            step_index = 4
            alpha = 40
            eps2 = 1.0e-3
            gver_slices5 = f_alias(self.projs, start_index, stop_index,
                                   self.center, alpha=alpha,
                                   step_index=step_index,
                                   chunk_size=10, ramp_filter="before",
                                   filter_name="hann", apply_log=False,
                                   gpu=gpu, show_progress=False)
            self.assertTrue(gver_slices5.shape == ver_slices5.shape and
                            np.mean(np.abs(gver_slices5 - ver_slices5)) < eps2)

    def test_vertical_reconstruction_different_angles(self):
        f_alias = vrec.vertical_reconstruction_different_angles
        gpu = False
        fbp_eps, bpf_eps = 0.05, 0.35
        eps = 1.0e-4
        indices = [self.width // 2 - 10, self.width // 2, self.width // 2 + 15]
        alphas = [0, 45, 90]
        ref_slices = []
        for i, alpha in enumerate(alphas):
            phantom_rot = ndi.rotate(self.phantom_3d, -alpha, axes=(1, 2),
                                     order=1, reshape=False)
            ref_slice = phantom_rot[:, indices[i], :]
            ref_slices.append(ref_slice)
        ref_slices = np.asarray(ref_slices)
        ver_slices1 = f_alias(self.projs, indices, alphas, self.center,
                              chunk_size=10, ramp_filter="before",
                              filter_name="hann", apply_log=False, gpu=gpu,
                              ncore=None, prefer="threads",
                              show_progress=False)
        num = np.mean(np.abs(ref_slices - ver_slices1))
        self.assertTrue(ver_slices1.shape[0] == len(indices)
                        and num <fbp_eps)

        f_alias2 = vrec.vertical_reconstruction
        idx = 0
        alpha = alphas[idx]
        slice_index = indices[idx]
        slice_1 = f_alias2(self.projs, slice_index, self.center,
                               alpha=alpha, chunk_size=10,
                               ramp_filter="before", filter_name="hann",
                               apply_log=False, gpu=gpu, ncore=None,
                               prefer="threads", show_progress=False)
        self.assertTrue(np.mean(np.abs(slice_1 - ver_slices1[idx])) < eps)

        idx = 1
        alpha = alphas[idx]
        slice_index = indices[idx]
        slice_2 = f_alias2(self.projs, slice_index, self.center,
                               alpha=alpha, chunk_size=10,
                               ramp_filter="before", filter_name="hann",
                               apply_log=False, gpu=gpu, ncore=None,
                               prefer="threads", show_progress=False)
        self.assertTrue(np.mean(np.abs(slice_2 - ver_slices1[idx])) < eps)

        idx = -1
        alpha = alphas[idx]
        slice_index = indices[idx]
        slice_3 = f_alias2(self.projs, slice_index, self.center,
                               alpha=alpha, chunk_size=10,
                               ramp_filter="before", filter_name="hann",
                               apply_log=False, gpu=gpu, ncore=None,
                               prefer="threads", show_progress=False)
        self.assertTrue(np.mean(np.abs(slice_3 - ver_slices1[idx])) < eps)

        if cuda.is_available() is True:
            gpu = True
            gver_slices1 = f_alias(self.projs, indices, alphas, self.center,
                                   chunk_size=10, ramp_filter="before",
                                   filter_name="hann", apply_log=False,
                                   gpu=gpu, ncore=None, prefer="threads",
                                   show_progress=False)
            num = np.mean(np.abs(ver_slices1 - gver_slices1))
            self.assertTrue(ver_slices1.shape == gver_slices1.shape
                            and num < eps)


    def test_find_center_vertical_slice(self):
        gpu = False
        f_alias = vrec.find_center_vertical_slice
        slice_index = self.width // 2
        start = self.center-5
        stop = self.center + 9
        cal_center = f_alias(self.projs + 1, slice_index, start, stop,
                             step=1.0, metric="autocorrelation", alpha=0.0,
                             angles=None, chunk_size=30, ramp_filter="after",
                             apply_log=True, gpu=gpu, show_progress=False,
                             invert_metric=True, masking=False,
                             return_metric=False)
        self.assertTrue(abs(cal_center - self.center) < 0.5)

        cal_center = f_alias(self.projs + 1, slice_index, start, stop,
                             step=1.0, metric="entropy", alpha=0.0,
                             angles=None, chunk_size=30, ramp_filter="after",
                             apply_log=True, gpu=gpu, show_progress=False,
                             invert_metric=True, masking=False,
                             return_metric=False)
        self.assertTrue(abs(cal_center - self.center) < 0.5)

        cal_center = f_alias(self.projs + 1, slice_index, start, stop,
                             step=1.0, metric="sharpness", alpha=0.0,
                             angles=None, chunk_size=30, ramp_filter="after",
                             apply_log=True, gpu=gpu, show_progress=False,
                             invert_metric=True, masking=False,
                             return_metric=False)
        self.assertTrue(abs(cal_center - self.center) < 0.5)

        def get_negative(mat, n=2):
            metric = np.abs(np.mean(mat[mat < 0.0])) ** n
            return metric

        cal_center = f_alias(self.projs + 1, slice_index, start, stop,
                             step=1.0, alpha=0.0, angles=None, chunk_size=30,
                             ramp_filter="after", apply_log=True, gpu=gpu,
                             block=(16, 16), ncore=None, prefer="threads",
                             show_progress=False, masking=True,
                             return_metric=False, invert_metric=True,
                             metric_function=get_negative, n=2)
        self.assertTrue(abs(cal_center - self.center) < 0.5)

        if cuda.is_available() is True:
            gpu = True
            cal_center = f_alias(self.projs + 1, slice_index, start, stop,
                                 step=1.0, metric="autocorrelation", alpha=0.0,
                                 angles=None, chunk_size=30,
                                 ramp_filter="after", apply_log=True, gpu=gpu,
                                 show_progress=False, invert_metric=True,
                                 masking=False, return_metric=False)
            self.assertTrue(abs(cal_center - self.center) < 0.5)

            cal_center = f_alias(self.projs + 1, slice_index, start, stop,
                                 step=1.0, metric="entropy", alpha=0.0,
                                 angles=None, chunk_size=30,
                                 ramp_filter="after", apply_log=True, gpu=gpu,
                                 show_progress=False, invert_metric=True,
                                 masking=False, return_metric=False)
            self.assertTrue(abs(cal_center - self.center) < 0.5)

            cal_center = f_alias(self.projs + 1, slice_index, start, stop,
                                 step=1.0, metric="sharpness", alpha=0.0,
                                 angles=None, chunk_size=30,
                                 ramp_filter="after", apply_log=True, gpu=gpu,
                                 show_progress=False, invert_metric=True,
                                 masking=False, return_metric=False)
            self.assertTrue(abs(cal_center - self.center) < 0.5)

            cal_center = f_alias(self.projs + 1, slice_index, start, stop,
                                 step=1.0, alpha=0.0, angles=None,
                                 chunk_size=30, ramp_filter="after",
                                 apply_log=True, gpu=gpu, block=(16, 16),
                                 ncore=None, prefer="threads",
                                 show_progress=False, masking=True,
                                 return_metric=False, invert_metric=True,
                                 metric_function=get_negative, n=2)
            self.assertTrue(abs(cal_center - self.center) < 0.5)

    def test_find_center_visual_vertical_slices(self):
        output = "./tmp"
        slice_index = self.width // 2
        start = self.center - 5
        stop = self.center + 5
        f_alias = vrec.find_center_visual_vertical_slices
        output_folder = f_alias(self.projs, output, slice_index, start, stop,
                                step=1, alpha=0.0, angles=None,
                                chunk_size=30, ramp_filter="after",
                                apply_log=False, gpu=False, display=False)
        files = losa.find_file(output_folder + "/*.tif*")
        img = losa.load_image(files[0])
        num_file = stop - start + 1
        self.assertTrue(img.shape == (self.height, self.width)
                        and len(files) == num_file)

        if cuda.is_available() is True:
            output2 = "./tmp/gpu/"
            output_folder = f_alias(self.projs, output2, slice_index, start,
                                    stop, step=1, alpha=0.0, angles=None,
                                    chunk_size=30, ramp_filter="after",
                                    apply_log=False, gpu=True, display=False)
            files = losa.find_file(output_folder + "/*.tif*")
            img = losa.load_image(files[0])
            num_file = stop - start + 1
            self.assertTrue(img.shape == (self.height, self.width)
                            and len(files) == num_file)
