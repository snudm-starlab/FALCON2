"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

FALCON: Lightweight and Accurate Convolution

File: unittest/unit_test.py
 - test units in FALCON on dummy data.

Version: 1.0
"""
import sys
import unittest
import random

import torch
import numpy as np
from torch import nn


sys.path.append("../")
from models.falcon import GEPdecompose

class TestFalcon(unittest.TestCase):
    """ A class for performing 10 tasks for unittest of FALCON2 """
    def tearDown(self):
        print("Test Case Complete!!!")

    def test1_pw_kernel_size_rank_1(self):
        """ Test 1: Check the kernel size of pointwise convolution in rank 1 """
        self.assertEqual(GEP_rank_1.point_wise.kernel_size, (1,1))

    def test2_pw_io_shape_rank_1(self):
        """ Test 2: Check the input/output shape of pointwise convolution in rank 1"""
        self.assertEqual(GEP_rank_1.point_wise.in_channels, IN_CHANNELS)
        self.assertEqual(GEP_rank_1.point_wise.out_channels, OUT_CHANNELS)

    def test3_dw_kernel_size_rank_1(self):
        """ Test 3: Check the kernel size of depthwise convolution in rank 1"""
        self.assertEqual(GEP_rank_1.depth_wise.kernel_size, kernel_size)

    def test4_dw_io_shape_rank_1(self):
        """ Test 4: Check the input/output shape of depthwise convolution in rank 1"""
        self.assertEqual(GEP_rank_1.depth_wise.in_channels, OUT_CHANNELS)
        self.assertEqual(GEP_rank_1.depth_wise.out_channels, OUT_CHANNELS)

    def test5_forward_rank_1(self):
        """ Test 5: Check the forward pass of rank 1"""
        computed = GEP_rank_1.forward(x)
        ans = GEP_rank_1.depth_wise(GEP_rank_1.point_wise(x))
        mse = torch.sum(torch.pow(computed - ans, 2))
        self.assertEqual(mse, 0)

    def test6_pw_kernel_size_rank_k(self):
        """ Test 6: Check the kernel size of pointwise convolution in rank k"""
        for i in range(GEP_rank_k.rank):
            _k = getattr(GEP_rank_k, f'pw{i}').kernel_size
            self.assertEqual(_k, (1,1))

    def test7_pw_io_shape_rank_k(self):
        """ Test 7: Check the input/output shape of pointwise convolution in rank k"""
        for i in range(GEP_rank_k.rank):
            pw_i = getattr(GEP_rank_k, f'pw{i}')
            self.assertEqual(pw_i.in_channels, IN_CHANNELS)
            self.assertEqual(pw_i.out_channels, OUT_CHANNELS)

    def test8_dw_kernel_size_rank_k(self):
        """ Test 8: Check the kernel size of depthwise convolution in rank k"""
        for i in range(GEP_rank_k.rank):
            _k = getattr(GEP_rank_k, f'dw{i}').kernel_size
            self.assertEqual(_k, kernel_size)

    def test9_dw_io_shape_rank_k(self):
        """ Test 9: Check the input/output shape of depthwise convolution in rank k"""
        for i in range(GEP_rank_k.rank):
            dw_i = getattr(GEP_rank_k, f'dw{i}')
            self.assertEqual(dw_i.in_channels, OUT_CHANNELS)
            self.assertEqual(dw_i.out_channels, OUT_CHANNELS)

    def test10_forward_rank_k(self):
        """ Test 10: Check the forward pass of rank k"""
        for i in range(GEP_rank_k.rank):
            if i == 0:
                out = getattr(GEP_rank_k, f'pw{i}')(x)
                out = getattr(GEP_rank_k, f'dw{i}')(out)
            else:
                out += getattr(GEP_rank_k, f'dw{i}')(getattr(GEP_rank_k, f'pw{i}')(x))
        ans = out
        computed = GEP_rank_k.forward(x)

        mse = torch.sum(torch.pow(computed-ans, 2))
        self.assertEqual(mse, 0)

if __name__ == '__main__':
    # Fix random seed for reproducibility
    RANDOM_SEED = 1997
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Problem Setting
    IN_CHANNELS = 2
    OUT_CHANNELS = 4
    kernel_size = (3,3)

    x = torch.rand((1, 2, 5, 5)) # N, C, H, W
    conv1 = nn.Conv2d(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,kernel_size=kernel_size,
                        bias=False, padding=kernel_size[0]//2)

    # Perform GEP_decompose for testing
    GEP_rank_1 = GEPdecompose(conv1, rank=1, init=False)
    k = 5
    GEP_rank_k = GEPdecompose(conv1, rank=k, init=False)

    unittest.main()
