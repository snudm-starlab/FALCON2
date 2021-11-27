import torch
import numpy as np
import torch.nn as nn

import unittest
import random
import sys
sys.path.append('../models/')
from falcon import *

# Fix random seed for reproducibility
random_seed = 1997
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# Problem Setting
input_x = 5
input_y = 5
in_channels = 2 
out_channels = 4
kernel_size = (3,3)

x = torch.rand((1, 2, 5, 5)) # N , C, H, W
conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=False, padding=kernel_size[0]//2)

# Perform GEP_decompose for testing 
GEP_rank_1 = GEPdecompose(conv1, rank=1, init=False)
k = 5
GEP_rank_k = GEPdecompose(conv1, rank=k, init=False)

class TestFalcon(unittest.TestCase):
    def test1_pw_kernel_size_rank_1(self):
        self.assertEqual(GEP_rank_1.pw.kernel_size, (1,1))
    
    def test2_pw_IO_shape_rank_1(self):
        self.assertEqual(GEP_rank_1.pw.in_channels, in_channels)
        self.assertEqual(GEP_rank_1.pw.out_channels, out_channels)

    def test3_dw_kernel_size_rank_1(self):
        self.assertEqual(GEP_rank_1.dw.kernel_size, kernel_size)
    
    def test4_dw_IO_shape_rank_1(self):
        self.assertEqual(GEP_rank_1.dw.in_channels, out_channels)
        self.assertEqual(GEP_rank_1.dw.out_channels, out_channels)

    def test5_forward_rank_1(self):
        computed = GEP_rank_1.forward(x)
        ans = GEP_rank_1.dw(GEP_rank_1.pw(x))
        mse = torch.sum(torch.pow(computed - ans, 2))
        self.assertEqual(mse, 0)

    def test6_pw_kernel_size_rank_k(self):
        for i in range(GEP_rank_k.rank):
            _k = getattr(GEP_rank_k, f'pw{i}').kernel_size
            self.assertEqual(_k, (1,1))
    
    def test7_pw_IO_shape_rank_k(self):
        for i in range(GEP_rank_k.rank):
            pw_i = getattr(GEP_rank_k, f'pw{i}')
            self.assertEqual(pw_i.in_channels, in_channels)
            self.assertEqual(pw_i.out_channels, out_channels)
    
    def test8_dw_kernel_size_rank_k(self):
        for i in range(GEP_rank_k.rank):
            _k = getattr(GEP_rank_k, f'dw{i}').kernel_size
            self.assertEqual(_k, kernel_size)
    
    def test9_dw_IO_shape_rank_k(self):
        for i in range(GEP_rank_k.rank):
            dw_i = getattr(GEP_rank_k, f'dw{i}')
            self.assertEqual(dw_i.in_channels, out_channels)
            self.assertEqual(dw_i.out_channels, out_channels)
    
    def test10_forward_rank_k(self):
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
