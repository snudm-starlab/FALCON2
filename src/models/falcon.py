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

File: models/falcon.py
- Contain source code for decomposing standard convolution kernel to FALCON kernels.

Version: 1.0
"""

# pylint: disable=C0103,W0223,R0902,R0913,R1725,R0201
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


class GEPdecompose(nn.Module):
    """
    GEP decomposition class
    """
    def __init__(self, conv_layer, rank=1, init=True, alpha=1.0, bn=False, relu=False, groups=1):
        """
        Initialize FALCON layer.
        
        :param conv_layer: standard convolution layer
        :param rank: rank of GEP
        :param init: whether initialize FALCON with decomposed tensors
        :param relu: whether use relu function
        :param groups: number of groups in 1*1 conv
        """

        super(GEPdecompose, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.bn_ = bn
        self.relu_ = relu
        self.device = conv_layer.weight.device

        # get weight and bias
        weight = conv_layer.weight.data
        bias = conv_layer.bias
        if bias is not None:
            bias = bias.data

        out_channels, in_channels, _, _ = weight.shape
        self.out_channels = int(out_channels * self.alpha)
        self.in_channels = int(in_channels * self.alpha)

        if self.rank == 1:
            self.pw = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False).to(self.device)
            self.dw = nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.kernel_size[0]//2,
                bias=False,
                groups=self.out_channels).to(self.device)
            self.bn = nn.BatchNorm2d(self.out_channels)
            if init:
                self.decompose(conv_layer.weight, self.pw.weight, self.dw.weight)

        else:
            for i in range(self.rank):
                setattr(self, 'pw'+str(i),
                        nn.Conv2d(in_channels=self.in_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias=False).to(self.device))
                setattr(self, 'dw'+str(i),
                        nn.Conv2d(in_channels=self.out_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=conv_layer.kernel_size,
                                  stride=conv_layer.stride,
                                  padding=conv_layer.kernel_size[0] // 2,
                                  bias=False,
                                  groups=self.out_channels).to(self.device))
            self.bn = nn.BatchNorm2d(self.out_channels)
            if init:
                if alpha == 1.0:
                    self.decompose_rank(conv_layer.weight)
                else:
                    self.width_mul(conv_layer.weight)
                    self.decompose_rank(conv_layer.weight)

        self.stride = conv_layer.stride

        if groups != 1:
            self.group1x1(groups)

    def forward(self, x):
        """
        Run forward propagation
        
        :param x: input feature maps
        :return: out: output tensor of forward propagation
        """
        if self.rank == 1:
            out = self.dw(self.pw(x))
        else:
#            names = locals()
            for i in range(self.rank):
                if i == 0:
                    out = getattr(self, 'pw' + str(i))(x)
                    out = getattr(self, 'dw' + str(i))(out)
                else:
                    out += getattr(self, 'dw' + str(i))(getattr(self, 'pw' + str(i))(x))
        if self.bn_:
            out = self.bn(out)
        if self.relu_:
            out = F.relu(out, True)
        return out

    def decompose(self, conv, pw, dw, lr=0.001, steps=600):
        """
        GEP decomposes standard convolution kernel
        
        :param conv: standard convolution kernel
        :param pw: decomposed pointwise convolution kernel
        :param dw: decomposed depthwise convolution kernel
        :param lr: learning rate
        :param steps: training steps for decomposing
        """

        conv.requires_grad = False
        pw.requires_grad = True
        dw.requires_grad = True

        criterion = nn.MSELoss()
        optimizer = AdamW({pw, dw}, lr=lr)
        st = time.time()
        for s in range(steps):
            if steps in (400, 700):
                lr = lr / 10
                optimizer = AdamW({pw, dw}, lr=lr)
            optimizer.zero_grad()
            kernel_pred = pw.cuda() * dw.cuda()
            loss = criterion(kernel_pred, conv.cuda())
            loss.backward()
            optimizer.step()
            if s % 100 == 99:
                print('loss = %f, time = %d' % (loss, (time.time() - st)))
                st = time.time()

    def decompose_rank(self, kernel, lr=5e-3, steps=600):
        """
        GEP decompose standard convolution kernel with different rank
        
        :param conv: standard convolution kernel
        :param lr: learning rate
        :param steps: training steps for decomposing
        """

        kernel.requires_grad = False
        param = {self.dw0.weight, self.pw0.weight}
        for i in range(self.rank):
            getattr(self, 'pw' + str(i)).weight.requires_grad = True
            getattr(self, 'dw' + str(i)).weight.requires_grad = True
            if i != 0:
                param.add(getattr(self, 'pw' + str(i)).weight)
                param.add(getattr(self, 'pw' + str(i)).weight)

        criterion = nn.MSELoss()
        optimizer = AdamW(param, lr=lr)
        st = time.time()
        for s in range(steps):
            if steps in (400, 700):
                lr = lr / 10
                optimizer = AdamW(param, lr=lr)
            optimizer.zero_grad()
            for i in range(self.rank):
                if i == 0:
                    kernel_pred = getattr(self, \
                            'pw' + str(i)).weight.cuda() * \
                getattr(self, 'dw' + str(i)).weight.cuda()
                else:
                    kernel_pred += getattr(self, \
                            'pw' + str(i)).weight.cuda() * getattr(self, \
                            'dw' + str(i)).weight.cuda()
            loss = criterion(kernel_pred, kernel.cuda())
            loss.backward()
            optimizer.step()
            if s % 100 == 99:
                print('step %d: loss = %f, time = %d' % ((s+1), loss, (time.time() - st)))
                st = time.time()

    def group1x1(self, group_num):
        """
        Replace 1x1 pointwise convolution in FALCON with 1x1 group convolution
        
        :param group_num: number of groups for 1x1 group
        """
        if self.rank == 1:
            pw = self.pw.weight.data
            if pw.shape[1] != 3:
                self.pw = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    groups=group_num).to(self.device)
                in_length = int(self.in_channels / group_num)
                out_length = int(self.out_channels / group_num)
                for i in range(group_num):
                    self.pw.weight.data[(i*out_length):((i+1)*out_length), 0:(in_length), :, :] =\
                    pw[(i*out_length):((i+1)*out_length), (i*in_length):((i+1)*in_length), :, :]
        else:
            for i in range(self.rank):
                pw = getattr(self, 'pw' + str(i)).weight.data
                if pw.shape[1] != 3:
                    setattr(self, 'pw'+str(i),
                            nn.Conv2d(in_channels=self.in_channels,
                                      out_channels=self.out_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=False,
                                      groups=group_num).to(self.device))
                    in_length = int(self.in_channels / group_num)
                    out_length = int(self.out_channels / group_num)
                    for j in range(group_num):
                        getattr(self, \
                                'pw'+str(i)).weight.data[(j*out_length):((j+1)*out_length),\
                                                                          0:(in_length), :, :] =\
                        pw[(j*out_length):((j+1)*out_length), (j*in_length):((j+1)*in_length), :, :]
