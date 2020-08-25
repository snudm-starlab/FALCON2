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

File: models/resnet.py
 - Contain ResNet class.

Version: 1.0
"""

import torch
import torch.nn as nn
from models.falcon import GEPdecompose

# configurations of ResNet
bottle_neck_channels = ((64, 256), (128, 512), (256, 1024), (512, 2048))
basic_channels = (64, 128, 256, 512)

cfgs = {
    '18': (2, 2, 2, 2),
    '34': (3, 4, 6, 3),
    '50': (3, 4, 6, 3),
    '101': (3, 4, 23, 3),
    '152': (3, 8, 36, 3)
}


class BottleNeckBlock(nn.Module):
    """
    BottleNeckBlock
    go through 1x1, 3x3, 1x1
    """

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initializ BottleNeckBlock
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: number of stride
        """
        super(BottleNeckBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        """Run forward propagation"""
        return self.conv(x)


class BasicBlock(nn.Module):
    """
    BasicBlock
    go through 3x3, 3x3
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initializ BasicBlock
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: number of stride
        """
        super(BasicBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        """Run forward propagation"""
        return self.conv(x)


class ResidualLayer(nn.Module):
    """
    add shortcut
    """

    def __init__(self, in_channels, out_channels, layer_num="34", stride=1):
        """
        Initialize Residual Layer (add shortcut to BottleNeckBlock or BasicBlock)
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param layer_num: number of layers in ResNet (default ResNet34)
        :param stride: number of stride
        """
        super(ResidualLayer, self).__init__()

        if layer_num == "18" or layer_num == "34":
            self.stacked = BasicBlock(in_channels, out_channels, stride)
        else:  # layer_num == "50" or layer_num == "101" or layer_num == "152":
            self.stacked = BottleNeckBlock(in_channels, out_channels, stride)

        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(True)

    def forward(self, x):
        """Run forward propagation"""
        stacked_out = self.stacked(x)
        shortcut_out = self.shortcut(x)
        return self.relu(stacked_out + shortcut_out)


class ResNet(nn.Module):
    """ResNet model"""

    def __init__(self, layer_num='18', num_classes=10):
        """
        Initialize ResNet
        :param layer_num: number of layers in ResNet (default ResNet34)
        :param num_classes: number of classes of datasets
        """
        super(ResNet, self).__init__()

        self.first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.residuals = self._make_layers(layer_num)

        self.avgpool_4 = nn.AvgPool2d(kernel_size=4)
        self.avgpool_2 = nn.AvgPool2d(kernel_size=2)

        if layer_num == "18" or layer_num == "34":
            last_channels = basic_channels[-1]
        else:
            last_channels = bottle_neck_channels[-1]
        self.fc = nn.Linear(last_channels, num_classes)

    def _make_layers(self, layer_num):
        """
        Make standard-conv Model layers.
        :param layer_num: number of convolution layers of ResNet-(18, 34, 50, 101, 152)
        """

        layers = []
        cfg = cfgs[layer_num]

        for i in range(4):
            for j in range(cfg[i]):
                if layer_num == "18" or layer_num == "34":
                    if j == 0:
                        if i != 0:
                            layers.append(ResidualLayer(basic_channels[i] // 2, basic_channels[i], layer_num=layer_num, stride=2))
                        else:
                            layers.append(ResidualLayer(basic_channels[i], basic_channels[i], layer_num=layer_num, stride=2))
                    else:
                        layers.append(ResidualLayer(basic_channels[i], basic_channels[i], layer_num=layer_num, stride=1))
                else:
                    if j == 0:
                        if i == 0:
                            layers.append(ResidualLayer(bottle_neck_channels[i], bottle_neck_channels[i] * 4, layer_num=layer_num, stride=2))
                        else:
                            layers.append(
                                ResidualLayer(bottle_neck_channels[i] * 2, bottle_neck_channels[i] * 4, layer_num=layer_num, stride=2))
                    else:
                        layers.append(ResidualLayer(bottle_neck_channels[i], bottle_neck_channels[i], layer_num=layer_num, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Run forward propagation"""
        out_conv = self.first(x)
        out_conv = self.residuals(out_conv)
        if out_conv.size(3) == 2:
            out_conv = self.avgpool_2(out_conv)
        else:
            out_conv = self.avgpool_4(out_conv)
        out = out_conv.reshape(out_conv.shape[0], -1)
        out = self.fc(out)
        return out, out_conv

    def falcon(self, rank, init=True, bn=False, relu=False, groups=1):
        """
        Replace standard convolution by FALCON
        :param rank: rank of GEP
        :param init: whether initialize FALCON with GEP decomposition tensors
        :param bn: whether add batch normalization after FALCON
        :param relu: whether add ReLU function after FALCON
        :param groups: number of groups for pointwise convolution
        """
        for i in range(len(self.residuals)):
            if isinstance(self.residuals[i].stacked.conv[0], nn.Conv2d):
                # print(self.residuals[i].stacked.conv[0])
                compress = GEPdecompose(self.residuals[i].stacked.conv[0], rank, init, groups=groups)
                self.residuals[i].stacked.conv[0] = compress
            if isinstance(self.residuals[i].stacked.conv[3], nn.Conv2d):
                # print(self.residuals[i].stacked.conv[3])
                compress = GEPdecompose(self.residuals[i].stacked.conv[3], rank, init, groups=groups)
                self.residuals[i].stacked.conv[3] = compress
            if isinstance(self.residuals[i].stacked.conv[1], nn.BatchNorm2d):
                device = self.residuals[i].stacked.conv[1].weight.device
                self.residuals[i].stacked.conv[1] = nn.BatchNorm2d(self.residuals[i].stacked.conv[1].num_features).to(device)
            if isinstance(self.residuals[i].stacked.conv[4], nn.BatchNorm2d):
                device = self.residuals[i].stacked.conv[4].weight.device
                self.residuals[i].stacked.conv[4] = nn.BatchNorm2d(self.residuals[i].stacked.conv[4].num_features).to(device)

