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

from torch import nn
from models.falcon import GEPdecompose

# Configurations of ResNet
BOTTLE_NECK_CHANNELS = ((64, 256), (128, 512), (256, 1024), (512, 2048))
BASIC_CHANNELS = (64, 128, 256, 512)

cfgs = {
    '18': (2, 2, 2, 2),
    '32': (3, 4, 6, 2),
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
        super().__init__()

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

    def forward(self, input_):
        """
        Run forward propagation
        
        :param input_: input feature maps
        :return: conv(input_): features of convolution
        """
        return self.conv(input_)


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
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, input_):
        """
        Run forward propagation
        
        :param input_: input feature maps
        :return: conv(input_): features of convolution
        """
        return self.conv(input_)


class ResidualLayer(nn.Module):
    """
    Add shortcut
    """

    def __init__(self, in_channels, out_channels, layer_num="34", stride=1):
        """
        Initialize Residual Layer (add shortcut to BottleNeckBlock or BasicBlock)
        
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param layer_num: number of layers in ResNet (default ResNet34)
        :param stride: number of stride
        """
        super().__init__()

        if layer_num in ("18", "32", "34"):
            self.stacked = BasicBlock(in_channels, out_channels, stride)
        else:  # layer_num == "50" or layer_num == "101" or layer_num == "152":
            self.stacked = BottleNeckBlock(in_channels, out_channels, stride)

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(True)

    def forward(self, input_):
        """
        Run forward propagation
        
        :param input_: input feature maps
        :return: self.relu(stacked_out + shortcut_out): features of residual
        """
        stacked_out = self.stacked(input_)
        shortcut_out = self.shortcut(input_)
        return self.relu(stacked_out + shortcut_out)


class ResNet(nn.Module):
    """
    ResNet model
    """

    def __init__(self, layer_num='18', num_classes=10, alpha=1.0):
        """
        Initialize ResNet
        
        :param layer_num: number of layers in ResNet (default ResNet34)
        :param num_classes: number of classes of datasets
        """
        super().__init__()

        self.alpha = alpha
        self.first = nn.Sequential(
            nn.Conv2d(3, int(64*alpha), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(64*alpha)),
            nn.ReLU(True),
        )

        self.residuals = self._make_layers(layer_num)

        self.avgpool_4 = nn.AvgPool2d(kernel_size=4)
        self.avgpool_2 = nn.AvgPool2d(kernel_size=2)

        if layer_num in ("18","32","34"):
            last_channels = BASIC_CHANNELS[-1]
        else:
            last_channels = BOTTLE_NECK_CHANNELS[-1]
        self.fc_layer = nn.Linear(last_channels, num_classes)

    def _make_layers(self, layer_num):
        """
        Make standard-conv Model layers.
        
        :param layer_num: number of convolution layers of ResNet-(18, 34, 50, 101, 152)
        :return: nn.Sequential(*layers): layers of resnet
        """

        layers = []
        cfg = cfgs[layer_num]
        global BASIC_CHANNELS
        global BOTTLE_NECK_CHANNELS

        if self.alpha != 1:
            tmp_basic = list(BASIC_CHANNELS)
            tmp_basic = list(BASIC_CHANNELS)
            for i, module in enumerate(tmp_basic):
                tmp_basic[i] = int(module*self.alpha)
            BASIC_CHANNELS = tuple(tmp_basic)
            tmp_bottle = list(BOTTLE_NECK_CHANNELS)

            for i, module in enumerate(tmp_bottle):
                tmp = list(module)
                for j, sub_module in enumerate(tmp):
                    tmp[j] = int(sub_module*self.alpha)
                tmp_bottle[i] = tuple(tmp)
            BOTTLE_NECK_CHANNELS = tuple(tmp_bottle)

        for i in range(4):
            for j in range(cfg[i]):
                if layer_num in ("18", "32", "34"):
                    if j == 0:
                        if i != 0:
                            layers.append(ResidualLayer(BASIC_CHANNELS[i] // 2, \
                                        BASIC_CHANNELS[i], layer_num=layer_num, stride=2))
                        else:
                            layers.append(ResidualLayer(BASIC_CHANNELS[i], BASIC_CHANNELS[i],\
                                        layer_num=layer_num, stride=2))
                    else:
                        layers.append(ResidualLayer(BASIC_CHANNELS[i], BASIC_CHANNELS[i],\
                                    layer_num=layer_num, stride=1))
                else:
                    if j == 0:
                        if i == 0:
                            layers.append(ResidualLayer(BOTTLE_NECK_CHANNELS[i], \
                                        BOTTLE_NECK_CHANNELS[i] * 4, layer_num=layer_num, stride=2))
                        else:
                            layers.append(
                                ResidualLayer(BOTTLE_NECK_CHANNELS[i] * 2, \
                                    BOTTLE_NECK_CHANNELS[i] * 4, layer_num=layer_num, stride=2))
                    else:
                        layers.append(ResidualLayer(BOTTLE_NECK_CHANNELS[i], \
                                    BOTTLE_NECK_CHANNELS[i], layer_num=layer_num, stride=1))
        return nn.Sequential(*layers)

    def forward(self, input_):
        """
        Run forward propagation
        
        :param input_: input feature maps
        :return: (out, out_conv): output features after the fully connected layer,
                                    and output features of residual layers
        """
        out_conv = self.first(input_)
        out_conv = self.residuals(out_conv)
        if out_conv.size(3) == 2:
            out_conv = self.avgpool_2(out_conv)
        else:
            out_conv = self.avgpool_4(out_conv)
        out = out_conv.reshape(out_conv.shape[0], -1)
        out = self.fc_layer(out)
        return out, out_conv

    def falcon(self, rank, init=False, groups=1, alpha=1.0):
        """
        Replace standard convolution by FALCON
        
        :param rank: rank of GEP
        :param init: whether initialize FALCON with GEP decomposition tensors
        :param groups: number of groups for pointwise convolution
        """
        for i, module in enumerate(self.residuals):
            if isinstance(module.stacked.conv[0], nn.Conv2d):
                # print(self.residuals[i].stacked.conv[0])
                compress = GEPdecompose(module.stacked.conv[0], rank, \
                        init, groups=groups, alpha=alpha)
                self.residuals[i].stacked.conv[0] = compress
            if isinstance(module.stacked.conv[3], nn.Conv2d):
                # print(self.residuals[i].stacked.conv[3])
                compress = GEPdecompose(module.stacked.conv[3], rank, \
                        init, groups=groups, alpha=alpha)
                self.residuals[i].stacked.conv[3] = compress
            if isinstance(module.stacked.conv[1], nn.BatchNorm2d):
                device = module.stacked.conv[1].weight.device
                self.residuals[i].stacked.conv[1] = nn.BatchNorm2d(\
                        int(module.stacked.conv[1].num_features*alpha)).to(device)
            if isinstance(module.stacked.conv[4], nn.BatchNorm2d):
                device = module.stacked.conv[4].weight.device
                self.residuals[i].stacked.conv[4] = nn.BatchNorm2d(\
                        int(module.stacked.conv[4].num_features*alpha)).to(device)
