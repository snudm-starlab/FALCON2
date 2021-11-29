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

File: models/stconv_branch.py
 - Contain model with stconv_branch and falcon compressed model.
Version: 1.0
"""

# pylint: disable=E1101,C0103,R0913,R1725,W0223
import torch
import torch.nn as nn

from models.falcon import GEPdecompose

def channel_shuffle(x, groups):
    """
    Description: channel shuffle operation
    
    :param x: output feature maps of last layer
    :param groups: number of groups for group convolution
    :return: x: channel shuffled feature maps
    """
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # Reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # Flatten
    x = x.view(batchsize, -1, height, width)

    return x


class StConv_branch(nn.Module):
    '''
    Description: Basic unit of ShuffleUnit with standard convolution
    '''
    def __init__(self, inp, oup, stride=1):
        """
        Initialize basic unit of StConv_branch.
        
        :param inp: the number of input channels
        :param oup: the number of output channels
        :param stride: stride of convolution
        """

        super(StConv_branch, self).__init__()
        self.benchmodel = 1 if inp == oup and stride == 1 else 2
        self.stride = stride

        oup_inc = oup // 2
        self.inp = inp
        self.oup = oup

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.branch2 = nn.Sequential(
                nn.Conv2d(oup_inc, oup_inc, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, oup_inc, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True)
            )

            self.branch2 = nn.Sequential(
                nn.Conv2d(inp, oup_inc, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True)
            )

    @staticmethod
    def _concat(x, out):
        """
        concatenate x and out
        
        :param x: input feature maps
        :param out: output feature maps
        :return: torch.cat((x, out), 1): concatenated feature maps along channel axis
        """
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        """
        Run forward propagation
        
        :param x: input feature maps
        :return: channel_shuffle(out, 2): channel shuffled output feature maps
        """
        if self.benchmodel == 1:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self.branch2(x2)
            out = self._concat(x1, out)
        elif self.benchmodel == 2:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            out = self._concat(x1, x2)

        return channel_shuffle(out, 2)

    def falcon(self, rank=1, init=True, alpha=1.0, bn=False, relu=False, groups=1):
        """
        Replace standard convolution by FALCON
        
        :param rank: rank of GEP
        :param init: whether initialize FALCON with GEP decomposition tensors
        :param alpha: width multiplier
        :param bn: whether add batch normalization after FALCON
        :param relu: whether add ReLU function after FALCON
        :param groups: number of groups for pointwise convolution
        """
        if self.benchmodel == 2:
            compress = GEPdecompose(self.branch1[0], rank, init, alpha=alpha,\
                    bn=bn, relu=relu, groups=groups)
            self.branch1[0] = compress
        compress = GEPdecompose(self.branch2[0], rank, init, alpha=alpha,\
                bn=bn, relu=relu, groups=groups)
        self.branch2[0] = compress


class VGG_StConv_branch(nn.Module):
    """
    Description: VGG model with VGG_StConv_branch.
    """

    # Configures of different models
    cfgs_VGG16 = [[64, 64, 2],
                  [64, 128], [128, 128, 2],
                  [128, 256], [256, 256], [256, 256, 2],
                  [256, 512], [512, 512], [512, 512, 2],
                  [512, 512], [512, 512], [512, 512, 2]]  # [3, 64],

    cfgs_VGG19 = [[64, 64, 2],
                  [64, 128], [128, 128, 2],
                  [128, 256], [256, 256], [256, 256], [256, 256, 2],
                  [256, 512], [512, 512], [512, 512], [512, 512, 2],
                  [512, 512], [512, 512], [512, 512], [512, 512, 2]]  # [3, 64],

    cfgs_VGG_en = [[64, 64], [64, 64, 2],
                   [64, 128], [128, 128], [128, 128, 2],
                   [128, 256], [256, 256], [256, 256],\
                       [256, 256], [256, 256], [256, 256, 2],
                   [256, 512], [512, 512], [512, 512],\
                       [512, 512], [512, 512], [512, 512, 2],
                   [512, 512], [512, 512], [512, 512],\
                       [512, 512], [512, 512], [512, 512, 2]]  # [3, 64],

    def __init__(self, num_classes=10, which='VGG16', alpha=1):
        """
        Initialize Model as argument configurations.
        
        :param num_classes: number of classification labels
        :param which: choose a model architecture from VGG16/VGG19/VGG_en
        :param alpha: width multiplier
        """

        super(VGG_StConv_branch, self).__init__()

        self.alpha = alpha
        first_output_channel = 64 if self.alpha == 1 else int(64 * self.alpha)
        self.conv = nn.Sequential(
            nn.Conv2d(3, first_output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(first_output_channel),
            nn.ReLU(True))
        self.layers = self._make_layers(which)
        self.avgPooling = nn.AvgPool2d(2, 2)

        last_output_channel = 512 if self.alpha == 1 else int(512 * self.alpha)
        if which == 'VGG_en':
            self.fc = nn.Linear(last_output_channel, num_classes)
        else:
            self.fc = nn.Sequential(
                nn.Linear(last_output_channel, 512),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )

    def _make_layers(self, which):
        """
        Make Model layers.
        
        :param which: choose a model architecture from VGG16/VGG19/VGG_en
        :return: nn.Sequential(*layers): vgg layers
        """

        layers = []
        if which == 'VGG16':
            self.cfgs = self.cfgs_VGG16
        elif which == 'VGG19':
            self.cfgs = self.cfgs_VGG19
        elif which == 'VGG_en':
            self.cfgs = self.cfgs_VGG_en
        else:
            pass

        if self.alpha != 1:
            for cfg in self.cfgs:
                cfg[0] = int(cfg[0] * self.alpha)
                cfg[1] = int(cfg[1] * self.alpha)

        for cfg in self.cfgs:
            layers.append(StConv_branch(cfg[0], cfg[1]))
            if len(cfg) == 3:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Run forward propagation
        
        :param x: input features
        :return: (out, out_conv): output features, and output features before the fully connected layer
        """
        out_conv = self.conv(x)
        out_conv = self.layers(out_conv)
        if out_conv.size(2) != 1:
            out_conv = self.avgPooling(out_conv)
        out = out_conv.view(out_conv.size(0), -1)
        out = self.fc(out)
        return out, out_conv

    def falcon(self, rank, init=True, alpha=1.0, bn=False, relu=False, groups=1):
        """
        Replace standard convolution by FALCON
        
        :param rank: rank of GEP
        :param init: whether initialize FALCON with GEP decomposition tensors
        :param alpha: width multiplier
        :param bn: whether add batch normalization after FALCON
        :param relu: whether add ReLU function after FALCON
        :param groups: number of groups for pointwise convolution
        """
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], StConv_branch):

                if self.layers[i].benchmodel == 2:
                    compress = GEPdecompose(self.layers[i].branch1[0], rank, init,\
                            alpha=alpha, bn=bn, relu=relu, groups=groups)
                    self.layers[i].branch1 = compress

                compress = GEPdecompose(self.layers[i].branch2[0], rank, init,\
                        alpha=alpha, bn=bn, relu=relu,
                                        groups=groups)
                self.layers[i].branch2[0] = compress


class BasicBlock(nn.Module):
    """
    Description: BasicBlock of ResNet with StConv_branch
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize BasicBlock with StConv_branch
        
        :param in_planes: the number of input channels
        :param out_planes: the number of output channels
        :param stride: stride of depthwise convolution
        """
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            StConv_branch(in_channels, out_channels, stride=stride),
            StConv_branch(out_channels, out_channels, stride=1)
        )

    def forward(self, x):
        """
        Run forward propagation
        
        :param x: input features
        :return: x: output features
        """
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x


class ResidualLayer(nn.Module):
    """
    Description: add shortcut to BasicBlock
    """

    def __init__(self, in_channels, out_channels, layer_num="34", stride=1):
        """
        Initialize basic unit of StConv_branch.
        
        :param in_planes: the number of input channels
        :param out_planes: the number of output channels
        :param stride: stride of depthwise convolution
        """
        super(ResidualLayer, self).__init__()

        self.stacked = BasicBlock(in_channels, out_channels, stride)
        self.layer_num = layer_num

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(True)

    def forward(self, x):
        """
        Run forward propagation
        
        :param x: input features
        :return: self.relu(stacked_out): output features of residual layer
        """
        stacked_out = self.stacked(x)
        return self.relu(stacked_out) # + shortcut_out)


class ResNet_StConv_branch(nn.Module):
    """
    Description: ResNet Model with StConv_branch
    """

    # Configurations for ResNet
    basic_channels = [64, 128, 256, 512]

    def __init__(self, layer_num='18', num_classes=10, alpha=1):
        """
        Initialize ResNet with StConv_branch.
        
        :param layer_num: number of layers in ResNet
        :param num_classes: number of classification categories
        :param alpha: width multiplier
        """

        super(ResNet_StConv_branch, self).__init__()

        self.alpha = alpha

        if self.alpha != 1:
            for i in range(len(self.basic_channels)):
                self.basic_channels[i] = int(self.basic_channels[i] * self.alpha)

        first_output_channel = 64 if self.alpha == 1 else int(64 * self.alpha)
        self.first = nn.Sequential(
            nn.Conv2d(3, first_output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(first_output_channel),
            nn.ReLU(True),
        )

        self.residuals = self._make_layers(layer_num)

        self.avgpool_4 = nn.AvgPool2d(kernel_size=4)
        self.avgpool_2 = nn.AvgPool2d(kernel_size=2)

        last_channels = self.basic_channels[-1]
        self.fc = nn.Linear(last_channels, num_classes)

    def _make_layers(self, layer_num):
        """
        Make Model layers.
        
        :param layer_num: number of convolution layers of ResNet-(18, 34, 50, 101, 152)
        :return: nn.Sequential(*layers): residual layers
        """
        layers = []
        cfg = (3, 4, 6, 3)

        for i in range(4):
            for j in range(cfg[i]):
                if j == 0:
                    if i != 0:
                        layers.append(ResidualLayer(self.basic_channels[i] // 2,\
                                    self.basic_channels[i], layer_num=layer_num, stride=2))
                    else:
                        layers.append(ResidualLayer(self.basic_channels[i],\
                                    self.basic_channels[i], layer_num=layer_num, stride=2))
                else:
                    layers.append(ResidualLayer(self.basic_channels[i],\
                                self.basic_channels[i], layer_num=layer_num, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Run forward propagation
        
        :param x: input features
        :return: (out, out_conv): output features, and output features before the fully connected layer
        """
        out_conv = self.first(x)
        out_conv = self.residuals(out_conv)

        if out_conv.size(3) == 2:
            out_conv = self.avgpool_2(out_conv)
        else:
            out_conv = self.avgpool_4(out_conv)
        out = out_conv.reshape(out_conv.shape[0], -1)
        out = self.fc(out)
        return out, out_conv

    def falcon(self, rank, init=True, alpha=1.0, bn=False, relu=False, groups=1):
        """
        Replace standard convolution by FALCON
        
        :param rank: rank of GEP
        :param init: whether initialize FALCON with GEP decomposition tensors
        :param bn: whether add batch normalization after FALCON
        :param relu: whether add ReLU function after FALCON
        :param groups: number of groups for pointwise convolution
        """
        for i in range(len(self.residuals)):
            if isinstance(self.residuals[i].stacked.conv[0], StConv_branch):
                if self.residuals[i].stacked.conv[0].benchmodel == 2:
                    compress = GEPdecompose(self.residuals[i].stacked.conv[0].branch1[0], \
                            rank, init, alpha=alpha, bn=bn, relu=relu, groups=groups)
                    self.residuals[i].stacked.conv[0].branch1[0] = compress
                compress = GEPdecompose(self.residuals[i].stacked.conv[0].branch2[0], \
                        rank, init, alpha=alpha, bn=bn, relu=relu, groups=groups)
                self.residuals[i].stacked.conv[0].branch2[0] = compress
            if isinstance(self.residuals[i].stacked.conv[1], StConv_branch):
                if self.residuals[i].stacked.conv[1].benchmodel == 2:
                    compress = GEPdecompose(self.residuals[i].stacked.conv[1].branch1[0],\
                            rank, init, alpha=alpha, bn=bn, relu=relu, groups=groups)
                    self.residuals[i].stacked.conv[1].branch1[0] = compress
                compress = GEPdecompose(self.residuals[i].stacked.conv[1].branch2[0], rank,\
                        init, alpha=alpha, bn=bn, relu=relu, groups=groups)
                self.residuals[i].stacked.conv[1].branch2[0] = compress
