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

import torch
from torch import nn

from models.falcon import GEPdecompose

def channel_shuffle(feature_map, groups):
    """
    Description: channel shuffle operation
    
    :param feature_map: output feature maps of last layer
    :param groups: number of groups for group convolution
    :return: feature_map: channel shuffled feature maps
    """
    batchsize, num_channels, height, width = feature_map.data.size()

    channels_per_group = num_channels // groups

    # Reshape
    feature_map = feature_map.view(batchsize, groups,
               channels_per_group, height, width)

    feature_map = torch.transpose(feature_map, 1, 2).contiguous()

    # Flatten
    feature_map = feature_map.view(batchsize, -1, height, width)

    return feature_map


class StConvBranch(nn.Module):
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

        super().__init__()
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
    def _concat(feature_map, out):
        """
        concatenate feature_map and out
        
        :param feature_map: input feature maps
        :param out: output feature maps
        :return: torch.cat((feature_map, out), 1): concatenated feature maps along channel axis
        """
        # concatenate along channel axis
        return torch.cat((feature_map, out), 1)

    def forward(self, input_):
        """
        Run forward propagation
        
        :param x: input feature maps
        :return: channel_shuffle(out, 2): channel shuffled output feature maps
        """
        if self.benchmodel == 1:
            tmp1 = input_[:, :(input_.shape[1] // 2), :, :]
            tmp2 = input_[:, (input_.shape[1] // 2):, :, :]
            out = self.branch2(tmp2)
            out = self._concat(tmp1, out)
        elif self.benchmodel == 2:
            tmp1 = self.branch1(input_)
            tmp2 = self.branch2(input_)
            out = self._concat(tmp1, tmp2)

        return channel_shuffle(out, 2)

    def falcon(self, rank=1, init=True, alpha=1.0, batch_norm=False, relu=False, groups=1):
        """
        Replace standard convolution by FALCON
        
        :param rank: rank of GEP
        :param init: whether initialize FALCON with GEP decomposition tensors
        :param alpha: width multiplier
        :param batch_norm: whether add batch normalization after FALCON
        :param relu: whether add ReLU function after FALCON
        :param groups: number of groups for pointwise convolution
        """
        if self.benchmodel == 2:
            compress = GEPdecompose(self.branch1[0], rank, init, alpha=alpha,\
                    bn=batch_norm, relu=relu, groups=groups)
            self.branch1[0] = compress
        compress = GEPdecompose(self.branch2[0], rank, init, alpha=alpha,\
                bn=batch_norm, relu=relu, groups=groups)
        self.branch2[0] = compress


class VGGStConvBranch(nn.Module):
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

        super().__init__()

        self.alpha = alpha
        first_output_channel = 64 if self.alpha == 1 else int(64 * self.alpha)
        self.conv = nn.Sequential(
            nn.Conv2d(3, first_output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(first_output_channel),
            nn.ReLU(True))
        self.layers = self._make_layers(which)
        self.avg_pooling = nn.AvgPool2d(2, 2)

        last_output_channel = 512 if self.alpha == 1 else int(512 * self.alpha)
        if which == 'VGG_en':
            self.fc_layer = nn.Linear(last_output_channel, num_classes)
        else:
            self.fc_layer = nn.Sequential(
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
            layers.append(StConvBranch(cfg[0], cfg[1]))
            if len(cfg) == 3:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, input_):
        """
        Run forward propagation
        
        :param input_: input features
        :return: (out, out_conv): output features,
                                    and output features before the fully connected layer
        """
        out_conv = self.conv(input_)
        out_conv = self.layers(out_conv)
        if out_conv.size(2) != 1:
            out_conv = self.avg_pooling(out_conv)
        out = out_conv.view(out_conv.size(0), -1)
        out = self.fc_layer(out)
        return out, out_conv

    def falcon(self, rank, init=True, alpha=1.0, batch_norm=False, relu=False, groups=1):
        """
        Replace standard convolution by FALCON
        
        :param rank: rank of GEP
        :param init: whether initialize FALCON with GEP decomposition tensors
        :param alpha: width multiplier
        :param batch_norm: whether add batch normalization after FALCON
        :param relu: whether add ReLU function after FALCON
        :param groups: number of groups for pointwise convolution
        """
        for idx, module in enumerate(self.layers):
            if isinstance(module, StConvBranch):
                if module.benchmodel == 2:
                    compress = GEPdecompose(module.branch1[0], rank, init,\
                            alpha=alpha, bn=batch_norm, relu=relu, groups=groups)
                    self.layers[idx].branch1 = compress

                compress = GEPdecompose(module.branch2[0], rank, init,\
                        alpha=alpha, bn=batch_norm, relu=relu,
                                        groups=groups)
                self.layers[idx].branch2[0] = compress


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
        super().__init__()
        self.conv = nn.Sequential(
            StConvBranch(in_channels, out_channels, stride=stride),
            StConvBranch(out_channels, out_channels, stride=1)
        )

    def forward(self, input_):
        """
        Run forward propagation
        
        :param input_: input features
        :return: out_: output features
        """
        out_ = self.conv[0](input_)
        out_ = self.conv[1](out_)
        return out_


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
        super().__init__()

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

    def forward(self, input_):
        """
        Run forward propagation
        
        :param input_: input features
        :return: self.relu(stacked_out): output features of residual layer
        """
        stacked_out = self.stacked(input_)
        return self.relu(stacked_out) # + shortcut_out)


class ResNetStConvBranch(nn.Module):
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

        super().__init__()

        self.alpha = alpha

        if self.alpha != 1:
            for i, module in enumerate(self.basic_channels):
                self.basic_channels[i] = int(module * self.alpha)

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
        self.fc_layer = nn.Linear(last_channels, num_classes)

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

    def forward(self, input_):
        """
        Run forward propagation
        
        :param input_: input features
        :return: (out, out_conv): output features,
                                    and output features before the fully connected layer
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

    def falcon(self, rank, init=True, alpha=1.0, batch_norm=False, relu=False, groups=1):
        """
        Replace standard convolution by FALCON
        
        :param rank: rank of GEP
        :param init: whether initialize FALCON with GEP decomposition tensors
        :param batch_norm: whether add batch normalization after FALCON
        :param relu: whether add ReLU function after FALCON
        :param groups: number of groups for pointwise convolution
        """
        for idx, module in enumerate(self.residuals):
            if isinstance(module.stacked.conv[0], StConvBranch):
                if module.stacked.conv[0].benchmodel == 2:
                    compress = GEPdecompose(module.stacked.conv[0].branch1[0], \
                            rank, init, alpha=alpha, bn=batch_norm, relu=relu, groups=groups)
                    self.residuals[idx].stacked.conv[0].branch1[0] = compress
                compress = GEPdecompose(module.stacked.conv[0].branch2[0], \
                        rank, init, alpha=alpha, bn=batch_norm, relu=relu, groups=groups)
                module.stacked.conv[0].branch2[0] = compress
            if isinstance(module.stacked.conv[1], StConvBranch):
                if module.stacked.conv[1].benchmodel == 2:
                    compress = GEPdecompose(module.stacked.conv[1].branch1[0],\
                            rank, init, alpha=alpha, bn=batch_norm, relu=relu, groups=groups)
                    self.residuals[idx].stacked.conv[1].branch1[0] = compress
                compress = GEPdecompose(module.stacked.conv[1].branch2[0], rank,\
                        init, alpha=alpha, bn=batch_norm, relu=relu, groups=groups)
                self.residuals[idx].stacked.conv[1].branch2[0] = compress
