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
under the License
.
FALCON: Lightweight and Accurate Convolution

File: models/model_imageNet.py
 - Contain source code for re-organize the structure of pre-trained model.

Version: 1.0
"""

from torch import nn

from falcon import GEPdecompose
from stconv_branch import StConvBranch

class VGGImageNet(nn.Module):
    """
    Description: Re-organize the structure of a given vgg model.
    """

    def __init__(self, model):
        """
        Initialize a given model.
        
        :param model: the given model
        """

        super().__init__()

        self.features = model.features
        self.classifier = model.classifier

    def forward(self, input_):
        """
        Run forward propagation
        
        :param input_: input feature maps
        :return: (out2_, out1_): output of forward propagation, \
                    and intermediate tensor after convolutions
        """
        out1_ = self.features(input_)
        out1_ = out1_.view(out1_.size(0), -1)
        out2_ = self.classifier(out1_)
        return out2_, out1_

    def falcon(self, init=True, rank=1, batch_norm=False, relu=False):
        """
        Replace standard convolution by FALCON
        
        :param rank: rank of GEP
        :param init: whether initialize FALCON with GEP decomposition tensors
        :param batch_norm: whether add batch normalization after FALCON
        :param relu: whether add ReLU function after FALCON
        """
        print('********** Compressing...... **********')
        for i, feature in enumerate(self.features):
            if isinstance(feature, nn.Conv2d):
                print(feature)
                compress = GEPdecompose(feature, rank, init, bn=batch_norm, relu=relu)
                self.features[i] = compress
            if isinstance(feature, nn.BatchNorm2d):
                device = feature.weight.device
                self.features[i] = nn.BatchNorm2d(feature.num_features).to(device)

    def stconv_branch(self, alpha=1):
        """
        Replace standard convolution by stconv_branch (vs shuffleunitv2)
        
        :param alpha: width multiplier
        """
        for i, feature in enumerate(self.features):
            if isinstance(feature, nn.Conv2d):
                shape = feature.weight.shape
                if shape[1] == 3:
                    self.features[i] = nn.Conv2d(3, \
                            int(feature.out_channels * alpha), \
                            kernel_size=3, padding=1)
                    self.features[i+1] = nn.BatchNorm2d(feature.out_channels)
                else:
                    compress = StConvBranch(int(feature.in_channels * alpha),
                                           int(feature.out_channels * alpha),
                                           stride=feature.stride[0])
                    self.features[i] = compress
        layers = []
        for i, feature in enumerate(self.features):
            if (isinstance(feature, nn.BatchNorm2d) and \
                    isinstance(self.features[i - 1], StConvBranch)) \
                    or (isinstance(feature, nn.ReLU) and \
                            isinstance(self.features[i - 2], StConvBranch)):
                pass
            else:
                layers.append(feature)
        if alpha != 1:
            layers.append(layers[-1])
            layers[-2] = nn.Conv2d(int(self.classifier[0].in_features * alpha / 49),
                                    int(self.classifier[0].in_features / 49),
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        self.features = nn.Sequential(*layers)

    def falcon_branch(self, init=True):
        """
        Replace standard convolution in stconv_branch by falcon
        
        :param init: whether initialize falcon
        """
        for i, module in enumerate(self.features.module):
            if isinstance(module, StConvBranch):
                self.features.module[i].falcon(init=init)


class BasicBlockStConvBranch(nn.Module):
    """
    Description: BasicBlock of ResNet with StConvBranch
    """

    def __init__(self, conv1, conv2, downsample=None):
        """
        Initialize BasicBlock_ShuffleUnit
        
        :param conv1: the first convolution layer in BasicBlock_StConvBranch
        :param conv2: the second convolution layer in BasicBlock_StConvBranch
        """
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_):
        """
        Run forward propagation
        
        :param input_: input feature maps
        :return: out: output tensor of forward propagation
        """
        out = self.conv1(input_)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(input_)
        else:
            identity = input_
        out += identity
        out = self.relu(out)
        return out


class ResNetImageNet(nn.Module):
    """
    Description: Re-organize the structure of a given resnet model.
    """

    def __init__(self, model):
        """
        Initialize a given model.
        
        :param model: the given model
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool
            ),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool
        )
        self.classifier = model.fc

    def forward(self, input_):
        """
        Run forward propagation
        
        :param input_: input feature maps
        :return: (out2_, out1_): output of forward propagation, \
                                and intermediate tensor after convolutions
        """
        out1_ = self.features(input_)
        out1_ = out1_.view(out1_.size(0), -1)
        out2_ = self.classifier(out1_)
        return out2_, out1_

    def falcon(self, rank=1, init=True, batch_norm=False, relu=False):
        """
        Replace standard convolution by FALCON
        
        :param rank: rank of GEP
        :param init: whether initialize FALCON with GEP decomposition tensors
        :param batch_norm: whether add batch normalization after FALCON
        :param relu: whether add ReLU function after FALCON
        """
        print('********** Compressing...... **********')
        for i in range(1, 5):
            for j in range(len(self.features[i])):
                if isinstance(self.features[i][j].conv1, nn.Conv2d):
                    print(self.features[i][j].conv1)
                    compress = GEPdecompose(self.features[i][j].conv1, rank, init, \
                            bn=batch_norm, relu=relu)
                    self.features[i][j].conv1 = compress
                if isinstance(self.features[i][j].conv2, nn.Conv2d):
                    print(self.features[i][j].conv2)
                    compress = GEPdecompose(self.features[i][j].conv2, rank, init, \
                            bn=batch_norm, relu=relu)
                    self.features[i][j].conv2 = compress
                if isinstance(self.features[i][j].bn1, nn.BatchNorm2d):
                    device = self.features[i][j].bn1.weight.device
                    self.features[i][j].bn1 = nn.BatchNorm2d(\
                            self.features[i][j].bn1.num_features).to(device)
                if isinstance(self.features[i][j].bn2, nn.BatchNorm2d):
                    device = self.features[i][j].bn2.weight.device
                    self.features[i][j].bn2 = nn.BatchNorm2d(\
                            self.features[i][j].bn2.num_features).to(device)

    def stconv_branch(self, alpha=1):
        """
        Replace standard convolution by StConvBranch
        
        :param alpha: width multiplier
        """
        self.features[0][0] = nn.Conv2d(3, int(self.features[0][0].out_channels * alpha),
                                        kernel_size=self.features[0][0].kernel_size,
                                        stride=self.features[0][0].stride,
                                        padding=self.features[0][0].padding,
                                        bias=False)
        self.features[0][1] = nn.BatchNorm2d(self.features[0][0].out_channels)
        for i in range(1, 5):
            for j in range(len(self.features[i])):
                if isinstance(self.features[i][j].conv1, nn.Conv2d):
                    compress = StConvBranch(int(self.features[i][j].conv1.in_channels * alpha),
                                           int(self.features[i][j].conv1.out_channels * alpha),
                                           stride=self.features[i][j].conv1.stride[0])
                    self.features[i][j].conv1 = compress
                if isinstance(self.features[i][j].conv2, nn.Conv2d):
                    compress = StConvBranch(int(self.features[i][j].conv2.in_channels * alpha),
                                           int(self.features[i][j].conv2.out_channels * alpha),
                                           stride=self.features[i][j].conv2.stride[0])
                    self.features[i][j].conv2 = compress
        layers = []
        layers.append(self.features[0])
        for i in range(1, 5):
            for j in range(len(self.features[i])):
                if self.features[i][j].downsample is not None:
                    self.features[i][j].downsample[0] = nn.Conv2d(\
                            int(self.features[i][j].downsample[0].in_channels * alpha), \
                            int(self.features[i][j].downsample[0].out_channels * alpha),\
                            kernel_size=self.features[i][j].downsample[0].kernel_size,\
                            stride=self.features[i][j].downsample[0].stride,\
                            padding=self.features[i][j].downsample[0].padding,\
                            bias=self.features[i][j].downsample[0].bias)
                    self.features[i][j].downsample[1] = \
    nn.BatchNorm2d(int(self.features[i][j].downsample[1].num_features * alpha))
                layers.append(BasicBlockStConvBranch(self.features[i][j].conv1,\
                            self.features[i][j].conv2, self.features[i][j].downsample))
        layers.append(self.features[5])
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Linear(int(self.classifier.in_features * alpha), 1000, bias=True)

    def falcon_branch(self, init=True):
        """
        Replace standard convolution in stconv_branch by falcon
        
        :param init: whether initialize falcon
        """
        for i, feature in enumerate(self.features):
            if isinstance(feature, BasicBlockStConvBranch):
                if isinstance(feature.conv1, StConvBranch):
                    self.features[i].conv1.falcon(init=init)
                if isinstance(feature.conv2, StConvBranch):
                    self.features[i].conv2.falcon(init=init)


class VGGImageNetINF(nn.Module):
    """
    Description: Re-organize the structure of a given vgg model.
    """

    def __init__(self, model):
        """
        Initialize a given model.
        
        :param model: the given model
        """
        super().__init__()

        self.features = model.features

    def forward(self, input_):
        """
        Run forward propagation
        
        :param input_: input feature maps
        :return: features(input_): output features of forward propagation
        """
        return self.features(input_)


class ResNetImageNetINF(nn.Module):
    """
    Description: Re-organize the structure of a given resnet model.
    """
    def __init__(self, model):
        """
        Initialize a given model.
        
        :param model: the given model
        """
        super().__init__()
        self.features = nn.Sequential(*list(model.features.children())[:-1])

    def forward(self, input_):
        """
        Run forward propagation
        
        :param input_: input feature maps
        :return: features(input_): output features of forward propagation
        """
        return self.features(input_)
