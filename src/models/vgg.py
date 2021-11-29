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

File: models/vgg.py
 - Contain VGG model.

Version: 1.0
"""

# pylint: disable=C0103, R0913,R1725,W0223
import torch.nn as nn
from models.falcon import GEPdecompose

class VGG(nn.Module):
    """
    Description: VGG class.
    """

    # configures of different models
    cfgs_VGG16 = [(3, 64), (64, 64, 2),
                  (64, 128), (128, 128, 2),
                  (128, 256), (256, 256), (256, 256, 2),
                  (256, 512), (512, 512), (512, 512, 2),
                  (512, 512), (512, 512), (512, 512, 2)]

    cfgs_VGG19 = [(3, 64), (64, 64, 2),
                  (64, 128), (128, 128, 2),
                  (128, 256), (256, 256), (256, 256), (256, 256, 2),
                  (256, 512), (512, 512), (512, 512), (512, 512, 2),
                  (512, 512), (512, 512), (512, 512), (512, 512, 2)]

    cfgs_VGG_en = [(3, 64), (64, 64), (64, 64, 2),
                  (64, 128), (128, 128), (128, 128, 2),
                  (128, 256), (256, 256), (256, 256), (256, 256), (256, 256), (256, 256, 2),
                  (256, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 512, 2),
                  (512, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 512, 2)]

    def __init__(self, num_classes=10, which='VGG16', alpha = 1.0):
        """
        Initialize VGG Model as argument configurations.
        
        :param num_classes: number of classification labels
        :param which: choose a model architecture from VGG16/VGG19/MobileNet
        """

        super(VGG, self).__init__()
        self.alpha = alpha
        self.conv = nn.Sequential()

        self.layers = self._make_layers(which)

        self.avgPooling = nn.AvgPool2d(2, 2)

        output_size = int(512*self.alpha)
        self.fc = nn.Sequential(
            nn.Linear(output_size, 512),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )


    def _make_layers(self, which):
        """
        Make standard-conv Model layers.
        
        :param which: choose a model architecture from VGG16/VGG19/MobileNet
        :return: nn.Sequential(*layers): vgg layers
        """

        layers = []
        if which == 'VGG16':
            self.cfgs = self.cfgs_VGG16
        elif which == 'VGG19':
            self.cfgs = self.cfgs_VGG19
        else:
            pass

        for cfg in self.cfgs:

            if cfg[0] == 3:
                layers.append(nn.Conv2d(int(cfg[0]), int(cfg[1]*self.alpha), kernel_size=3, stride=1, padding=1))
            else:
                layers.append(nn.Conv2d(int(cfg[0]*self.alpha), int(cfg[1]*self.alpha), kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(int(cfg[1]*self.alpha)))
            layers.append(nn.ReLU())
            if len(cfg) == 3:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Run forward propagation
        
        :param x: input feature maps
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
        :param bn: whether add batch normalization after FALCON
        :param relu: whether add ReLU function after FALCON
        :param groups: number of groups for pointwise convolution
        """
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], nn.Conv2d):
                compress = GEPdecompose(self.layers[i], rank, init, \
                        alpha=alpha, bn=bn, relu=relu, groups=groups)
                self.layers[i] = compress
            if isinstance(self.layers[i], nn.BatchNorm2d):
                device = self.layers[i].weight.device
                self.layers[i] = nn.BatchNorm2d(self.layers[i].num_features).to(device)
