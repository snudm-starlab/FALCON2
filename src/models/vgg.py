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

from torch import nn
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

        super().__init__()
        self.alpha = alpha
        self.conv = nn.Sequential()

        self.layers = self._make_layers(which)

        self.avg_pooling = nn.AvgPool2d(2, 2)

        output_size = int(512*self.alpha)
        self.fc_layers = nn.Sequential(
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
                layers.append(nn.Conv2d(int(cfg[0]), int(cfg[1]*self.alpha),
                                        kernel_size=3,stride=1, padding=1))
            else:
                layers.append(nn.Conv2d(int(cfg[0]*self.alpha), int(cfg[1]*self.alpha),
                                        kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(int(cfg[1]*self.alpha)))
            layers.append(nn.ReLU())
            if len(cfg) == 3:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, input_):
        """
        Run forward propagation
        
        :param input_: input feature maps
        :return: (out, out_conv): output features,
                                    and output features before the fully connected layer
        """
        out_conv = self.conv(input_)
        out_conv = self.layers(out_conv)
        if out_conv.size(2) != 1:
            out_conv = self.avg_pooling(out_conv)
        out = out_conv.view(out_conv.size(0), -1)
        out = self.fc_layers(out)
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
        for idx, module in enumerate(self.layers):
            if isinstance(module, nn.Conv2d):
                compress = GEPdecompose(module, rank, init, \
                        alpha=alpha, bn=batch_norm, relu=relu, groups=groups)
                self.layers[idx] = compress
            if isinstance(module, nn.BatchNorm2d):
                device = module.weight.device
                self.layers[idx] = nn.BatchNorm2d(module.num_features).to(device)
