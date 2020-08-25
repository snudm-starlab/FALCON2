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

File: utils/default_param.py
 - Contain source code for receiving arguments .

Version: 1.0
"""

import argparse


def get_default_param():
    """
    Receive arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-train", "--is_train",
                        help="whether train_test the model (train_test-True; test-False)",
                        action="store_true")

    parser.add_argument("-bs", "--batch_size",
                        help="batch size of training",
                        type=int,
                        default=128)

    parser.add_argument("-ep", "--epochs",
                        help="epochs of training",
                        type=int,
                        default=350)

    parser.add_argument("-lr", "--learning_rate",
                        help="set beginning learning rate",
                        type=float,
                        default=0.1)

    parser.add_argument("-op", "--optimizer",
                        help="choose optimizer",
                        choices=["SGD", "Adagrad", "Adam", "RMSprop"],
                        type=str,
                        default='SGD')

    parser.add_argument("-conv", "--convolution",
                        help="choose convolution",
                        choices=["StandardConv",
                                 "FALCON",
                                 "RankFALCON",
                                 "StConvBranch",
                                 "FALCONBranch"],
                        type=str,
                        default="StandardConv")

    parser.add_argument("-k", "--rank",
                        help="if the model is Rank K, the rank(k) in range {1,2,3}",
                        choices=[1, 2, 3, 4],
                        type=int,
                        default=1)

    parser.add_argument("-al", "--alpha",
                        help="Width Multiplier in range (0,1]",
                        # choices=[1, 0.75, 0.5, 0.33, 0.25],
                        type=float,
                        default=1)

    parser.add_argument("-m", "--model",
                        help="model type - VGG16/VGG19/ResNet",
                        choices=['VGG16', 'VGG19', 'ResNet'],
                        type=str,
                        default='VGG19')

    parser.add_argument("-data", "--datasets",
                        help="specify datasets - cifar100/svhn/imagenet",
                        choices=['cifar100', 'svhn', 'imagenet'],
                        type=str,
                        default='cifar100')

    parser.add_argument("-ns", "--not_save",
                        help="do not save the model",
                        action="store_true")

    parser.add_argument("-b", "--beta",
                        help="balance between classification loss and transfer loss",
                        type=float,
                        default=0.0)

    parser.add_argument('-bn', '--bn',
                        action='store_true',
                        help='add batch_normalization after FALCON')

    parser.add_argument('-relu', '--relu',
                        action='store_true',
                        help='add relu function after FALCON')

    parser.add_argument("-lrd", "--lr_decay_rate",
                        help="learning rate dacay rate",
                        type=int,
                        default=10)

    parser.add_argument("-exp", "--expansion",
                        help="expansion ration in MobileConvV2",
                        type=float,
                        default=6.0)

    parser.add_argument('-init', '--init',
                        action='store_true',
                        help='Whether initialize FALCON')

    parser.add_argument("-g", "--groups",
                        help="groups number for pointwise convolution",
                        type=int,
                        default=1)

    parser.add_argument("--stconv_path",
                        help="restore StConv model from the path",
                        type=str,
                        default='')

    parser.add_argument("--restore_path",
                        help="restore model from the path",
                        type=str,
                        default='')

    return parser
