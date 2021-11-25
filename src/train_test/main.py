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

File: train_test/main.py
 - receive arguments and train/test the model.

Version: 1.0
"""
# pylint: disable=redefined-outer-name,C0103,R0912,R0915,E1101
# pylint: disable=wrong-import-position

import sys
sys.path.append('../')
import torch
from torch.autograd import Variable
from models.vgg import VGG
from models.resnet import ResNet
from models.stconv_branch import VGG_StConv_branch, ResNet_StConv_branch
from utils.timer import Timer
from utils.default_param import get_default_param
from utils.save_restore import save_specific_model, load_specific_model, \
    init_with_alpha_resnet, init_with_alpha_vgg
from utils.compression_cal import print_model_parm_nums, print_model_parm_flops
from train_test.train import train
from train_test.test import test




def main(args):
    '''
    main function of FALCON2
    :param args: arguments for a model
    '''

    # Choose dataset
    if args.datasets == "svhn":
        num_classes = 10
    elif args.datasets == "cifar100":
        num_classes = 100
    else:
        pass

    # Choose model ResNet
    if "ResNet" in args.model:
        if args.convolution == "FALCON":
            net = ResNet(layer_num=str(args.layer_num), num_classes=num_classes)
            if args.is_train:
                if args.alpha == 1:
                    if args.init:
                        load_specific_model(net, args, convolution='StandardConv',
                                            input_path=args.stconv_path)
                        net.falcon(rank=args.rank, init=args.init, bn=args.bn,
                                   relu=args.relu, groups=args.groups)
                    else:
                        net.falcon(rank=args.rank, init=False, bn=args.bn,
                                   relu=args.relu, groups=args.groups)
                else:
                    if args.init:
                        net2 = ResNet(layer_num=str(args.layer_num), num_classes=num_classes)
                        net = ResNet(layer_num=str(args.layer_num), num_classes=num_classes,
                                     alpha=args.alpha)
                        load_specific_model(net2, args, convolution='StandardConv',
                                            input_path=args.stconv_path)
                        net = init_with_alpha_resnet(net2, net, args.alpha)
                        net.falcon(rank=args.rank, init=args.init, bn=args.bn,
                                   relu=args.relu, groups=args.groups)
                    else:
                        net = ResNet(layer_num=str(args.layer_num), num_classes=num_classes,
                                     alpha=args.alpha)
                        net.falcon(rank=args.rank, init=False, bn=args.bn, relu=args.relu,
                                   groups=args.groups)
            else:
                if args.alpha == 1:
                    net = ResNet(layer_num=str(args.layer_num), num_classes=num_classes,
                                 alpha=args.alpha)
                    net.falcon(rank=args.rank, init=False, bn=args.bn, relu=args.relu,
                               groups=args.groups)
                else:
                    net = ResNet(layer_num=str(args.layer_num), num_classes=num_classes,
                                 alpha=args.alpha)
                    net.falcon(rank=args.rank, init=False, bn=args.bn, relu=args.relu,
                               groups=args.groups)
        elif args.convolution == "StConvBranch":
            net = ResNet_StConv_branch(layer_num=str(args.layer_num), num_classes=num_classes,
                                       alpha=args.alpha)
        elif args.convolution == 'FALCONBranch':
            net = ResNet_StConv_branch(layer_num=str(args.layer_num), num_classes=num_classes,
                                       alpha=args.alpha)
            if args.init:
                load_specific_model(net, args, convolution='StConvBranch',
                                    input_path=args.stconv_path)
            net.falcon(rank=args.rank, init=False, bn=args.bn,
                       relu=args.relu, groups=args.groups)
        elif args.convolution == "StandardConv":
            net = ResNet(layer_num=str(args.layer_num), num_classes=num_classes)
        else:
            pass
    # Choose model VGG
    elif "VGG" in args.model:
        if args.convolution == "FALCON":
            net = VGG(num_classes=num_classes, which=args.model)
            if args.is_train:
                if args.alpha == 1:
                    if args.init:
                        load_specific_model(net, args, convolution='StandardConv',
                                            input_path=args.stconv_path)
                        net.falcon(rank=args.rank, init=args.init, bn=args.bn,
                                   relu=args.relu, groups=args.groups)
                    else:
                        net.falcon(rank=args.rank, init=False, bn=args.bn,
                                   relu=args.relu, groups=args.groups)
                else:
                    if args.init:
                        net2 = VGG(num_classes=num_classes, which=args.model)
                        net = VGG(num_classes=num_classes, which=args.model,
                                  alpha=args.alpha)
                        load_specific_model(net2, args, convolution='StandardConv',
                                            input_path=args.stconv_path)
                        net = init_with_alpha_vgg(net2, net, args.alpha)
                        net.falcon(rank=args.rank, init=args.init, bn=args.bn,
                                   relu=args.relu, groups=args.groups)
                    else:
                        net = VGG(num_classes=num_classes, which=args.model,
                                  alpha=args.alpha)
                        net.falcon(rank=args.rank, init=False, bn=args.bn,
                                   relu=args.relu, groups=args.groups)
            else:
                if args.alpha == 1:
                    net = VGG(num_classes=num_classes, which=args.model,
                              alpha=args.alpha)
                    net.falcon(rank=args.rank, init=False, bn=args.bn,
                               relu=args.relu, groups=args.groups)
                else:
                    net = VGG(num_classes=num_classes, which=args.model,
                              alpha=args.alpha)
                    net.falcon(rank=args.rank, init=False, bn=args.bn,
                               relu=args.relu, groups=args.groups)

        elif args.convolution == 'StConvBranch':
            net = VGG_StConv_branch(num_classes=num_classes, which=args.model,
                                    alpha=args.alpha)
        elif args.convolution == 'FALCONBranch':
            net = VGG_StConv_branch(num_classes=num_classes, which=args.model,
                                    alpha=args.alpha)
            if args.init:
                load_specific_model(net, args, convolution='StConvBranch',
                                    input_path=args.stconv_path)
            net.falcon(rank=args.rank, init=args.is_train, bn=args.bn,
                       relu=args.relu, groups=args.groups)
        elif args.convolution == "StandardConv":
            net = VGG(num_classes=num_classes, which=args.model)
        else:
            pass
    else:
        pass

    net = net.cuda()

    print_model_parm_nums(net)
    print_model_parm_flops(net)

    if args.is_train:
        # training
        best = train(net,
                     lr=args.learning_rate,
                     optimizer_option=args.optimizer,
                     epochs=args.epochs,
                     batch_size=args.batch_size,
                     is_train=args.is_train,
                     data=args.datasets,
                     lrd=args.lr_decay_rate)
        if not args.not_save:
            save_specific_model(best, args)
        test(net, batch_size=args.batch_size, data=args.datasets)
    else:
        # testing
        load_specific_model(net, args, input_path=args.restore_path)
        inference_time = 0
        inference_time += \
            test(net, batch_size=args.batch_size, data=args.datasets)
#        print("Average Inference Time: %f" % (float(inference_time) / float(1)))

    # calculate number of parameters & FLOPs
    print_model_parm_nums(net)
    print_model_parm_flops(net)

    # time of forwarding 100 data sample (ms)
    x = torch.rand(100, 3, 32, 32)
    x = Variable(x.cuda())
    net(x)
    timer = Timer()
    timer.tic()
    for _ in range(100):
        net(x)
    timer.toc()
#    print('Do once forward need %.3f ms.' % (timer.total_time * 1000 / 100.0))


if __name__ == "__main__":
    parser = get_default_param()
    args = parser.parse_args()

    # print configuration
    print(args)

    main(args)
