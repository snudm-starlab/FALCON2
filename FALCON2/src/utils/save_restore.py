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

File: utils/save_restore.py
 - Contain source code for saving and restoring the model.

Version: 1.0
"""
# pylint: disable=R0912
import os
import sys
import torch


def create_log(args):
    """
    Save the training process.
    
    :param args: arguments of the trained model
    :return open: open file
    """
    name = 'conv=' + str(args.convolution) + \
           ',model=' + str(args.model) + \
           ',data=' + str(args.datasets) + \
           ',rank=' + str(args.rank) + \
           ',alpha=' + str(args.alpha)
    path = 'training_log/'
    mkdir(path)
    return open(path + name + '.txt', "w+")


def save_model(best, args, log):
    """
    Save the trained model.
    
    :param best: best trained model (to be saved)
    :param args: arguments of the trained model
    """
    name = 'conv=' + str(args.convolution) + \
           ',model=' + str(args.model) + \
           ',data=' + str(args.datasets) + \
           ',rank=' + str(args.rank) + \
           ',alpha=' + str(args.alpha)
    path = 'trained_model/'
    mkdir(path)
    torch.save(best, path + name + '.pkl')
    print("model saved in %s" % (path + name + '.pkl'))
    log.write("model saved in %s\n" % (path + name + '.pkl'))


def load_model(net, args):
    """
    Restore the pre-trained model.
    
    :param net: model architecture without parameters
    :param args: arguments of the trained model
    """
    name = 'conv=' + str(args.convolution) + \
           ',model=' + str(args.model) + \
           ',data=' + str(args.datasets) + \
           ',rank=' + str(args.rank) + \
           ',alpha=' + str(args.alpha)

    path = 'trained_model/'
    file = path + name + '.pkl'
    if os.path.exists(file):
        state_dict = torch.load(file)
        net.load_state_dict(state_dict)
        print("model restored from %s" % (file))
    else:
        print(name + '.pkl does not exist.')
        print('Testing can only be done when the trained model exists.')
        sys.exit()


def mkdir(path):
    """
    Make a directory if it doesn't exist.
    
    :param path: directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def load_specific_model(net, args, convolution='', input_path=''):
    """
    Restore the pre-trained model.
    
    :param net: model architecture without parameters
    :param args: arguments for a current model
    :param convolution: kinds of convolution
    :param input_path: path of a stored model
    """
    if convolution != '':
        name = 'conv=' + str(convolution)
    else:
        name = 'conv=' + str(args.convolution)
    name += ',model=' + str(args.model) + \
            ',data=' + str(args.datasets)
    if args.convolution != 'StandardConv' and convolution != 'StandardConv':
        name += ',rank=' + str(args.rank)
    else:
        name += ',rank=' + str(1)

    if convolution == 'StandardConv':
        name += ',alpha=' + str(1)
    else:
        name += ',alpha=' + str(args.alpha)
    if args.convolution == 'FALCON' and convolution == '' and args.init:
        name += ',init'
    if args.convolution == 'FALCONBranch' and convolution == '' and args.init:
        name += ',init'
    if args.convolution == 'FALCON' and convolution == '' and args.beta != 0:
        name += ',beta='
        name += str(args.beta)
    if args.convolution == 'FALCON' and convolution != 'StandardConv' and args.groups != 1:
        name += ',groups='
        name += str(args.groups)
    if args.convolution == 'MobileConvV2':
        name += ',exp='
        name += str(args.expansion)
    if args.convolution == 'ShuffleUnit':
        name += ',groups='
        name += str(args.groups)
    name += ',opt='
    name += str(args.optimizer)
    name += ',lr='
    name += str(args.learning_rate)

    path = 'trained_model/'
    file = path + name + '.pkl'
    if input_path != '':
        file = input_path
    if os.path.exists(file):
        state_dict = torch.load(file)
        net.load_state_dict(state_dict)
        print("model restored from %s" % (file))
    else:
        print(file + 'does not exist.')
        sys.exit()


def save_specific_model(best, args, convolution=''):
    """
    Save the trained model.
    
    :param best: best trained model (to be saved)
    :param args: arguments for the trained model
    """
    if convolution != '':
        name = 'conv=' + str(convolution)
    else:
        name = 'conv=' + str(args.convolution)
    name += ',model=' + str(args.model) + \
            ',data=' + str(args.datasets) + \
            ',rank=' + str(args.rank) + \
            ',alpha=' + str(args.alpha)
    if args.init:
        name += ',init'
    if args.convolution != 'StandardConv' and args.beta != 0:
        name += ',beta='
        name += str(args.beta)
    if args.convolution == 'FALCON' and args.groups != 1:
        name += ',groups='
        name += str(args.groups)
    if args.convolution == 'MobileConvV2':
        name += ',exp='
        name += str(args.expansion)
    if args.convolution == 'ShuffleUnit':
        name += ',groups='
        name += str(args.groups)

    name += ',opt='
    name += str(args.optimizer)
    name += ',lr='
    name += str(args.learning_rate)

    path = 'trained_model/'
    mkdir(path)
    torch.save(best, path + name + '.pkl')
    print("model saved in %s" % (path + name + '.pkl'))

def init_with_alpha_resnet(source_net, dest_net, alpha):
    """
    Initialize with width multiplier for resnet.
    
    :param source_net: a base model
    :param dest_net: a model to be compressed
    :param alpha: width multiplier.
                  if alpha is less than 1,
                  a part of standard convolution kernel
                  is used for initializing kernels of FALCON.
    :param dest_net: a model to be compressed
    """
    for i in range(len(source_net.first)):
        if isinstance(source_net.first[i], torch.nn.Conv2d):
            shape = source_net.first[i].weight.shape
#            last_conv = i
            if i == 0:
                dest_net.first[i].weight = torch.nn.Parameter(\
                        source_net.first[i].weight[:int(shape[0]*alpha), :shape[1], :,:])
            else:
                dest_net.first[i].weight = torch.nn.Parameter(\
                        source_net.first[i].weight[:int(shape[0]*alpha), :int(shape[1]*alpha), :,:])

    for i in range(len(source_net.residuals)):
        for j in range(len(source_net.residuals[i].stacked.conv)):
            if isinstance(source_net.residuals[i].stacked.conv[j], torch.nn.Conv2d):
                shape = source_net.residuals[i].stacked.conv[j].weight.shape
                dest_net.residuals[i].stacked.conv[j].weight = torch.nn.Parameter(\
                        source_net.residuals[i].stacked.conv[j].weight[:int(shape[0]*alpha),\
                        :int(shape[1]*alpha), :,:])

        for j in range(len(source_net.residuals[i].shortcut)):
            if isinstance(source_net.residuals[i].shortcut[j], torch.nn.Conv2d):
                shape = source_net.residuals[i].shortcut[j].weight.shape
                dest_net.residuals[i].shortcut[j].weight = torch.nn.Parameter(\
                        source_net.residuals[i].shortcut[j].weight[:int(shape[0]*alpha),\
                        :int(shape[1]*alpha), :,:])

    return dest_net

def init_with_alpha_vgg(source_net, dest_net, alpha):
    """
    Initialize with width multiplier for vgg.
    
    :param source_net: a base model
    :param dest_net: a model to be compressed
    :param alpha: width multiplier.
                  if alpha is less than 1,
                  a part of standard convolution kernel
                  is used for initializing kernels of FALCON.
    :param dest_net: a model to be compressed
    """

    for i in range(len(source_net.first)):
        if isinstance(source_net.first[i], torch.nn.Conv2d):
            shape = source_net.first[i].weight.shape
#            last_conv = i
            if i == 0:
                dest_net.first[i].weight = torch.nn.Parameter(\
                        source_net.first[i].weight[:int(shape[0]*alpha), :shape[1], :,:])
            else:
                dest_net.first[i].weight = torch.nn.Parameter(\
                        source_net.first[i].weight[:int(shape[0]*alpha), :int(shape[1]*alpha), :,:])

    for i in range(len(source_net.residuals)):
        for j in range(len(source_net.residuals[i].stacked.conv)):
            if isinstance(source_net.residuals[i].stacked.conv[j], torch.nn.Conv2d):
                shape = source_net.residuals[i].stacked.conv[j].weight.shape
                dest_net.residuals[i].stacked.conv[j].weight = torch.nn.Parameter(\
                        source_net.residuals[i].stacked.conv[j].weight[:int(shape[0]*alpha),\
                        :int(shape[1]*alpha), :,:])

        for j in range(len(source_net.residuals[i].shortcut)):
            if isinstance(source_net.residuals[i].shortcut[j], torch.nn.Conv2d):
                shape = source_net.residuals[i].shortcut[j].weight.shape
                dest_net.residuals[i].shortcut[j].weight = torch.nn.Parameter(\
                        source_net.residuals[i].shortcut[j].weight[:int(shape[0]*alpha),\
                        :int(shape[1]*alpha), :,:])

    return dest_net
