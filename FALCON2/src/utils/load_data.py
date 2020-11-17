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

File: utils/load_data.py
 - Contain source code for loading data.

Version: 1.0
"""
import sys
sys.path.append('../')
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import os, shutil, random



# CIFAR100 data
def load_cifar100(is_train=True, batch_size=128):
    """
    Load cifar-100 datasets.
    :param is_train: if true, load train_test/val data; else load test data.
    :param batch_size: batch_size of train_test data
    """

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    if is_train:
        # dataset
        trainset = torchvision.datasets.CIFAR100(root='./data',
                                                 train=True,
                                                 download=True,
                                                 transform=transform_train)

        # dataloader
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  # sampler=train_sampler,
                                                  num_workers=2,
                                                  shuffle=True)
        # valloader = torch.utils.data.DataLoader(valset,
        #                                         batch_size=batch_size,
        #                                         sampler=valid_sampler,
        #                                         num_workers=2)
        return trainloader #, valloader

    else:
        testset = torchvision.datasets.CIFAR100(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=2)
        return testloader


# SVHN data
def load_svhn(is_train=True, batch_size=128):
    """
    Load SVHN datasets.
    :param is_train: if true, load train_test/val data; else load test data.
    :param batch_size: batch_size of train_test data
    """

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    if is_train:
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=2)
        return trainloader

    else:
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=2)
        return testloader



# imagenet
def load_imagenet(is_train=True, batch_size=128):
    """
        Load imagenet datasets.
        :param is_train: if true, load train_test/val data; else load test data.
        :param batch_size: batch_size of train_test data
    """
    # path
    data_path = './data/ImageNet_data/'

    # transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if is_train:
        # path
        train_dir = data_path + 'train/'
        val_dir = data_path + 'val/'

        # transforms
        transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transforms_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        # dataset
        train_set = datasets.ImageFolder(train_dir, transforms_train)
        val_set = datasets.ImageFolder(val_dir, transforms_val)

        # dataloader
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=2)

        return train_loader, val_loader

    else:
        # path
        val_dir = data_path + 'val/'

        # transforms
        transforms_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        # dataset
        val_set = datasets.ImageFolder(val_dir, transforms_val)

        # dataloader
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=2)

        return val_loader
