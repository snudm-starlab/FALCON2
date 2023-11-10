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

File: train_test/train.py
 - Contain training code for execution for model.

Version: 1.0
"""

import time
import sys
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.optimizer_option import get_optimizer
from utils.load_data import load_cifar100, load_svhn
from utils.lr_decay import adjust_lr
from train_test.validation import validation

def train(net,
          learning_rate,
          log=None,
          optimizer_option='SGD',
          data='cifar100',
          epochs=350,
          batch_size=128,
          is_train=True,
          net_st=None,
          beta=0.0,
          lrd=10):
    """
    Train a model.
    
    :param net: model to be trained
    :param learning_rate: learning rate
    :param optimizer_option: type of optimizer
    :param data: datasets used to train
    :param epochs: number of training epochs
    :param batch_size: batch size
    :param is_train: Whether it is a training process
    :param net_st: uncompressed model
    :param beta: transfer parameter
    :return: best_param: the parameters of the model that achieves the best accuracy
    """

    net.train()
    if net_st is not None:
        net_st.eval()

    if data == 'cifar100':
        trainloader = load_cifar100(is_train, batch_size)
        valloader = load_cifar100(False, batch_size)
    elif data == 'svhn':
        trainloader = load_svhn(is_train, batch_size)
        valloader = load_svhn(False, batch_size)
    else:
        sys.exit()

    # Cross entropy loss for classification
    criterion = nn.CrossEntropyLoss()
    # MSE loss for approximation
    criterion_mse = nn.MSELoss()
    # Get optimizer according to your optimizer_option
    optimizer = get_optimizer(net, learning_rate, optimizer_option)

    start_time = time.time()
    last_time = 0

    best_acc = 0
    best_param = net.state_dict()

    iteration = 0
    for epoch in range(epochs):
        print(f"{'*'*18} EPOCH = {epoch} {'*'*18}")
        if log is not None:
            log.write(f"{'*'*18} EPOCH = {epoch} {'*'*18}")

        total = 0
        correct = 0
        loss_sum = 0

        # Change learning rate
        if epoch in (150, 250):
            learning_rate = adjust_lr(learning_rate, lrd=lrd, log=log)
            optimizer = get_optimizer(net, learning_rate, optimizer_option)

        for _, batch in enumerate(trainloader, 0):
            iteration += 1

            # Foward
            inputs, labels = batch
            inputs_var, labels_var = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs, outputs_conv = net(inputs_var)
            loss = criterion(outputs, labels_var)
            if net_st is not None:
                _, outputs_st_conv = net_st(inputs_var)
                for idx, module in enumerate(outputs_st_conv):
                    if idx != (len(outputs_st_conv)-1):
                        loss += beta / 50 * criterion_mse(outputs_conv[idx], module.detach())
                    else:
                        loss += beta * criterion_mse(outputs_conv[idx], module.detach())

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(F.softmax(outputs, -1), 1)
            total += labels_var.size(0)
            correct += (predicted == labels_var).sum()
            loss_sum += loss

            if iteration % 100 == 99:
                now_time = time.time()
                print(f"accuracy: {float(100) * float(correct) / float(total):.3f}%; \
                    loss: {loss:.3f}; time: {now_time - last_time:.3f}s")
                # if log is not None:
                #     log.write(f"accuracy: {float(100) * float(correct) / float(total):.3f} %; \
                #         loss: {loss:.3f}; time: {now_time - last_time:.3f}s\n")

                total = 0
                correct = 0
                loss_sum = 0
                last_time = now_time

        # Validation and save the best model
        net.eval()
        val_acc = validation(net, valloader, log)
        net.train()
        if val_acc >= best_acc:
            best_acc = val_acc
            # Store the current parameters in the parameters of the best model
            best_param = copy.deepcopy(net.state_dict())

    print(f"Training finished. It took {time.time() - start_time}s in total")
    if log is not None:
        log.write(f"Training finished. It took {time.time() - start_time}s in total\n")

    return best_param
