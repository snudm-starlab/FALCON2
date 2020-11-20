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

# pylint: disable=C0103,R0912,R0913,R0914,R0915,R1704,C0200,W0621,E1101
import time
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from utils.optimizer_option import get_optimizer
from utils.load_data import load_cifar100, load_svhn
from utils.lr_decay import adjust_lr
from train_test.validation import validation



def train(net,
          lr,
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
    Train the model.
    :param net: model to be trained
    :param lr: learning rate
    :param optimizer_option: optimizer type
    :param data: datasets used to train
    :param epochs: number of training epochs
    :param batch_size: batch size
    :param is_train: whether it is a training process
    :param net_st: uncompressed model
    :param beta: transfer parameter
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

    criterion = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    optimizer = get_optimizer(net, lr, optimizer_option)

    start_time = time.time()
    last_time = 0

    best_acc = 0
    best_param = net.state_dict()

    iteration = 0
    for epoch in range(epochs):
        print("****************** EPOCH = %d ******************" % epoch)
        if log is not None:
            log.write("****************** EPOCH = %d ******************\n" % epoch)

        total = 0
        correct = 0
        loss_sum = 0

        # change learning rate
        if epoch in (150, 250):
            lr = adjust_lr(lr, lrd=lrd, log=log)
            optimizer = get_optimizer(net, lr, optimizer_option)

        for i, data in enumerate(trainloader, 0):
            iteration += 1

            # foward
            inputs, labels = data
            inputs_var, labels_var = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs, outputs_conv = net(inputs_var)
            loss = criterion(outputs, labels_var)
            if net_st is not None:
                _, outputs_st_conv = net_st(inputs_var)
                for i in range(len(outputs_st_conv)):
                    if i != (len(outputs_st_conv)-1):
                        loss += beta / 50 * criterion_mse(outputs_conv[i], \
                                outputs_st_conv[i].detach())
                    else:
                        loss += beta * criterion_mse(outputs_conv[i], \
                                outputs_st_conv[i].detach())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(F.softmax(outputs, -1), 1)
            total += labels_var.size(0)
            correct += (predicted == labels_var).sum()
            loss_sum += loss

            if iteration % 100 == 99:
                now_time = time.time()
                print('accuracy: %f %%; loss: %f; time: %ds'
                      % ((float(100) * float(correct) / float(total)), loss, \
                          (now_time - last_time)))
                if log is not None:
                    log.write('accuracy: %f %%; loss: %f; time: %ds\n'
                              % ((float(100) * float(correct) / float(total)), loss, \
                                  (now_time - last_time)))

                total = 0
                correct = 0
                loss_sum = 0
                last_time = now_time

        # validation
        net.eval()
        val_acc = validation(net, valloader, log)
        net.train()
        if val_acc > best_acc:
            best_acc = val_acc
            # Store the current parameters in the parameters of the best model
            best_param = copy.deepcopy(net.state_dict())

    print('Finished Training. It took %ds in total' % (time.time() - start_time))
    if log is not None:
        log.write('Finished Training. It took %ds in total\n' % (time.time() - start_time))

    return best_param
