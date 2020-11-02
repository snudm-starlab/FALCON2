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

File: train_test/test.py
 - test the pre-trained model on test datasets.
 - print the test accuracy and inference time.

Version: 1.0
"""
import torch
import torch.nn.functional as F

import time

from utils.load_data import load_cifar100, load_svhn


def test(net, log=None, batch_size=128, data='cifar100'):
    """
    Test on trained model.
    :param net: model to be tested
    :param log: log dir
    :param batch_size: batch size
    :param data: datasets used
    """

    net.eval()
    is_train = False

    # data
    if data == 'cifar100':
        test_loader = load_cifar100(is_train, batch_size)
    elif data == 'svhn':
        test_loader = load_svhn(is_train, batch_size)
    else:
        exit()

    correct = 0
    total = 0
    inference_start = time.time()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            outputs, outputs_conv = net(inputs.cuda())
            _, predicted = torch.max(F.softmax(outputs, -1), 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum()
    inference_time = time.time() - inference_start
    print('Accuracy: %f %%; Inference time: %fs' % (float(100) * float(correct) / float(total), inference_time))

    if log != None:
        log.write('Accuracy of the network on the 10000 test images: %f %%\n' % (float(100) * float(correct) / float(total)))
        log.write('Inference time is: %fs\n' % inference_time)

    return inference_time
