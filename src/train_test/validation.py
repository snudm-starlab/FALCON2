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

File: train_test/validation.py
 - Contain validation code for execution for model.

Version: 1.0
"""
# pylint: disable=E1101, R0902, R0913, R0914
import time
import torch
import torch.nn.functional as F



def validation(net, val_loader, log=None):
    """
    Validation process.
    
    :param net: model to be validated
    :param val_loader: data loader for validation
    :param log: log dir
    :return: accuracy: accuracy for validation dataset
    """

    # Set testing mode
    net.eval()

    correct = 0
    total = 0
    inference_start = time.time()
    with torch.no_grad():
        for _, data in enumerate(val_loader, 0):
            inputs, labels = data
            outputs, _ = net(inputs.cuda())
            _, predicted = torch.max(F.softmax(outputs, -1), 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum()
    inference_time = time.time() - inference_start
    accuracy = float(100) * float(correct) / float(total)

    print(f"{'*'*15} Validation {'*'*15}")
    print(f"Accuracy of the network validation images: {accuracy} %")
    print(f"Validation time is: {inference_time:.3f}s")

    if log is not None:
        log.write(f"{'*'*15} Validation {'*'*15}")
        log.write(f"Accuracy of the network validation images: {accuracy} %")
        log.write(f"Validation time is: {inference_time:.2f}s\n")

    return accuracy
