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

File: utils/optimizer_option.py
 - Contain source code for choosing optimizer.

Version: 1.0
"""

import sys
import torch.optim as optim


def get_optimizer(net, lr, optimizer='SGD',  weight_decay=1e-4, momentum=0.9):
    """
    Get optimizer accroding to arguments.
    There are four optimizers, SGD, Adagrad, Adam, and RMSprop.
    :param net: the model
    :param lr: learing rate
    :param optimizer: choose an optimizer - SGD/Adagrad/Adam/RMSprop
    :param weight_decay: weight decay rate
    :param momentum: momentum of the optimizer
    """
    if optimizer == 'SGD':
        return optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                         lr=lr,
                         weight_decay=weight_decay,
                         momentum=momentum)
    elif optimizer == 'Adagrad':
        return optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()),
                             lr=lr,
                             weight_decay=weight_decay)
    elif optimizer == 'Adam':
        return optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=lr,
                          weight_decay=weight_decay,
                          )
    elif optimizer == 'RMSprop':
        return optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()),
                             lr=lr,
                             weight_decay=weight_decay,
                             momentum=momentum)
    else:
        sys.exit('Wrong Instruction! '
                 'The optimizer must be one of SGD/Adagrad/Adam/RMSprop.')
