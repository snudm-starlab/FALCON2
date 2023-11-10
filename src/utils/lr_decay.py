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

File: utils/lr_decay.py
 - Contain source code for updating learning rate.

Version: 1.0
"""

def adjust_lr(learning_rate, lrd=10, log=None):
    """
    Update learning rate.
    
    :param learning_rate: original learning rate.
    :param lrd: decrease ratio of learning rate.
    :param log: if log is not None, print the comments for changing learning rate.
    :return lr: adjusted learning rate.
    """
    learning_rate = learning_rate / lrd
    print(f"learning rate change to {learning_rate}")
    if log is not None:
        log.write(f"learning rate change to {learning_rate}\n")
    return learning_rate
