#!/bin/bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# FALCON: Lightweight and Accurate Convolution
#
# File: scripts/demo.sh
#  - Test trained model
#  - Trained model saved in ../train_test/trained_model.
#
# Version: 1.0
#==========================================================================================
cd ../src/train_test

CUDA_VISIBLE_DEVICES=0 python main.py -m ResNet -conv FALCON -init -data cifar100 -al 0.77
