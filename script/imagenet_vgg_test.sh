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
# File: scripts/imagenet_vgg_test.sh
#  - Test trained vgg model for imagenet dataset
#
# Version: 1.0
#==========================================================================================
cd ../src/train_test

python imagenet.py \
    -a vgg16_bn \
    -e \
	-conv FALCON \
    -b 96 \
	-init \
    --resume ../checkpoints/model_best.pth.tar \
     /data/ImageNet_data/
