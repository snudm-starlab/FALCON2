FALCON2: Lightweight and Accurate Convolution
===

This package provides implementations of FALCON2/FALCONBranch convolution with their corresponding CNN model.

## Overview
#### Code structure
``` unicode
FALCON2
  │ 
  ├── src
  │    │     
  │    ├── models
  │    │    ├── vgg.py: VGG model
  │    │    ├── resnet.py: ResNet model
  │    │    ├── model_imagenet.py: Pretrained model (from pytorch) 
  │    │    ├── falcon.py: FALCON
  │    │    └── stconv_branch.py: StConvBranch & FALCONBranch
  │    │      
  │    ├── train_test
  │    │    ├── imagenet.py: train/validate on ImageNet 
  │    │    ├── main.py: train/test on CIFAR10/CIFAR100/SVHN 
  │    │    ├── train.py: training process
  │    │    ├── test.py: testing process
  │    │    └── validation.py: validation process
  │    ├── imagenetutils: this code are from https://github.com/d-li14/mobilenetv2.pytorch
  │    │    ├── dataloaders.py: dataloader for ImageNet 
  │    │    ├── eval.py: evaluate function for ImageNet 
  │    │    ├── logger.py: logger functions 
  │    │    ├── misc.py: helper function
  │    │    └── visualize.py: visualization functions
  │    │     
  │    └── utils
  │         ├── compression_cal.py: calculate the number of parameters and FLOPs
  │         ├── default_param.py: default cfgs 
  │         ├── load_data.py: load datasets
  │         ├── lr_decay.py: control learning rate
  │         ├── optimizer_option.py: choose optimizer 
  │         ├── save_restore.py: save and restore trained model
  │         └── timer.py: timer for inference time
  │
  └── script: shell scripts for execution of training/testing codes
```

#### Naming convention
**StandardConv**: Standard Convolution (baseline)

**FALCON2**: Lightweight and Accurate Convolution - the new convolution architecture we proposed

**Rank**: Rank of convolution. Copy the conv layer for k times, run independently and add output together at the end of the layer. This hyperparameter helps balace compression rate/ accelerate rate and accuracy.

**FALCON-branch**: New version of FALCON - for fitting FALCON into ShuffleUnitV2 architecture.

#### Data description
* CIFAR-100 datasets
* SVHN
* ImageNet
* Note that: 
    * CIFAR and SVHN datasets depend on torchvision (https://pytorch.org/docs/stable/torchvision/datasets.html#cifar). You don't have to download anything. When executing the source code, the datasets will be automaticly downloaded if it is not detected.
    * ImageNet is downloaded from http://www.image-net.org/challenges/LSVRC/

#### Output
* For CIFAR datasets, the trained model will be saved in `train_test/trained_model/` after training.
* For ImageNet, the checkpoint will be saved in `train_test/checkpoint`
* You can test the model only if there is a trained model in `train_test/trained_model/`.

## Install
#### Environment 
* Unbuntu
* CUDA 10.1
* Python 3.6
* torch
* torchvision
* [DALI (NVIDIA Data Loading Library)](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html)
#### Dependence Install
    pip install torch torchvision

## How to use 
### CIFAR100/SVHN
#### Training & Testing
* To train the model on CIFAR-100/SVHN datasets, run script:
    ```    
    cd src/train_test
    python main.py -train -conv StandardConv -m VGG19 -data cifar100
    python main.py -train -conv FALCON -m VGG19 -data cifar100 -init
    ```
    The trained model will be saved in `src/train_test/trained_model/`
* To test the model, run script:
    ```
    cd src/train_test
    python main.py -conv StandardConv -m VGG19 -data cifar100
    python main.py -conv FALCON -m VGG19 -data cifar100 -init
    ```
    The testing accuracy, inference time, number of parameters and number of FLOPs will be printed on the screen.
* Pre-trained model is saved in `src/train_test/trained_model/`
    * For example:
        * Standard model:
            conv=StandardConv,model=VGG19,data=cifar100,rank=1,alpha=1,opt=SGD,lr=0.1.pkl
        * FALCON model:
            conv=FALCON,model=VGG19,data=cifar100,rank=1,alpha=1,init,opt=SGD,lr=0.1.pkl

#### DEMO
* There are two demo scripts: `script/train.sh` and `script/inference.sh`.
* You can change arguments in `.sh` files to train/test different model.
    * `train.sh`: Execute training process of the model
        * Accuracy/ loss/ training time for 100 iteration will be printed on the screen during training.
        * Accuracy/ inference time/ number of parameters/ number of FLOPs will be printed on the screen after training.
    * `inference.sh`: Execute inference process of the model
        * Accuracy/ inference time/ number of parameters/ number of FLOPs will be printed on the screen.
        * You can run this file only when the trained model exist.
        * Sample trained model is provided in `src/train_test/trained_model/`.
        
### ImageNet
#### Training & Testing
* To train the model on ImageNet dataset, run script:
    ```    
    python imagenet.py \
    -a <architecture> \
    -conv <convolution> \
    -b <batch-size> \
    -init \
    -c <checkpoint directory> \
    --pretrained \
    --epochs <number of epochs> \
    --lr-decay <decay strategey> \
    --lr <learning rate> \
    data directory
    ```
    The trained model will be saved in `src/checkpoint directory/`
    
    Note: dataloader function is implemented based on DALI library; hence, you install DALI library before training a model with ImageNet dataset. Our dataloader function is from https://github.com/d-li14/mobilenetv2.pytorch
* To test the modelon ImageNet dataset, run script:
    ```    
    python imagenet.py \
    -a <architecture> \
    -e \
    -conv <convolution> \
    -b <batch-size> \
    -init \
    --resume <model location> \
    data directory
    ```
    The testing accuracy, inference time, number of parameters and number of FLOPs will be printed on the screen.
* Pre-trained model is saved in `src/checkpoint directory/` for ImageNet dataset
    * For example:
        * FALCON model:
            checkpoints/model_best.pth.tar

#### DEMO
* There are four demo scripts: `script/train.sh`, `script/inference.sh`, `script/imagenet_vgg_train.sh`, and `script/imagenet_vgg_test.sh`.
* You can change arguments in `.sh` files to train/test different model.
    * `imagenet_vgg_train.sh`: Execute training process of vgg model for ImageNet
    * `imagenet_vgg_test.sh`: Execute inference process of vgg model for ImageNet
        * Sample trained model is provided in `src/checkpoints/`.

## License
Licensed under the Apache License, Version 2.0
