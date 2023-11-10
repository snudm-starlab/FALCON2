'''
The following codes are from https://github.com/d-li14/mobilenetv2.pytorch
Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''

import errno
import os

import torch
from torch import nn
from torch.nn import init

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter']


def get_mean_and_std(dataset):
    '''
    Compute the mean and std value of dataset.
    
    :param dataset: input data
    :return: (mean, std): mean value of dataset, and standard deviation value of dataset
    '''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''
    Init layer parameters.
    
    :param net: input network to be initialized
    '''
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal(module.weight, mode='fan_out')
            if module.bias:
                init.constant(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            init.constant(module.weight, 1)
            init.constant(module.bias, 0)
        elif isinstance(module, nn.Linear):
            init.normal(module.weight, std=1e-3)
            if module.bias:
                init.constant(module.bias, 0)

def mkdir_p(path):
    '''
    make dir if not exist
    
    :param path: directory path we make
    '''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter:
    """
    Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """ reset function """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        update function
        
        :param val: input current average value
        :param n: number of items for val
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count
