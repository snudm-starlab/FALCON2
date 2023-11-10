"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: train_test/imagenet.py
 - receive arguments and train_test/test the model on imagenet.
 - Code is modified from https://github.com/pytorch/examples/tree/master/imagenet

Version: 1.0

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

"""

import argparse
import os
import random
import shutil
import time
import warnings
from math import cos, pi
import sys
sys.path.append('../')

import torch
from torch import nn
import torch.nn.parallel
from torch.backends import cudnn
from torch import distributed as dist
import torch.optim
from torch import multiprocessing as mp
from torch.autograd import Variable
from torchvision import models
from utils.timer import Timer
from utils.compression_cal import print_model_parm_nums, print_model_parm_flops
from models.model_imagenet import VGGImageNet, ResNetImageNet, \
    VGGImageNetINF, ResNetImageNetINF
from imagenetutils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from imagenetutils.dataloaders import get_dali_train_loader, get_dali_val_loader
from tensorboardX import SummaryWriter

model_names = sorted(name for name, value in models.__dict__.items()
    if name.islower() and not name.startswith("__")
    and callable(value))

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-decay', type=str, default='cos',
                    help='mode for learning rate decay')
parser.add_argument('--opt', '--optimizer',
                    choices=["SGD", "Adagrad", "Adam", "RMSprop"],
                    type=str,
                    default="SGD")
parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-conv', '--convolution', default='StandardConv', type=str,
                    choices=['StandardConv', 'FALCON', 'DepSepConv', 'MobileConvV2',
                    'ShuffleUnit', 'ShuffleUnitV2', 'StConvBranch', 'FALCONBranch'],
                    help='convolution type')
parser.add_argument('-init', '--init', action='store_true',
                    help='initialize model with EHP decomposed tensors')
parser.add_argument("-k", "--falconrank", type=int, default=1, choices=[1, 2, 3],
                    help="expansion ration in MobileConvV2")
parser.add_argument('-tucker', '--tucker', action='store_true',
                    help='initialize model with tucekr decomposed tensors')
parser.add_argument("-exp", "--expansion", type=float, default=6.0,
                    help="expansion ration in MobileConvV2")
parser.add_argument("-g", "--groups", type=int, default=2,
                    help="groups number in ShuffleUnit")
parser.add_argument("-al", "--alpha", type=float, default=1,
                    help="Width Multiplier in range (0,1]")
parser.add_argument('-inf', '--inference_time', action='store_true',
                    help='test inference time of the model')
parser.add_argument("-is", "--input_size", type=int, default=224, choices=[32, 64, 128, 180, 224],
                    help="Size of input data")
parser.add_argument("-in", "--input_num", type=int, default=100,
                    help="Number of input data")
parser.add_argument('--stconv_branch_model', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

BEST_ACC1 = 0

def main():
    """
    Discription: main
    """
    args = parser.parse_args()
    print(f"{'#' * 30} {args.input_num} * {args.input_size} * {args.input_size} {'#' * 30}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    """
    Discription: load model and train/test
    
    :param gpu: gpu id to use
    :param ngpus_per_node: number of gpus
    :param args: arguments from input
    """
    global BEST_ACC1
    args.gpu = gpu

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # Create a model
    if args.pretrained:
        print(f"=> using pre-trained model '{args.arch}'")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print(f"=> creating model '{args.arch}'")
        model = models.__dict__[args.arch]()

    # Re-organize the model and compress
    if 'vgg' in args.arch:
        model = VGGImageNet(model)
    elif 'resnet' in args.arch:
        model = ResNetImageNet(model)
    else:
        print('ONLY VGG like or ResNet like model is accepted!')
        sys.exit(0)

    # Count the number of parameters and flops
    print_model_parm_nums(model)
    print_model_parm_flops(model, imagenet=True)

    # Compress
    if args.convolution == 'FALCON':
        model.falcon(rank=args.falconrank, init=args.init)
    elif args.convolution == 'DepSepConv':
        model.dsc()
    elif args.convolution == 'MobileConvV2':
        model.mobileconvv2(expansion=args.expansion)
    elif args.convolution == 'ShuffleUnit':
        model.shuffleunit(groups=args.groups, alpha=args.alpha)
    elif args.convolution == 'ShuffleUnitV2':
        model.shuffleunitv2(alpha=args.alpha)
    elif args.convolution == 'StConvBranch':
        model.stconv_branch(alpha=args.alpha)
    elif args.convolution == 'FALCONBranch':
        model.stconv_branch(alpha=args.alpha)
        # model.falcon_branch(init=args.init)
    elif args.convolution == 'StandardConv' and args.tucker:
        model.compress_tucker()
    else:
        pass

    if args.convolution != 'FALCONBranch':
        print('*********** compressed model ***********')
        print(list(model.children()))

        # count the number of parameters and flops
        print_model_parm_nums(model)
        print_model_parm_flops(model, imagenet=True)

    # Time of forwarding 100 data sample (ms)
    if args.inference_time:
        dummy_input = torch.rand(args.input_num, 3, args.input_size, args.input_size)
        dummy_input = Variable(dummy_input.cuda())
        if 'vgg' in args.arch or 'alex' in args.arch:
            model = VGGImageNetINF(model).cuda()
        elif 'resnet' in args.arch:
            model = ResNetImageNetINF(model).cuda()
        else:
            pass
        timer = Timer()
        timer.tic()
        model.eval()
        for _ in range(1000):
            model(dummy_input)
        timer.toc()
        print(f'Do once forward need {timer.total_time * 1000 / 1000.0:.3f} ms.')
        sys.exit(0)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # print the model
    # print('********** model **********')
    # print(list(model.children()))

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    weight_decay=args.weight_decay)

    title = 'ImageNet-' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Optionally resume from a checkpoint
    if args.resume and args.convolution != 'FALCONBranch':
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            BEST_ACC1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                BEST_ACC1 = BEST_ACC1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        print('hi')


    if args.convolution == 'FALCONBranch':
        if args.stconv_branch_model:
            if os.path.isfile(args.stconv_branch_model):
                print(f"=> loading checkpoint '{args.stconv_branch_model}'")
                checkpoint = torch.load(args.stconv_branch_model)
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{args.stconv_branch_model}' \
                    (epoch {checkpoint['epoch']})")
            else:
                print(f"=> no checkpoint found at '{args.stconv_branch_model}'")
        if isinstance(model, torch.nn.DataParallel):
            model.module.falcon_branch(init=args.init)
        else:
            model.falcon_branch(init=args.init)
        model.cuda()

        print('*********** compressed model ***********')
        print(list(model.children()))

        # Count the number of parameters and flops
        print_model_parm_nums(model)
        print_model_parm_flops(model, imagenet=True)

        # Define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

        if args.opt == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        if args.opt == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                         weight_decay=args.weight_decay)
        if args.resume:
            if os.path.isfile(args.resume):
                print(f"=> loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                BEST_ACC1 = checkpoint['best_acc1']
                if args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    BEST_ACC1 = BEST_ACC1.to(args.gpu)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
                logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
            else:
                print(f"=> no checkpoint found at '{args.resume}'")
        else:
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss',\
                    'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])


    get_train_loader = get_dali_train_loader(dali_cpu=False)
    get_val_loader = get_dali_val_loader()
    train_loader, train_loader_len = get_train_loader(args.data, args.batch_size,\
            workers=args.workers)
    val_loader, val_loader_len = get_val_loader(args.data, args.batch_size, \
            workers=args.workers)

    if args.evaluate:
        inf_times = 0
        for _ in range(1):
            _, inf_time = validate(val_loader, val_loader_len, model, criterion)
            inf_times += inf_time
        print(f"\nAverage Inference Time: {float(inf_times) / 1.0}")
        return

    writer = SummaryWriter(os.path.join(args.checkpoint, 'logs'))

    for epoch in range(args.start_epoch, args.epochs):

        # Train for one epoch
        train_loss, train_acc = train(train_loader, train_loader_len, model, \
                criterion, optimizer, epoch, args)

        # Evaluate on validation set
        val_loss, prec1 = validate(val_loader, val_loader_len, model, criterion)

        learning_rate = optimizer.param_groups[0]['lr']

        # Append logger file
        logger.append([learning_rate, train_loss, val_loss, train_acc, prec1])

        # tensorboardX
        writer.add_scalar('learning rate', learning_rate, epoch + 1)
        writer.add_scalars('loss', {'train loss': train_loss, \
                'validation loss': val_loss}, epoch + 1)
        writer.add_scalars('accuracy', {'train accuracy': train_acc, \
                'validation accuracy': prec1}, epoch + 1)

        is_best = prec1 > BEST_ACC1
        BEST_ACC1 = max(prec1, BEST_ACC1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': BEST_ACC1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    writer.close()

    print('Best accuracy:')
    print(BEST_ACC1)

def train(train_loader, train_loader_len, model, criterion, optimizer, epoch, args):
    '''
    train function for imagenet
    
    :param train_loader: train data
    :param train_loader_len: length of train data
    :param model: which model we use
    :param criterion: loss function
    :param optimizer: which optimizer we use
    :param epoch: number of epochs
    :param args: arguments for training
    :return: (losses.avg, top1.avg): average loss of training, and average top1 accuracy of training
    '''
    pbar = Bar('Processing', max=train_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, i, train_loader_len, args)

        # measure data loading time
        data_time.update(time.time() - end)

        targets = targets.cuda(non_blocking=True)

        # compute output
        output, _ = model(inputs)
        loss = criterion(output, targets)

        # measure accuracy and record loss
        results = accuracy(output, targets, topk=(1, 5))
        prec1 = results[0]
        prec5 = results[1]
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        pbar.suffix  = f'({i + 1}/{train_loader_len}) Data: {data_time.avg:.3f}s | \
                        Batch: {batch_time.avg:.3f}s | Total: {pbar.elapsed_td} | \
                        ETA: {pbar.eta_td:} | Loss: {losses.avg:.4f} | \
                        top1: {top1.avg: .4f} | top5: {top5.avg: .4f}'
        pbar.next()
    pbar.finish()
    return (losses.avg, top1.avg)


def validate(val_loader, val_loader_len, model, criterion):
    '''
    validation function for our model
    
    :param val_loader: validation data
    :param val_loader_len: length of validation data
    :param model: our model to be validated
    :param criterion: loss function
    :return: (losses.avg, top1.avg): average loss of validation,
                                        and average top1 accuracy of validation
    '''
    pbar = Bar('Processing', max=val_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        targets = targets.cuda(non_blocking=True)

        with torch.no_grad():
            # Compute output
            output, _ = model(inputs)
            loss = criterion(output, targets)

        # Measure accuracy and record loss
        results = accuracy(output, targets, topk=(1, 5))
        prec1 = results[0]
        prec5 = results[1]
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Plot progress
        pbar.suffix  = f'({i + 1}/{val_loader_len}) Data: {data_time.avg:.3f}s | \
                        Batch: {batch_time.avg:.3f}s | Total: {pbar.elapsed_td:} | \
                        ETA: {pbar.eta_td:} | Loss: {losses.avg:.4f} | \
                        top1: {top1.avg: .4f} | top5: {top5.avg: .4f}'
        pbar.next()
    pbar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    '''
    this function saves check point during training
    
    :param state: current state of a model
    :param is_best: best model or not
    :param checkpoint: directory for checkpoint
    :param filename: filename for checkpoint
    '''
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch, iteration, num_iter, args):
    '''
    adjust learning rate over epochs
    
    :param optimizer: optimizer class
    :param epoch: the number of epochs
    :param iteration: current iterations in an epoch
    :param num_iter: number of iterations in an epoch
    :param args: arguments
    '''
    learning_rate = optimizer.param_groups[0]['lr']

    warmup_epoch = 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        learning_rate = args.lr * (args.gamma ** \
            ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        learning_rate = args.lr * (1 + cos(pi * \
            (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        learning_rate = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum(1 for s in args.schedule if s <= epoch)
        learning_rate = args.lr * pow(args.gamma, count)
    else:
        raise ValueError(f'Unknown lr mode {args.lr_decay}')

    if epoch < warmup_epoch:
        learning_rate = args.lr * current_iter / warmup_iter


    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

if __name__ == '__main__':
    main()
