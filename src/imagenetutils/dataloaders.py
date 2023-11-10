'''
The following codes are from https://github.com/d-li14/mobilenetv2.pytorch
'''

import os
import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms

DATA_BACKEND_CHOICES = ['pytorch']
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali import ops
    from nvidia.dali import types
    DATA_BACKEND_CHOICES.append('dali-gpu')
    DATA_BACKEND_CHOICES.append('dali-cpu')
except ImportError:
    print("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


class HybridTrainPipe(Pipeline):
    """
        Pipeline for training
    """
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        '''
        Init function for training pipeline
        
        :param batch_size: batch size for training
        :param num_threads: number of threads to use
        :param device_id: device id
        :param data_dir: name of data directory
        :param crop: shape of cropped image
        :param dali_cpu: whether running on cpu or not
        '''
        super().__init__(batch_size, num_threads, \
                device_id, seed = 12 + device_id)
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        self.input = ops.FileReader(
                file_root = data_dir,
                shard_id = local_rank,
                num_shards = world_size,
                random_shuffle = True)

        self.jpegs, self.labels = self.input(name = "Reader")

        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoderRandomCrop(device=dali_device, output_type=types.RGB,
                                                    random_aspect_ratio=[0.75, 4./3.],
                                                    random_area=[0.08, 1.0],
                                                    num_attempts=100)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to
            # be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.ImageDecoderRandomCrop(device="mixed", \
                    output_type=types.RGB, device_memory_padding=211025920, \
                    host_memory_padding=140544512,
                                                      random_aspect_ratio=[0.75, 4./3.],
                                                      random_area=[0.08, 1.0],
                                                      num_attempts=100)

        self.res = ops.Resize(device=dali_device, resize_x=crop,\
                resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT,
                                            output_layout = types.NCHW,
                                            crop = (crop, crop),
                                            image_type = types.RGB,
                                            mean = [0.485 * 255,0.456 * 255,0.406 * 255],
                                            std = [0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability = 0.5)

    def define_graph(self):
        """
        Define graph function
        
        :return: (output, labels): images of data, and labels of data
        """
        rng = self.coin()
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror = rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    """
    Pipeline for validation
    """
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        '''
        Init function for validation pipeline
        
        :param batch_size: batch size for training
        :param num_threads: number of threads to use
        :param device_id: device id
        :param data_dir: name of data directory
        :param crop: shape of cropped image
        :param size: size of images
        '''
        super().__init__(batch_size, num_threads,\
                device_id, seed = 12 + device_id)
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        self.input = ops.FileReader(
                file_root = data_dir,
                shard_id = local_rank,
                num_shards = world_size,
                random_shuffle = False)

        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.res = ops.Resize(device = "gpu", resize_shorter = size)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                output_dtype = types.FLOAT,
                output_layout = types.NCHW,
                crop = (crop, crop),
                image_type = types.RGB,
                mean = [0.485 * 255,0.456 * 255,0.406 * 255],
                std = [0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        """
        Define graph function
        
        :return: (output, labels): images of data, and labels of data
        """
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


class DALIWrapper:
    """
    Wrapper Class
    """
    def __init__(self, dalipipeline):
        """
        Init function for initialization
        """
        self.dalipipeline = dalipipeline

    def __iter__(self):
        """
        Iterator function
        """
        return DALIWrapper.gen_wrapper(self.dalipipeline)

    def gen_wrapper(self):
        """
        generate wrapper function
        
        :param dalipipeline: dali pipeline
        """
        for data in self.dalipipeline:
            inputs = data[0]["data"]
            targets = data[0]["label"].squeeze().cuda().long()
            yield inputs, targets
        self.dalipipeline.reset()

def get_dali_train_loader(dali_cpu=False):
    """
    DALI train loader
    
    :param dali_cpu: whether cpus is used
    :return: gdtl: output of gdtl function
    """
    def gdtl(data_path, batch_size, workers=5, _worker_init_fn=None):
        """
        DALI train loader function
        
        :param data_path: image data path
        :param batch_size: batch size in training phase
        :param workers: how much workers we use
        :param _worker_init_fn: initialize worker function
        
        :return: DALIWrapper(train_loader) or int(pipe.epoch_size("Reader")/(world_size*batch_size))
                : wrapper of train loader, or number of batch
        """
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        traindir = os.path.join(data_path, 'train')

        pipe = HybridTrainPipe(batch_size=batch_size, num_threads=workers,
                device_id = local_rank,
                data_dir = traindir, crop = 224, dali_cpu=dali_cpu)

        pipe.build()
        train_loader = DALIClassificationIterator(pipe, size \
                = int(pipe.epoch_size("Reader") / world_size))

        return DALIWrapper(train_loader), \
            int(pipe.epoch_size("Reader") / (world_size * batch_size))

    return gdtl

def get_dali_val_loader():
    """
    DALI valid loader
    
    :return: gdvl: output of gdvl function
    """
    def gdvl(data_path, batch_size, workers=5, _worker_init_fn=None):
        """
        DALI valid loader function
        
        :param data_path: image data path
        :param batch_size: batch size in validation phase
        :param workers: how much workers we use
        :param _worker_init_fn: initialize worker function
        :return: DALIWrapper(val_loader) or int(pipe.epoch_size("Reader") / (world_size*batch_size))
                : wrapper of validation loader, or number of batch
        """
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        valdir = os.path.join(data_path, 'val')

        pipe = HybridValPipe(batch_size=batch_size, num_threads=workers,
                device_id = local_rank,
                data_dir = valdir,
                crop = 224, size = 256)
        pipe.build()
        val_loader = DALIClassificationIterator(pipe, size \
                = int(pipe.epoch_size("Reader") / world_size), fill_last_batch=False)

        return DALIWrapper(val_loader), \
            int(pipe.epoch_size("Reader") / (world_size * batch_size))
    return gdvl


def fast_collate(batch):
    """
    collate function
    
    :param batch: batch images
    :return: (tensor, targets): tensor for images, and labels of images
    """
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    width = imgs[0].size[0]
    height = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, height, width), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


class PrefetchedWrapper:
    """
    Prefetched Wrapper
    """
    def prefetched_loader(self,loader):
        """
        Prefetched loader function
        
        :param loader: loading data
        :return: (tensor, targets): tensor for images, and labels of images
        """
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float()
                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield current_input, current_target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            current_input = next_input
            current_target = next_target

        yield current_input, current_target

    def __init__(self, dataloader):
        """
        Init function
        
        :param dataloader: loading data
        """
        self.dataloader = dataloader
        self.epoch = 0

    def __iter__(self):
        """
        Iterator function
        
        :return: PrefetchedWrapper.prefetched_loader(self.dataloader): wrapper of prefetched loader
        """
        if (self.dataloader.sampler is not None and
            isinstance(self.dataloader.sampler,
                       torch.utils.data.distributed.DistributedSampler)):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self, self.dataloader)

def get_pytorch_train_loader(data_path, batch_size, workers=5, \
        _worker_init_fn=None, input_size=224):
    """
    train loader function
    
    :param data_path: image data path
    :param batch_size: batch size in training phase
    :param workers: how much workers we use
    :param _worker_init_fn: initialize worker function
    :param input_size: image size
    :return: (PrefetchedWrapper(train_loader), len(train_loader))
            : prefetcehd wrapper of training loader, and length of training loader
    """
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                ]))

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, worker_init_fn=_worker_init_fn, \
            pin_memory=True, sampler=train_sampler, collate_fn=fast_collate)

    return PrefetchedWrapper(train_loader), len(train_loader)


def get_pytorch_val_loader(data_path, batch_size, workers=5, _worker_init_fn=None, input_size=224):
    """
    validation loader function
    
    :param data_path: image data path
    :param batch_size: batch size in training phase
    :param workers: how much workers we use
    :param _worker_init_fn: initialize worker function
    :param input_size: image size
    :return: (PrefetchedWrapper(val_loader), len(val_loader)
            : prefetcehd wrapper of validation loader, and length of validation loader
    """
    valdir = os.path.join(data_path, 'val')
    val_dataset = datasets.ImageFolder(
            valdir, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                ]))

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
            collate_fn=fast_collate)

    return PrefetchedWrapper(val_loader), len(val_loader)
