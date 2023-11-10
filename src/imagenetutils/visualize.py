'''
The following codes are from https://github.com/d-li14/mobilenetv2.pytorch
'''

import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
# from .misc import *

__all__ = ['make_image', 'show_batch', 'show_mask', 'show_mask_single']

# functions to show an image
def make_image(img, mean=(0,0,0), std=(1,1,1)):
    """
    image generation function
    
    :param img: target image
    :param mean: mean value of the image
    :param std: standard deviation of the image
    :return: np.transpose(npimg, (1, 2, 0)): unnormalized image
    """
    for i in range(0, 3):
        img[i] = img[i] * std[i] + mean[i]    # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def gauss(x_data,cons,mean,std):
    """
    gauss function
    refer to https://en.wikipedia.org/wiki/Gaussian_function
    
    return torch.exp(-torch.pow(torch.add(x,-b),2).div(2*c*c)).mul(a): normalized value
    """
    return torch.exp(-torch.pow(torch.add(x_data,-mean),2).div(2*std*std)).mul(cons)

def colorize(image):
    ''' 
    Converts a one-channel grayscale image to a color heatmap image 
    
    :param x: image
    :return: cl: color heatmap image
    '''
    if image.dim() == 2:
        torch.unsqueeze(image, 0, out=image)
    if image.dim() == 3:
        color_image = torch.zeros([3, image.size(1), image.size(2)])
        color_image[0] = gauss(image,.5,.6,.2) + gauss(image,1,.8,.3)
        color_image[1] = gauss(image,1,.5,.3)
        color_image[2] = gauss(image,1,.2,.3)
        color_image[color_image.gt(1)] = 1
    elif image.dim() == 4:
        color_image = torch.zeros([image.size(0), 3, image.size(2), image.size(3)])
        color_image[:,0,:,:] = gauss(image,.5,.6,.2) + gauss(image,1,.8,.3)
        color_image[:,1,:,:] = gauss(image,1,.5,.3)
        color_image[:,2,:,:] = gauss(image,1,.2,.3)
    return color_image

def show_batch(images, mean=(2, 2, 2), std=(0.5,0.5,0.5)):
    """
    batch images showing function
    
    :param images: batch images
    :param mean: mean value of the image
    :param std: standard deviation of the image
    """
    images = make_image(torchvision.utils.make_grid(images), mean, std)
    plt.imshow(images)
    plt.show()


def show_mask_single(images, mask, mean=(2, 2, 2), std=(0.5,0.5,0.5)):
    """
    batch images with mask showing function
    
    :param images: batch images
    :param mask: mask
    :param mean: mean value of the image
    :param std: standard deviation of the image
    """
    im_size = images.size(2)

    # save for adding mask
    im_data = images.clone()
    for i in range(0, 3):
        im_data[:,i,:,:] = im_data[:,i,:,:] * std[i] + mean[i]    # unnormalize

    images = make_image(torchvision.utils.make_grid(images), mean, std)
    plt.subplot(2, 1, 1)
    plt.imshow(images)
    plt.axis('off')

    mask_size = mask.size(2)
    mask = tuple([upsampling(mask, scale_factor=im_size/mask_size)])

    mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask.expand_as(im_data)))
    plt.subplot(2, 1, 2)
    plt.imshow(mask)
    plt.axis('off')

def show_mask(images, masklist, mean=(2, 2, 2), std=(0.5,0.5,0.5)):
    """
    mask showing function
    
    :param images: batch images
    :param masklist: target masks
    :param mean: mean value of the image
    :param std: standard deviation of the image
    """
    im_size = images.size(2)

    # save for adding mask
    im_data = images.clone()
    for i in range(0, 3):
        im_data[:,i,:,:] = im_data[:,i,:,:] * std[i] + mean[i]    # unnormalize

    images = make_image(torchvision.utils.make_grid(images), mean, std)
    plt.subplot(1+len(masklist), 1, 1)
    plt.imshow(images)
    plt.axis('off')

    for i, maskdata in enumerate(masklist):
        mask = maskdata.data.cpu()
        # for b in range(mask.size(0)):
        #     mask[b] = (mask[b] - mask[b].min())/(mask[b].max() - mask[b].min())
        mask_size = mask.size(2)
        # print('Max %f Min %f' % (mask.max(), mask.min()))
        mask = tuple([upsampling(mask, scale_factor=im_size/mask_size)])
        # mask = colorize(upsampling(mask, scale_factor=im_size/mask_size))
        # for c in range(3):
        #     mask[:,c,:,:] = (mask[:,c,:,:] - Mean[c])/Std[c]

        # print(mask.size())
        mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask.expand_as(im_data)))
        # mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask), Mean, Std)
        plt.subplot(1+len(masklist), 1, i+2)
        plt.imshow(mask)
        plt.axis('off')



# x = torch.zeros(1, 3, 3)
# out = colorize(x)
# out_im = make_image(out)
# plt.imshow(out_im)
# plt.show()
