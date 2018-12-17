"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss

Credits: Aurelien Lucchi, ETH Zürich
"""

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import helpers as h
import os,sys
from PIL import Image

# Helper functions

def make_patches(patch_size,imgs,gt_imgs,step_size,padding):
    # Extract patches from input images
    img_patches = [img_crop_padded(imgs[i], patch_size, patch_size,padding, step_size) for i in range(len(imgs))]
    gt_patches = [img_crop2(gt_imgs[i], patch_size, patch_size,step_size) for i in range(len(imgs))]
    # Linearize list of patches
    #shape is 10*625 (10 images, cut up into 625 images with 16*16)
    img_patches=np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])    
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    return img_patches, gt_patches

def add_rotated_imgs(img_patches,Y,rotations):
    original_picture_count = len(img_patches)
    new_imgs = img_patches.copy()
    new_Y = Y.copy()
    for i in range(1,rotations):
        flip_img = img_patches.copy()
        for j in range (original_picture_count):
            flip_img[j] = np.rot90(img_patches[j], i)
        new_imgs = np.concatenate((new_imgs, flip_img), axis=0)
        new_Y = np.concatenate((new_Y, Y.copy()), axis=0)
    return new_imgs, new_Y

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):#400/16 = 25 -> 25 iterations over each axis 400*400 image becomes 25*25 images
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def img_crop2(im, w, h, step):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight-h+step,step):#400/16 = 25 -> 25 iterations over each axis 400*400 image becomes 25*25 images
        for j in range(0,imgwidth-w+step, step):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches
def img_crop_padded(im, w, h, pad, step):
    
    list_patches = []

    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    im = np.lib.pad(im, ((pad, pad), (pad, pad), (0,0)), 'reflect')
    is_2d = len(im.shape) < 3
    for i in range(pad,imgheight-h+step+pad,step):#400/16 = 25 -> 25 iterations over each axis 400*400 image becomes 25*25 images
        for j in range(pad,imgwidth-w+step+pad, step):
            if is_2d:
                im_patch = im[j-pad:j+w+pad, i-pad:i+h+pad]
            else:
                im_patch = im[j-pad:j+w+pad, i-pad:i+h+pad, :]
            list_patches.append(im_patch)
    return list_patches