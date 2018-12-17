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
import random 
from keras import backend as K


# Helper functions

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract features for a given image
def extract_img_features(filename):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([ extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    return X



def value_to_class(v, foreground_threshold):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
def split_data(x, y, ratio, seed=1):
    assert(len(x)==len(y))
    lists = list(zip(x, y))
    random.shuffle(lists)  
    x, y = zip(*lists)
    #select a cutting point based on ratio
    cut = int(round(len(x)*ratio))
    x_test = x[cut:]
    x_train = x[:cut]
    y_test = y[cut:]
    y_train = y[:cut]    
    return x_train, x_test, y_train, y_test



# Convert array of labels to an image

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def scores(pred,truth):
    pred = np.nonzero(pred)[0]
    truth = np.nonzero(truth)[0]
    TP = len(list(set(truth) & set(pred)))
    FP = len(pred)-TP
    FN = len(truth)-TP
    Precision = TP / (TP+FP+K.epsilon())
    Recall = TP / (TP+FN+K.epsilon())
    F1 = 2*Recall*Precision/(Recall+Precision+K.epsilon())
    return F1, Precision, Recall

def load_train_images(n,directory):
    image_dir = directory+ "images/"
    files = os.listdir(image_dir)
    n = min(n, len(files)) # Load maximum 100 images
    print("Loading " + str(n) + " images")
    imgs = [load_image(image_dir + files[i]) for i in range(n)]
    gt_dir = directory + "groundtruth/"
    print("Loading " + str(n) + " ground truth images")
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
    return imgs, gt_imgs