"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss

Credits: Aurelien Lucchi, ETH Zürich
"""

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
import os,sys
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,Conv2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
from keras.layers import LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Helper functions

def make_model(img_shape,pad):
    #make sequential model
    model = Sequential()
    window_size = img_shape+ 2*pad
    pool_size = (2, 2)
    #Sier noe om størrelse på det som går inn
    input_shape = (window_size, window_size, 3)
    # 64 5x5 filters
    model.add(Conv2D(64, (5,5),
                     border_mode="same",
                     input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))


    model.add(Conv2D(256, (3, 3), 
                     border_mode="same"
                   ))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), 
                     border_mode="same"
                   ))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))



    #model.add(Conv2D(256, (3, 3), 
    #                 #border_mode="same"
     #               ))
    #model.add(Activation("relu"))
    #model.add(MaxPooling2D(pool_size=pool_size))


    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss = "binary_crossentropy",
                 optimizer ="adam",
                 metrics =["accuracy"])

    return model
