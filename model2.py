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

def make_model(window_size, classes):
    #make sequential model
    model = Sequential()

    pool_size = (2, 2)
    #Sier noe om størrelse på det som går inn
    input_shape = (window_size, window_size, 3)
    # 64 5x5 filters
        
    # Size of pooling area for max pooling

    reg = 1e-6 # L2 regularization factor (used on weights, but not biases)


    model.add(Convolution2D(64, 5, 5, # 64 5x5 filters
                            border_mode='same',
                            input_shape=input_shape
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, # 128 3x3 filters
                            border_mode='same'
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, 3, 3, # 256 3x3 filters
                            border_mode='same'
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, 3, 3, # 256 3x3 filters
                            border_mode='same'
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, W_regularizer=l2(reg)
                        )) # Fully connected layer (128 neurons)
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss = "binary_crossentropy",
             optimizer ="adam",
             metrics =["accuracy"])

    return model
