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

def remove_singles(array,n_imgs,img_shape):
    copy = array.copy()
    for i in range(n_imgs):
        copy[i*(img_shape**2):(i+1)*(img_shape**2)] = take_isolated(copy[i*(img_shape**2):(i+1)*(img_shape**2)], img_shape)
    return copy 

def remove_small_cliques(array,n_imgs,img_shape, min_clique):
    copy = array.copy()
    for i in range(n_imgs):
        copy[i*(img_shape**2):(i+1)*(img_shape**2)] = find_clique(copy[i*(img_shape**2):(i+1)*(img_shape**2)], img_shape, min_clique)
    return copy 

def take_isolated(array, image_shape):
    new = array.copy()
    change = True
    while(change):
        change = False
        for i in range(len(new)):
            road = False
            if(new[i]==1):
                road = True
            row = i%image_shape
            col = i//image_shape
            #print("col: ",col, "row: ", row, "Road: ",array[i])
            score = 0
            for j in range(row-1,row+2):
                for k in range(col-1,col+2):
                    if ((j>=image_shape) or (j <0) or(k <0) or (k>=image_shape)):
                        score +=1
                    elif((road and new[k*image_shape+j] ==0) or (not road and new[k*image_shape+j] ==1)):
                        score += 1
            if(score >=8):
                change = True
                if(road):
                    new[col*image_shape+row] = 0
                else:                
                    new[col*image_shape+row] = 1
    return new
            
def find_clique(array, image_shape, min_size):
    neighbours = []
    clique_size = array.copy()
    new = array.copy()
    for i in range(len(array)):
        if(array[i] == 1):
            row = i%image_shape
            col = i//image_shape
            neighbours.append(set())
            neighbours[i].add(i)
            if((row>=image_shape-1) or (row<=0) or (col>=image_shape-1) or (col<=0)):
                neighbours[i].add(-1)
                #large value if on edge of image
        else:
            neighbours.append(None)
   
    for col in range(1,image_shape-1):
        for row in range(1,image_shape-1):
            if(neighbours[col*image_shape+row] is not None): 
                for j in range(row-1,row+2):
                    #if(col>20):
                     #   print("tjong")
                    for k in range(col-1,col+2):
                        if(neighbours[k*image_shape+j] is not None):
                            neighbours[col*image_shape+row].update(neighbours[k*image_shape+j]) 
    #backwards
    for col in range(image_shape-2,0,-1):
        for row in range(image_shape-2,0,-1):
            if(neighbours[col*image_shape+row] is not None): 
                for j in range(row-1,row+2):
                    for k in range(col-1,col+2):
                        if(neighbours[k*image_shape+j] is not None):
                            neighbours[col*image_shape+row].update(neighbours[k*image_shape+j]) 
    for i in range(len(array)):
        if(neighbours[i] is None):
            clique_size[i] =0
        elif(-1 in neighbours[i]):
            clique_size[i] =100
            #dont change
        else: 
            clique_size[i] = len(neighbours[i])
            if(clique_size[i]<min_size):
                new[i]=0
            
    return new
