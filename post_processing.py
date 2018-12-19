import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import helpers as h
import os,sys
from PIL import Image



#method used to remove single points from array coresponding to several images
def remove_singles(array,n_imgs,img_shape):
    copy = array.copy()
    for i in range(n_imgs):
        copy[i*(img_shape**2):(i+1)*(img_shape**2)] = take_isolated(copy[i*(img_shape**2):(i+1)*(img_shape**2)], img_shape)
    return copy 


#method used to call find_cliques several times by splitting an array consisting of several images into smaller batches
def remove_small_cliques(array,n_imgs,img_shape, min_clique):
    copy = array.copy()
    for i in range(n_imgs):
        #method performed on part coresponding to one image
        copy[i*(img_shape**2):(i+1)*(img_shape**2)] = find_clique(copy[i*(img_shape**2):(i+1)*(img_shape**2)], img_shape, min_clique)
    return copy 

#flip points that are surrounded by points of another type
def take_isolated(array, image_shape):
    new = array.copy()
    for i in range(len(new)):
        road = False
        if(new[i]==1):
            road = True
        row = i%image_shape
        col = i//image_shape
        score = 0
        #loop through all neighbouring points and add to score if they have different prediction than current point
        for j in range(row-1,row+2):
            for k in range(col-1,col+2):
                if ((j>=image_shape) or (j <0) or(k <0) or (k>=image_shape)):
                    score +=1
                elif((road and new[k*image_shape+j] ==0) or (not road and new[k*image_shape+j] ==1)):
                    score += 1
        #if score is 8 then all surrounding points are predicted differently
        if(score >=8):
            if(road):
                new[col*image_shape+row] = 0
            else:                
                new[col*image_shape+row] = 1
    return new
 
#method used to append sets to other sets in neighbourhood    
def add_neighbour_sets(neighbours, start, stop,imgage_shape):
    step =1
    if(start>stop):
        step *= -1
    for col in range(start,stop,step):
        for row in range(start,stop,step):
            if(neighbours[col*image_shape+row] is not None): 
                for j in range(row-1,row+2):
                    for k in range(col-1,col+2):
                        if(neighbours[k*image_shape+j] is not None):
                            neighbours[col*image_shape+row].update(neighbours[k*image_shape+j]) 
    return neighbours 
                           

#method used to find how many connecting road squares every square has 
def find_clique(array, image_shape, min_size):
    #initialize several arrays
    neighbours = []
    clique_size = array.copy()
    new = array.copy()
    #loop through every predicted point 
    for i in range(len(array)):
        #if a point at an edge is predicted to be road we make a set for this point and add itself to it
        if(array[i] == 1):
            row = i%image_shape
            col = i//image_shape
            neighbours.append(set())
            neighbours[i].add(i)
            #if road is at the edge of image we add -1 to the set of the point
            if((row>=image_shape-1) or (row<=0) or (col>=image_shape-1) or (col<=0)):
                neighbours[i].add(-1)
                #large value if on edge of image
        else:
            #points not predicted to be road does not get a set as it is not part of a road-clique
            neighbours.append(None)
   
    #add every neighbouring set to the set of the point being checked. Points at the "end" of a clique will have the entire clique as its set
    neighbours = add_neighbour_sets(neighbours, 1,image_shape-1,image_shape)
    #do the same thing the other way around to ensure that all points has its entire clique in its set
    neighbours = add_neighbour_sets(neighbours, image_shape-2,0,image_shape)
    
    #set clique size for every point
    for i in range(len(array)):
        #points that are not road does not have a clique
        if(neighbours[i] is None):
            clique_size[i] =0
        #points that are part of a road going towards edge get a large clique size
        elif(-1 in neighbours[i]):
            clique_size[i] =100
            #dont change
        else: 
            #other points get the size of their clique, if this size is too small then these points are predicted to not be road
            clique_size[i] = len(neighbours[i])
            if(clique_size[i]<min_size):
                new[i]=0
    #array where small cliques are removed is returned       
    return new
