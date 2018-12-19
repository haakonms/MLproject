
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import helpers as h
import os,sys
import skimage.io as io
from PIL import Image

#this method is used in method beneath
def load_image(infilename):
    return mpimg.imread(infilename)

#method to load testing images
def load_test_images():
    root_dir = "test_set_images/"
    directory = root_dir
    # Get filenames and images for all the 50 submission images
    image_dir = [directory + "test_{}/".format(i) for i in range(1, 51)]
    filenames = [fn for imdir in image_dir for fn in os.listdir(imdir)]
    test_images = [load_image(image_dir[i] + filenames[i]) for i in range(0,50)]
    return test_images

#method to save predicted images
def save_result(save_path,npyfile):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)


#method used to make a deliverable csv file
def save_submission(final_pred, submission_filename, patch_size = 16):
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i in range(final_pred.shape[0]):
            for j in range(final_pred.shape[1]):
                for k in range(final_pred.shape[2]):
                    name = '{:03d}_{}_{},{}'.format(i+1, j * patch_size, k * patch_size, int(final_pred[i,j,k]))
                    f.write(name + '\n')    
