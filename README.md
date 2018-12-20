# Road Segmentation 
Second project of CS 433 Machine Learning.

Our final best F1-score was 0.877, Username: kjoerda, ScoreID: XXXXX

This includes
a clear ReadMe le explaining how to reproduce your setup, including precise training, prediction and
installation instructions if additional libraries are used - the same way as if you would ideally instruct
a new teammate to start working with your code.

## Libraries
* Keras 
    * Used for fitting and predicting using convolutional neural networks.    
    * Based on TensorFlow
    * First install TensorFlow using ```"$ pip3 install tensorflow"```
    * Then install  Keras using "$ pip3 install keras"
    
Be sure to have pip3 installed as well. If running Python 2.7.9+ or 3.4+ you have it installed. Else you could run "$sudo apt-get install python-pip".

## How to run
* The file used to replicate the prediction of our best F1-score download the file ["weights.hdf5"](https://drive.google.com/open?id=1-4xn3jNalba3myY4BKHee0Ri6j-vB7OZ). The size of the weights are approx. The weights are around 300MB so it will take some time.
* When the file is downloaded put it in the same repository as the file run.ipynb, eventually having it in your own Google Drive.
* Run all cells in this jupyter notebook.
* If wanted, predicted images can be saved by setting "save_predicted_images" to True.
* The resulting file "predictions.csv" contains our predictions.
* Google Colaboratory may be used if predicting the model takes too long time.



## How to train.
* Open train.ipynb, set paths to images to train on and select name of output. 
* Parameters for fitting and image augmentation can be adjusted.
* To train the data we strongly recommend using Google CoLab or a robust GPU. 
* The fit generator has been run with several epochs, but due to runtime errors we had to run the cell several times on repeat. 

## Files in the repository
* train.ipynb - Used to train model on training set, and output weights of model.
* run.ipynb - Used to create a submission based on a previously trained model.
* helpers.py - Several methods used at different points in the code.
* save_and_load.py - methods used for either loading data from repositories or to save data to repositories.
* image_augmentation.py - methods used for the training data generator
* model.py - file containing the model class, makes use of the unet design.
* post_processing.py - file containing post processing methods that have been used at various points in the project.


Authors: Haakon Melgaard Sveen and Haakon Skirstad Grini.

## Credits
Many of the methods in helpers.py is written by Aurelien Lucchi, ETH Zürich,  [HERE](https://github.com/epfml/ML_course/blob/master/projects/project2/project_road_segmentation/tf_aerial_images.py). The U-Net architechture is based on Zhi Xu Hao's implementation, found [HERE](https://github.com/zhixuhao/unet).
