# Road Segmentation 
Second project of CS 433 Machine Learning.

Our final best F1-score was 0.891, Username: Hakon, ScoreID: 25281.

This ReadMe explains in detail which libraries that is needed, and how to run the files that gives the same F1-score as abovementioned as well as how to run the training files that lead to this model.

## Libraries
* Keras 
    * Used for fitting and predicting using convolutional neural networks.    
    * Based on TensorFlow
    * First install TensorFlow using ```"$ pip3 install tensorflow"```
    * Then install  Keras using ```"$ pip3 install keras"```
    
Be sure to have pip3 installed as well. If running Python 2.7.9+ or 3.4+ you have it installed. Else you could run ```"$sudo apt-get install python-pip"```.

## How to run
* The file used to replicate the prediction of our best F1-score download the file [weights.hdf5](https://drive.google.com/open?id=1-IdwPV2q1kpjeCTEmK8cCWHJbitM4W2F). The size of the weights are approximately 300MB so it will take some time.
* When the file is downloaded put it in the same repository as the file run.ipynb, eventually having it in your own Google Drive. In that case, notice that you have to set the directories to either local or an own Google Drive-directory.
* Run all cells in run.ipynb.
* If wanted, predicted images can be saved by setting "save_predicted_images" to True.
* The resulting file "predictions.csv" contains our predictions.
* Google Colaboratory or a robust GPU may be used if predicting the model takes too long time.


## How to train.
* Open train.ipynb, set paths to images to train on and select name of output. Be careful about the differences in running the file locally compared to Google Drive, in having the correct directories.
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
Many of the methods in helpers.py is written by Aurelien Lucchi, ETH Zürich, and can be found[here](https://github.com/epfml/ML_course/blob/master/projects/project2/project_road_segmentation/tf_aerial_images.py). The U-Net architechture is based on Zhi Xu Hao's [implementation](https://github.com/zhixuhao/unet).
