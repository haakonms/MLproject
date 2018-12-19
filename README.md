# Road Segmentation 
Second project of CS 433 Machine Learning

This includes
a clear ReadMe le explaining how to reproduce your setup, including precise training, prediction and
installation instructions if additional libraries are used - the same way as if you would ideally instruct
a new teammate to start working with your code.

## Libraries
* Keras
* 

## How to run
The file used to replicate the prediction of our best F1-score download the file "weights.hdf5" *HERE ADD LINK*. The size of the weights are approx. 300MB so it might take some time.
When the file is downloaded put it in the same repository as the file run.ipynb and run all cells in this jupyter notebook.
It is also possible to save the predicted images to a repository.
The resulting file "predictions.csv" contains our predictions.




## How to train.
Open train.ipynb, set paths to images to train on and select name of output. 
Parameters for fitting and image augmentation can be adjusted.
To train the data we strongly recommend using Google CoLab or a robust GPU. 
The fit generator has been run with several epochs, but due to runtime errors we had to run the cell several times on repeat. 

## Files in the repository



Authors: Haakon Melgaard Sveen and Haakon Skirsta Grini
