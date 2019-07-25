# AnomalousDiffusion
This repository is dedicated for the project of the Patter Recognition course from the master in Applied Physics of the University of Bologna

##Pattern Recognition
In this project i'm trying to evaluate if the process of cell migration is an anomalous diffusion process.
Given a set of frames of a monolayer of cells acting migration,
in the first part of the project i try to recognize the cells from the background through a texture matching method, using the locally binary pattern (LBP) of the images. 
First the image is divided in 10000 subimages.Then for each subimage  the LBP image is computed with the method 'uniform' which recognizes the number of uniform patterns formed by the nearest neighbours of the pixels.
What we obtain is an histogram for each subimage, containing the number of uniform patterns.
So at the end we have 10000 histogram with 10 bins each.
Then i use this histograms as a dataframe with 10000 samples and 10 features.
Then i compute the PCA for this Dataframe and take the first 5 principal components.
After that i compute the Gaussian-mixture in order to classify two main clusters, which belong to the background and to the cells.
With this two clusters i label the subimages and obtain a binary image where 0 is for the cells and 1 is for the background.

To recognize the borders i use opencv which keeps the coordinates of the borders and then i save them into a dataframe.
After that in order to have smoother borders make a grid for the fronts, and for every interval i take the max value in that interval.


In the second part of the project i use the fronts of the cells in each frame found in the first part to classify the type of this
diffusion process.
The main measures for the diffusion process are the Mean Square Displacement (MSD) and the Velocity AutoCorrelation Function (VACF).

Both of these measure in an anomalous diffusion processes should have an exponential behaviour, so i fit them with an exponential function and evaluate the
 diffusion coefficent D and the exponent \alpha.

##Faster method
Using PCA and Gaussian mixture for all the images is quite slow, so i tried to implement a faster method to do so.
This method is based on thresholding and morphological operations on the images, such as dilation, opening and closing.
This method can be resumed in:
- Using the adaptive histogram equalization on the image
- Thresholding the image to make it binary
- Using dilation and opening to reduce noise in the centre of the frames and to make the fronts smoother
- Track the fronts with OpenCV
- Save the coordinates of the fronts in file txt

## andif Application
To do so in a fast way there is the plumbum application 'andif' which allow to do this from command line.
####Read the images from nd2 file
if you want to read the images from a nd2 file and save them you can use the command 'andif read <N>', where '<N>' is the number of frames you want to save.

