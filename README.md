# AnomalousDiffusion

In this project we try to evaluate if the process of cell migration is an anomalous diffusion process.
Given a set of frames of a monolayer of cells acting migration,
in the first part of the project we try to recognize the cells from the background through a texture matching method, using the locally binary pattern (LBP) of the images.
First the image is divided in 10000 subimages.Then for each subimage  the LBP image is computed with the method 'uniform' which recognizes the number of uniform patterns formed by the nearest neighbours of the pixels.
What we obtain is an histogram for each subimage, containing the number of uniform patterns.
So at the end we have 10000 histogram with 10 bins each.
Then we use this histograms as a dataframe with 10000 samples and 10 features.
Then we compute the PCA for this Dataframe and take the first 5 principal components.

After that we compute the Gaussian-mixture in order to classify two main clusters, which belong to the background and to the cells.
With this two clusters we label the subimages and obtain a binary image where 0 is for the cells and 1 is for the background.

To recognize the borders we use opencv which keeps the coordinates of the borders and then we save them into a dataframe.
After that in order to have smoother borders make a grid for the fronts, and for every interval we take the max value in that interval.


In the second part of the project we use the fronts of the cells in each frame found in the first part to classify the type of this
diffusion process.
The main measures for the diffusion process are the Mean Square Displacement (MSD) and the Velocity AutoCorrelation Function (VACF).

Both of these measure in an anomalous diffusion processes should have an exponential behaviour, so we fit them with an exponential function and evaluate the
 diffusion coefficent D and the exponent \alpha.

## Faster method

Using PCA and Gaussian mixture for all the images is quite slow, so we tried to implement a faster method to do so.
This method is based on thresholding and morphological operations on the images, such as dilation, opening and closing.
This method can be resumed in:
- Using the adaptive histogram equalization on the image
- Thresholding the image to make it binary
- Using dilation and opening to reduce noise in the centre of the frames and to make the fronts smoother
- Track the fronts with OpenCV
- Save the coordinates of the fronts in file txt


To do so in a fast way there is the plumbum application `andif` which allow to do this from command line.
## andif Application
## Installation
to install the application clone the repository [AnomalousDiffusion](https://github.com/riccardoscheda/AnomalousDiffusion) and use pip:
```
git clone https://github.com/riccardoscheda/AnomalousDiffusion
cd AnomalousDiffusion
pip install -r requirements.txt
pip install --editable andif
```


## Usage
When installed you can use it from command line using `andif  <subcommand>`. The main commands are:

###### Read the images from nd2 file
if you want to read the images from a nd2 file and save them you can use the command

```
andif read <file.nd2> <frames> <field>
```

 where:
 - `<file.nd2>` is the nd2 file you want to read;
 - `<frames>` is the number of frames you want to save
- `<field>` is the number of fields of view you want to save.
Since this command is thought to read a lot of images in different directories, you can use this command in the parent directory using the flag `--all`, and it will save the images for all the directories with the nd2 file in the directory `images/`.


###### Modify the images with histogram equalization
If you want to modify an image with adaptive histogram equalization you can use the command `andif modify <image>` and it saves the modified image in a new directory named `modified_images/`

If you want to modify all the images in a directory, use
```
cd <directory>
andif modify .
```


###### Binarize the images
This command binarize the images classifying two main labels, using PCA and Gaussian-mixture algorithm.
Use the command `andif label <image>` or `andif label .` to binarize all the images in the current directory. It will save the labelled image in the directory `labelled_images/`



###### Track the borders
In this project we had to track the borders from a layer of cells, so the command `andif fronts <image>` will track the longest border in the image which should correspond to the center of the image, i.e. the background, and save the x and y coordinates in a txt file.
To do this for all the images use `andif fronts .` .This command will save the txt files in the directory `fronts/`


###### Divide the front
To divide the front found with the command `fronts`, you can use the command `andif divide`, which takes the txt file with the coordinates of the front and divide it in left and right borders, and save them in two different txt files in the directory `divided_fronts/`.
To do this for all the fronts use `andif divide .`.

###### All in one
If you don't need intermediate steps and you want immediately the fronts you can use `andif fast <directory>` which uses the second method to binarize the images.
`<directory>` has to be a directory with inside the subdirectory `images/`:
```
<directory/>
            images/
```
if you want to use the command for all the directories you have to use `andif fast .` in the parent directory with the directories with the subdirectories `images/`. It will search the directory `images/` in all the directories.
```
<parent folder>
                <dir1/>
                        images/
                <dir2/>     
                        images/
                <dir3/>     
                        images/
```

Since in some nd2 files there are more than one field of view, saving all the images in png format is slow and inefficient. So there is the command `andif faster <file.nd2>` which takes the data directly from the nd2 files and will track the borders for all the fields of view.
To do this for all the directories use `andif faster <file.nd2> --all` in the parent directory with all the directories with the nd2 files and it will save the coordinates of the different frames in the file `coordinates.txt`, which correspond to the x and y coordinates for the two borders, left and right.

###### Areas
The command `andif areas` will compute the areas between the left and right borders and normalize it with respect to the area found in the first frame. You have to use this command in the parent directory with the directories with the different images.

###### Mean Squared Displacement
The command `andif msd` will compute the MSD. You have to use this command in the parent directory with the directories with the different images.

###### Fit
The command `andif fit` will fit the msd. You have to use this command in the parent directory with the directories with the different images. It will save a txt file with the fitted parameters D which is the Diffusion coefficent and the exponent alpha:
MSD = Dt^{alpha}
