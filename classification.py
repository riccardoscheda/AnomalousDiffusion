import cv2
import os # To perform path manipulations
from skimage.feature import local_binary_pattern # Local Binary Pattern function
import matplotlib.pyplot as plt # For plotting
import numpy as np
from nd2reader import ND2Reader
import pandas as pd

#creating output directories
outfolder = "images/"
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

modified_images = "modified_images/"
if not os.path.exists(modified_images):
    os.makedirs(modified_images)


#Reading images from nd2 file
n_images = 60
image = []

with ND2Reader("../data_prova/Sham_8_2_18_Field_5.nd2") as images:
    for i in range(n_images):
        image.append(np.asmatrix(images[i]).astype('uint16'))

#creating the set of images i will modify : maybe is not necessary to save them
def create_set():
    for i in range(n_images):
        plt.imsave(outfolder+str(i)+".png",image[i],cmap="gray")
       #plt.imshow(image)
      # plt.show()

#create_set()


def adaptive_contrast_enhancement(image,grid_size):
    """
    This function uses adaptive histogram equalization in order
    to enhance fronts of the cells. Returns the modified image.
    Parameters
    -----------------------
    image : image in matrix format
    grid_size : tuple with the size of the grid
    References
    ---------------------
    [1] https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
    """
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=grid_size)
    cl1 = clahe.apply(image)
    return cl1


#contrast enhancement on the images
modimg = []
grid_size = (100,100)

def create_modified_images():
    for i in range(n_images):
        im = adaptive_contrast_enhancement(image[i],grid_size)
        modimg.append(im)
        #plt.imsave("m_"+str(i)+".png",im,cmap="gray")

#create_modified_images()

# maybe is not important to save them. try not to use saved images
# maybe is better to save them because adaptive_contrast_enhancement(image, grid_size) takes too much time.


def LBP(image):
    """
    Computes the local binary pattern of an image
    Returns the normalized histogram of the local binary pattern image.
    Parameters
    --------------------------------
    image : image in matrix format
    References
    -------------------------------
    [1] http://hanzratech.in/2015/05/30/local-binary-patterns.html
    """
    radius = 1
    # Number of points to be considered as neighbours
    no_points = 8 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(image,no_points,radius,method="uniform")
    #converts lbp with integer values (maybe is not necessary)
    #lbp = [[int(item) for item in items] for items in lbp]

    #make the histogram of pixel intensities
    hist = np.unique(lbp, return_counts=True)
    hist = np.asarray(hist)

    # Normalize the histogram
    hist = hist[1,:]/sum(hist[1,:])
    return hist
