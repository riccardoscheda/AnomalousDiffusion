import cv2
import os # To perform path manipulations
from skimage.feature import local_binary_pattern # Local Binary Pattern function
import matplotlib.pyplot as plt # For plotting
import numpy as np
from nd2reader import ND2Reader
import pandas as pd

#creating output directories
outfolder= "images/"
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
