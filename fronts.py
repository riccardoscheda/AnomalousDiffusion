import cv2
import pandas as pd
import numpy as np
import pylab as plt


import itertools as it
from collections import Counter
from functools import partial


def fronts(path):
    """
    Takes the longest border inside an image and saves in a text file
    the (x,y) coordinates.
    Return also the dataframe of the coordinates
    The input image is modifief by morphological transformation in order to have a smoother border

    --------
    Parameters
    path : the path of the image in the directory
    ------------
    References
    [1] https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
    [2] https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

    """
    #reading the image
    image =  cv2.imread(path)
    #make it gray
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #make it binary
    ret, thresh = cv2.threshold(imgray, 100, 255, 0)

    #create the kernel for the morphological transformations
    struct = [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]
    kernel = make_kernel(struct,30)
    kernel = np.array(kernel,np.uint8)

    #using erosion to enhance the fronts
    erode = cv2.erode(thresh,kernel,iterations = 1)

    #finding all the contours and select the longest one
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lencontours = np.array([len(x) for x in contours])
    sel = [x in np.sort(lencontours)[-1:] for x in lencontours]
    maxcontours = np.array(contours)
    maxcontours = maxcontours[sel]

    #plt.imsave("fr0.png",cv2.drawContours(imgray, maxcontours, -1, (0,255,0), 3))

    #making maxcontours as array so it can ben putted in a dataframe
    maxcontours = list(it.chain.from_iterable(maxcontours))
    maxcontours = list(it.chain.from_iterable(maxcontours))
    maxcontours = np.array(maxcontours)
    coordinates = pd.DataFrame(maxcontours)


    return coordinates

def make_kernel(struct,length):
    """
    Makes the Structuring Element for the morphological operations
    """
    kernel = []
    #making a matrix with rows all equal to the struct
    for i in range(length):
        kernel.append(struct)
    return np.matrix(kernel)

def divide(coord):
    """
    Divides the found border in two different borders: the left one and the
    right one.
    Parameters
    ------------
    coord : pandas Dataframe which contains the coordinates of the border

    Returns two pandas dataframes one for the left border and one for the right
    border.
    """
    #takes the left upper corner and keep what there is before
    leftup = np.min(np.where(coord[1]== np.max(coord[1])))
    leftdown = np.min(np.where(coord[1]== np.min(coord[1])))
    sx = coord.iloc[leftdown:leftup, :]
    #takes the right upper corner and takes what there is after
    rightup = np.max(np.where(coord[1]== np.max(coord[1])))
    #takes not the last value but the second last because some times there are
    #problems with the lowest border
    a = np.where(coord[1]== np.min(coord[1]))[0]
    if len(a)>1:
        rightdown = a[1]
    else:
        rightdown = a[0]
    dx = coord.iloc[rightup:rightdown,:]

    return sx, dx
