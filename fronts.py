import cv2
import pandas as pd
import numpy as np
import pylab as plt
from scipy import stats


import itertools as it
from collections import Counter
from functools import partial

import classification as cl
def fronts(path):
    """
    Takes the longest border inside an image and saves in a text file
    the (x,y) coordinates.
    Return also the dataframe of the coordinates
    The input image is modified by morphological transformation in order to have a smoother border

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
    #apply adaptive histogram histogram_equalization
    imgray = cl.adaptive_contrast_enhancement(imgray)
    #make it binary
    ret, thresh = cv2.threshold(imgray, 100, 255, 0)

    #create the kernel for the morphological transformations
    struct = [0,0,0,1,1,1,1,1,1,1,0,0,0]
    kernel = make_kernel(struct,100)
    kernel = np.array(kernel,np.uint8)

    #using erosion to enhance the fronts
    erode = cv2.erode(thresh,kernel,iterations = 1)

    #finding all the contours and select the longest one
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lencontours = np.array([len(x) for x in contours])
    sel = [x in np.sort(lencontours)[-1:] for x in lencontours]
    maxcontours = np.array(contours)
    maxcontours = maxcontours[sel]

    image_with_fronts = cv2.drawContours(imgray, maxcontours, -1, (0,255,0), 3)

    #making maxcontours as array so it can ben putted in a dataframe
    maxcontours = list(it.chain.from_iterable(maxcontours))
    maxcontours = list(it.chain.from_iterable(maxcontours))
    maxcontours = np.array(maxcontours)
    coordinates = pd.DataFrame(maxcontours)
    coordinates.columns = ["x", "y"]

    return coordinates, image_with_fronts

def make_kernel(struct,length):
    """
    Makes the Structuring Element for the morphological operations
    """
    kernel = []
    #making a matrix with rows all equal to the struct
    for i in range(length):
        kernel.append(struct)
    return np.matrix(kernel)

def fast_fronts(path, outdir = "fronts/", size = 20, threshold = 127, length_struct = 10,iterations = 2, save = False):
    """
    Takes the two longest borders inside an image and saves in a text file
    the (x,y) coordinates.
    The input image is modified by morphological transformation in order to have a smoother border
    -----------

    Parameters
    ---------
    path : the path of the image in the directory
    outdir : the ouput directory where it saves the txt files
    size : the size of the window for the adaptive histogram equalization
    threshold : the value of the threshold to binarize the image
    length_struct : the length for the kernel for the morphological transformations
    iterations : the number of times the dilation is applied
    save : boolean var for saving the coordinates in a txt file
    ------------
    Returns a list with the two dataframes with the coordinates of the longest borders

    References
    [1] https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
    [2] https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

    """
    im = cv2.imread(path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    struct = [0,0,0,1,1,1,1,0,0,0]
    kernel = make_kernel(struct,length_struct)
    kernel = np.array(kernel,np.uint8)
    #apply adaptive histogram histogram_equalization
    grid_size = (int(size),int(size))
    gray = cl.adaptive_contrast_enhancement(gray, grid_size= grid_size)
    mean = np.mean(gray)
    threshold = mean + 30
    ret, thresh = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)
    ###################
    thresh[0:3,] = 255
    thresh[len(thresh)-3:len(thresh)-1,:] = 255
    thresh[:,0:10] = 255
    thresh[:,-10:] = 255

    #now i try using closing to make the cells more uniform
    cstruct = np.ones(length_struct)
    ckernel = make_kernel(cstruct,length_struct*2)
    ckernel = np.array(ckernel,np.uint8)
    dilate = cv2.dilate(thresh,ckernel,iterations = iterations)
    opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, ckernel)
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lencontours = np.array([len(x) for x in contours])
    sel = [x in np.sort(lencontours)[-2:] for x in lencontours]
    maxcontours = np.array(contours)
    maxcontours = maxcontours[sel]
    coord = maxcontours
    image_with_fronts = cv2.drawContours(thresh, maxcontours, -1, (255,255,255), 3)
    dfs = []
    #maxcontours = list(it.chain.from_iterable(maxcontours))
    coord[0] = list(it.chain.from_iterable(coord[0]))
    coord[1] = list(it.chain.from_iterable(coord[1]))

    #maxcontours = list(it.chain.from_iterable(maxcontours))
    #maxcontours = np.array(maxcontours)
    #maxcontours = pd.DataFrame(maxcontours)
    coordinates = pd.DataFrame(coord[1])

    coordinates.columns = ["x", "y"]
    #takes the left upper corner and keep what there is before
    leftup = np.min(np.where(coordinates["y"] == np.max(coordinates["y"])))
    leftdown = np.max(np.where(coordinates["y"]== np.min(coordinates["y"])))
    dx = coordinates.iloc[leftdown:leftup , :]
    dfs.append(dx)
    #takes the right upper corner and takes what there is after
    rightup = np.max(np.where(coordinates["y"] == np.max(coordinates["y"])))
    #takes not the last value but the second last because some times there are
    #problems with the lowest border

    # if len(a)>1:
    #     rightdown = a[1]
    # else:
    #     rightdown = a[0]
    sx = coordinates.iloc[rightup:   ,:]
    dfs.append(sx)
    name = path.split(".")[0]
    name = path.split("/")[-1]

    if save:
        np.savetxt(outdir + name + "_dx.txt", dx,fmt = '%d', delimiter=' ')
        np.savetxt(outdir + name + "_sx.txt", sx,fmt = '%d', delimiter=' ')

    return  dfs , maxcontours   , opening


def divide(coord):
    """
    Divides the found border in two different borders: the left one and the
    right one.
    Parameters
    ------------
    coord : pandas Dataframe which contains the coordinates of the border

    Returns (first sx and second dx) two pandas dataframes one for the left border and one for the right
    border.
    """
    coord.columns = ["x", "y"]
    #takes the left upper corner and keep what there is before
    leftup = np.min(np.where(coord["y"] == np.max(coord["y"])))
    sx = coord.iloc[:leftup + 1, :]
    #takes the right upper corner and takes what there is after
    rightup = np.max(np.where(coord["y"] == np.max(coord["y"])))
    #takes not the last value but the second last because some times there are
    #problems with the lowest border
    a = np.where(coord["y"]== np.min(coord["y"]))[0]
    if len(a)>1:
        rightdown = a[1]
    else:
        rightdown = a[0]
    dx = coord.iloc[rightup  :rightdown ,:]

    return sx, dx
