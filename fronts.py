import cv2
import pandas as pd
import numpy as np
import pylab as plt


import itertools as it
from collections import Counter
from functools import partial


def fronts(path,output_name):
    """
    Takes the longest border inside an image and saves in a text file
    the (x,y) coordinates.
    Return also the dataframe of the coordinates
    The input image is modifief by morphological transformation in order to have a smoother border

    --------
    Parameters
    path : the path of the image in the directory
    output_name : the name of the output file

    References
    [1] https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
    [2] https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

    """


    ret, thresh = cv2.threshold(path, 100, 255, 0)
    struct = [0,0,0,1,1,1,0,0,0]
    kernel = make_kernel(struct,20)
    kernel = np.array(kernel,np.uint8)

    erode = cv2.erode(path,kernel,iterations = 1)

    contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lencontours = np.array([len(x) for x in contours])
    sel = [x in np.sort(lencontours)[-1:] for x in lencontours]
    maxcontours = np.array(contours)
    maxcontours = maxcontours[sel]
    #cv2.drawContours(path, maxcontours, -1, (0,255,0), 3)
    maxcontours = list(it.chain.from_iterable(maxcontours))
    maxcontours = list(it.chain.from_iterable(maxcontours))
    maxcontours = np.array(maxcontours)
    coordinates = pd.DataFrame(maxcontours)
    np.savetxt(output_name, coordinates.values, fmt='%d')

    return coordinates

def make_kernel(struct,length):
    """
    Makes the Structuring Element for the morphological operations
    """
    kernel = []
    for i in range(length):
        kernel.append(struct)
    return np.matrix(kernel)
