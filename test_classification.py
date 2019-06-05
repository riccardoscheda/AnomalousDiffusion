import os
import numpy as np
import matplotlib as plt
import cv2
import pandas as pd

import classification as cl
import fronts as fr

filepath = "Data/"
test_image =  cv2.imread( filepath + "images/1.png")
im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

#######################################################
##TESTS FOR classification.py



def test_create_set():
    """
    Tests if the number of images in the set is correct
    """
    assert len(os.listdir(filepath + "images/")) == 60

def test_create_modified_images():
    """
    Tests if the number of images in the set is correct
    """
    assert len(os.listdir(filepath + "modified_images/")) == 60

def test_adaptive_contrast_enhancement():
    """
    Tests if the function returns a np.ndarray
    """
    assert isinstance(cl.adaptive_contrast_enhancement(im_gray,(10,10)), np.ndarray) == True

def test_LBP():
    """
    Tests:
    if the output histogram is normalized
    if the length of the histogram is 10
    if the output is a np.ndarray
    """
    assert isinstance(cl.LBP(im_gray), np.ndarray) == True
    assert sum(cl.LBP(im_gray)) <= 1.0
    assert sum(cl.LBP(im_gray)) >= 0.999
    assert len(cl.LBP(im_gray)) == 10

def test_Principal_components_analysis():
    """
    Tests:
    if the function returns a dataframe
    if does not return Nan values
    """
    assert isinstance(cl.Principal_components_analysis(im_gray), pd.DataFrame) == True
    assert any(np.where(np.isnan(cl.Principal_components_analysis(im_gray)))) == False

def test_classification():
    """
    Tests:
    if the output is numpy array
    if the output image is binary (there are only 0s and 1s)
    """
    binary = cl.classification(im_gray, cl.Principal_components_analysis(im_gray))
    assert isinstance(binary,np.ndarray) == True
    assert len(np.where(binary == 0 )[1]) + len(np.where(binary == 1)[1]) == len(binary)*len(binary.T)

###############################################################################
#TESTS FOR fronts.py


def test_fronts():
    """
    Tests:
    if the output is a pandas DataFrame
    """
    assert isinstance(fr.fronts(filepath + "images/1.png"), pd.DataFrame) == True



def test_make_kernel():
    """
    Tests:
    if the output of make_kernel is a numpy matrix of 0s and 1s
    """
    struct = [0,0,0,1,0,0,0]
    binary = fr.make_kernel(struct, 1)
    assert isinstance(binary, np.matrix) == True
    assert len(np.where(binary == 0 )[1]) + len(np.where(binary == 1)[1]) == len(binary)*len(binary.T)
