import os
import numpy as np
import matplotlib as plt
import cv2
import pandas as pd
import classification as cl



test_image =  cv2.imread("images/1.png")
im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)


def test_create_set():
    """
    Tests if the number of images in the set is correct
    """
    assert len(os.listdir("images/")) == 60

def test_create_modified_images():
    """
    Tests if the number of images in the set is correct
    """
    assert len(os.listdir("modified_images/")) == 60

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
    if the output is a binary numpy array
    """
    assert isinstance(cl.classification(cl.Principal_components_analysis(im_gray)),np.ndarray) == True
