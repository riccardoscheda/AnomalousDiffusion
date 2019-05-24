import os
import numpy as np
import matplotlib as plt
import cv2
import pandas as pd
import classification as cl

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
    Tests if the function return a np.ndarray
    """
    test_image =  cv2.imread("images/0.png")
    im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    assert isinstance(cl.adaptive_contrast_enhancement(im_gray,(10,10)), np.ndarray) == True

def test_LBP():
    """
    Tests:
    if the output histogram is normalized
    if the output is a np.ndarray
    """
    test_image =  cv2.imread("images/1.png")
    im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    assert isinstance(cl.LBP(im_gray), np.ndarray) == True
    assert sum(cl.LBP(im_gray)) == 1.0

def test_Principal_components_analysis():
    """
    Test if the function return a dataframe
    """
    test_image =  cv2.imread("images/1.png")
    im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    assert isinstance(cl.Principal_components_analysis(im_gray), pd.DataFrame) == True
    assert any(np.where(np.isnan(cl.Principal_components_analysis(im_gray)))) == False
