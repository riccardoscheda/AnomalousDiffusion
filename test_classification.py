import os
import numpy as np

import classification as cl

def test_create_set():
    """
    Tests if the number of images in the set is correct
    """
    assert len(os.listdir(cl.outfolder)) == cl.n_images

def test_create_modified_images():
    """
    Tests if the number of images in the set is correct
    """
    assert len(os.listdir(cl.modified_images)) == cl.n_images

def test_adaptive_contrast_enhancement():
    """
    Tests if the function return a np.ndarray
    """
    assert isinstance(cl.adaptive_contrast_enhancement(cl.image[0],(10,10)), np.ndarray) == True

def test_LBP():
    """
    Tests:
    if the output histogram is normalized
    if the output is a np.ndarray
    """
    assert isinstance(cl.LBP(cl.image[0]), np.ndarray) == True
    assert sum(cl.LBP(cl.image[0])) == 1
