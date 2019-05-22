import os
import classification as cl
import numpy as np

def test_create_set():
    """
    Tests if the number of images in the set is correct
    """
    assert len(os.listdir("images/")) == cl.n_images

def test_create_modified_images():
    """
    Testa if the number of images in the set is correct
    """
    assert len(os.listdir("modified_images/")) == cl.n_images

def test_adaptive_contrast_enhancement():
    """
    Tests if the number of images in the set is correct
    """
    assert isinstance(cl.adaptive_contrast_enhancement(cl.image[0],(10,10)), np.ndarray) == True
