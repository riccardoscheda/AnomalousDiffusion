import os
import classification


def test_create_set():
    """
    Verifies if the number of images in the set is correct
    """
    assert len(os.listdir("images/")) == classification.n_images
